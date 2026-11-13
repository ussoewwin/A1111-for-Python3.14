"""ConvRot INT8 IP-Adapter support for A1111 (isolated, IP-Adapter-only).

modules/controlnet_int8_convrot.py covers the ControlNet graph that
extensions-builtin/sd-webui-controlnet/scripts/cldm.py builds (PlugableControlModel).
An IP-Adapter / PuLID pack carries a *second* graph - ``image_proj`` and
``ip_adapter`` - which scripts/ipadapter/ipadapter_model.py builds with plain float
``load_state_dict`` calls. A ConvRot INT8 pack therefore failed there:

    RuntimeError: Error(s) in loading state_dict for PuLIDEncoder:
        Unexpected key(s) in state_dict: "body.0.weight_scale", "body.0.comfy_quant", ...

This module gives the IP-Adapter graph the same runtime INT8 treatment the ControlNet
graph already has:

  - Linear only. The int8 payload plus its per-output-channel scale are installed in
    place and an instance-level ``forward`` is bound, so ``type(mod) is nn.Linear``
    still holds and no caller sees a different class.
  - CUDA fp16/bf16 activations go through comfy_kitchen.int8_linear(convrot=True,
    groupsize) exactly like the ControlNet path. float32 activations (PuLID's
    image_proj is fed float32 by IPAdapterModel) are cast to bfloat16 for the kernel
    and cast back, which is the activation dtype the same pack sees on the ComfyUI
    side. Other dtypes / CPU use a dequantised weight that is cached per
    (device, dtype) instead of being rebuilt on every denoise step.
  - Sidecar keys (``*.weight_scale`` / ``*.comfy_quant``) are stripped before the
    float load, so the float load can never overwrite the installed int8 payload.

Separation:
  - IP-Adapter graph only. modules/controlnet_int8_convrot.py is imported for its
    pure Hadamard / comfy_quant helpers and is not modified; the UNet runtimes
    (modules/sd15_int8_convrot.py, modules/sdxl_int8_convrot.py) are not imported.
  - Gate: a layer counts as INT8 only when the pack carries its ``.comfy_quant``
    marker with ``format == int8_tensorwise``, its weight is int8, and the scale
    exists. Any pack without such markers is a strict no-op.
  - Layers whose comfy_quant marker exists but whose module is absent from the graph
    are dequantised to float rather than left as a raw int8 tensor, because
    load_state_dict would otherwise cast int8 into a float parameter and silently
    corrupt the model.
"""

from __future__ import annotations

import logging
import types

import torch
import torch.nn.functional as F

from modules.controlnet_int8_convrot import (
    _get_layer_conf,
    _unrotate_last_dim,
    build_hadamard,
    decode_comfy_quant_conf,
)

logger = logging.getLogger(__name__)

_LOG = "[IPAdapter INT8 ConvRot]"

_DEQ_CACHE_ATTR = "_hswq_deq_cache"


# ---------------------------------------------------------------------------
# comfy_quant metadata / payload discovery
# ---------------------------------------------------------------------------

def split_convrot_int8(state_dict: dict) -> tuple[dict, dict]:
    """Split an IP-Adapter sub state dict into (float_state_dict, layers).

    ``layers`` maps layer prefix -> (int8 weight, weight scale, groupsize).
    ``float_state_dict`` keeps every key that is not part of an int8 payload, so it
    can be handed to a plain float ``load_state_dict`` / ``To_KV``.
    """
    if not state_dict:
        return state_dict or {}, {}

    layers: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {}
    for key in list(state_dict.keys()):
        if not key.endswith(".comfy_quant"):
            continue
        prefix = key[: -len(".comfy_quant")]
        qw = state_dict.get(prefix + ".weight")
        sc = state_dict.get(prefix + ".weight_scale")
        if not isinstance(qw, torch.Tensor) or qw.dtype != torch.int8 or sc is None:
            continue
        try:
            conf = _get_layer_conf(decode_comfy_quant_conf(state_dict[key]), prefix)
        except Exception:
            conf = {}
        fmt = conf.get("format", "")
        if fmt and fmt != "int8_tensorwise":
            logger.warning(
                "%s %s is format=%s - only int8_tensorwise is supported; the layer "
                "is left untouched", _LOG, prefix, fmt,
            )
            continue
        gs = int(conf.get("convrot_groupsize", 256) or 256)
        layers[prefix] = (qw, sc, gs)

    if not layers:
        return state_dict, {}

    float_sd = {}
    for key, value in state_dict.items():
        if key.endswith(".comfy_quant"):
            continue
        stripped = False
        for prefix in layers:
            if key == prefix + ".weight" or key == prefix + ".weight_scale":
                stripped = True
                break
        if not stripped:
            float_sd[key] = value
    return float_sd, layers


def layer_shapes(layers: dict) -> dict:
    """prefix -> (out_features, in_features) for the int8 layers of a sub graph.

    To_KV builds its Linears from the *float* state dict, so a fully quantised
    ip_adapter sub graph contributes no float key at all. These shapes are all To_KV
    needs to create the modules; the int8 payload is installed right after.
    """
    return {
        prefix: (int(qw.shape[0]), int(qw.shape[1]))
        for prefix, (qw, _sc, _gs) in layers.items()
    }


def state_dict_has_convrot_int8(state_dict: dict) -> bool:
    """True if the sub state dict carries at least one int8_tensorwise layer."""
    if not state_dict:
        return False
    for key in state_dict:
        if key.endswith(".comfy_quant"):
            prefix = key[: -len(".comfy_quant")]
            qw = state_dict.get(prefix + ".weight")
            if isinstance(qw, torch.Tensor) and qw.dtype == torch.int8:
                return True
    return False


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------

def _dequant(mod, dtype, device) -> torch.Tensor:
    gs = int(mod._hswq_gs)
    out_f, in_f = mod.weight.shape
    w = mod.weight.to(device=device, dtype=torch.float32) * mod.weight_scale.to(
        device=device, dtype=torch.float32
    ).reshape(out_f, -1)
    if in_f % gs == 0:
        w = _unrotate_last_dim(w, build_hadamard(gs, device=device, dtype=torch.float32), gs)
    return w.to(dtype)


def _dequant_cached(mod, dtype, device) -> torch.Tensor:
    key = (str(device), dtype)
    cached = getattr(mod, _DEQ_CACHE_ATTR, None)
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == key:
        return cached[1]
    weight = _dequant(mod, dtype, device)
    setattr(mod, _DEQ_CACHE_ATTR, (key, weight))
    return weight


def _int8_forward(mod, x):
    """comfy_kitchen ConvRot int8 linear, or None when this call cannot use it."""
    if not x.is_cuda or not mod.weight.is_cuda or mod.weight.device != x.device:
        return None
    gs = int(mod._hswq_gs)
    if mod.in_features % gs != 0:
        return None
    src_dtype = x.dtype
    if src_dtype not in (torch.float16, torch.bfloat16):
        if src_dtype != torch.float32:
            return None
        x = x.to(torch.bfloat16)
    from comfy_kitchen.backends import cuda as kc

    scale = mod.weight_scale.to(device=x.device, dtype=torch.float32)
    bias = None
    if mod.bias is not None:
        bias = mod.bias.detach().to(device=x.device, dtype=x.dtype)
    out = kc.int8_linear(
        x, mod.weight, scale, bias,
        out_dtype=x.dtype, convrot=True, convrot_groupsize=gs,
    )
    if out.dtype != src_dtype:
        out = out.to(src_dtype)
    return out


def _linear_forward_shim(self, x):
    try:
        out = _int8_forward(self, x)
        if out is not None:
            return out
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "%s int8 kernel failed on %s (%s); float fallback",
            _LOG, getattr(self, "_hswq_name", "?"), e,
        )
    bias = None
    if self.bias is not None:
        bias = self.bias.to(device=x.device, dtype=x.dtype)
    return F.linear(x, _dequant_cached(self, x.dtype, x.device), bias)


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def _resolve_linear(model, prefix):
    """Find the nn.Linear for a state-dict prefix inside an already built module.

    Exact dotted-path match first (image_proj: ``body.0``). To_KV flattens keys into
    a ModuleDict (``1.to_k_ip`` -> ``to_kvs.1_to_k_ip``), so a flattened tail match is
    accepted as the second form.
    """
    named = dict(model.named_modules())
    mod = named.get(prefix)
    if isinstance(mod, torch.nn.Linear) and type(mod) is torch.nn.Linear:
        return mod
    flat = prefix.replace(".", "_")
    for name, candidate in named.items():
        if not (isinstance(candidate, torch.nn.Linear) and type(candidate) is torch.nn.Linear):
            continue
        if name.split(".")[-1] == flat:
            return candidate
    return None


def _install_convrot_linear(mod, name, qw, sc, gs):
    out_f = qw.shape[0]
    mod.weight = torch.nn.Parameter(qw.to(torch.int8).clone(), requires_grad=False)
    mod.weight_scale = sc.reshape(-1).float()[:out_f].reshape(-1, 1).clone()
    mod._hswq_kind = "linear"
    mod._hswq_gs = int(gs)
    mod._hswq_name = name
    setattr(mod, _DEQ_CACHE_ATTR, None)
    mod.forward = types.MethodType(_linear_forward_shim, mod)


def install_runtime_int8(model, layers: dict) -> tuple[dict, set]:
    """Install the int8 payload into the modules of an already built graph."""
    stats = {"linear": 0, "skipped": 0, "int8_bytes": 0}
    installed: set[str] = set()
    for prefix, (qw, sc, gs) in layers.items():
        mod = _resolve_linear(model, prefix)
        if mod is None:
            stats["skipped"] += 1
            continue
        _install_convrot_linear(mod, prefix, qw, sc, gs)
        installed.add(prefix)
        stats["linear"] += 1
        stats["int8_bytes"] += qw.numel()
    return stats, installed


def prepare_state_dict(model, state_dict: dict):
    """Install int8 modules on ``model`` and return (loadable_state_dict, stats).

    Returns the unchanged state dict and ``None`` when the graph has no ConvRot INT8
    layer, so callers can keep the strict float load.
    """
    float_sd, layers = split_convrot_int8(state_dict)
    if not layers:
        return state_dict, None

    stats, installed = install_runtime_int8(model, layers)
    if not installed:
        return state_dict, None

    # Layers that carry an int8 marker but have no module in this graph must not reach
    # load_state_dict as int8, otherwise the float load casts them silently.
    for prefix, (qw, sc, gs) in layers.items():
        if prefix in installed:
            continue
        w = qw.to(torch.float32) * sc.to(torch.float32).reshape(qw.shape[0], -1)
        if qw.shape[1] % gs == 0:
            w = _unrotate_last_dim(w, build_hadamard(gs, dtype=torch.float32), gs)
        float_sd[prefix + ".weight"] = w
        stats["dequantised"] = stats.get("dequantised", 0) + 1

    if stats["skipped"]:
        logger.warning(
            "%s %d int8 layer(s) had no module in the graph and were dequantised to float",
            _LOG, stats["skipped"],
        )
    stats["installed"] = sorted(installed)
    logger.info(
        "%s Runtime INT8 modules installed: linear=%d skipped=%d int8_weight_mb=%.1f",
        _LOG, stats["linear"], stats["skipped"], stats["int8_bytes"] / 1e6,
    )
    return float_sd, stats


def load_prepared(model, state_dict: dict, stats: dict):
    """load_state_dict for a graph that had int8 payloads installed.

    The installed layers keep their int8 Parameter and their ``.weight`` key was
    stripped, so they can never appear in the state dict again. ``strict=False`` is
    therefore required, and the result is verified here: any missing key that is not
    exactly an installed ``<prefix>.weight``, and any unexpected key, is an error.
    """
    result = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {prefix + ".weight" for prefix in stats.get("installed", ())}
    bad_missing = [k for k in result.missing_keys if k not in allowed_missing]
    if bad_missing or result.unexpected_keys:
        raise RuntimeError(
            f"IP-Adapter ConvRot INT8 load mismatch: "
            f"missing={bad_missing} unexpected={result.unexpected_keys}"
        )
    return result
