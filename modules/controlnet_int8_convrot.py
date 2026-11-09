"""ConvRot INT8 ControlNet support for A1111 (isolated, ControlNet-only).

The UNet runtime INT8 hooks (modules/sd15_int8_convrot.py, modules/sdxl_int8_convrot.py)
are installed from modules/sd_models.py, which only ever sees the UNet. A ControlNet is
loaded by the ControlNet extension on its own path
(extensions-builtin/sd-webui-controlnet/scripts/cldm.py, PlugableControlModel), so a
ConvRot INT8 ControlNet reached that path with its int8 payload untouched:
`load_state_dict` cast the int8 weights to float, the ControlNet ran on garbage weights
and the sampling produced NaN.

This module gives the ControlNet the same runtime INT8 treatment:

  - Linear: comfy_kitchen.int8_linear(x, q, wscale, convrot=True, groupsize)

The Conv2d / Linear modules are not replaced by new classes: the `weight` Parameter is
swapped in place for the int8 payload and an instance-level `forward` is bound, so
`type(mod) is torch.nn.Linear` still holds (the A1111 LoRA extension's exact-type checks
keep working). LoRA deltas are baked into the int8 payload the same way the UNet runtime
does it (restore the pristine int8 base, accumulate `module.calc_updown(base)` for every
loaded network, requantize back into ConvRot int8) - the instance-level forward shadows
the class forward, so the LoRA extension's global float patch never runs on these modules.

Separation:
  - ControlNet only. The UNet runtimes are not imported and not modified.
  - Gate: the state dict must carry `.comfy_quant` markers for ControlNet layers
    (int8_tensorwise) and the ControlNet layer names must exist in the model.
  - Strict no-op for any ControlNet without ConvRot INT8 markers.

Canonical reference for the ComfyUI side of the same feature:
  ComfyUI-HSWQ-Loader-and-Tools/nodes/hswq_load_convrot_int8_controlnet.py
  (it forces a float architecture dtype and passes the comfy_quant-aware ops class so the
  int8 payload is consumed; here the equivalent is done in A1111's own runtime).
"""

from __future__ import annotations

import json
import logging
import math
import types

import torch

logger = logging.getLogger(__name__)

_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

_LOG = "[ControlNet INT8 ConvRot]"

# ControlNet layer names live directly under the ControlNet module
# (input_blocks.*, middle_block.*, zero_convs.*, time_embed.*, input_hint_block.*),
# i.e. no "model.diffusion_model." prefix like the UNet. There is deliberately no
# name whitelist here: a layer counts as a candidate when the pack carries its
# comfy_quant marker and the ControlNet graph actually has that module. A name
# whitelist hid time_embed.2 for one pack and left an int8 payload to be cast to
# float by load_state_dict (NaN).


# ---------------------------------------------------------------------------
# Hadamard (same construction as the HSWQ converter / comfy_kitchen)
# ---------------------------------------------------------------------------

def build_hadamard(size: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be power of 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=torch.float32,
        device=device,
    )
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    h = h / (size ** 0.5)
    if dtype != torch.float32:
        h = h.to(dtype=dtype)
    _HADAMARD_CACHE[cache_key] = h
    return h


def _rotate_last_dim(w2d, h, gs):
    o, i = w2d.shape
    return torch.matmul(w2d.reshape(o, i // gs, gs), h.t().to(w2d.dtype)).reshape(o, i)


def _unrotate_last_dim(w2d, h, gs):
    o, i = w2d.shape
    return torch.einsum("okg,gh->okh", w2d.reshape(o, i // gs, gs), h.to(w2d.dtype)).reshape(o, i)


# ---------------------------------------------------------------------------
# comfy_quant metadata
# ---------------------------------------------------------------------------

def decode_comfy_quant_conf(raw) -> dict:
    if isinstance(raw, torch.Tensor):
        raw = raw.tolist()
    if isinstance(raw, (list, tuple)):
        raw = bytes(raw).decode("utf-8")
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, dict):
        return raw
    raise TypeError(f"comfy_quant config must be dict/str/bytes, got {type(raw).__name__}")


def _get_layer_conf(conf: dict, prefix: str) -> dict:
    if not isinstance(conf, dict):
        return {}
    if "format" in conf:
        return conf
    if "layers" in conf and isinstance(conf["layers"], dict):
        lc = conf["layers"].get(prefix, {})
        if isinstance(lc, dict) and "format" in lc:
            return lc
    return conf


def _collect_layer_confs(state_dict: dict) -> dict:
    """prefix -> comfy_quant conf, for ControlNet layers with an int8 payload."""
    confs: dict[str, dict] = {}
    for key in list(state_dict.keys()):
        if not key.endswith(".comfy_quant"):
            continue
        prefix = key[: -len(".comfy_quant")]
        if (prefix + ".weight") not in state_dict:
            continue
        try:
            conf = _get_layer_conf(decode_comfy_quant_conf(state_dict[key]), prefix)
        except Exception:
            conf = {}
        fmt = conf.get("format", "")
        if fmt and fmt != "int8_tensorwise":
            logger.warning("%s %s is format=%s - only int8_tensorwise is supported; "
                           "the layer is left untouched", _LOG, prefix, fmt)
            continue
        w = state_dict[prefix + ".weight"]
        if not isinstance(w, torch.Tensor) or w.dtype != torch.int8:
            continue
        confs[prefix] = conf
    return confs


def state_dict_is_controlnet_convrot_int8(state_dict: dict) -> bool:
    return bool(_collect_layer_confs(state_dict))


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------

def _scale_on(device, scale):
    return scale if scale.device == device else scale.to(device)


def _dequant_linear(mod, dtype):
    gs = mod._hswq_gs
    o, i = mod.weight.shape
    w = mod.weight.float() * _scale_on(mod.weight.device, mod.weight_scale).reshape(o, -1).float()
    if i % gs != 0:
        return w.to(dtype)
    return _unrotate_last_dim(w, build_hadamard(gs, device=w.device, dtype=torch.float32), gs).to(dtype)


def _int8_linear_forward(mod, x):
    if x.dtype not in (torch.float16, torch.bfloat16) or not x.is_cuda:
        return None
    gs = int(mod._hswq_gs)
    if mod.in_features % gs != 0:
        return None
    from comfy_kitchen.backends import cuda as kc
    bias = mod.bias.detach() if mod.bias is not None else None
    scale = _scale_on(x.device, mod.weight_scale)
    return kc.int8_linear(
        x, mod.weight, scale, bias,
        out_dtype=x.dtype, convrot=True, convrot_groupsize=gs,
    )


def _fp16_linear_forward(mod, x):
    import torch.nn.functional as F
    return F.linear(x, _dequant_linear(mod, x.dtype), mod.bias)


# ---------------------------------------------------------------------------
# LoRA (bake into int8; read-only on the LoRA extension)
# ---------------------------------------------------------------------------

def _get_loaded_networks():
    try:
        import networks as _net
        return list(getattr(_net, "loaded_networks", []) or [])
    except Exception:
        return []


def _snapshot_base(mod):
    if getattr(mod, "_hswq_base_weight_cpu", None) is None:
        mod._hswq_base_weight_cpu = mod.weight.detach().to("cpu", copy=True).clone()
    if getattr(mod, "_hswq_base_scale_cpu", None) is None:
        mod._hswq_base_scale_cpu = mod.weight_scale.detach().to("cpu", copy=True).clone()


def _requantize_linear(mod, w_float):
    gs = int(mod._hswq_gs)
    w_rot = _rotate_last_dim(w_float.to(torch.float32),
                             build_hadamard(gs, device=w_float.device, dtype=torch.float32), gs)
    amax = torch.clamp(w_rot.abs().amax(dim=1), min=1e-6)
    new_scale = (amax / 127.0).reshape(-1, 1)
    q = torch.round(w_rot / new_scale).clamp(-127, 127).to(torch.int8)
    mod.weight.data.copy_(q.to(mod.weight.device))
    mod.weight_scale = new_scale.to(mod.weight_scale.device)


def _sync_lora(mod):
    nets = _get_loaded_networks()
    wanted = tuple((n.name, n.te_multiplier, n.unet_multiplier, n.dyn_dim) for n in nets)
    if getattr(mod, "_hswq_lora_state", "unset") == wanted:
        return
    lname = getattr(mod, "network_layer_name", None)
    _snapshot_base(mod)
    with torch.no_grad():
        mod.weight.data.copy_(mod._hswq_base_weight_cpu.to(mod.weight.device))
        mod.weight_scale = mod._hswq_base_scale_cpu.clone()
    if lname and nets:
        try:
            base = _dequant_linear(mod, torch.float32)
            changed = False
            for net in nets:
                module = net.modules.get(lname, None)
                if module is None or not hasattr(module, "calc_updown"):
                    continue
                try:
                    updown, ex_bias = module.calc_updown(base)
                    if updown is not None:
                        base = base + updown.to(base.dtype)
                        changed = True
                    if ex_bias is not None and mod.bias is not None:
                        mod.bias.data.add_(ex_bias.to(mod.bias.dtype))
                        changed = True
                except Exception as e:  # noqa: BLE001
                    logger.warning("%s LoRA updown failed on %s: %s", _LOG, lname, e)
            if changed:
                _requantize_linear(mod, base)
        except Exception as e:  # noqa: BLE001
            logger.warning("%s LoRA bake failed on %s: %s", _LOG, lname, e)
    mod._hswq_lora_state = wanted


def _linear_forward_shim(self, x):
    _sync_lora(self)
    try:
        out = _int8_linear_forward(self, x)
        if out is not None:
            return out
    except Exception as e:  # noqa: BLE001
        logger.warning("%s Linear runtime INT8 path failed (%s); fp16 fallback", _LOG, e)
    return _fp16_linear_forward(self, x)


def _install_convrot_linear(mod, qw, sc, gs, bias):
    mod.weight = torch.nn.Parameter(qw.to(torch.int8).clone(), requires_grad=False)
    mod.weight_scale = sc.reshape(-1).float()[: mod.weight.shape[0]].reshape(-1, 1).clone()
    if bias is not None and mod.bias is not None:
        mod.bias.data.copy_(bias.detach())
    mod._hswq_kind = "linear"
    mod._hswq_gs = int(gs)
    mod._hswq_lora_state = None
    mod._hswq_base_weight_cpu = None
    mod._hswq_base_scale_cpu = None
    mod.forward = types.MethodType(_linear_forward_shim, mod)


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def install_runtime_int8_modules(model, state_dict) -> dict:
    """Swap the int8 payload into the ControlNet's Linear modules."""
    confs = _collect_layer_confs(state_dict)
    name_to_mod = dict(model.named_modules())
    stats = {"linear": 0, "skipped": 0, "int8_bytes": 0, "layers": len(confs)}

    for prefix, conf in confs.items():
        mod = name_to_mod.get(prefix)
        if mod is None:
            stats["skipped"] += 1
            continue
        gs = int(conf.get("convrot_groupsize", 256) or 256)
        qw = state_dict.get(prefix + ".weight")
        sc = state_dict.get(prefix + ".weight_scale")
        b = state_dict.get(prefix + ".bias")
        if qw is None or sc is None:
            stats["skipped"] += 1
            continue
        if isinstance(mod, torch.nn.Linear) and type(mod) is torch.nn.Linear:
            _install_convrot_linear(mod, qw, sc, gs, b)
            stats["linear"] += 1
            stats["int8_bytes"] += qw.numel()
        else:
            stats["skipped"] += 1

    return stats


def prepare_controlnet_state_dict(model, state_dict):
    """Install the runtime INT8 Linear modules and strip the int8 sidecar keys.

    Returns the state dict that must be handed to `load_state_dict`: the int8 layers
    keep their installed payload (their float keys are removed so the float load cannot
    overwrite it), every other tensor is passed through unchanged.
    """
    if not state_dict:
        return state_dict, None
    confs = _collect_layer_confs(state_dict)
    if not confs:
        return state_dict, None

    stats = install_runtime_int8_modules(model, state_dict)
    if stats["linear"] == 0:
        return state_dict, None

    reduced = {}
    name_to_mod = dict(model.named_modules())
    installed = set(prefix for prefix in confs if prefix in name_to_mod)

    # Any quantised layer that could not be installed must not reach
    # load_state_dict as a raw int8 tensor (it would be cast to float and
    # silently corrupt the model). Dequantise it to float instead.
    for prefix in confs:
        if prefix in installed:
            continue
        qw = state_dict.get(prefix + ".weight")
        sc = state_dict.get(prefix + ".weight_scale")
        if qw is None or sc is None:
            continue
        gs = int(confs[prefix].get("convrot_groupsize", 256) or 256)
        w = qw.float() * sc.reshape(-1, 1).float()[: qw.shape[0]]
        if qw.shape[1] % gs == 0:
            w = _unrotate_last_dim(w, build_hadamard(gs, dtype=torch.float32), gs)
        state_dict[prefix + ".weight"] = w
        stats["dequantised"] = stats.get("dequantised", 0) + 1
    for k, v in state_dict.items():
        if k.endswith(".comfy_quant"):
            continue
        stripped = False
        for prefix in installed:
            for suffix in (".weight", ".weight_scale"):
                if k == prefix + suffix:
                    stripped = True
                    break
            if stripped:
                break
        if stripped:
            continue
        reduced[k] = v

    logger.info("%s Runtime INT8 modules installed: linear=%d skipped=%d int8_weight_mb=%.1f",
                _LOG, stats["linear"], stats["skipped"], stats["int8_bytes"] / 1e6)
    return reduced, stats
