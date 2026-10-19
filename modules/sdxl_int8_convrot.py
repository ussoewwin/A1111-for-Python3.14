"""SDXL ConvRot INT8 checkpoint support for A1111 (isolated, SDXL-only).

Runtime INT8 design (weight stays int8 in VRAM; forward is a true W8A8 GEMM):

  - Linear  : comfy_kitchen.int8_linear(x, q, wscale, convrot=True, groupsize)
  - Conv2d  : rotate activation (last-dim Hadamard blocks) -> im2col view ->
              rowwise int8 quant (kitchen) -> torch._int_mm vs the offline
              pre-rotated int8 weight -> dequant epilogue (+bias)

This replaces the previous SDXL dequantize-to-fp16 hook, which blew the int8
payload back up to fp16 (no VRAM saving).

The UNet Conv2d / Linear modules are NOT replaced by new classes: their
`weight` Parameter is swapped in place for the int8 payload and an
instance-level `forward` is bound. This keeps `type(mod) is torch.nn.Conv2d /
torch.nn.Linear`, which the A1111 LoRA extension requires for its exact-type
checks, while still executing true runtime INT8.

LoRA: because the instance-level forward shadows the class forward, the LoRA
extension's global forward patch never runs on these modules, so its float
delta application never corrupts the int8 weight. Instead this module reads the
A1111 LoRA extension's already-loaded networks (read-only) and bakes each
delta into the int8 payload: it restores the pristine int8 base, accumulates
`module.calc_updown(base)` for every loaded network (handles all adapter types,
stacking and dyn_dim), then requantizes the result back into ConvRot int8.

Separation:
  - SDXL only. Gate = SDXL family signature (`conditioner.` prefix present or
    `model.is_sdxl`) AND `.comfy_quant` markers under model.diffusion_model.
  - Strict no-op for any checkpoint without `.comfy_quant` diffusion markers.
  - SD1.5 / SD2 / SD3 / fp16 / fp8 checkpoints are never touched.
  - The SD1.5 hook (modules/sd15_int8_convrot.py) is a separate file and a
    separate branch in sd_models.py.
  - No other file is modified: the LoRA extension is read-only here.
"""

from __future__ import annotations

import json
import logging
import math
import types

import torch

logger = logging.getLogger(__name__)

_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

_LOG = "[SDXL INT8 ConvRot]"
_PREFIX = "model.diffusion_model."


# ---------------------------------------------------------------------------
# Hadamard (matches the HSWQ converter/kitchen: kron h4 in float32, normalized)
# ---------------------------------------------------------------------------

def build_hadamard(size: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be power of 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=torch.float32, device=device,
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


def _valid_group_size(n: int, preferred: int) -> int | None:
    if n < 4:
        return None
    if preferred >= 4 and n % preferred == 0 and math.log(preferred, 4) % 1 == 0:
        return preferred
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def _rotate_last_dim(w2d, h, gs):
    o, i = w2d.shape
    return torch.matmul(w2d.reshape(o, i // gs, gs), h.t().to(w2d.dtype)).reshape(o, i)


def _unrotate_last_dim(w2d, h, gs):
    o, i = w2d.shape
    return torch.einsum("okg,gh->okh", w2d.reshape(o, i // gs, gs), h.to(w2d.dtype)).reshape(o, i)


def _rotate_weight_convrot(w, gs):
    o, i, kh, kw = w.shape
    flat = w.permute(0, 2, 3, 1).reshape(o * kh * kw, i)
    flat = _rotate_last_dim(flat, build_hadamard(gs, device=w.device, dtype=torch.float32), gs)
    return flat.view(o, kh, kw, i).permute(0, 3, 1, 2).contiguous()


def _unrotate_weight_convrot(w, gs):
    o, i, kh, kw = w.shape
    flat = w.permute(0, 2, 3, 1).reshape(o * kh * kw, i)
    flat = _unrotate_last_dim(flat, build_hadamard(gs, device=w.device, dtype=torch.float32), gs)
    return flat.view(o, kh, kw, i).permute(0, 3, 1, 2).contiguous()


def _rotate_act_lastdim_blocks(x_nhwc: torch.Tensor, h: torch.Tensor, gs: int) -> torch.Tensor:
    B, Hh, Ww, C = x_nhwc.shape
    xg = x_nhwc.reshape(B, Hh * Ww, C // gs, gs)
    hh = h.to(dtype=xg.dtype, device=xg.device)
    return torch.matmul(xg, hh).reshape(B, Hh, Ww, C)


# ---------------------------------------------------------------------------
# comfy_quant metadata decode
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


# ---------------------------------------------------------------------------
# Detection (SDXL-only gate)
# ---------------------------------------------------------------------------

def is_sdxl_state_dict(state_dict: dict) -> bool:
    """True for an SDXL-family checkpoint (conditioner.* present)."""
    if "model.diffusion_model.x_embedder.proj.weight" in state_dict:
        return False  # SD3
    return any(k.startswith("conditioner.") for k in state_dict)


def state_dict_has_comfy_quant(state_dict: dict) -> bool:
    for key in state_dict:
        if key.startswith(_PREFIX) and key.endswith(".comfy_quant"):
            return True
    return False


def state_dict_is_sdxl_convrot_int8(state_dict: dict) -> bool:
    return is_sdxl_state_dict(state_dict) and state_dict_has_comfy_quant(state_dict)


def _collect_layer_confs(state_dict: dict) -> dict:
    confs = {}
    for k in list(state_dict.keys()):
        if not (k.startswith(_PREFIX) and k.endswith(".comfy_quant")):
            continue
        prefix = k[: -len(".comfy_quant")]
        if (prefix + ".weight") not in state_dict:
            continue
        w = state_dict[prefix + ".weight"]
        if not isinstance(w, torch.Tensor) or w.dtype != torch.int8:
            continue
        try:
            confs[prefix] = _get_layer_conf(decode_comfy_quant_conf(state_dict[k]), prefix)
        except Exception:
            confs[prefix] = {}
    return confs


def _get_loaded_networks():
    try:
        import networks as _net
        return list(getattr(_net, "loaded_networks", []) or [])
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Weight helpers (per module instance)
# ---------------------------------------------------------------------------

def _scale_on(device, scale):
    return scale if scale.device == device else scale.to(device)


def _dequant_conv2d(mod, dtype):
    gs = mod._hswq_gs
    w = mod.weight.float() * _scale_on(mod.weight.device, mod.weight_scale).reshape(-1, 1, 1, 1).float()
    if mod.weight.shape[1] % gs != 0:
        return w.to(dtype)
    return _unrotate_weight_convrot(w, gs).to(dtype)


def _dequant_linear(mod, dtype):
    gs = mod._hswq_gs
    o, i = mod.weight.shape
    w = mod.weight.float() * _scale_on(mod.weight.device, mod.weight_scale).reshape(o, -1).float()
    if i % gs != 0:
        return w.to(dtype)
    return _unrotate_last_dim(w, build_hadamard(gs, device=w.device, dtype=torch.float32), gs).to(dtype)


def _int8_conv2d_forward(mod, x):
    import torch.nn.functional as F
    if x.dtype not in (torch.float16, torch.bfloat16) or not x.is_cuda or x.ndim != 4:
        return None
    if mod.groups != 1 or any(d != 1 for d in mod.dilation):
        return None
    if mod.padding_mode != "zeros":
        return None
    gs = int(mod._hswq_gs)
    B, C, H, W = x.shape
    kh, kw = mod.kernel_size
    sh, sw = mod.stride
    ph, pw = mod.padding
    if C % gs != 0:
        return None
    k_dim = kh * kw * C
    q = mod.weight
    o, ic, khw, kww = q.shape
    if (ic, khw, kww) != (C, kh, kw) or k_dim % 8 != 0 or o % 8 != 0:
        return None
    hs = (H + 2 * ph - kh) // sh + 1
    ws_ = (W + 2 * pw - kw) // sw + 1
    t = B * hs * ws_
    if t < 16:
        return None

    h = build_hadamard(gs, device=x.device, dtype=x.dtype)
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    x_rot = _rotate_act_lastdim_blocks(x_nhwc, h, gs)
    if ph or pw:
        x_rot = F.pad(x_rot, (0, 0, pw, pw, ph, ph))
    Hp, Wp = H + 2 * ph, W + 2 * pw
    row_stride = Wp * C
    img_stride = Hp * Wp * C
    im2col = x_rot.as_strided(
        (B, hs, ws_, kh, kw, C),
        (img_stride, sh * row_stride, sw * C, row_stride, C, 1),
    ).reshape(t, k_dim).contiguous()

    wmat = q.permute(2, 3, 1, 0).reshape(k_dim, o).contiguous()
    wscale = _scale_on(x.device, mod.weight_scale).reshape(o).float()

    from comfy_kitchen.backends import cuda as kc
    aq, ascale = kc.quantize_int8_rowwise(im2col)
    acc = torch._int_mm(aq.contiguous(), wmat).float()
    out = acc.mul_(ascale.float().view(-1, 1)).mul_(wscale.view(1, -1))
    if mod.bias is not None:
        out = out.add_(mod.bias.detach().float().view(1, -1))
    if x.dtype == torch.float16:
        out = out.clamp(-65504.0, 65504.0)
    out = out.to(x.dtype).view(B, hs, ws_, o).permute(0, 3, 1, 2)
    return out.contiguous()


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


def _fp16_conv2d_forward(mod, x):
    import torch.nn.functional as F
    w = _dequant_conv2d(mod, x.dtype)
    if mod.padding_mode != "zeros":
        return F.conv2d(F.pad(x, mod._reversed_padding_repeated_twice, mode=mod.padding_mode),
                        w, mod.bias, mod.stride, (0, 0), mod.dilation, mod.groups)
    return F.conv2d(x, w, mod.bias, mod.stride, mod.padding, mod.dilation, mod.groups)


def _fp16_linear_forward(mod, x):
    import torch.nn.functional as F
    return F.linear(x, _dequant_linear(mod, x.dtype), mod.bias)


# ---------------------------------------------------------------------------
# LoRA (bake into int8; read-only on the LoRA extension)
# ---------------------------------------------------------------------------

def _snapshot_base(mod):
    if getattr(mod, "_hswq_base_weight_cpu", None) is None:
        mod._hswq_base_weight_cpu = mod.weight.detach().to("cpu", copy=True).clone()
        mod._hswq_base_scale_cpu = mod.weight_scale.detach().to("cpu", copy=True).clone()


def _requantize_conv2d(mod, w_float):
    gs = int(mod._hswq_gs)
    w_rot = _rotate_weight_convrot(w_float.to(torch.float32), gs)
    amax = torch.clamp(w_rot.abs().amax(dim=(1, 2, 3)), min=1e-6)
    new_scale = (amax / 127.0)
    q = torch.round(w_rot / new_scale.reshape(-1, 1, 1, 1)).clamp(-127, 127).to(torch.int8)
    mod.weight.data.copy_(q.to(mod.weight.device))
    mod.weight_scale = new_scale.to(mod.weight_scale.device)


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
            is_conv = mod._hswq_kind == "conv2d"
            base = _dequant_conv2d(mod, torch.float32) if is_conv else _dequant_linear(mod, torch.float32)
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
                if is_conv:
                    _requantize_conv2d(mod, base)
                else:
                    _requantize_linear(mod, base)
        except Exception as e:  # noqa: BLE001
            logger.warning("%s LoRA bake failed on %s: %s", _LOG, lname, e)
    mod._hswq_lora_state = wanted


# ---------------------------------------------------------------------------
# Instance forward shims
# ---------------------------------------------------------------------------

def _conv2d_forward_shim(self, x):
    _sync_lora(self)
    try:
        out = _int8_conv2d_forward(self, x)
        if out is not None:
            return out
    except Exception as e:  # noqa: BLE001
        logger.warning("%s Conv2d runtime INT8 path failed (%s); fp16 fallback", _LOG, e)
    return _fp16_conv2d_forward(self, x)


def _linear_forward_shim(self, x):
    _sync_lora(self)
    try:
        out = _int8_linear_forward(self, x)
        if out is not None:
            return out
    except Exception as e:  # noqa: BLE001
        logger.warning("%s Linear runtime INT8 path failed (%s); fp16 fallback", _LOG, e)
    return _fp16_linear_forward(self, x)


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def _install_convrot(mod, qw, sc, gs, bias):
    mod.weight = torch.nn.Parameter(qw.to(torch.int8).clone(), requires_grad=False)
    mod.weight_scale = sc.reshape(-1).float()[: mod.weight.shape[0]].clone()
    if bias is not None and mod.bias is not None:
        mod.bias.data.copy_(bias.detach())
    mod._hswq_kind = "conv2d"
    mod._hswq_gs = int(gs)
    mod._hswq_lora_state = None
    mod._hswq_base_weight_cpu = None
    mod._hswq_base_scale_cpu = None
    mod.forward = types.MethodType(_conv2d_forward_shim, mod)


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


def install_runtime_int8_modules(model, state_dict) -> dict:
    """Swap the int8 payload into the existing Conv2d / Linear modules."""
    confs = _collect_layer_confs(state_dict)
    name_to_mod = dict(model.named_modules())
    stats = {"conv2d": 0, "linear": 0, "skipped": 0, "int8_bytes": 0}

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
        if isinstance(mod, torch.nn.Conv2d) and type(mod) is torch.nn.Conv2d:
            _install_convrot(mod, qw, sc, gs, b)
            stats["conv2d"] += 1
            stats["int8_bytes"] += qw.numel()
        elif isinstance(mod, torch.nn.Linear) and type(mod) is torch.nn.Linear:
            _install_convrot_linear(mod, qw, sc, gs, b)
            stats["linear"] += 1
            stats["int8_bytes"] += qw.numel()
        else:
            stats["skipped"] += 1

    return stats


def build_runtime_int8(model, state_dict) -> dict:
    """Entry point from sd_models.py: install runtime INT8 modules then strip the
    int8 sidecar keys so the subsequent load_state_dict only sees the plain-float
    tensors it still needs (CLIP / VAE / non-converted)."""
    confs = _collect_layer_confs(state_dict)
    stats = install_runtime_int8_modules(model, state_dict)
    for prefix in confs:
        for suffix in (".weight", ".weight_scale", ".comfy_quant",
                       ".input_scale", ".weight_scale_2", ".pre_quant_scale"):
            state_dict.pop(prefix + suffix, None)
    return stats
