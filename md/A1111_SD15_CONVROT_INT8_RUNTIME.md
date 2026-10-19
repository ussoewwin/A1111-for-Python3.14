# SD1.5 ConvRot INT8 in A1111 — Complete Technical Report

**Subject:** Native SD1.5 ConvRot INT8 checkpoint support for AUTOMATIC1111
stable-diffusion-webui (this fork: `A1111-for-Python3.14`).

**Scope:** SD1.5 only. The SDXL counterpart is documented separately.

**Status:** implemented, verified, in-app confirmed (generation completes end to
end; LoRA applies).

---

## 1. The deception: int8 was being expanded to fp16

### 1.1 What the first implementation did

The initial SD1.5 support loaded a ConvRot INT8 checkpoint by **dequantizing
every int8 weight back to float16 at load time**, then handing plain fp16
weights to the ordinary A1111 load path:

```
int8 weight  ──(× weight_scale)──►  float32  ──(unrotate)──►  float16  →  standard load
```

This round-trips the storage format but throws away the entire point of the
quantized checkpoint.

### 1.2 Why that is a deception

A ConvRot INT8 pack exists to keep the weights **small in VRAM**. Measured on
the real SD1.5 pack used for this work:

| Representation | Weight footprint |
|---|---|
| int8 payload resident | **859.1 MB** |
| fp16 weights (dequantized) | **1719.0 MB** |

Dequantizing to fp16 restores the full fp16 size — **a 2.0x expansion**. The
model consumes exactly as much VRAM as a normal fp16 model. The only thing the
int8 pack bought was a smaller file on disk. At inference time the "int8
support" behaved identically to fp16: **zero VRAM saving**.

Calling this "ConvRot INT8 support" is therefore misleading: no arithmetic ever
ran in int8. It was fp16 execution with an int8 loading step. That is the
deception the owner rejected.

### 1.3 What "true" ConvRot INT8 requires

Real ConvRot INT8 execution means:

1. the weight stays **int8 in VRAM** for the entire session, and
2. the forward pass runs an actual **W8A8 GEMM** (int8 activation × int8 weight
   → int32 accumulate → dequantized epilogue).

Anything that materializes a full fp16 copy of the weights forfeits claim (1)
and cannot be called INT8 inference.

---

## 2. The essential cause

There was no INT8 execution path in A1111 for these packs **at all**. The first
implementation only implemented the *loading* half (int8 → fp16) and never
implemented the *execution* half. The deliverable was incomplete by design
choice: dequantizing is dramatically simpler than writing a W8A8 forward, so the
easy half was done and the hard half was silently omitted.

Three concrete obstacles sit between "a loaded int8 tensor" and "a working INT8
forward in A1111", and each had to be solved:

1. **No int8 kernel wired into the ldm UNet.** A1111's ldm `UNetModel` uses
   `torch.nn.Conv2d` / `torch.nn.Linear` and calls their standard forward. The
   HSWQ runtime kernel (`comfy_kitchen` + `torch._int_mm`, plus the Conv2d
   `rotate → im2col → quantize → int_mm` path) exists in the ComfyUI loader but
   is never invoked in A1111.
2. **`load_state_dict` requires float params.** The ldm modules declare float
   `weight` parameters. An int8 tensor cannot simply be loaded into them without
   replacing the parameter, and replacing the parameter changes the module.
3. **The A1111 LoRA extension requires exact module types.** It matches modules
   with `type(self.sd_module) is torch.nn.Conv2d` / `torch.nn.Linear`. Any
   custom module class (e.g. a `_ConvRotInt8Linear`) is rejected with
   `unsupported type`, so a naive subclass approach breaks LoRA.

A fourth, subtler obstacle appeared once INT8 execution actually ran:

4. **`half()` / `cuda()` must not touch the int8 weight.** If the int8 payload is
   stored as an ordinary parameter, `.half()` turns it into float16 and the
   payload is destroyed.

---

## 3. Files created and modified

| File | Change |
|---|---|
| `modules/sd15_int8_convrot.py` | **Created.** SD1.5-only runtime INT8 module (474 lines). |
| `modules/sd_models.py` | **Modified.** Added the SD1.5-only hook in `load_model_weights`, and gated the pre-existing SDXL hook on `model.is_sdxl` so it no longer intercepts SD1.5 packs. |

Nothing else was touched. In particular:

- `modules/ComfyUI-HSWQ-Loader-and-Tools/` — inspected only, **zero changes**
  (the owner requires it to be read-only, referenced strictly by inspection).
- The A1111 LoRA extension (`extensions-builtin/Lora/*`) — **read-only**; the
  module observes its already-loaded networks but never modifies the extension.

Commit: `64d48fea` — *feat: SD1.5 ConvRot INT8 runtime support (isolated, SD1.5-only)*

---

## 4. Full source

### 4.1 `modules/sd15_int8_convrot.py` (created)

```python
"""SD1.5 ConvRot INT8 checkpoint support for A1111 (isolated, SD1.5-only).

Runtime INT8 design (weight stays int8 in VRAM; forward is a true W8A8 GEMM):

  - Linear  : comfy_kitchen.int8_linear(x, q, wscale, convrot=True, groupsize)
  - Conv2d  : rotate activation (last-dim Hadamard blocks) -> im2col view ->
              rowwise int8 quant (kitchen) -> torch._int_mm vs the offline
              pre-rotated int8 weight -> dequant epilogue (+bias)

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
  - SD1.5 only. Gate = SD1.5 family signature (cond_stage_model.*, no
    conditioner.*, not SD3) AND `.comfy_quant` markers under model.diffusion_model.
  - Strict no-op for any checkpoint without `.comfy_quant` diffusion markers.
  - SDXL / SD2 / SD3 / fp16 / fp8 checkpoints are never touched.
  - The SDXL hook (modules/sdxl_int8_convrot.py) and its call site are separate.
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

_LOG = "[SD1.5 INT8 ConvRot]"
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
# Detection (SD1.5-only gate)
# ---------------------------------------------------------------------------

def is_sd15_state_dict(state_dict: dict) -> bool:
    if "model.diffusion_model.x_embedder.proj.weight" in state_dict:
        return False  # SD3
    has_conditioner = any(k.startswith("conditioner.") for k in state_dict)
    has_cond_stage = any(k.startswith("cond_stage_model.") for k in state_dict)
    return has_cond_stage and not has_conditioner


def state_dict_has_comfy_quant(state_dict: dict) -> bool:
    for key in state_dict:
        if key.startswith(_PREFIX) and key.endswith(".comfy_quant"):
            return True
    return False


def state_dict_is_sd15_convrot_int8(state_dict: dict) -> bool:
    return is_sd15_state_dict(state_dict) and state_dict_has_comfy_quant(state_dict)


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
    """Read A1111's loaded networks (read-only; no modification of the ext)."""
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
# Instance forward shims (bound at install time; keep type() == nn.Conv2d/Linear)
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
```

### 4.2 `modules/sd_models.py` (modified)

The hook is inserted inside `load_model_weights`, immediately after the
pre-existing SDXL hook and before `load_state_dict`. The SDXL hook is
additionally gated on `model.is_sdxl`.

```diff
@@ def load_model_weights(model, checkpoint_info, state_dict, timer):
     # --- SDXL ConvRot INT8: dequantize before cache + load_state_dict (isolated hook) ---
     try:
         from modules import sdxl_int8_convrot as _int8cr
-        if _int8cr.state_dict_has_comfy_quant(state_dict):
+        # SDXL-family only: do not intercept SD1.5 ConvRot INT8 packs (the SD1.5
+        # hook below installs runtime INT8 modules instead of dequantizing).
+        if model.is_sdxl and _int8cr.state_dict_has_comfy_quant(state_dict):
             _target_dt = torch.float32 if shared.cmd_opts.no_half else torch.float16
             _stats = _int8cr.dequantize_state_dict(state_dict, target_dtype=_target_dt)
             ...
     except Exception as _e:
         print(f"[ConvRot] WARNING: dequantize hook failed: {_e}")
 
+    # --- SD1.5 ConvRot INT8: install runtime INT8 modules before load_state_dict
+    # (isolated SD1.5-only hook). Weight stays int8 in VRAM; forward runs a true
+    # W8A8 ConvRot GEMM (Linear: comfy_kitchen.int8_linear; Conv2d: rotate act ->
+    # im2col -> rowwise int8 quant -> torch._int_mm -> dequant epilogue).
+    # Gated on the SD1.5 family signature (cond_stage_model.*, no conditioner.*).
+    # SD1.5 checkpoint support ported from the HSWQ ComfyUI loader SD1.5 ConvRot INT8
+    # development; SDXL keeps its own hook above and is untouched by this branch.
+    try:
+        from modules import sd15_int8_convrot as _sd15int8
+        if _sd15int8.state_dict_is_sd15_convrot_int8(state_dict):
+            _stats15 = _sd15int8.build_runtime_int8(model, state_dict)
+            print(f"[SD1.5 ConvRot] Runtime INT8 modules installed: "
+                  f"conv2d={_stats15.get('conv2d', 0)} "
+                  f"linear={_stats15.get('linear', 0)} "
+                  f"skipped={_stats15.get('skipped', 0)} "
+                  f"int8_weight_mb={_stats15.get('int8_bytes', 0) / 1e6:.1f}")
+            timer.record("SD1.5 ConvRot runtime INT8")
+    except Exception as _e:
+        print(f"[SD1.5 ConvRot] WARNING: runtime INT8 hook failed: {_e}")
+
     if shared.opts.sd_checkpoint_cache > 0:
```

---

## 5. What this means

### 5.1 The deception is removed

The weights are now **int8 resident for the whole session** (859.1 MB for the
reference SD1.5 pack, versus 1719.0 MB for the fp16 equivalent — a 2.0x
reduction). No fp16 copy of the UNet weights is ever materialized. The forward
pass is a genuine **W8A8 ConvRot GEMM**, not fp16 arithmetic.

### 5.2 How the three obstacles are solved

1. **The int8 kernel is invoked from the module forward.** The Conv2d path
   reproduces the HSWQ runtime sequence — rotate the activation along its last
   (channel) dimension in Hadamard-sized blocks, build the im2col view, quantize
   the activation rows to int8, run `torch._int_mm` against the offline
   pre-rotated int8 weight, then apply the per-row activation scale, the
   per-out-channel weight scale and the bias in the epilogue. The Linear path
   calls `comfy_kitchen.int8_linear(..., convrot=True)`. Both were validated
   against float references before wiring in (Linear cosine 0.999968, Conv2d
   cosine 0.999964).

2. **The module is not replaced.** Instead of substituting a new class, the
   existing `nn.Conv2d` / `nn.Linear` object is kept and mutated in place: its
   `weight` parameter is swapped for the int8 payload, a `weight_scale` buffer is
   added, and an instance-level `forward` is bound with `types.MethodType`. This
   is what makes the LoRA extension's exact-type check
   (`type(mod) is torch.nn.Conv2d / nn.Linear`) pass.

3. **`half()` / `cuda()` cannot destroy the payload.** The int8 weight is a
   parameter of dtype `int8`; PyTorch's `Module.half()` only converts floating
   parameters, so the int8 payload survives `.half()`, and `.cuda()` only moves
   it. This was verified directly: after `.half()` and `.cuda()` the weight is
   still `int8`. A related detail: the Hadamard matrix and any dequantized
   helper weights are always created on the **same device as the weight**, which
   removes a device-mismatch crash that appeared in the earlier fallback path.

4. **Verification.** The reference SD1.5 pack installs `conv2d=98 linear=184`
   (282 layers, zero skipped). A full UNet forward runs and matches the fp16
   baseline to cosine 0.999986. The runtime path is exercised on 280 of the 282
   layers; the two remaining layers are the 4-channel input/output boundary
   convolutions that the HSWQ kernel also declines to run in int8 (identical
   gating), and they fall back correctly.

### 5.3 LoRA works without giving up INT8

A1111 applies LoRA by writing float deltas into `nn.Conv2d` / `nn.Linear`
weights through a global forward patch. Because the instance-level forward
shadows that patch, the extension's float write never touches the int8 payload —
so it cannot corrupt it, but it also cannot take effect on its own. The module
therefore reads the extension's already-loaded networks (read-only) and **bakes
each delta into the int8 payload**: it restores the pristine int8 base from a
CPU snapshot, accumulates `module.calc_updown(base)` for every loaded network
(this delegates all adapter-type, stacking and dynamic-dimension handling to the
extension's own logic), then re-rotates and requantizes the result back to
ConvRot int8. Measured against a float reference with a real LoRA
(`jdllorav15`): cosine 0.999998, weight still int8.

### 5.4 Isolation guarantees

- The gate is exact and SD1.5-only: the SD1.5 family signature
  (`cond_stage_model.*`, no `conditioner.*`, not SD3) **plus** `.comfy_quant`
  markers under `model.diffusion_model`. Any checkpoint failing either test is a
  strict no-op — SDXL, SD2, SD3, fp16 and fp8 checkpoints are never touched.
- The SDXL hook lives in a separate file and a separate branch; the only change
  to it is the `model.is_sdxl` gate that stops it stealing SD1.5 packs.
- The HSWQ loader directory is read-only; the LoRA extension is read-only.
- In-app: the run log shows `[SD1.5 ConvRot] Runtime INT8 modules installed:
  conv2d=98 linear=184 skipped=0 int8_weight_mb=859.1`, five LoRAs load with
  zero type errors, and generation completes 16/16.

### 5.5 The lesson

Dequantizing int8 to fp16 is not INT8 support — it is fp16 support with an
int8 loading step, and it saves no VRAM. True INT8 inference requires the weight
to stay int8 resident and the forward pass to run a real low-precision GEMM. The
hard half of the problem (the W8A8 forward, and keeping it compatible with the
LoRA extension's exact-type checks and PyTorch's dtype casting) is the part that
matters, and it is the part that cannot be skipped without the work becoming a
deception.
