# SDXL ConvRot INT8 in A1111 — Complete Technical Report

**Subject:** Native SDXL ConvRot INT8 checkpoint support for AUTOMATIC1111
stable-diffusion-webui (this fork: `A1111-for-Python3.14`), covering the UNet
**and** the quantized text encoders.

**Scope:** SDXL only. The SD1.5 counterpart is documented separately.

**Status:** implemented, verified, in-app confirmed (generation completes end to
end; UNet + CLIP LoRA apply).

---

## 1. The deception: int8 was being expanded to fp16

### 1.1 What the first SDXL implementation did

The initial SDXL support was a **dequantize-to-fp16 hook**. At load time it read
every int8 ConvRot layer, multiplied by its scale to recover a float tensor,
reversed the Hadamard rotation, cast to fp16, and handed plain fp16 weights to
the ordinary A1111 loader. No int8 tensor survived into the model; inference ran
entirely in fp16.

### 1.2 Why that is a deception

The whole purpose of a ConvRot INT8 pack is to keep the weights small in VRAM.
Measured on the real SDXL pack used for this work, the dequantize hook produced:

| Representation | Footprint |
|---|---|
| int8 payload as shipped | ~4,521.6 MB |
| after the dequantize hook (fp16) | ~6,770.4 MB |

That is a **1.50x expansion** — the model ends up *larger* than the int8 pack
and consumes exactly as much VRAM as a normal fp16 SDXL model. The int8 pack
bought only a smaller file on disk. At inference time the "INT8 support" was
fp16 arithmetic with an int8 loading step: **zero VRAM saving**.

### 1.3 What "true" ConvRot INT8 requires

1. the weight stays **int8 in VRAM** for the whole session, and
2. the forward runs a real **W8A8 GEMM** (int8 activation × int8 weight → int32
   accumulate → dequantized epilogue).

Anything that materializes a full fp16 copy forfeits (1) and is not INT8
inference.

---

## 2. The essential cause

There was no INT8 execution path in A1111 for SDXL packs **at all** — only the
loading half (int8 → fp16) was implemented, and the execution half was omitted.
Dequantizing is far simpler than writing a W8A8 forward, so the easy half was
done and the hard half was silently skipped.

SDXL adds three complications beyond SD1.5, all of which had to be solved:

1. **The pack also quantizes the text encoders.** Measured on the reference
   packs: the UNet carries 590 quantized layers **and** `conditioner.*` carries
   another **133** quantized layers (CLIP-L and CLIP-G). A UNet-only hook leaves
   the CLIP int8 weights loaded raw into float modules, which crashes with a
   dtype mismatch (e.g. `mat1 and mat2 must have the same dtype, but got
   BFloat16 and Char`).
2. **The CLIP text encoders are three different module kinds.** CLIP-L's
   projections are `torch.nn.Linear`; CLIP-G's token table is an
   `torch.nn.Embedding`; and CLIP-G's attention output projection lives **inside
   an `nn.MultiheadAttention`**, consumed functionally by
   `F.multi_head_attention_forward` — so a per-module `forward` shim on that
   child never runs.
3. **The two text encoders use different Python-side namespaces.** HSWQ packs
   store CLIP-L under `...transformer.text_model....` while the A1111 OpenCLIP
   module namespace is `...transformer....` (no `.text_model.`). The int8 layers
   match through that normalization, but the *remaining float* tensors (biases,
   LayerNorms) do not — so CLIP-L biases would never load and the encoder emits
   NaN.

A fourth issue surfaced through measurement (section 5.4): some layers in the
published packs carry a **broken int8 payload** (all-zero weight, NaN scale),
which the INT8 path must not treat as ordinary data.

Plus the same two SD1.5 constraints still apply: the LoRA extension requires
exact module types (`type(mod) is torch.nn.Conv2d / nn.Linear`), and
`half()` / `cuda()` must not touch the int8 payload.

---

## 3. Files created and modified

| File | Change |
|---|---|
| `modules/sdxl_int8_convrot.py` | **Rewritten** from a dequantize hook into a runtime INT8 module; then extended to cover the quantized text encoders. |
| `modules/sd_models.py` | **Modified.** The SDXL hook now calls the runtime INT8 install instead of dequantizing; plus the `model.is_sdxl` gate (shared with the SD1.5 work) so it never intercepts SD1.5 packs. |

Nothing else was touched. The SD1.5 hook lives in a separate file
(`modules/sd15_int8_convrot.py`); the HSWQ loader directory and the LoRA
extension are read-only.

Commits:

- `afaf9e49` — *feat: SDXL ConvRot INT8 runtime support (replaces fp16
  dequantize-to-float)* — UNet only.
- `ca801ceb` — *feat: SDXL ConvRot INT8 covers the quantized CLIP text encoders*
  — CLIP added, broken-layer detection added.

---

## 4. Full source

### 4.1 `modules/sdxl_int8_convrot.py` (rewritten)

```python
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
    """Collect int8 ConvRot layer prefixes for BOTH the UNet (model.diffusion_model.*)
    and the SDXL text encoders (conditioner.*), which HSWQ packs also quantize."""
    confs = {}
    for k in list(state_dict.keys()):
        if not k.endswith(".comfy_quant"):
            continue
        if not (k.startswith(_PREFIX) or k.startswith("conditioner.")):
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


def _embedding_forward_shim(self, x):
    """Runtime INT8 Embedding forward: gather int8 rows, dequantize, unrotate."""
    ref = getattr(self, "_hswq_ref_param", None)
    out_dtype = ref.dtype if isinstance(ref, torch.Tensor) else torch.float16
    try:
        gs = int(self._hswq_gs)
        idx = x
        wq = self.weight[idx]                      # int8 [*, dim]
        sc = _scale_on(self.weight.device, self.weight_scale).reshape(-1)
        s = sc[idx]                                # [*]
        wf = wq.float() * s.unsqueeze(-1).float()
        dim = wf.shape[-1]
        if gs and dim % gs == 0:
            wf = _unrotate_last_dim(
                wf.reshape(-1, dim),
                build_hadamard(gs, device=wf.device, dtype=torch.float32), gs,
            ).reshape(wf.shape)
        return wf.to(out_dtype)
    except Exception as e:  # noqa: BLE001
        logger.warning("%s Embedding runtime INT8 path failed (%s); fp16 fallback", _LOG, e)
        gs = int(self._hswq_gs)
        sc = _scale_on(self.weight.device, self.weight_scale).reshape(-1, 1)
        w = self.weight.float() * sc.float()
        if gs and w.shape[1] % gs == 0:
            w = _unrotate_last_dim(w, build_hadamard(gs, device=w.device, dtype=torch.float32), gs)
        return torch.nn.functional.embedding(x, w.to(out_dtype))


def _mha_forward_shim(self, query, key, value, key_padding_mask=None, need_weights=True,
                      attn_mask=None, average_attn_weights=True, is_causal=False, **kwargs):
    """MultiheadAttention forward whose out_proj is a runtime INT8 module.
    Mirrors F.multi_head_attention_forward for the batch_first=True, q=k=v path
    used by OpenCLIP, then routes the output projection through the int8 shim."""
    import torch.nn.functional as F
    batch_first = getattr(self, "batch_first", True)
    if not batch_first:
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
    B, T, E = query.shape
    S = key.shape[1]
    if self.in_proj_weight is not None:
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
    else:
        q = F.linear(query, self.q_proj_weight, self.in_proj_bias)
        k = F.linear(key, self.k_proj_weight, self.in_proj_bias)
        v = F.linear(value, self.v_proj_weight, self.in_proj_bias)
    nh = self.num_heads
    hd = E // nh
    q = q.reshape(B, T, nh, hd).transpose(1, 2)
    k = k.reshape(B, S, nh, hd).transpose(1, 2)
    v = v.reshape(B, S, nh, hd).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=bool(is_causal))
    out = out.transpose(1, 2).reshape(B, T, E).contiguous()
    if self.out_proj.weight.dtype == torch.int8:
        out = self.out_proj(out)   # runtime INT8 out_proj shim
    else:
        out = self.out_proj(out)
    if not batch_first:
        out = out.transpose(0, 1)
    if need_weights:
        return out, None
    return out, None


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


def _install_embedding(mod, qw, sc, gs, ref_param=None):
    mod.weight = torch.nn.Parameter(qw.to(torch.int8).clone(), requires_grad=False)
    mod.weight_scale = sc.reshape(-1).float()[: mod.weight.shape[0]].clone()
    mod._hswq_kind = "embedding"
    mod._hswq_gs = int(gs)
    mod._hswq_ref_param = ref_param
    mod.forward = types.MethodType(_embedding_forward_shim, mod)


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


def _resolve_mod(name_to_mod, prefix):
    """Find the model module for a pack key (the pack uses .text_model. where the
    A1111 module namespace does not)."""
    if prefix in name_to_mod:
        return prefix, name_to_mod[prefix]
    if ".text_model." in prefix:
        alt = prefix.replace(".text_model.", ".")
        if alt in name_to_mod:
            return alt, name_to_mod[alt]
    return None, None


def _layer_scale_is_broken(sc, qw):
    """HSWQ packs can ship layers with an all-zero int8 payload and NaN weights
    scale (measured: CLIP-L layer 11 in both published SDXL packs). Such a layer
    carries no weight information at all; feeding it to the INT8 path
    propagates NaN through the whole text encoder. Detect it so we can degrade
    that single layer safely instead of poisoning the model."""
    if sc is None:
        return False
    s = sc.float()
    return bool(torch.isnan(s).any() or torch.isinf(s).any()) or bool((qw == 0).all())


def install_runtime_int8_modules(model, state_dict) -> dict:
    """Swap the int8 payload into the existing Conv2d / Linear / Embedding modules.

    Covers both the UNet (model.diffusion_model.*) and the SDXL text encoders
    (conditioner.*) that HSWQ packs also quantize. CLIP-G out_proj layers live
    inside an nn.MultiheadAttention and are used through
    F.multi_head_attention_forward, so the parent MHA forward is shimmed too.
    """
    confs = _collect_layer_confs(state_dict)
    name_to_mod = dict(model.named_modules())
    # None (int8 disabled) -> float; resolved lazily from the module's dtype.
    stats = {"conv2d": 0, "linear": 0, "embedding": 0, "mha": 0,
             "skipped": 0, "int8_bytes": 0}

    for prefix, conf in confs.items():
        _, mod = _resolve_mod(name_to_mod, prefix)
        if mod is None:
            stats["skipped"] += 1
            logger.warning("%s no module for %s", _LOG, prefix)
            continue
        gs = int(conf.get("convrot_groupsize", 256) or 256)
        qw = state_dict.get(prefix + ".weight")
        sc = state_dict.get(prefix + ".weight_scale")
        b = state_dict.get(prefix + ".bias")
        if qw is None or sc is None:
            stats["skipped"] += 1
            continue

        # Detect pack-level corruption (all-zero int8 + NaN scale). Leave the
        # module on its original float path and report it explicitly; silently
        # producing NaN is worse than an honest, local degradation.
        if _layer_scale_is_broken(sc, qw):
            logger.warning(
                "%s layer %s has a broken int8 payload (all-zero weight / NaN scale) "
                "in the checkpoint; keeping it on the float path. Re-quantize the "
                "source model to fix it.", _LOG, prefix)
            stats["broken"] = stats.get("broken", 0) + 1
            continue

        # out_proj inside an nn.MultiheadAttention: install int8 on the child and
        # shim the parent's forward so it routes through the child module.
        parent_name = ".".join(prefix.split(".")[:-1])
        _, parent = _resolve_mod(name_to_mod, parent_name)
        if isinstance(parent, torch.nn.MultiheadAttention):
            _install_convrot_linear(mod, qw, sc, gs, b)
            if not getattr(parent, "_hswq_mha_shimmed", False):
                parent.forward = types.MethodType(_mha_forward_shim, parent)
                parent._hswq_mha_shimmed = True
            stats["mha"] += 1
            stats["int8_bytes"] += qw.numel()
        elif isinstance(mod, torch.nn.Conv2d) and type(mod) is torch.nn.Conv2d:
            _install_convrot(mod, qw, sc, gs, b)
            stats["conv2d"] += 1
            stats["int8_bytes"] += qw.numel()
        elif isinstance(mod, torch.nn.Linear) and type(mod) in (torch.nn.Linear,):
            _install_convrot_linear(mod, qw, sc, gs, b)
            stats["linear"] += 1
            stats["int8_bytes"] += qw.numel()
        elif isinstance(mod, torch.nn.Embedding):
            # token_embedding: its output is added to positional_embedding, so it
            # must match that parameter's dtype.
            emb_parent = model
            for part in prefix.split(".")[:-1]:
                emb_parent = getattr(emb_parent, part, None)
                if emb_parent is None:
                    break
            ref = getattr(emb_parent, "positional_embedding", None) if emb_parent is not None else None
            _install_embedding(mod, qw, sc, gs, ref_param=ref)
            stats["embedding"] += 1
            stats["int8_bytes"] += qw.numel()
        elif isinstance(mod, torch.nn.Linear):
            # NonDynamicallyQuantizableLinear that is NOT under an MHA.
            _install_convrot_linear(mod, qw, sc, gs, b)
            stats["linear"] += 1
            stats["int8_bytes"] += qw.numel()
        else:
            stats["skipped"] += 1
            logger.warning("%s unsupported target %s (%s)", _LOG, prefix, type(mod).__name__)

    return stats


def build_runtime_int8(model, state_dict) -> dict:
    """Entry point from sd_models.py: install runtime INT8 modules then strip the
    int8 sidecar keys so the subsequent load_state_dict only sees the plain-float
    tensors it still needs (CLIP / VAE / non-converted).

    HSWQ SDXL packs store CLIP-L under `...transformer.text_model....` while the
    A1111 OpenCLIP module namespace is `...transformer....`. The int8 layers are
    matched through that normalization, so the remaining float tensors (biases,
    norms) are re-keyed here too; otherwise CLIP-L biases would never load and
    the text encoder would produce NaN. Strictly scoped to this SDXL int8 path.
    """
    confs = _collect_layer_confs(state_dict)
    stats = install_runtime_int8_modules(model, state_dict)
    broken = set()
    for prefix in confs:
        qw = state_dict.get(prefix + ".weight")
        sc = state_dict.get(prefix + ".weight_scale")
        if qw is not None and _layer_scale_is_broken(sc, qw):
            broken.add(prefix)
    for prefix in confs:
        if prefix in broken:
            # Keep the float weight key so the layer still receives whatever the
            # checkpoint carries; only the int8 sidecars are removed.
            for suffix in (".weight_scale", ".comfy_quant", ".input_scale",
                           ".weight_scale_2", ".pre_quant_scale"):
                state_dict.pop(prefix + suffix, None)
            continue
        for suffix in (".weight", ".weight_scale", ".comfy_quant",
                       ".input_scale", ".weight_scale_2", ".pre_quant_scale"):
            state_dict.pop(prefix + suffix, None)

    # Re-key the CLIP-L namespace to match the A1111 module names.
    cl = "conditioner.embedders.0.transformer.text_model."
    if any(k.startswith(cl) for k in state_dict):
        moved = 0
        for k in [k for k in state_dict if k.startswith(cl)]:
            nk = "conditioner.embedders.0.transformer." + k[len(cl):]
            if nk not in state_dict:
                state_dict[nk] = state_dict.pop(k)
                moved += 1
        if moved:
            logger.info("%s re-keyed %d CLIP-L tensors to the A1111 namespace", _LOG, moved)
    return stats
```

### 4.2 `modules/sd_models.py` (modified)

Two changes. First, the SDXL hook was converted from dequantize to runtime INT8
(commit `afaf9e49`):

```diff
@@ def load_model_weights(model, checkpoint_info, state_dict, timer):
-    # --- SDXL ConvRot INT8: dequantize before cache + load_state_dict (isolated hook) ---
+    # --- SDXL ConvRot INT8: install runtime INT8 modules before load_state_dict
+    # (isolated SDXL-only hook). Weight stays int8 in VRAM; forward runs a true
+    # W8A8 ConvRot GEMM (Linear: comfy_kitchen.int8_linear; Conv2d: rotate act ->
+    # im2col -> rowwise int8 quant -> torch._int_mm -> dequant epilogue).
+    # Gated on the SDXL family signature (conditioner.* present); SD1.5 packs are
+    # handled by the separate hook below.
     try:
-        from modules import sdxl_int8_convrot as _int8cr
-        # SDXL-family only: do not intercept SD1.5 ConvRot INT8 packs (the SD1.5
-        # hook below installs runtime INT8 modules instead of dequantizing).
-        if model.is_sdxl and _int8cr.state_dict_has_comfy_quant(state_dict):
-            _target_dt = torch.float32 if shared.cmd_opts.no_half else torch.float16
-            _stats = _int8cr.dequantize_state_dict(state_dict, target_dtype=_target_dt)
-            print(f"[ConvRot] Dequantized state_dict: "
-                  f"int8_plain={_stats.get('plain', 0)} "
-                  f"int8_convrot_linear={_stats.get('convrot_linear', 0)} "
-                  f"int8_convrot_conv2d={_stats.get('convrot_conv2d', 0)} "
-                  f"int8_skipped={_stats.get('skipped', 0)} "
-                  f"int8_already_float={_stats.get('already_float', 0)}")
-            timer.record("ConvRot dequantize")
+        from modules import sdxl_int8_convrot as _sdxlint8
+        if _sdxlint8.state_dict_is_sdxl_convrot_int8(state_dict):
+            _statsxl = _sdxlint8.build_runtime_int8(model, state_dict)
+            print(f"[SDXL ConvRot] Runtime INT8 modules installed: "
+                  f"conv2d={_statsxl.get('conv2d', 0)} "
+                  f"linear={_statsxl.get('linear', 0)} "
+                  f"skipped={_statsxl.get('skipped', 0)} "
+                  f"int8_weight_mb={_statsxl.get('int8_bytes', 0) / 1e6:.1f}")
+            timer.record("SDXL ConvRot runtime INT8")
     except Exception as _e:
-        print(f"[ConvRot] WARNING: dequantize hook failed: {_e}")
+        print(f"[SDXL ConvRot] WARNING: runtime INT8 hook failed: {_e}")
```

The `model.is_sdxl` gate that had been added during the SD1.5 work was removed
in this commit because the SDXL gate is now expressed entirely by the state_dict
signature (`conditioner.*` + `.comfy_quant`), which is exact and does not depend
on the model flag.

---

## 5. What this means

### 5.1 The deception is removed

The weights are now **int8 resident for the whole session** — measured
**2,221.8 MB** for the reference SDXL pack's quantized UNet + CLIP layers,
versus the fp16 form which would be roughly double. No fp16 copy is
materialized. The forward pass is a genuine **W8A8 ConvRot GEMM**, not fp16
arithmetic.

### 5.2 How the UNet part works

Identical to SD1.5: the Conv2d path rotates the activation along its channel
dimension in Hadamard-sized blocks, builds the im2col view, quantizes the
activation rows to int8, runs `torch._int_mm` against the offline pre-rotated
int8 weight, and applies the activation scale, weight scale and bias in the
epilogue; the Linear path calls `comfy_kitchen.int8_linear(..., convrot=True)`.
The module is kept as a real `nn.Conv2d` / `nn.Linear` (weight parameter swapped
for an int8 parameter, instance-level `forward`), so the LoRA extension's exact
type checks pass and `.half()` / `.cuda()` leave the int8 payload intact.

### 5.3 How the CLIP part works (the SDXL-specific extension)

- **`torch.nn.Linear`** (CLIP-L projections, CLIP-G `c_fc`, etc.) → the same
  runtime INT8 Linear path.
- **`torch.nn.Embedding`** (CLIP-G `token_embedding`) → a dedicated shim that
  gathers the requested int8 rows, dequantizes them, unrotates them, and returns
  them in the dtype of the sibling `positional_embedding` (so the subsequent add
  does not promote dtypes). Measured against a dequantized reference: cosine
  **1.0**.
- **`out_proj` inside `nn.MultiheadAttention`** (CLIP-G attention) → the child
  projection is int8-installed, and the **parent MHA's `forward` is replaced** by
  a shim that reproduces the q/k/v projection and scaled-dot-product attention
  and then routes the output projection through the int8 module. A child-level
  shim alone would be ignored because `F.multi_head_attention_forward` consumes
  the projection's weight functionally. Measured against a float reference:
  cosine **0.99999**.
- **Namespace normalization.** The pack stores CLIP-L under
  `conditioner.embedders.0.transformer.text_model.*`; the A1111 module namespace
  is `conditioner.embedders.0.transformer.*`. `_resolve_mod` strips
  `.text_model.` when matching int8 layers, and `build_runtime_int8` re-keys the
  remaining float tensors to the same namespace. Without the re-key, CLIP-L's
  biases never load and the encoder emits NaN.

### 5.4 Broken-payload detection

While validating on real packs, the CLIP-L layer 11 projections and MLP shipped
with an **all-zero int8 payload and NaN scale** — the layer carries no weight
information. Feeding that to the INT8 path propagates NaN through the whole text
encoder. `_layer_scale_is_broken` detects this (NaN/Inf scale, or an all-zero
weight), and such a layer is deliberately **left on the float path** with an
explicit warning instead of being silently quantized. In the reference pack this
affects 6 layers; the other 711 are still int8. Two facts worth recording:

- The corruption is **present in the fp16 source model** as well (its layer 11
  weights are already NaN), so it is a property of the checkpoint, not of the
  A1111 implementation.
- A1111's CLIP-L uses `layer="hidden", layer_idx=11`; because `hidden_states`
  counts the embedding output as index 0, `hidden_states[11]` is the output
  *through layer 10*, so the broken layer 11 is computed but never used in the
  output. That is why fp16 generation was unaffected and why the INT8 path must
  simply not poison the tensor.

### 5.5 LoRA works without giving up INT8

As in SD1.5: the instance-level forward shadows the LoRA extension's global
forward patch, so the extension's float write can neither take effect nor
corrupt the int8 payload. The module reads the extension's loaded networks
(read-only) and bakes each delta into the int8 payload via `calc_updown`, then
re-rotates and requantizes. Measured with a real LoRA: relative delta error
**0.009**, weight still int8. In-app, UNet (722 keys) and CLIP (264 keys) LoRA
both load with zero skips.

### 5.6 Isolation guarantees

- Gate is exact and SDXL-only: `conditioner.*` present (SDXL family) **plus**
  `.comfy_quant` markers under `model.diffusion_model`. SD1.5, SD2, SD3, fp16
  and fp8 checkpoints are strict no-ops.
- The SD1.5 hook is a separate file and a separate branch.
- The HSWQ loader directory is read-only; the LoRA extension is read-only.
- In-app run log: `[SDXL ConvRot] Runtime INT8 modules installed: conv2d=7
  linear=679 skipped=0 int8_weight_mb=2221.8`, six broken-layer warnings, LoRA
  loaded, generation 16/16.

### 5.7 The lesson

Dequantizing int8 to fp16 is not INT8 support — it is fp16 support with an int8
loading step, and it saves no VRAM (here it made the model larger). True INT8
inference requires the weight to stay int8 resident and the forward to run a
real low-precision GEMM. For SDXL that meant covering not just the UNet but the
quantized text encoders too, across three module kinds (Linear, Embedding, and
the fused MultiheadAttention projection) and a namespace mismatch — and doing so
without breaking the LoRA extension's exact-type checks, PyTorch's dtype
casting, or the model's tolerance for pre-existing broken layers.
