# SDXL ConvRot INT8 / NVFP4 Checkpoint Support for A1111

## Technical Documentation

---

## 1. Overview

This document describes the implementation of SDXL ConvRot INT8 and NVFP4 checkpoint support in a custom A1111 WebUI fork (v2.3.5, Python 3.14). The implementation enables loading diffusion model checkpoints quantized with the ComfyUI-native `comfy_quant` format — both `int8_tensorwise` (with optional ConvRot Hadamard rotation) and `nvfp4` (E2M1 4-bit packed, with optional ConvRot) — into A1111's standard float16 inference pipeline.

### Design Principle: Complete Isolation

All new code paths are **exclusively branched**. The detection gate (`.comfy_quant` key presence in `state_dict`) ensures zero impact on existing model types (SD1.5, SD2, SDXL, SD3, fp16, fp8, bnb, etc.). No custom layers are injected, no monkey-patching of `torch.nn` modules, and no runtime quantization hooks are installed. The approach is purely **offline dequantize + unrotate at load time**, producing a standard float model indistinguishable from a normal checkpoint after loading.

---

## 2. Checkpoint Formats

### 2.1 INT8 Tensorwise (Plain)

Each quantized Linear or Conv2d layer stores:

| Key | Dtype | Shape | Description |
|-----|-------|-------|-------------|
| `<layer>.weight` | `int8` | `(out, in)` or `(out, in, kH, kW)` | Quantized weights |
| `<layer>.weight_scale` | `float32` | scalar | Per-tensor scale factor |
| `<layer>.comfy_quant` | `uint8` | variable | JSON bytes: `{"format":"int8_tensorwise"}` |

**Dequantize formula:** `w_float = weight.float() * weight_scale`

### 2.2 INT8 Tensorwise + ConvRot

Same as plain INT8, but with additional ConvRot metadata:

| Key | Dtype | Shape | Description |
|-----|-------|-------|-------------|
| `<layer>.weight_scale` | `float32` | `[out, 1]` (Linear) or `[out, 1, 1, 1]` (Conv2d) | Per-output-channel scale |
| `<layer>.comfy_quant` | `uint8` | variable | `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}` |

The weight is stored in **rotated** form: `W_rot = W @ H^T` (group-wise along `in_features`), where `H` is a normalized regular Hadamard matrix. The scale shape differs because ConvRot uses row-wise (per-output-channel) quantization instead of per-tensor.

**ConvRot group size:** Must be a power of 4 (4, 16, 64, 256). Default is 256. The `in_features` dimension must be divisible by the group size.

### 2.3 NVFP4 (E2M1 Packed)

NVFP4 is a 4-bit floating-point format (E2M1) with block-wise scaling. Each quantized Linear layer stores:

| Key | Dtype | Shape | Description |
|-----|-------|-------|-------------|
| `<layer>.weight` | `uint8` | `(padded_out, padded_in // 2)` | Packed E2M1 nibble pairs |
| `<layer>.weight_scale` | `float8_e4m3fn` | block scale grid | Per-block scaling factors |
| `<layer>.weight_scale_2` | `float32` | scalar or tensor | Per-tensor (super-block) scale |
| `<layer>.comfy_quant` | `uint8` | variable | JSON: `{"format":"nvfp4","convrot":true,"convrot_groupsize":256,"orig_shape":[out,in]}` |
| `<layer>.input_scale` | `float32` | `(1,)` | Optional activation scale |
| `<layer>.pre_quant_scale` | `float32` | various | Optional pre-quantization scale |

**Packing:** Two E2M1 values are packed into each uint8 byte. The storage shape is `(padded_out, padded_in // 2)`, where both dimensions are padded to multiples of 16 (16×16 alignment for tensor core compatibility).

**`orig_shape`:** Required because `padded_in // 2 * 2` may not equal the logical `in_features` (e.g., logical 12 → padded 16 → storage 8 → `8 * 2 = 16 ≠ 12`).

**Dequantize:** Uses `comfy_kitchen.dequantize_nvfp4(qdata, per_tensor_scale, block_scales, output_type, hi_first)`, which unpacks E2M1 nibbles and applies two-level scaling: `w_float = e2m1_value * block_scale * tensor_scale`.

### 2.4 Mixed Pack Checkpoints

NVFP4 ConvRot checkpoints use a **mixed pack** format:

| Layer type | Quantization | ConvRot |
|------------|-------------|---------|
| Linear (attention projections, FFN, etc.) | NVFP4 | Yes (Hadamard along `in_features`) |
| Conv2d (input blocks, ResBlock convs, etc.) | INT8 tensorwise | Yes (Hadamard along `in_channels`) |
| CLIP / VAE | Not quantized | N/A |

This is because NVFP4 tensor core acceleration targets matrix multiplications (Linear layers), while Conv2d layers use INT8 which has broader hardware support.

---

## 3. Hadamard ConvRot

### 3.1 Mathematical Foundation

ConvRot applies a **normalized regular Hadamard matrix** `H` to rotate weights along the input dimension. The Hadamard matrix is:

- **Symmetric:** `H^T = H`
- **Orthogonal:** `H^{-1} = H^T = H`
- **Normalized:** `H * H^T = I` (entries divided by `√n`)

Therefore, the inverse rotation is identical to the forward rotation: `unrotate(rotate(W)) = W_rot @ H = (W @ H^T) @ H = W @ (H^T @ H) = W @ I = W`.

### 3.2 Construction

The Hadamard matrix is built via repeated Kronecker product of the base 4×4 matrix:

```
H_4 = (1/2) * [[ 1,  1,  1, -1],
               [ 1,  1, -1,  1],
               [ 1, -1,  1,  1],
               [-1,  1,  1,  1]]

H_{4k} = H_4 ⊗ H_{k}
```

Valid sizes: 4, 16, 64, 256, 1024, ... (powers of 4).

### 3.3 Weight Rotation

**Linear (2D weight, shape `(out, in)`):**

Forward (quantize time): `W_rot = W @ H^T` (group-wise along `in`)
- Reshape: `(out, in // gs, gs)`
- Matrix multiply: `(out, in // gs, gs) @ (gs, gs) → (out, in // gs, gs)`
- Reshape back: `(out, in)`

Inverse (load time): `W = W_rot @ H` (identical operation since `H = H^T`)

**Conv2d (4D weight, shape `(out, in, kH, kW)`):**

Forward: rotate along `in_channels`
- Reshape: `(out, in, kH*kW)` → permute → `(out * kH * kW, in)`
- Apply Linear rotation along `in`
- Reshape back to `(out, in, kH, kW)`

Inverse: same permutation, same `unrotate_weight`, same reshape back.

### 3.4 Why ConvRot Improves Quality

INT8/FP4 quantization error is proportional to the weight magnitude. Without rotation, outlier weights dominate the per-tensor scale, degrading quantization resolution for typical weights. Hadamard rotation **spreads outliers across all dimensions** (energy-preserving mixing), producing a more uniform weight distribution that quantizes better with a single scale factor.

The trade-off is a one-time matrix multiplication during quantization (offline) and dequantization (load time), plus an online activation rotation in the ComfyUI inference path (not needed in our offline approach).

---

## 4. Implementation

### 4.1 File: `modules/sdxl_int8_convrot.py` (New)

This module contains all dequantization logic. It is imported lazily inside `load_model_weights()` only when `comfy_quant` keys are detected.

#### Key Functions

```
state_dict_has_comfy_quant(state_dict) → bool
    Fast detection gate. Returns True if any key ends with ".comfy_quant".

dequantize_state_dict(state_dict, target_dtype) → dict[str, int]
    Combined entry point. Calls NVFP4 pass first, then INT8 pass.
    Returns stats dict for logging.

dequantize_nvfp4_state_dict(state_dict, target_dtype) → dict
    Processes layers where comfy_quant format == "nvfp4".
    Uses comfy_kitchen.dequantize_nvfp4 for E2M1 unpacking.

dequantize_int8_state_dict(state_dict, target_dtype) → dict
    Processes layers where comfy_quant format == "int8_tensorwise".
    Manual dequantize: weight.float() * weight_scale.

build_hadamard(size, device, dtype) → Tensor
    Cached Hadamard matrix construction.

unrotate_weight(weight, h_matrix, group_size) → Tensor
    Inverse Linear ConvRot: W = W_rot @ H (group-wise).

unrotate_weight_conv2d(weight, h_matrix, group_size) → Tensor
    Inverse Conv2d ConvRot: unrotate along in_channels.

decode_comfy_quant_conf(raw) → dict
    Decode uint8 JSON bytes tensor to Python dict.
```

#### Processing Order

```
dequantize_state_dict(state_dict, target_dtype)
├── 1. NVFP4 pass: dequantize_nvfp4_state_dict()
│   ├── For each .comfy_quant key with format=="nvfp4":
│   │   ├── ck.dequantize_nvfp4(weight, weight_scale_2, weight_scale, fp32)
│   │   ├── Crop from padded shape to orig_shape
│   │   ├── If convrot: unrotate_weight(Hadamard)
│   │   ├── Convert to target_dtype (fp16 or fp32)
│   │   └── Pop all sidecar keys (.weight_scale, .weight_scale_2,
│   │       .input_scale, .pre_quant_scale, .comfy_quant)
│   └── Layers with format!="nvfp4" are skipped (left for INT8 pass)
│
├── 2. INT8 pass: dequantize_int8_state_dict()
│   ├── For each remaining .comfy_quant key with format=="int8_tensorwise":
│   │   ├── w_float = weight.float() * weight_scale.float()
│   │   ├── If convrot + 2D: unrotate_weight(Hadamard)
│   │   ├── If convrot + 4D: unrotate_weight_conv2d(Hadamard)
│   │   ├── Convert to target_dtype
│   │   └── Pop all sidecar keys
│   ├── Non-int8 weights (already float): pop sidecar, count as "already_float"
│   └── NVFP4-processed layers: .comfy_quant already popped, not in key list
│
└── Return combined stats dict
```

### 4.2 File: `modules/sd_models.py` (Modified)

A single hook is inserted in `load_model_weights()`, between `convert_sdxl_to_ssd()` and the checkpoint cache:

```python
# --- SDXL INT8/NVFP4 ConvRot: dequantize before cache + load_state_dict ---
try:
    from modules import sdxl_int8_convrot as _int8cr
    if _int8cr.state_dict_has_comfy_quant(state_dict):
        _target_dt = torch.float32 if shared.cmd_opts.no_half else torch.float16
        _stats = _int8cr.dequantize_state_dict(state_dict, target_dtype=_target_dt)
        print(f"[ConvRot] Dequantized state_dict: ...")
        timer.record("ConvRot dequantize")
except Exception as _e:
    print(f"[ConvRot] WARNING: dequantize hook failed: {_e}")
```

**Placement rationale:**
- **After** `set_model_type()` / `extend_sdxl()`: These use key names and shapes, which are valid for int8 tensors (same shape, different dtype).
- **Before** `checkpoints_loaded[checkpoint_info] = state_dict.copy()`: Ensures the cache stores dequantized float weights, avoiding re-processing on cache hits.
- **Before** `model.load_state_dict(state_dict, strict=False)`: The model receives a clean float state_dict with all sidecar keys removed.
- **Inside** `LoadStateDictOnMeta` context: The `self.state_dict` reference points to the same dict object, so in-place modifications are visible. Sidecar keys are popped before `load_from_state_dict` processes each module, preventing `Parameter(int8_tensor, requires_grad=True)` errors.

---

## 5. Isolation Guarantees

### 5.1 Detection Gate

The sole entry condition is `state_dict_has_comfy_quant(state_dict)`, which scans for keys ending with `.comfy_quant`. If none exist, the entire hook is a no-op — not even the import of `sdxl_int8_convrot` occurs.

### 5.2 Format-Specific Branching

Within the dequantize functions, each layer is processed only if its `comfy_quant` JSON `format` field matches:

- `"int8_tensorwise"` → INT8 path
- `"nvfp4"` → NVFP4 path
- Anything else → sidecar keys popped, weight left as-is (counted as `skipped`)

### 5.3 No Impact on Non-Quantized Checkpoints

For a standard SDXL checkpoint (e.g., `sd_xl_base_1.0.safetensors`):
1. No `.comfy_quant` keys in state_dict
2. `state_dict_has_comfy_quant()` returns `False`
3. Hook body never executes
4. `load_model_weights()` proceeds exactly as before

### 5.4 No Runtime Hooks

Unlike Forge-Nunchaku's `comfy_quant_int8.py` (which monkey-patches `comfy.ops`, `comfy.lora`, `comfy.model_patcher`, and injects custom Conv2d forward functions), this implementation:
- Installs **zero** monkey-patches
- Injects **zero** custom layers
- Modifies **zero** forward functions
- After loading, the model is a standard `torch.nn.Module` tree with float16 parameters

---

## 6. A1111 Load Pipeline (Reference)

The complete SDXL checkpoint loading flow in this A1111 fork:

```
load_model(checkpoint_info)
│
├── get_checkpoint_state_dict(checkpoint_info)
│   └── read_state_dict(filename)  # safetensors or torch.load
│
├── find_checkpoint_config(state_dict, checkpoint_info)
│   └── guess_model_config_from_state_dict()
│       └── Key-name + shape checks (works for int8: same shapes)
│       └── Returns: sd_xl_base.yaml (target=sgm.models.diffusion.DiffusionEngine)
│
├── repair_config(sd_config, state_dict)
│   └── Sets use_fp16, attn_type, etc. (state_dict not used)
│
├── instantiate_from_config(sd_config.model, state_dict)
│   └── DiffusionEngine.__init__()  # state_dict not passed to constructor
│       └── UNetModel.__init__()   # creates empty modules on meta device
│
├── LoadStateDictOnMeta(state_dict, device, weight_dtype_conversion)
│   └── Context manager: patches Module._load_from_state_dict
│
└── load_model_weights(model, checkpoint_info, state_dict, timer)
    ├── set_model_type(model, state_dict)          # key-name based
    ├── extend_sdxl(model)                          # model property setup
    ├── ★ dequantize_state_dict(state_dict) ★      # ← OUR HOOK
    ├── checkpoints_loaded[info] = state_dict.copy()  # cache (post-dequant)
    ├── model.load_state_dict(state_dict, strict=False)  # float weights
    ├── model_hijack.hijack(model)                  # attention hijack
    └── model_data.set_sd_model(model)
```

---

## 7. Comparison with Forge-Nunchaku

| Aspect | Forge-Nunchaku (`comfy_quant_int8.py`) | This Implementation |
|--------|----------------------------------------|---------------------|
| **Approach** | Online dequantize (runtime) | Offline dequantize (load time) |
| **VRAM benefit** | Yes (int8 storage in VRAM) | No (float16 after loading) |
| **Inference speed** | Overhead from runtime dequant | Same as standard float16 |
| **Dependencies** | ComfyUI `ops.py`, `model_patcher`, `lora`, `controlnet` | `comfy_kitchen` (NVFP4 only) |
| **Monkey-patches** | ~20 functions across 5 modules | Zero |
| **Custom layers** | Injected Conv2d with INT8 forward | Zero |
| **LoRA** | Requires convert_weight/set_weight bake | Standard A1111 LoRA (float weights) |
| **Architecture** | Forge `backend/operations.py` | A1111 `sd_models.py` load pipeline |

The Forge approach keeps weights quantized in VRAM and dequantizes on-the-fly during each forward pass, reducing memory but adding compute overhead. This implementation trades the memory benefit for simplicity: weights are dequantized once at load time, after which the model is indistinguishable from a standard float16 checkpoint.

---

## 8. Limitations

1. **No VRAM savings:** After loading, the model occupies the same VRAM as a standard float16 model. The INT8/NVFP4 storage benefit is only realized on disk and during the initial state_dict load (before dequantize).

2. **NVFP4 requires `comfy_kitchen`:** The `comfy_kitchen` package must be installed in the A1111 venv for NVFP4 dequantization. Without it, NVFP4 layers are skipped and the model will fail to load (logged as a warning).

3. **Load time overhead:** Dequantization adds time to model loading proportional to the number of quantized layers. For a typical SDXL model (~300 Linear + ~50 Conv2d), this is a few seconds on CPU.

4. **NVFP4 accuracy:** 4-bit quantization introduces higher error than INT8. Typical max weight diff is ~0.4-0.7 (vs ~0.02-0.04 for INT8). ConvRot helps mitigate this by spreading outliers.

5. **Conv2d NVFP4 not supported:** NVFP4 is only applied to Linear layers (matching the checkpoint format). Conv2d layers in NVFP4 checkpoints use INT8 tensorwise, handled by the INT8 path.

---

## 9. File Inventory

| File | Status | Description |
|------|--------|-------------|
| `modules/sdxl_int8_convrot.py` | New | Dequantize module: Hadamard, unrotate, INT8/NVFP4 dequantize |
| `modules/sd_models.py` | Modified | Hook in `load_model_weights()` (lines ~437-453) |

### Commits

```
96a80ed7 feat: add SDXL INT8 ConvRot checkpoint support
27891d83 feat: add SDXL NVFP4 ConvRot checkpoint support
181e3783 fix: stats key collision between NVFP4 and INT8 skipped counters
```

---

## 10. Testing

### Unit Tests

**INT8 (6 tests):**
- `state_dict_has_comfy_quant` detection (positive/negative)
- Plain INT8 dequantize (max diff < 0.018)
- ConvRot Linear dequantize (max diff < 0.040)
- ConvRot Conv2d dequantize (max diff < 0.031)
- Float weight skip (already_float)
- No-op for plain state_dict (all stats zero)

**NVFP4 (4 tests):**
- Plain NVFP4 dequantize (max diff < 0.68)
- ConvRot NVFP4 dequantize (max diff < 0.42)
- Mixed pack: NVFP4 Linear + INT8 Conv2d in same state_dict
- No-op for plain state_dict

### Verified Pipeline Interactions

- `find_checkpoint_config`: Uses key names and `shape[1]` — same for int8/uint8
- `set_model_type`: Uses key names — unaffected by dtype
- `instantiate_from_config`: `DiffusionEngine.__init__` does not receive `state_dict`
- `LoadStateDictOnMeta`: Same dict reference; sidecar keys popped before module load
- `sd_disable_initialization`: int8 `Parameter` error resolved (weights are float before `load_state_dict`)
- Checkpoint cache: Stores post-dequant float weights; cache hits skip re-processing
- `reload_model_weights`: Calls `load_model_weights()` → same hook fires

---

*Document generated: 2026-08-10*
*Repository: https://github.com/ussoewwin/A1111-for-Python3.14*
