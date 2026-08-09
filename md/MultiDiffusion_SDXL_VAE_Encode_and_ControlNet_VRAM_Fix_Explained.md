# MultiDiffusion SDXL — VAE Encode Crash & ControlNet VRAM Spill Fix

**Repository:** `ussoewwin/A1111-for-Python3.12`  
**Commit:** `7b5b2130` (`fix: SDXL VAE encode tensor and MD ControlNet caching allocator spill`)  
**Date:** 2026-06-29  

**Files changed (3):**

| File | Role |
|------|------|
| `modules/forge_tiled_vae.py` | Forge 3-pass tiled VAE (SDXL `DiffusionEngine` encode path) |
| `extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/multidiffusion.py` | MultiDiffusion `sample_one_step` tile loop |
| `extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/abstractdiffusion.py` | Noise Inversion Euler loop |

**Typical pipeline that triggered both issues:**

- img2img upscale with **MultiDiffusion**
- **ControlNet tile** (SDXL, ~3 GB model weights)
- **Noise Inversion (NI)** enabled
- **Forge Tiled VAE** active (3-pass encode, CPU accumulation)
- ~16 GB dedicated VRAM (e.g. RTX 4080 Super)
- Example canvas after upscale: **1536×2688** px → latent **336×192**

**Out of scope for this commit (separate docs):**

- Forge encode NaN / narrow crash → `md/A1111_Forge_Tiled_VAE_Encode_NaN_Fix_Explained.md`
- Older MultiDiffusion AttnBlock OOM → `md/MULTIDIFFUSION_OOM_FIX_EXPLANATION.md`

---

## 1. Summary

This commit fixes **two independent failures** in the same SDXL upscale pipeline:

| # | Failure | Symptom | Fix in one line |
|---|---------|---------|-----------------|
| **A** | SDXL VAE encode type confusion | Crash at encode with `AttributeError: 'torch.return_types.mode' object has no attribute 'float'` | Branch on `isinstance(Tensor)` for SDXL; never call `Tensor.mode()` |
| **B** | ControlNet + tiled sampling allocator spill | No hard OOM, but **117 s/it** on NI step 2; VRAM ~15.4 GB + **shared GPU memory** swap | `torch.cuda.empty_cache()` after each tile batch and each NI step (ControlNet only) |

Both fixes are **minimal and gated**: Forge 3-pass tiled encode/decode logic is unchanged; SD 1.5 paths keep prior behaviour; `empty_cache` runs only when `enable_controlnet` is true.

---

## 2. Problem A — SDXL Forge VAE Encode Crash

### 2-1. Error symptoms

During Forge tiled VAE **encode** (before MultiDiffusion sampling starts), the WebUI aborts:

```text
AttributeError: 'torch.return_types.mode' object has no attribute 'float'
```

Stack trace points at `_vae_encode_latent_tensor` in `modules/forge_tiled_vae.py`, inside the per-tile `encode_fn` called from `_forge_3pass_tiled`.

Log context (encode succeeds through 3-pass diagnostics, then sampling never starts):

```text
[Forge VAE] SD1.5/2.x Forge encode/decode patch active — skipping multidiffusion VAEHook ...
[MD-DIAG] 3pass[0] tile=(192,192) ov=24 part shape=(1, 4, 336, 192) nan=0 ...
[Tiled VAE]: Executing Encoder Task Queue: 100%|...| 394/394
```

### 2-2. Root cause (surface)

**Before this commit**, `_vae_encode_latent_tensor` used a single return path:

```python
encoded = vae.encode(x)
if hasattr(encoded, "mode"):
    return encoded.mode().float()
return encoded.float()
```

For **SD 1.5 / 2.x** (`LatentDiffusion`), `vae.encode()` returns a `DiagonalGaussianDistribution` object. Its `.mode()` returns a **latent tensor** — correct.

For **SDXL** (`DiffusionEngine` via `encode_pixels`), Forge's scaled encode path can return a **`torch.Tensor` directly** (already the latent mean, BCHW). That tensor is the value we want.

However, **`torch.Tensor` also has a `.mode()` method** — it is PyTorch's **statistical mode** along a dimension, not the VAE posterior mean. Calling `encoded.mode()` on a plain tensor returns a **`torch.return_types.mode` named tuple** (values + indices), which has **no `.float()` method** → `AttributeError`.

The old code used `hasattr(encoded, "mode")` as a type probe. That probe is **true for both** `DiagonalGaussianDistribution` and `torch.Tensor`, so SDXL tensors took the wrong branch.

### 2-3. Root cause (essential / architectural)

SDXL and SD1.5 do not share the same VAE encode **return contract** in this fork:

| Model family | Entry point | Typical `vae.encode()` return | Correct extraction |
|--------------|-------------|----------------------------------|--------------------|
| SD 1.5 / 2.x | `forge_ldm_encode_first_stage` → `_encode_tiled(..., diffusion_engine=False)` | `DiagonalGaussianDistribution` | `.mode().float()` |
| SDXL | `encode_pixels` → `_encode_tiled(..., diffusion_engine=True)` | `torch.Tensor` **or** distribution | If tensor → `.float()` only; if distribution → `.mode().float()` |

Using a **duck-typed** `hasattr(..., "mode")` check collapses two different meanings of "mode" into one code path. That is the essential bug: **name collision between distribution mean and tensor statistics**.

### 2-4. Countermeasure

1. Add an explicit keyword flag `diffusion_engine: bool` to `_vae_encode_latent_tensor`, `_encode_tiled`, and `_encode_full`.
2. Pass `diffusion_engine=True` **only** from `encode_pixels` (SDXL `DiffusionEngine` patch).
3. On the SDXL path:
   - If `isinstance(encoded, torch.Tensor)` → `return encoded.float()` (**do not** call `.mode()`).
   - Else (distribution object) → `return encoded.mode().float()`.
4. On the SD 1.5 path (`diffusion_engine=False`, default): **unchanged** — always `return encoded.mode().float()`.

Forge **3-pass tiled encode**, overlap, CPU accumulation, decode paths, and OOM→tiled fallback are **not modified** except for threading the flag through encode helpers.

### 2-5. Full added/changed code — `modules/forge_tiled_vae.py`

#### `_vae_encode_latent_tensor` (core fix)

```python
def _vae_encode_latent_tensor(
    vae, x: torch.Tensor, *, diffusion_engine: bool = False
) -> torch.Tensor:
    """VAE.encode -> BCHW latent mean for tiled/full encode helpers.

    diffusion_engine=False (SD 1.5 / 2.x LatentDiffusion via forge_ldm_encode_first_stage):
        vae.encode always returns DiagonalGaussianDistribution — use .mode() only (unchanged).
    diffusion_engine=True (SDXL DiffusionEngine via encode_pixels):
        vae.encode may return a latent tensor (Forge/scaled path); never call Tensor.mode().
    """
    encoded = vae.encode(x)
    if diffusion_engine:
        if isinstance(encoded, torch.Tensor):
            return encoded.float()
        return encoded.mode().float()
    return encoded.mode().float()
```

#### `_encode_tiled` — signature + `encode_fn` wiring

```python
def _encode_tiled(
    vae,
    pixel_samples: torch.Tensor,
    dtype,
    device,
    *,
    diffusion_engine: bool = False,
) -> torch.Tensor:
    orig_device = pixel_samples.device
    accum_device = _tiled_accum_device()
    if orig_device != accum_device:
        pixel_samples = pixel_samples.to(device=accum_device)
    _log_encode_tile_grid(pixel_samples)
    _, _, h, w = pixel_samples.shape
    base = _effective_encode_base(h, w)
    overlap = _encode_overlap_for_base(base)
    encode_passes = _encode_passes(base, overlap)
    output_device = _tiled_accum_device()

    def encode_fn(a):
        a = a.to(dtype=dtype, device=device)
        try:
            return _vae_encode_latent_tensor(vae, a, diffusion_engine=diffusion_engine)
        finally:
            devices.torch_gc()

    upscale = 1.0 / DOWNSCALE_RATIO
    total_steps = sum(
        _count_tiled_scale_steps(pixel_samples.shape, tile, ov)
        for tile, ov in encode_passes
    )
    pbar = _vae_progress_bar(is_decoder=False, total=total_steps)
    try:
        samples = _forge_3pass_tiled(
            pixel_samples,
            encode_fn,
            encode_passes,
            upscale_amount=upscale,
            out_channels=LATENT_CHANNELS,
            output_device=output_device,
            downscale=False,
            pbar=pbar,
        )
    finally:
        if pbar is not None:
            pbar.close()
    devices.torch_gc()
    return samples.to(device=orig_device)
```

#### `_encode_full` — signature + per-chunk encode

```python
def _encode_full(
    vae,
    pixel_samples: torch.Tensor,
    dtype,
    device,
    *,
    diffusion_engine: bool = False,
) -> torch.Tensor:
    memory_used = MEMORY_USED_ENCODE(pixel_samples.shape[2], pixel_samples.shape[3], dtype)
    free = _get_free_memory(device)
    batch_number = max(1, int(free / max(1, memory_used)))

    out = None
    for start in range(0, pixel_samples.shape[0], batch_number):
        chunk = pixel_samples[start : start + batch_number].to(dtype=dtype, device=device)
        encoded = _vae_encode_latent_tensor(vae, chunk, diffusion_engine=diffusion_engine)
        if out is None:
            out = torch.empty(
                (pixel_samples.shape[0],) + tuple(encoded.shape[1:]),
                device=encoded.device,
                dtype=encoded.dtype,
            )
        out[start : start + batch_number] = encoded
    return out.to(pixel_samples.device)
```

#### `encode_pixels` — SDXL-only `diffusion_engine=True`

```python
def encode_pixels(vae, pixel_samples: torch.Tensor) -> torch.Tensor:
    """Forge VAE.encode() — BCHW pixels in [-1, 1] (A1111 encode_first_stage input)."""
    device, dtype = _vae_device_dtype(vae)

    with _bypass_vae_hooks(vae):
        if VAE_ALWAYS_TILED:
            devices.torch_gc()
            return _encode_tiled(
                vae, pixel_samples, dtype, device, diffusion_engine=True
            )

        try:
            return _encode_full(
                vae, pixel_samples, dtype, device, diffusion_engine=True
            )
        except OOM_EXCEPTIONS:
            print(
                "Warning: Encountered Out of Memory during VAE Encoding; "
                "Retrying with Tiled VAE Encoding..."
            )
            devices.torch_gc()
            return _encode_tiled(
                vae, pixel_samples, dtype, device, diffusion_engine=True
            )
```

### 2-6. Meaning of Problem A fix

| Element | Meaning |
|---------|---------|
| `diffusion_engine=True` | "We are on the SDXL `DiffusionEngine.encode_first_stage` path; apply SDXL return-type rules." |
| `isinstance(encoded, torch.Tensor)` | Detect Forge's direct latent tensor **before** any `.mode()` call. |
| Default `diffusion_engine=False` | SD 1.5 / 2.x callers of `_encode_tiled` / `_encode_full` need no changes; behaviour frozen. |
| No decode changes | Decode already used `vae.decode(a).float()`; only **encode** had the type bug. |

---

## 3. Problem B — ControlNet + Noise Inversion VRAM Spill & Extreme Slowdown

### 3-1. Error symptoms

After Problem A was fixed, encode completes and **Noise Inversion** starts. Observed on 16 GB VRAM:

```text
MultiDiffusion hooked into 'Euler' sampler, Tile size: 96x96, Tile count: 18, Batch size: 4, Tile batches: 5 (ext: NoiseInv, ContrlNet)
[MD-DIAG] init_latent shape=(1, 4, 336, 192) dtype=torch.float32 nan=False ...
[MD-DIAG] NI i=1 x_in_scaled ... | 0/50 [00:00<?, ?it/s]
MultiDiffusion Sampling: : 0it [00:42, ?it/s]
[MD-DIAG] NI i=1 eps nan=False inf=False absmax=3.684
Noise Inversion:   4%|██▋| 2/50 [02:16<55:14, 69.05s/it]
```

Characteristics:

- **No** immediate `torch.OutOfMemoryError` in the log.
- Task Manager / GPU metrics: **~15.4 GB dedicated VRAM** used, plus spill into **shared GPU memory** (system RAM backing the GPU on Windows).
- First NI step can take **30–117 seconds**; later steps may speed up once the caching allocator stabilizes — but the first steps are unusably slow.
- User constraint: **Noise Inversion must stay enabled**; settings must not change; Forge tiled speedup must remain.

### 3-2. Root cause (surface)

MultiDiffusion with ControlNet enabled sets `micro_plan = [1] * tb` — **one tile at a time** (line 184–185 in `multidiffusion.py`). For each tile in a NI step:

1. `switch_controlnet_tensors` binds ControlNet hints for that tile.
2. `repeat_func` → UNet forward + ControlNet forward on the tile.
3. Activations and temporary buffers are freed from Python (`del`, scope exit) but **PyTorch's CUDA caching allocator retains the underlying GPU blocks** for reuse.

With **18 tiles × 50 NI steps**, the allocator's **reserved** memory grows across serial tile runs. On a 16 GB card already holding UNet (~6 GB FP16), VAE, ControlNet weights (~3 GB), and full latent canvas, the allocator pushes past dedicated VRAM into **shared GPU memory**. Shared memory is **orders of magnitude slower** than GDDR6 → wall-clock per step explodes (69–117 s/it).

Noise Inversion amplifies this: **`find_noise_for_image_sigma_adjustment`** calls `get_noise` → `sample_one_step` over **all tiles per Euler step**, so allocator pressure repeats **every NI iteration**, not only during the main 50-step sample.

### 3-3. Root cause (essential)

This is **not** primarily "the model needs 158 GiB" (that was a different failure mode from un-tiled global attention). Here the essential issue is:

> **Serial tile processing + ControlNet + caching allocator retention → memory footprint grows monotonically within a step, spilling to shared GPU RAM.**

PyTorch's allocator **by design** does not return freed blocks to the OS immediately. Without an explicit release hint, **peak reserved memory ≈ sum of peak per-tile allocations**, not max single tile.

On Windows WDDM, when dedicated VRAM is exhausted, the driver uses **shared GPU memory** (reclaimed system RAM). The process still "fits" but becomes **swap-bound** — hence slow without a clean OOM traceback.

### 3-4. Countermeasure

Insert **`torch.cuda.empty_cache()`** at two points, **only when ControlNet is active**:

1. **`multidiffusion.py`** — inside the `micro_plan` loop, **after each tile batch** completes in `sample_one_step`. Returns unused cached blocks from the caching allocator to the CUDA driver (and on Windows, reduces pressure to expand into shared memory before the next tile).

2. **`abstractdiffusion.py`** — at the **end of each Noise Inversion Euler iteration**, after `del` of step temporaries. Prevents NI step *N* from starting with allocator bloat left over from step *N−1*.

Guards:

```python
if self.enable_controlnet and torch.cuda.is_available():
    torch.cuda.empty_cache()
```

- **SD 1.5 without ControlNet:** guard is false → **zero behaviour change**.
- **Main sampling without ControlNet:** micro_plan uses `[2,2,...]` batching; no new `empty_cache` in that path from this commit.
- **Forge tiled VAE:** untouched — user requirement "do not remove Forge tiled speedup" preserved.

### 3-5. What `empty_cache()` does **not** do

| Expectation | Reality |
|-------------|---------|
| Lowers **peak** VRAM needed for one UNet forward | **No** — peak is set by the largest single forward pass (UNet + ControlNet on one tile). |
| Frees **model weights** | **No** — weights stay loaded. |
| Always speeds up **first** kernel after call | **Sometimes slower** — next allocation may re-request from OS; amortized over many tiles it prevents shared-memory spill. |
| Replaces proper batching | **No** — it only trims **retained but unused** cache between serial tiles. |

Observed after fix: VRAM meter may still show **~15 GB** (weights + working set), but **69 s/it → ~5 s/it** once spill is avoided — consistent with eliminating shared-memory thrashing, not reducing model size.

### 3-6. Full added code — MultiDiffusion

#### `tile_methods/multidiffusion.py` (inside `sample_one_step`, background grid loop)

Context: when `enable_controlnet`, `micro_plan = [1]*tb` processes tiles one-by-one.

```python
                else:
                    outs = []
                    k = 0
                    for m in micro_plan:
                        bb = bboxes[k:k+m]
                        xt = x_tile[k * N:(k + m) * N, :, :, :]
                        self.switch_controlnet_tensors(batch_id, N, m, tile_offset=k)
                        outs.append(repeat_func(xt, bb))
                        k += m
                        # ControlNet tile + per-tile UNet activations otherwise
                        # accumulate in the caching allocator and spill into shared
                        # GPU memory between tiles (NoiseInv path: 18 tiles serial,
                        # 117s/it observed on 16GB SDXL ControlNet tile runs).
                        if self.enable_controlnet and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    x_tile_out = torch.cat(outs, dim=0)
```

**Why inside the loop (after each `repeat_func`), not once per batch row:**

- With `micro_plan = [1,1,...,1]`, each iteration is **one tile**; releasing after each tile prevents 18 consecutive UNet+ControlNet peaks from stacking in the allocator before accumulation into `x_buffer`.

#### `tile_methods/abstractdiffusion.py` (`find_noise_for_image_sigma_adjustment`)

```python
            # This is neccessary to save memory before the next iteration
            del x_in, sigma_in, c_out, c_in, t,
            del eps, denoised, d, dt

            # Per-step allocator release: NoiseInv runs sample_one_step over
            # every tile + ControlNet on the full latent, so without this the
            # allocator keeps growing across NI steps and spills into shared
            # GPU memory (RAM swap), turning each step into 100+ s.
            if self.enable_controlnet and torch.cuda.is_available():
                torch.cuda.empty_cache()

            pbar.update(1)
```

**Why after `del`:**

- Python references to large tensors must be dropped first; otherwise `empty_cache` cannot reclaim their blocks.
- NI runs **before** the main MultiDiffusion sample loop; without per-step release, step 2 inherits step 1's allocator footprint × 18 tiles.

### 3-7. Meaning of Problem B fix

| Insertion point | What it prevents |
|-----------------|------------------|
| Per-tile in `sample_one_step` | Cross-tile accumulation of **temporary** activation cache within one denoise/NI forward |
| Per-step in NI Euler loop | Cross-**iteration** accumulation across 50 NI steps, each iterating all tiles |

Together they address the **serial** nature of ControlNet tiling (`micro_plan = [1]*n`) without changing tile size, batch size, or disabling NI.

---

## 4. End-to-end pipeline flow (where fixes apply)

```text
img2img + MultiDiffusion upscale
    │
    ├─► [Forge Tiled VAE encode]  ── Problem A fix (_vae_encode_latent_tensor)
    │       3-pass, ~394 encoder steps, latent (1,4,336,192)
    │
    ├─► [Noise Inversion loop]     ── Problem B fix (abstractdiffusion empty_cache)
    │       for each of ~50 sigmas:
    │           get_noise → sample_one_step (all 18 tiles)
    │               └── per-tile empty_cache (multidiffusion.py)
    │
    └─► [Main MultiDiffusion sample]  ── same sample_one_step path; per-tile empty_cache when ControlNet on
            50 steps, Euler
```

---

## 5. Design constraints preserved

| Constraint | How this commit respects it |
|------------|----------------------------|
| No user setting changes | Code-only; no UI / config defaults touched |
| Noise Inversion mandatory | NI loop kept; only adds allocator release |
| SD 1.5 unaffected | `diffusion_engine=False` default; `empty_cache` gated on `enable_controlnet` |
| Keep Forge tiled speedup | No bypass of `_forge_3pass_tiled`; no revert to global VAE encode |
| Minimal diff | +54 / −11 lines across 3 files |

---

## 6. Verification checklist

After pulling `7b5b2130`:

1. **SDXL img2img** — MultiDiffusion + ControlNet tile + NI + Forge Tiled VAE.
2. Encode finishes without `AttributeError` on `.mode().float()`.
3. `[MD-DIAG] init_latent` shows `nan=False`.
4. NI progress: step time **not** stuck at 60–120 s/it on 16 GB (target: single-digit seconds per step after warm-up).
5. Dedicated VRAM may remain high; **shared GPU memory** should not climb monotonically through NI.
6. **SD 1.5** img2img without ControlNet: regression smoke test (encode/decode unchanged).

---

## 7. Related documentation

| Document | Topic |
|----------|-------|
| `md/A1111_Forge_Tiled_VAE_Encode_NaN_Fix_Explained.md` | Encode NaN on wide 3-pass tiles; `floor`/`round` latent sizing |
| `md/MULTIDIFFUSION_OOM_FIX_EXPLANATION.md` | Legacy Tiled VAE "tiny skip" → global AttnBlock OOM |
| `md/MULTIDIFFUSION_OOM_FIX_2026-04-30.md` | Earlier MultiDiffusion OOM notes |

---

## 8. Quick reference — commit diff stats

```text
 extensions-builtin/.../abstractdiffusion.py  |  7 ++++++++
 extensions-builtin/.../multidiffusion.py     |  6 ++++++
 modules/forge_tiled_vae.py                   | 52 +++++++++++++++++++++------
 3 files changed, 54 insertions(+), 11 deletions(-)
```

**Author / committer:** ussoewwin  
**Push target:** `origin/main` (2026-06-29)
