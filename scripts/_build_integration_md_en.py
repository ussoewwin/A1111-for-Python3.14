# One-off: build English A1111_RES4LYF_INTEGRATION.md (documentation only)
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
samplers_src = (root / "modules/a1111_res4lyf_samplers.py").read_text(encoding="utf-8")
shim_src = (root / "modules/a1111_res4lyf_shim.py").read_text(encoding="utf-8")

replacements = [
    (
        "Forge (Stable-Diffusion-WebUI-Forge-Nunchaku) 側の\n`modules_forge/forge_res4lyf_samplers.py` を A1111 向けに書き直したもの。",
        "A1111 port of Forge (Stable-Diffusion-WebUI-Forge-Nunchaku)\n`modules_forge/forge_res4lyf_samplers.py`.",
    ),
    ("Forge との主な差分:", "Main differences from Forge:"),
    (
        "- A1111 には ``sd_samplers.add_sampler()`` が無いため\n"
        "  ``all_samplers`` を直接拡張し ``set_samplers()`` を再実行する",
        "- A1111 has no ``sd_samplers.add_sampler()``;\n"
        "  extend ``all_samplers`` directly and call ``set_samplers()`` again",
    ),
    (
        "- ComfyUI ランタイム (``ComfyUI-master``) は ``modules/paths.py`` では\n"
        "  sys.path に追加されないため、本モジュールが自前で追加する",
        "- ComfyUI runtime (``ComfyUI-master``) is not added to sys.path by\n"
        "  ``modules/paths.py``; this module adds it",
    ),
    (
        "- ``bong_tangent`` は Forge の ``sd_schedulers.py`` にはネイティブ実装があったが\n"
        "  A1111 には無いため、``beta57`` と併せて本モジュール側で登録する",
        "- ``bong_tangent`` was native in Forge ``sd_schedulers.py`` but\n"
        "  missing on A1111; registered here with ``beta57``",
    ),
    (
        "- ``comfy.samplers.beta_scheduler`` / RES4LYF ``sigmas.bong_tangent_scheduler`` は\n"
        "  ComfyUI 由来のシグネチャなので、A1111 の\n"
        "  ``(n, sigma_min, sigma_max, inner_model, device)`` にラップする",
        "- ``comfy.samplers.beta_scheduler`` / RES4LYF ``sigmas.bong_tangent_scheduler`` use\n"
        "  ComfyUI signatures; wrapped for A1111\n"
        "  ``(n, sigma_min, sigma_max, inner_model, device)``",
    ),
    (
        '"""``ComfyUI-master`` を sys.path に挿入。既にあれば何もしない。"""',
        '"""Insert ``ComfyUI-master`` on sys.path if not already present."""',
    ),
    (
        '"""``pywavelets`` / ``comfy-kitchen`` を可能なら pip で入れる。失敗しても致命的にしない。"""',
        '"""Install ``pywavelets`` / ``comfy-kitchen`` via pip when possible; non-fatal on failure."""',
    ),
    (
        "    ``folder_paths`` と ``server.PromptServer`` を、\n"
        "    ComfyUI 本体を起動していない状態でも import できるように差し込む。",
        "    Inject ``folder_paths`` and ``server.PromptServer`` so RES4LYF can import\n"
        "    without running a full ComfyUI server.",
    ),
    (
        "    ``ComfyUI-master`` に本物の ``folder_paths.py`` があればそれを優先する。",
        "    Prefer the real ``folder_paths.py`` from ``ComfyUI-master`` when available.",
    ),
    (
        "    # A1111 の標準サンプラーが参照する ``k_diffusion.sampling.sample_<name>``\n"
        "    # と衝突する名前は絶対に上書きしない。",
        "    # Never overwrite names used by A1111 standard samplers in\n"
        "    # ``k_diffusion.sampling.sample_<name>``.",
    ),
    (
        "    # ``RK_SAMPLER_NAMES_BETA_NO_FOLDERS`` には ``dpmpp_2m`` / ``dpmpp_3m`` /\n"
        "    # ``dpmpp_2s`` / ``dpmpp_sde_2s`` / ``dpmpp_3s`` / ``euler`` / ``ddim`` 等、\n"
        "    # A1111 の DPM++ 2M / Euler / DDIM の実体名と重なるエントリが含まれる。",
        "    # ``RK_SAMPLER_NAMES_BETA_NO_FOLDERS`` includes ``dpmpp_2m``, ``dpmpp_3m``,\n"
        "    # ``dpmpp_2s``, ``dpmpp_sde_2s``, ``dpmpp_3s``, ``euler``, ``ddim``, etc.—same\n"
        "    # names as A1111 DPM++ 2M / Euler / DDIM.",
    ),
    (
        "    # これらを ``setattr(k_diffusion.sampling, \"sample_<name>\", ...)`` すると、\n"
        "    # A1111 標準サンプラーが起動する ``getattr(k_diffusion.sampling, \"sample_<name>\")``\n"
        "    # の返り値が RES4LYF のクロージャに差し替わり、標準サンプラーが破壊される。",
        "    # Overwriting via ``setattr(k_diffusion.sampling, \"sample_<name>\", ...)``\n"
        "    # replaces ``getattr(k_diffusion.sampling, \"sample_<name>\")`` used by\n"
        "    # standard samplers and breaks them.",
    ),
    ("            # 標準サンプラー名と衝突。上書き禁止。", "            # Collision with standard sampler name; skip."),
    (
        '    """``SamplerData.constructor`` 用のクロージャを返す。\n\n'
        "    A1111 の ``KDiffusionSampler`` に渡す ``func`` を、\n"
        "    RES4LYF が ComfyUI 互換 API を要求する箇所を吸収するための\n"
        "    :func:`modules.a1111_res4lyf_shim.res4lyf_shim_context` で包む。\n"
        '    """',
        '    """Return a closure for ``SamplerData.constructor``.\n\n'
        "    Wraps the ``func`` passed to A1111 ``KDiffusionSampler`` with\n"
        "    :func:`modules.a1111_res4lyf_shim.res4lyf_shim_context` where RES4LYF\n"
        "    needs ComfyUI-compatible APIs.\n"
        '    """',
    ),
    (
        "        # KDiffusionSampler は ``inspect.signature(self.func).parameters`` で\n"
        "        # ``n`` / ``sigmas`` / ``sigma_min`` / ``sigma_max`` などを検査するため、\n"
        "        # 元関数の署名情報を wrapper に引き継ぐ。",
        "        # KDiffusionSampler inspects signature(self.func).parameters for\n"
        "        # n, sigmas, sigma_min, sigma_max, etc.; preserve metadata.",
    ),
    (
        '    """RES4LYF のサンプラーを A1111 の ``all_samplers`` に追加する。"""',
        '    """Register RES4LYF samplers into A1111 ``all_samplers``."""',
    ),
    (
        "        # RES4LYF 本体を import。ここで __init__.py 末尾の add_samplers() が走り、\n"
        "        # comfy.k_diffusion.sampling.sample_* が生える。",
        "        # Import RES4LYF; __init__.py add_samplers() builds comfy.k_diffusion.sample_*",
    ),
    (
        "        # Forge fork の RES4LYF は beta/__init__.py で全 RK サンプラーを動的登録\n"
        "        # するのに対し、A1111 に置いた upstream 版は 17 個しか登録しない。\n"
        "        # ここで足りない分を A1111 側から追加登録する（RES4LYF 本体は無編集）。",
        "        # Forge fork registers all RK samplers in beta/__init__.py; upstream A1111\n"
        "        # copy registers ~17. Add the rest here without editing RES4LYF.",
    ),
    ("        # comfy.k_diffusion.sampling → k_diffusion.sampling へ関数コピー", "        # Copy comfy.k_diffusion.sampling -> k_diffusion.sampling"),
    (
        "    RES4LYF スケジューラーを A1111 の ``sd_schedulers`` に登録する。",
        "Register RES4LYF schedulers into A1111 ``sd_schedulers``.",
    ),
    (
        "    - ``beta57``: ComfyUI の ``beta_scheduler(model_sampling, steps, alpha=0.5, beta=0.7)`` を\n"
        "      A1111 シグネチャ ``(n, sigma_min, sigma_max, inner_model, device)`` にラップ",
        "    - ``beta57``: wrap ComfyUI ``beta_scheduler(model_sampling, steps, alpha=0.5, beta=0.7)`` to\n"
        "      A1111 ``(n, sigma_min, sigma_max, inner_model, device)``",
    ),
    (
        "    - ``bong_tangent``: RES4LYF ``sigmas.bong_tangent_scheduler(model_sampling, steps, ...)`` を\n"
        "      同じくラップ",
        "    - ``bong_tangent``: wrap RES4LYF ``sigmas.bong_tangent_scheduler(...)`` similarly",
    ),
]
samplers_doc = samplers_src
for old, new in replacements:
    samplers_doc = samplers_doc.replace(old, new)

prose = f'''# A1111 RES4LYF Integration — Complete Technical Guide

**Target repository:** `D:\\USERFILES\\A1111`  
**Created:** 2026-07-01  
**Goal:** Expose RES4LYF (ComfyUI custom node) samplers and schedulers in the native A1111 UI.  
**Scope:** `modules/RES4LYF/` and `ComfyUI-master/` are **unchanged** in this integration (vendored/placed only).

---

## Table of contents

1. [Overview and design](#1-overview-and-design)
2. [Changed files](#2-changed-files)
3. [Architecture](#3-architecture)
4. [Data flow from startup to generation](#4-data-flow-from-startup-to-generation)
5. [New files (full text and notes)](#5-new-files-full-text-and-notes)
6. [Modified files (full text and notes)](#6-modified-files-full-text-and-notes)
7. [Errors encountered and fixes](#7-errors-encountered-and-fixes)
8. [Differences from Forge](#8-differences-from-forge)
9. [Limitations and future work](#9-limitations-and-future-work)

---

## 1. Overview and design

### 1.1 What was achieved

- Added RES4LYF samplers (100+ entries, including Forge-parity dynamic RK registration) to the A1111 **Sampling method** dropdown
- Added `beta57` and `bong_tangent` to A1111 **Schedule type**
- Image generation **completes successfully** when a RES4LYF sampler is selected (fixed missing `model_sampling`, `CFGDenoiser` kwargs mismatch, name collisions, etc.)

### 1.2 Design principles

| Principle | Description |
|-----------|-------------|
| Do not modify RES4LYF core | `modules/RES4LYF/**` stays a ComfyUI node copy |
| Minimal A1111 core changes | Only a short startup hook in `initialize.py`; `sd_samplers.py` etc. untouched |
| Glue + shim pattern | Same as Forge `modules_forge/forge_res4lyf_samplers.py` |
| Runtime-only compatibility | `res4lyf_shim_context` injects ComfyUI-expected attributes temporarily, then restores |
| Protect standard samplers | On name collision (`dpmpp_2m`, `euler`, etc.), skip RES4LYF registration |

### 1.3 External dependencies (placement only; out of scope for code edits in this doc)

- `ComfyUI-master/` — `comfy.*` runtime (added to `sys.path` manually)
- `modules/RES4LYF/` — RES4LYF implementation
- Extra venv packages: `torchaudio`, `av`, `pywavelets`, `comfy-kitchen` (glue may try pip or user installs manually)

### 1.4 Manual environment change (outside integration code)

In `requirements_versions_py314.txt` / `requirements_versions_py314_windows.txt`,  
relaxed `einops==0.4.1` to **`einops>=0.4.1`** (ComfyUI / spandrel need `from einops import einsum`).

---

## 2. Changed files

### 2.1 New files (3)

| Path | Lines | Role |
|------|-------|------|
| `modules/a1111_res4lyf_samplers.py` | 426 | Registration glue (path, mock, dynamic RK, UI, schedulers) |
| `modules/a1111_res4lyf_shim.py` | 292 | Runtime shim (`model_sampling`, `diffusion_model` alias, `CFGDenoiser` patch) |
| `md/A1111_RES4LYF_SHIM_PLAN.md` | — | Phase 1 technical plan (reference) |

### 2.2 Modified (1 file)

| Path | Change |
|------|--------|
| `modules/initialize.py` | RES4LYF registration hook after `sd_samplers.set_samplers()` (+10 lines) |

### 2.3 Out of scope for this document

- `modules/RES4LYF/**`
- `ComfyUI-master/**`

---

## 3. Architecture

### 3.1 Two-layer structure (same as Forge)

```
┌─────────────────────────────────────────────────────────────┐
│  A1111 UI (txt2img / img2img)                                │
│    Sampling method  ← sd_samplers.all_samplers               │
│    Schedule type    ← sd_schedulers.schedulers               │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  modules/a1111_res4lyf_samplers.py  (GLUE)                     │
│    · Add ComfyUI-master to sys.path                            │
│    · Mock folder_paths / server                                │
│    · import modules.RES4LYF → build extra_samplers             │
│    · _register_extra_rk_beta_samplers (Forge-parity dynamic RK)  │
│    · Sync comfy.k_diffusion → k_diffusion.sampling             │
│    · Append SamplerData to all_samplers                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  modules/a1111_res4lyf_shim.py  (SHIM)                        │
│    · res4lyf_shim_context: temporary LatentDiffusion attrs     │
│    · patch_cfg_denoiser_forward: drop unknown kwargs           │
│    · ensure_res4lyf_extra_args: model_options nesting          │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  modules/RES4LYF/  (ComfyUI node, unmodified)                │
│    rk_sampler_beta.sample_rk_beta, etc.                        │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Model object hierarchy (core issue)

RES4LYF expects ComfyUI `BaseModel`. A1111’s innermost object is `LatentDiffusion` with different attributes.

| Path | ComfyUI / Forge | A1111 |
|------|-----------------|-------|
| `model` | `CFGDenoiser` | `CFGDenoiser` ✓ |
| `model.inner_model` | `CompVisDenoiser` | `CompVisDenoiser` ✓ |
| `model.inner_model.inner_model` | `BaseModel` | **`LatentDiffusion`** ✗ |
| `.model_sampling` | `comfy.model_sampling.EPS` etc. | **missing** → injected by shim |
| `.diffusion_model` | direct U-Net ref | **nested under `model.diffusion_model`** → `__dict__` alias |
| `.device` | attribute | **read-only `@property`** (no assignment needed) |

---

## 4. Data flow from startup to generation

### 4.1 At startup (`initialize_rest`)

1. A1111 standard `sd_samplers.set_samplers()`
2. `a1111_res4lyf_samplers.register_res4lyf_samplers()`
   - `_ensure_comfyui_on_path()` → `ComfyUI-master` at `sys.path[0]`
   - `_mock_comfyui_globals()` → stub missing modules
   - `from modules import RES4LYF` → `add_samplers()` in `__init__.py` builds `extra_samplers`
   - `_register_extra_rk_beta_samplers()` → add Forge-parity RK names (skip collisions)
   - Copy `sample_*` from `comfy.k_diffusion.sampling` to `k_diffusion.sampling`
   - Append `SamplerData` per name → call `set_samplers()` again
   - Apply `patch_cfg_denoiser_forward()` once
3. `register_res4lyf_schedulers()` → add `beta57`, `bong_tangent` to `sd_schedulers`

### 4.2 At generation (RES4LYF sampler selected)

1. UI → `SamplerData.constructor(model)` → closure from `_build_res4lyf_constructor`
2. Build `KDiffusionSampler(wrapped_func, model)`
3. On `wrapped_func(cfg_denoiser, x, ...)`:
   - `ensure_res4lyf_extra_args(kwargs["extra_args"])`
   - `with res4lyf_shim_context(cfg_denoiser):` inject `model_sampling` / `diffusion_model`
   - `k_diffusion.sampling.sample_<name>(...)` → `rk_sampler_beta.sample_rk_beta` internally
4. RES4LYF calls `self.model(x, sigma, **extra_args)`
5. Patched `CFGDenoiser.forward` ignores extra kwargs such as `model_options`

---

## 5. New files (full text and notes)

### 5.1 `modules/a1111_res4lyf_samplers.py` (full text)

```python
{samplers_doc}
```

#### 5.1.1 Function reference

| Function | Technical role |
|----------|----------------|
| `_ensure_comfyui_on_path` | RES4LYF requires `import comfy.*`. A1111 does not add `ComfyUI-master` automatically; glue inserts `script_path/ComfyUI-master` at `sys.path[0]` |
| `_install_optional_deps` | Some RES4LYF features need `pywavelets` / `comfy-kitchen`. Try pip at startup (Forge-style); registration continues on failure |
| `_mock_comfyui_globals` | Stub `folder_paths` / `server.PromptServer` when missing; prefer real modules if importable |
| `_register_extra_rk_beta_samplers` | **Forge parity.** Register all `RK_SAMPLER_NAMES_BETA_NO_FOLDERS` via `sample_rk_beta` closures. No `_ode` for implicit RK families. **Skip** names that collide with A1111 standard `sample_*` (critical bugfix) |
| `_build_res4lyf_constructor` | A1111 `SamplerData` uses `constructor(model) -> Sampler`. Returns `KDiffusionSampler` whose `func` is wrapped with shim for ComfyUI-compatible attributes at runtime only |
| `register_res4lyf_samplers` | Orchestrates glue: extend `all_samplers` / `all_samplers_map`, call `set_samplers()` |
| `register_res4lyf_schedulers` | Wrap ComfyUI scheduler signatures to A1111 `(n, sigma_min, sigma_max, inner_model, device)` |

#### 5.1.2 Dual `k_diffusion` module issue

- RES4LYF defines `sample_<name>` on **`comfy.k_diffusion.sampling`**
- A1111 `KDiffusionSampler` uses **`k_diffusion.sampling`** (Crowsonkb fork, different module object)
- Same process, different module instances → copy attributes in `register_res4lyf_samplers`:

```python
setattr(k_diffusion.sampling, f"sample_{{sampler_name}}", fn)
```

#### 5.1.3 Name collision protection (standard samplers)

`RK_SAMPLER_NAMES_BETA_NO_FOLDERS` includes `dpmpp_2m`, `euler`, `ddim`, etc.  
Overwriting `k_diffusion.sampling` would replace A1111 DPM++ 2M / Euler with RES4LYF implementations and **break standard samplers**.

Fix: snapshot existing `sample_*` names from `sd_samplers_kdiffusion.samplers_k_diffusion` and **skip** colliding RES4LYF entries (log: `Skipped N name(s)...`).

---

### 5.2 `modules/a1111_res4lyf_shim.py` (full text)

```python
{shim_src}
```

#### 5.2.1 Role of `A1111ModelSamplingShim`

- `rk_sampler_beta.py` calls `isinstance(model_sampling, EPS)` and `calculate_denoised` via **ComfyUI EPS**
- A1111 `CompVisDenoiser` has equivalent math but **different API shape**
- Shim delegates `sigmas` / `sigma_to_t` / `t_to_sigma` and subclasses EPS
- `sigma_data = 1.0` is ComfyUI EPS convention (SD1/SDXL EPS)

#### 5.2.2 `diffusion_model` alias via `__dict__`

Assigning `inner_ldm.diffusion_model = real_unet` normally goes through `torch.nn.Module.__setattr__` and may **double-register** the U-Net, breaking `.parameters()`.

```python
inner_ldm.__dict__["diffusion_model"] = real_unet
```

Lookup succeeds; `_modules` is not updated.

#### 5.2.3 `patch_cfg_denoiser_forward`

RES4LYF: `self.model(x, sigma, **extra_args)`  
After `ensure_res4lyf_extra_args`, `extra_args` includes `model_options`.  
A1111 `CFGDenoiser.forward` does not accept `**kwargs` → `TypeError`.

Idempotent monkey-patch: `inspect.signature` → filter kwargs. No-op for standard samplers without foreign keys.

#### 5.2.4 `functools.update_wrapper`

In `_build_res4lyf_constructor`, `update_wrapper(wrapped_func, original_func)` preserves signature metadata (`n`, `sigmas`, `sigma_min`, `sigma_max`, etc.) that `KDiffusionSampler` inspects.

---

## 6. Modified files (full text and notes)

### 6.1 `modules/initialize.py` (changed section only)

**Before (conceptual):**

```python
    from modules import sd_samplers
    sd_samplers.set_samplers()
    startup_timer.record("set samplers")

    from modules import extensions
```

**After (actual file):**

```python
    from modules import sd_samplers
    sd_samplers.set_samplers()
    startup_timer.record("set samplers")

    # Register RES4LYF samplers and schedulers (mirrors Forge's forge_res4lyf_samplers hook)
    try:
        from modules import a1111_res4lyf_samplers
        a1111_res4lyf_samplers.register_res4lyf_samplers()
        a1111_res4lyf_samplers.register_res4lyf_schedulers()
        startup_timer.record("register RES4LYF")
    except Exception:
        import traceback
        traceback.print_exc()

    from modules import extensions
```

#### 6.1.1 Why this placement

- **After** `sd_samplers.set_samplers()` — standard sampler list is fixed before RES4LYF is **appended**
- **Before** `extensions.list_extensions()` — native registration before extensions stabilizes UI rebuild timing
- `try/except` + `traceback` — RES4LYF failure does not block A1111 startup (Forge-style)
- Same path on `initialize_rest(reload_script_modules=True)` reload

---

## 7. Errors encountered and fixes

| # | Error | Cause | Fix |
|---|-------|-------|-----|
| 1 | `ModuleNotFoundError: torchaudio` | RES4LYF → `comfy.sd` dependency | `pip install torchaudio` in venv (manual) |
| 2 | `ImportError: cannot import name 'einsum' from 'einops'` | `einops==0.4.1` pin too old | Relax requirements to `einops>=0.4.1` |
| 3 | `ModuleNotFoundError: av` | ComfyUI dependency | `pip install av` (manual) |
| 4 | `AttributeError: ... has no attribute 'model_sampling'` | `LatentDiffusion` lacks ComfyUI attrs | `a1111_res4lyf_shim.py` + `res4lyf_shim_context` |
| 5 | `AttributeError: property 'device' ... has no setter` | shim tried to assign `device` | Use existing `LatentDiffusion.device` property |
| 6 | `TypeError: CFGDenoiser.forward() got an unexpected keyword argument 'model_options'` | Strict A1111 forward signature | `patch_cfg_denoiser_forward()` |
| 7 | `AttributeError: ... has no attribute 'diffusion_model'` | U-Net one level deeper on A1111 | `__dict__["diffusion_model"]` alias |
| 8 | Standard DPM++ / Euler broken | RES4LYF overwrote `sample_dpmpp_2m` etc. | Collision skip in `_register_extra_rk_beta_samplers` |

---

## 8. Differences from Forge

| Item | Forge-Nunchaku | A1111 (this integration) |
|------|----------------|--------------------------|
| Glue file | `modules_forge/forge_res4lyf_samplers.py` | `modules/a1111_res4lyf_samplers.py` |
| Startup hook | `initialize_forge()` | `initialize_rest()` |
| ComfyUI path | Forge backend prepared | `_ensure_comfyui_on_path()` |
| Sampler registration API | Extend `all_samplers` + `set_samplers()` | Same |
| Sampler class | `RES4LYFSampler(KDiffusionSampler)` | `_build_res4lyf_constructor` + shim `wrapped_func` |
| Dynamic RK registration | In Forge `RES4LYF/beta/__init__.py` | A1111 glue `_register_extra_rk_beta_samplers()` |
| `bong_tangent` | Native in `sd_schedulers.py` | Wrapped registration in glue |
| Model shim | Forge uses Comfy `BaseModel` | A1111-specific `a1111_res4lyf_shim.py` required |
| Name collision | Less problematic on Forge | **Explicit skip** required |

---

## 9. Limitations and future work

### 9.1 Current scope (Phase 1)

- Shim targets **SD1 / SDXL, EPS** parameterization
- Flux / HiDream paths (`double_stream_blocks`, etc.) **not supported** (usually not reached on A1111)

### 9.2 Names skipped from UI due to collision

RES4LYF variants of names that match A1111 standards (e.g. `dpmpp_2m`, `euler`, `ddim`) are **intentionally not registered**. Use the standard A1111 samplers for those names.

### 9.3 Related documentation

- `md/A1111_RES4LYF_SHIM_PLAN.md` — shim design, API surface, phases
- Forge reference: `Stable-Diffusion-WebUI-Forge-Nunchaku/modules_forge/forge_res4lyf_samplers.py`

---

## Appendix A: File tree (integration touch points only)

```
D:\\USERFILES\\A1111\\
├── modules\\
│   ├── a1111_res4lyf_samplers.py   [new]
│   ├── a1111_res4lyf_shim.py         [new]
│   ├── initialize.py                 [modified: +10 lines]
│   └── RES4LYF\\                    [unchanged, vendored]
├── ComfyUI-master\\                 [unchanged, vendored]
└── md\\
    ├── A1111_RES4LYF_INTEGRATION.md  [this document]
    └── A1111_RES4LYF_SHIM_PLAN.md    [plan]
```

---

## Appendix B: Verification log markers

Successful registration typically logs:

```
[RES4LYF] Added ComfyUI to sys.path: ...
[RES4LYF] Added N extra RK samplers (Forge parity)
[RES4LYF] Registered M samplers
[RES4LYF] Registered scheduler: beta57
[RES4LYF] Registered scheduler: bong_tangent
[RES4LYF shim] Patched CFGDenoiser.forward to drop unknown kwargs
```

On collision skip:

```
[RES4LYF] Skipped K name(s) that would overwrite A1111 standard samplers: [...]
```

---

*This document reflects `modules/a1111_res4lyf_samplers.py`, `modules/a1111_res4lyf_shim.py`, and `modules/initialize.py` as implemented in the repository.*
'''

out = root / "md/A1111_RES4LYF_INTEGRATION.md"
if __name__ == "__main__":
    out.write_text(prose, encoding="utf-8", newline="\n")
    remaining = len(re.findall(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]", prose))
    print(f"Wrote {out} ({len(prose.splitlines())} lines), remaining CJK: {remaining}")
