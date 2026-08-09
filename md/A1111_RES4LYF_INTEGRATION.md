# A1111 RES4LYF Integration — Complete Technical Guide

**Target repository:** `D:\USERFILES\A1111`  
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
"""
A1111 integration for RES4LYF samplers and schedulers.

A1111 port of Forge (Stable-Diffusion-WebUI-Forge-Nunchaku)
`modules_forge/forge_res4lyf_samplers.py`.

Main differences from Forge:
- A1111 has no ``sd_samplers.add_sampler()``;
  extend ``all_samplers`` directly and call ``set_samplers()`` again
- ComfyUI runtime (``ComfyUI-master``) is not added to sys.path by
  ``modules/paths.py``; this module adds it
- ``bong_tangent`` was native in Forge ``sd_schedulers.py`` but
  missing on A1111; registered here with ``beta57``
- ``comfy.samplers.beta_scheduler`` / RES4LYF ``sigmas.bong_tangent_scheduler`` use
  ComfyUI signatures; wrapped for A1111
  ``(n, sigma_min, sigma_max, inner_model, device)``
"""

from __future__ import annotations

import logging
import os
import sys
import types
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def _ensure_comfyui_on_path() -> Optional[str]:
    """Insert ``ComfyUI-master`` on sys.path if not already present."""
    from modules.paths_internal import script_path
    comfyui_path = os.path.join(script_path, "ComfyUI-master")
    if not os.path.isdir(comfyui_path):
        logger.warning(f"[RES4LYF] ComfyUI-master not found at: {comfyui_path}")
        return None
    if comfyui_path not in sys.path:
        sys.path.insert(0, comfyui_path)
        logger.info(f"[RES4LYF] Added ComfyUI to sys.path: {comfyui_path}")
    return comfyui_path


def _install_optional_deps() -> None:
    """Install ``pywavelets`` / ``comfy-kitchen`` via pip when possible; non-fatal on failure."""
    try:
        from modules import launch_utils
    except ImportError:
        return
    is_installed = launch_utils.is_installed
    run_pip = launch_utils.run_pip
    if not is_installed("pywavelets"):
        try:
            run_pip("install pywavelets", "pywavelets")
            logger.info("[RES4LYF] Installed pywavelets")
        except Exception as e:
            logger.warning(f"[RES4LYF] Failed to install pywavelets: {e}")
    if not is_installed("comfy-kitchen") and not is_installed("comfy_kitchen"):
        try:
            run_pip("install comfy-kitchen", "comfy-kitchen")
            logger.info("[RES4LYF] Installed comfy-kitchen")
        except Exception as e:
            logger.warning(f"[RES4LYF] Failed to install comfy-kitchen: {e}")


def _mock_comfyui_globals() -> None:
    """
    Inject ``folder_paths`` and ``server.PromptServer`` so RES4LYF can import
    without running a full ComfyUI server.
    Prefer the real ``folder_paths.py`` from ``ComfyUI-master`` when available.
    """
    if 'folder_paths' not in sys.modules:
        try:
            import folder_paths  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            mock_module = types.ModuleType('folder_paths')

            mock_module.get_filename_list = lambda *_a, **_kw: []

            def _raise_not_found(folder_type, filename):
                raise FileNotFoundError(f"Mock folder_paths: {folder_type}/{filename}")

            mock_module.get_full_path_or_raise = _raise_not_found
            mock_module.get_input_directory = lambda: ""
            mock_module.get_output_directory = lambda: ""
            mock_module.get_save_image_path = lambda *a, **kw: ("", "", 0, "", "")
            sys.modules['folder_paths'] = mock_module

    if 'server' not in sys.modules:
        class MockRoutes:
            def post(self, path):
                def decorator(func):
                    return func
                return decorator

        class MockPromptServerInstance:
            def __init__(self):
                self.routes = MockRoutes()
                self.client_id = None
                self.supports = set()

            def send_sync(self, event_type, data):
                pass

            async def send(self, event_type, data):
                pass

        class MockPromptServer:
            instance = MockPromptServerInstance()

        server_module = types.ModuleType('server')
        server_module.PromptServer = MockPromptServer
        sys.modules['server'] = server_module


# Sampler names that must not have an auto-generated ``_ode`` variant.
# These are implicit RK families where ``eta=0`` is meaningless / redundant.
# Mirrors the keyword list used in
# ``modules_forge/../modules/RES4LYF/beta/__init__.py`` (Forge fork).
_IMPLICIT_RK_KEYWORDS = (
    "gauss-legendre",
    "radau",
    "lobatto",
    "irk_exp_diag",
    "kraaijevanger",
    "qin_zhang",
    "pareschi",
    "crouzeix",
)


def _register_extra_rk_beta_samplers() -> list:
    """
    Add the dynamically-generated RK sampler set that the Forge fork of
    RES4LYF exposes.

    The RES4LYF copy shipped in A1111's ``modules/RES4LYF`` is closer to
    upstream and its ``beta/__init__.py`` only registers ~17 hand-picked
    names. The Forge fork instead iterates
    ``RK_SAMPLER_NAMES_BETA_NO_FOLDERS`` and creates one sampler per
    entry (plus an ``_ode`` variant for non-implicit families).

    Both copies share the same sampler-name list
    (``rk_coefficients_beta.py`` is byte-identical), so we can reproduce
    the Forge behaviour on the A1111 side without editing RES4LYF at
    all.

    Returns
    -------
    list[str]
        Sampler names newly registered by this call. Empty if all were
        already registered (which happens if the RES4LYF source has been
        updated to the Forge behaviour).
    """
    try:
        from modules.RES4LYF.beta.rk_coefficients_beta import (
            RK_SAMPLER_NAMES_BETA_NO_FOLDERS,
        )
        from modules.RES4LYF.beta import rk_sampler_beta
        from modules import RES4LYF
    except ImportError as e:
        logger.warning(f"[RES4LYF] Failed to import RK sampler internals: {e}")
        return []

    extra = getattr(RES4LYF, "extra_samplers", None)
    if extra is None:
        logger.warning("[RES4LYF] RES4LYF.extra_samplers not available")
        return []

    # Never overwrite names used by A1111 standard samplers in
    # ``k_diffusion.sampling.sample_<name>``.
    # ``RK_SAMPLER_NAMES_BETA_NO_FOLDERS`` includes ``dpmpp_2m``, ``dpmpp_3m``,
    # ``dpmpp_2s``, ``dpmpp_sde_2s``, ``dpmpp_3s``, ``euler``, ``ddim``, etc.—same
    # names as A1111 DPM++ 2M / Euler / DDIM.
    # Overwriting via ``setattr(k_diffusion.sampling, "sample_<name>", ...)``
    # replaces ``getattr(k_diffusion.sampling, "sample_<name>")`` used by
    # standard samplers and breaks them.
    protected_funcnames = set()
    try:
        from modules import sd_samplers_kdiffusion
        for entry in sd_samplers_kdiffusion.samplers_k_diffusion:
            fn = entry[1]
            if isinstance(fn, str):
                protected_funcnames.add(fn)
    except Exception:
        logger.exception("[RES4LYF] Failed to snapshot A1111 standard sampler funcnames")

    def _make_sample_fn(rk_type):
        def sample_fn(model, x, sigmas, extra_args=None, callback=None, disable=None):
            return rk_sampler_beta.sample_rk_beta(
                model, x, sigmas, None, extra_args, callback, disable,
                rk_type=rk_type,
            )
        return sample_fn

    def _make_sample_ode_fn(rk_type):
        def sample_ode_fn(model, x, sigmas, extra_args=None, callback=None, disable=None):
            return rk_sampler_beta.sample_rk_beta(
                model, x, sigmas, None, extra_args, callback, disable,
                rk_type=rk_type, eta=0.0, eta_substep=0.0,
            )
        return sample_ode_fn

    added_names = []
    skipped_collisions = []
    for name in RK_SAMPLER_NAMES_BETA_NO_FOLDERS:
        if name == "none":
            continue
        if f"sample_{name}" in protected_funcnames:
            # Collision with standard sampler name; skip.
            skipped_collisions.append(name)
            continue
        if name not in extra:
            extra[name] = _make_sample_fn(name)
            added_names.append(name)
        if not any(kw in name for kw in _IMPLICIT_RK_KEYWORDS):
            ode_name = f"{name}_ode"
            if f"sample_{ode_name}" in protected_funcnames:
                skipped_collisions.append(ode_name)
                continue
            if ode_name not in extra:
                extra[ode_name] = _make_sample_ode_fn(name)
                added_names.append(ode_name)
    if skipped_collisions:
        logger.info(
            f"[RES4LYF] Skipped {len(skipped_collisions)} name(s) that would "
            f"overwrite A1111 standard samplers: {skipped_collisions}"
        )

    # Mirror RES4LYF/__init__.py::add_samplers so that comfy.k_diffusion.sampling
    # also carries a ``sample_<name>`` attribute for the newly-added samplers.
    # This is required for the later comfy -> k_diffusion sync step to find
    # the function.
    if added_names:
        try:
            from comfy.samplers import KSampler, k_diffusion_sampling
            for name in added_names:
                if name not in KSampler.SAMPLERS:
                    try:
                        idx = KSampler.SAMPLERS.index("uni_pc_bh2")
                        KSampler.SAMPLERS.insert(idx + 1, name)
                    except ValueError:
                        pass
                setattr(k_diffusion_sampling, f"sample_{name}", extra[name])
        except Exception:
            logger.exception("[RES4LYF] Failed to publish extra RK samplers to comfy.k_diffusion.sampling")

    return added_names


def _build_res4lyf_constructor(sampler_key: str) -> Callable:
    """Return a closure for ``SamplerData.constructor``.

    Wraps the ``func`` passed to A1111 ``KDiffusionSampler`` with
    :func:`modules.a1111_res4lyf_shim.res4lyf_shim_context` where RES4LYF
    needs ComfyUI-compatible APIs.
    """
    def constructor(model):
        import functools

        from modules import sd_samplers_kdiffusion
        import k_diffusion.sampling
        from modules.a1111_res4lyf_shim import (
            ensure_res4lyf_extra_args,
            res4lyf_shim_context,
        )

        original_func = getattr(k_diffusion.sampling, f"sample_{sampler_key}", None)
        if original_func is None:
            raise ValueError(f"Unknown RES4LYF sampler: {sampler_key}")

        def wrapped_func(cfg_denoiser, x, *args, **kwargs):
            ensure_res4lyf_extra_args(kwargs.get("extra_args"))
            with res4lyf_shim_context(cfg_denoiser):
                return original_func(cfg_denoiser, x, *args, **kwargs)

        # KDiffusionSampler inspects signature(self.func).parameters for
        # n, sigmas, sigma_min, sigma_max, etc.; preserve metadata.
        functools.update_wrapper(wrapped_func, original_func)

        return sd_samplers_kdiffusion.KDiffusionSampler(wrapped_func, model)
    return constructor


def register_res4lyf_samplers() -> None:
    """Register RES4LYF samplers into A1111 ``all_samplers``."""
    try:
        _install_optional_deps()
        _mock_comfyui_globals()
        comfyui_path = _ensure_comfyui_on_path()
        if comfyui_path is None:
            logger.warning("[RES4LYF] Aborting sampler registration (ComfyUI-master missing)")
            return

        # Import RES4LYF; __init__.py add_samplers() builds comfy.k_diffusion.sample_*
        from modules import RES4LYF
        import comfy.k_diffusion.sampling as comfy_k_diffusion_sampling
        import k_diffusion.sampling

        # Forge fork registers all RK samplers in beta/__init__.py; upstream A1111
        # copy registers ~17. Add the rest here without editing RES4LYF.
        extra_added = _register_extra_rk_beta_samplers()
        if extra_added:
            logger.info(f"[RES4LYF] Added {len(extra_added)} extra RK samplers (Forge parity)")

        extra_samplers = getattr(RES4LYF, 'extra_samplers', {})
        if not extra_samplers:
            logger.info("[RES4LYF] No samplers found in extra_samplers")
            return

        # Copy comfy.k_diffusion.sampling -> k_diffusion.sampling
        if k_diffusion.sampling is not comfy_k_diffusion_sampling:
            for sampler_name in extra_samplers:
                fn = getattr(comfy_k_diffusion_sampling, f"sample_{sampler_name}", None)
                if fn is not None:
                    setattr(k_diffusion.sampling, f"sample_{sampler_name}", fn)

        from modules import sd_samplers, sd_samplers_common

        existing_names = {x.name for x in sd_samplers.all_samplers}
        added = 0
        for sampler_name in extra_samplers:
            if sampler_name in existing_names:
                continue
            if getattr(k_diffusion.sampling, f"sample_{sampler_name}", None) is None:
                logger.warning(
                    f"[RES4LYF] sample_{sampler_name} not found in k_diffusion.sampling after sync"
                )
                continue
            data = sd_samplers_common.SamplerData(
                sampler_name,
                _build_res4lyf_constructor(sampler_key=sampler_name),
                [sampler_name],
                {}
            )
            sd_samplers.all_samplers.append(data)
            sd_samplers.all_samplers_map[data.name] = data
            added += 1

        if added > 0:
            sd_samplers.set_samplers()
            logger.info(f"[RES4LYF] Registered {added} samplers")
        else:
            logger.info("[RES4LYF] No new samplers to register")

        # Install CFGDenoiser.forward wrapper so ComfyUI-style kwargs that
        # RES4LYF pushes through ``**extra_args`` don't crash A1111's strict
        # forward signature. Idempotent.
        try:
            from modules.a1111_res4lyf_shim import patch_cfg_denoiser_forward
            patch_cfg_denoiser_forward()
        except Exception as e:
            logger.warning(f"[RES4LYF] Failed to patch CFGDenoiser.forward: {e}")

    except ImportError as e:
        logger.warning(f"[RES4LYF] Failed to import RES4LYF: {e}")
    except Exception as e:
        logger.error(f"[RES4LYF] Error registering samplers: {e}", exc_info=True)


def register_res4lyf_schedulers() -> None:
    """
Register RES4LYF schedulers into A1111 ``sd_schedulers``.

    - ``beta57``: wrap ComfyUI ``beta_scheduler(model_sampling, steps, alpha=0.5, beta=0.7)`` to
      A1111 ``(n, sigma_min, sigma_max, inner_model, device)``
    - ``bong_tangent``: wrap RES4LYF ``sigmas.bong_tangent_scheduler(...)`` similarly
    """
    try:
        _ensure_comfyui_on_path()
        _mock_comfyui_globals()

        from modules import sd_schedulers, RES4LYF  # noqa: F401
        from modules.RES4LYF import sigmas as res4lyf_sigmas
        from comfy import samplers as comfy_samplers
        import torch

        def _to_tensor(sigs, device):
            if isinstance(sigs, torch.Tensor):
                return sigs.to(device)
            return torch.as_tensor(sigs, dtype=torch.float32).to(device)

        def beta57_scheduler(n, sigma_min, sigma_max, inner_model, device):
            sigs = comfy_samplers.beta_scheduler(inner_model, n, alpha=0.5, beta=0.7)
            return _to_tensor(sigs, device)

        def bong_tangent(n, sigma_min, sigma_max, inner_model, device):
            sigs = res4lyf_sigmas.bong_tangent_scheduler(inner_model, n)
            return _to_tensor(sigs, device)

        existing = {s.name for s in sd_schedulers.schedulers}
        registered = 0

        if "beta57" not in existing:
            sch = sd_schedulers.Scheduler(
                "beta57", "Beta57", beta57_scheduler, need_inner_model=True
            )
            sd_schedulers.schedulers.append(sch)
            sd_schedulers.schedulers_map[sch.name] = sch
            sd_schedulers.schedulers_map[sch.label] = sch
            registered += 1
            logger.info("[RES4LYF] Registered scheduler: beta57")

        if "bong_tangent" not in existing:
            sch = sd_schedulers.Scheduler(
                "bong_tangent", "Bong Tangent", bong_tangent, need_inner_model=True
            )
            sd_schedulers.schedulers.append(sch)
            sd_schedulers.schedulers_map[sch.name] = sch
            sd_schedulers.schedulers_map[sch.label] = sch
            registered += 1
            logger.info("[RES4LYF] Registered scheduler: bong_tangent")

        if registered > 0:
            logger.info(f"[RES4LYF] Registered {registered} schedulers")
        else:
            logger.info("[RES4LYF] No new schedulers to register")

    except ImportError as e:
        logger.warning(f"[RES4LYF] Failed to import RES4LYF for schedulers: {e}")
    except Exception as e:
        logger.error(f"[RES4LYF] Error registering schedulers: {e}", exc_info=True)

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
setattr(k_diffusion.sampling, f"sample_{sampler_name}", fn)
```

#### 5.1.3 Name collision protection (standard samplers)

`RK_SAMPLER_NAMES_BETA_NO_FOLDERS` includes `dpmpp_2m`, `euler`, `ddim`, etc.  
Overwriting `k_diffusion.sampling` would replace A1111 DPM++ 2M / Euler with RES4LYF implementations and **break standard samplers**.

Fix: snapshot existing `sample_*` names from `sd_samplers_kdiffusion.samplers_k_diffusion` and **skip** colliding RES4LYF entries (log: `Skipped N name(s)...`).

---

### 5.2 `modules/a1111_res4lyf_shim.py` (full text)

```python
"""
Bridging shim so RES4LYF samplers can run against A1111's model stack.

RES4LYF was written against ComfyUI. Its sampler code (e.g.
``modules/RES4LYF/beta/rk_sampler_beta.py``) traverses
``model.inner_model.inner_model.model_sampling`` and
``model.inner_model.inner_model.device``. In A1111 that innermost
object is ``ldm.models.diffusion.ddpm.LatentDiffusion`` which has no
such attributes.

This module provides:

- ``A1111ModelSamplingShim``: subclass of ``comfy.model_sampling.EPS``
  that borrows sigmas / sigma-t transforms from A1111's k-diffusion
  ``CompVisDenoiser`` and passes ``isinstance(_, EPS)`` checks used by
  RES4LYF.
- ``res4lyf_shim_context``: context manager that temporarily attaches
  ``model_sampling`` and ``device`` onto the ``LatentDiffusion`` object
  for the duration of one sampler run, then restores the original state.

Scope (Phase 1): SD1 / SDXL, EPS parameterization only. See
``md/A1111_RES4LYF_SHIM_PLAN.md`` for the full plan and roadmap.
"""

from __future__ import annotations

import inspect
import logging
import sys
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)


def _get_eps_base():
    """Import ``comfy.model_sampling.EPS`` lazily.

    Import is deferred so this module can be imported at A1111 startup
    even before ``ComfyUI-master`` has been added to ``sys.path``
    (``a1111_res4lyf_samplers._ensure_comfyui_on_path`` runs at
    registration time).
    """
    from comfy.model_sampling import EPS  # noqa: WPS433
    return EPS


class _ShimBase:
    """Placeholder replaced by the real EPS-derived class at first call.

    We can't inherit from ``EPS`` at module import time because
    ``comfy.model_sampling`` may not yet be importable. Instead we build
    the concrete class lazily inside :func:`build_shim`.
    """


_shim_cls_cache = None


def build_shim(comp_vis_denoiser):
    """Return an instance of the EPS-derived shim.

    Class is cached so ``isinstance(shim, EPS)`` returns True consistently
    across calls (a fresh class per call would break equality-style
    checks in RES4LYF).
    """
    global _shim_cls_cache
    if _shim_cls_cache is None:
        eps_cls = _get_eps_base()

        class A1111ModelSamplingShim(eps_cls):
            """
            Bridges A1111's ``CompVisDenoiser`` into ComfyUI's
            ``ModelSamplingDiscrete + EPS`` API.

            Only EPS parameterization (SD1 / SDXL). V_PREDICTION and
            later families are Phase 2+.
            """

            sigma_data = 1.0

            def __init__(self, cvd):
                self._cvd = cvd

            @property
            def sigmas(self):
                return self._cvd.sigmas

            @property
            def log_sigmas(self):
                return self._cvd.log_sigmas

            @property
            def sigma_min(self):
                return self._cvd.sigmas[0]

            @property
            def sigma_max(self):
                return self._cvd.sigmas[-1]

            def timestep(self, sigma):
                return self._cvd.sigma_to_t(sigma)

            def sigma(self, timestep):
                return self._cvd.t_to_sigma(timestep)

            def percent_to_sigma(self, percent):
                if percent <= 0.0:
                    return 999999999.9
                if percent >= 1.0:
                    return 0.0
                percent = 1.0 - percent
                return self.sigma(torch.tensor(percent * 999.0)).item()

            # calculate_input / calculate_denoised / noise_scaling /
            # inverse_noise_scaling are inherited from EPS.

        _shim_cls_cache = A1111ModelSamplingShim

    return _shim_cls_cache(comp_vis_denoiser)


_cfg_forward_patched = False


def patch_cfg_denoiser_forward():
    """Idempotently wrap ``CFGDenoiser.forward`` to silently drop kwargs
    it does not accept.

    RES4LYF sampler internals do ``self.model(x, sigma, **extra_args)``.
    Once :func:`ensure_res4lyf_extra_args` has added ``model_options``
    (etc.) to ``extra_args`` on the sampler side, those keys leak into
    ``CFGDenoiser.forward`` and blow up because A1111's forward has a
    strict signature (``x, sigma, uncond, cond, cond_scale,
    s_min_uncond, image_cond``).

    The wrapper filters keyword arguments to only those accepted by the
    real ``forward`` signature. For all non-RES4LYF sampler paths on
    A1111 the ``extra_args`` dict never contains foreign keys, so the
    wrapper is a no-op there.

    Safe to call multiple times. First call installs the wrapper;
    subsequent calls are no-ops.
    """
    global _cfg_forward_patched
    if _cfg_forward_patched:
        return
    try:
        from modules.sd_samplers_cfg_denoiser import CFGDenoiser
    except ImportError:
        logger.warning("[RES4LYF shim] CFGDenoiser not importable; skip forward patch")
        return

    original_forward = CFGDenoiser.forward
    sig = inspect.signature(original_forward)
    has_var_keyword = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if has_var_keyword:
        _cfg_forward_patched = True
        return
    accepted = {
        name
        for name, p in sig.parameters.items()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and name != "self"
    }

    def patched_forward(self, *args, **kwargs):
        if kwargs:
            filtered = {k: v for k, v in kwargs.items() if k in accepted}
        else:
            filtered = kwargs
        return original_forward(self, *args, **filtered)

    CFGDenoiser.forward = patched_forward
    _cfg_forward_patched = True
    logger.info("[RES4LYF shim] Patched CFGDenoiser.forward to drop unknown kwargs")


def ensure_res4lyf_extra_args(extra_args):
    """Ensure ``extra_args`` has the ComfyUI-style nesting RES4LYF expects.

    RES4LYF assigns into ``extra_args['model_options']['transformer_options']``
    directly. A1111's ``sampler_extra_args`` only contains ``cond`` /
    ``image_cond`` / ``uncond`` / ``cond_scale`` / ``s_min_uncond``, so we
    add the missing keys with empty dicts. Idempotent; safe to call
    multiple times.

    ``extra_args`` is mutated in place. Non-dict input is a no-op.
    """
    if not isinstance(extra_args, dict):
        return
    extra_args.setdefault("model_options", {})
    mo = extra_args["model_options"]
    if isinstance(mo, dict):
        mo.setdefault("transformer_options", {})


@contextmanager
def res4lyf_shim_context(cfg_denoiser):
    """Attach ComfyUI-compatible attributes for one RES4LYF sampler run.

    Parameters
    ----------
    cfg_denoiser :
        The ``CFGDenoiser`` instance passed as the first positional
        argument to any k-diffusion ``sample_*`` function on A1111.
        We expect:
        - ``cfg_denoiser.inner_model`` -> k-diffusion ``CompVisDenoiser``
        - ``cfg_denoiser.inner_model.inner_model`` -> ``LatentDiffusion``

    Notes
    -----
    ``LatentDiffusion.device`` already exists as a read-only ``@property``,
    so we do **not** attempt to set it. RES4LYF's ``model.inner_model.inner_model.device``
    read will resolve to the property getter, which returns the model's
    parameter device.

    Yields
    ------
    None
        The context body runs with the shim installed. Original state
        is restored on exit (both normal and exceptional).
    """
    inner_ldm = cfg_denoiser.inner_model.inner_model
    comp_vis = cfg_denoiser.inner_model

    # --- 1. install model_sampling ------------------------------------
    had_model_sampling = hasattr(inner_ldm, "model_sampling")
    original_model_sampling = getattr(inner_ldm, "model_sampling", None)

    try:
        inner_ldm.model_sampling = build_shim(comp_vis)
    except Exception:
        logger.exception("[RES4LYF shim] Failed to build model_sampling shim")
        yield
        return

    # --- 2. install diffusion_model alias (A1111 LDM has model.diffusion_model,
    #        ComfyUI BaseModel has .diffusion_model directly) ---------
    #
    # We assign via ``__dict__`` directly so torch.nn.Module.__setattr__ does
    # not register the U-Net twice under _modules (which would double up in
    # .parameters()/.state_dict()). Attribute lookup on the instance still
    # finds __dict__ before Module.__getattr__ fires.
    diffusion_installed = False
    original_diffusion_present = "diffusion_model" in inner_ldm.__dict__
    original_diffusion_val = inner_ldm.__dict__.get("diffusion_model", None)
    try:
        wrapper = getattr(inner_ldm, "model", None)
        real_unet = getattr(wrapper, "diffusion_model", None) if wrapper is not None else None
        if real_unet is not None:
            inner_ldm.__dict__["diffusion_model"] = real_unet
            diffusion_installed = True
    except Exception:
        logger.exception("[RES4LYF shim] Failed to alias diffusion_model")

    # --- 3. patch RES4LYF rk_sampler_beta torch.zeros for data_prev_* buffers ---
    #
    # RES4LYF allocates ``data_prev_`` / ``data_prev_x_`` / ``data_prev_y_``
    # as ``torch.zeros(4, *x.shape, ...)`` (rk_sampler_beta.py lines 744, 746,
    # 1134, 1135) — hardcoded size 4. For hybrid samplers like
    # ``lawson45-gen-mod_4h4s`` where ``hybrid_stages = 4``, ``recycled_stages``
    # becomes 4 (rk_sampler_beta.py line 725), and the rotation loop at line
    # 2028-2029 writes ``data_prev_[4]`` → IndexError (size 4, valid 0..3).
    #
    # We cannot edit ``modules/RES4LYF/``. Instead, we replace ``torch.zeros``
    # **only in the ``rk_sampler_beta`` module's globals** (not the global
    # ``torch`` package) with a wrapper that grows any size-4 first-dim
    # allocation to 8 — enough for all current hybrid_stages values
    # (max 4 → need 5 slots, 8 gives headroom). Other torch.zeros calls
    # (different leading dim) pass through unchanged.
    rk_sampler_beta_mod = None
    try:
        import importlib
        # RES4LYF is imported as a subpackage of ``modules``; the fully
        # qualified name is ``RES4LYF.beta.rk_sampler_beta``. The bare
        # ``beta.rk_sampler_beta`` form only resolves when CWD is inside
        # the RES4LYF package, which is not the case at runtime.
        for _modname in (
            "RES4LYF.beta.rk_sampler_beta",
            "modules.RES4LYF.beta.rk_sampler_beta",
            "beta.rk_sampler_beta",
        ):
            _m = sys.modules.get(_modname)
            if _m is not None:
                rk_sampler_beta_mod = _m
                break
        if rk_sampler_beta_mod is None:
            for _modname in (
                "RES4LYF.beta.rk_sampler_beta",
                "modules.RES4LYF.beta.rk_sampler_beta",
                "beta.rk_sampler_beta",
            ):
                try:
                    rk_sampler_beta_mod = importlib.import_module(_modname)
                    break
                except Exception:
                    continue
    except Exception:
        rk_sampler_beta_mod = None

    zeros_patched = False
    _rk_torch = None
    if rk_sampler_beta_mod is not None:
        # Inject a module-level helper that shadows torch.zeros only inside
        # rk_sampler_beta's global scope. We do this by replacing the
        # ``torch`` attribute the module sees with a lightweight namespace
        # that proxies all torch.* calls but intercepts ``zeros``.
        try:
            _rk_torch = rk_sampler_beta_mod.torch
            _orig_torch_zeros = _rk_torch.zeros

            def _patched_zeros(*args, **kwargs):
                if args and isinstance(args[0], int) and args[0] == 4:
                    args = (8,) + args[1:]
                return _orig_torch_zeros(*args, **kwargs)

            # Build a proxy that forwards every attribute to real torch except
            # ``zeros`` which goes through our wrapper.
            class _TorchProxy:
                def __getattr__(self, name):
                    return getattr(_rk_torch, name)

                def zeros(self, *args, **kwargs):
                    return _patched_zeros(*args, **kwargs)

            rk_sampler_beta_mod.torch = _TorchProxy()
            zeros_patched = True
        except Exception:
            logger.debug("[RES4LYF shim] Failed to patch rk_sampler_beta.torch.zeros", exc_info=True)

    try:
        yield
    finally:
        # --- restore rk_sampler_beta.torch ---
        if zeros_patched and rk_sampler_beta_mod is not None:
            try:
                rk_sampler_beta_mod.torch = _rk_torch
            except Exception:
                logger.debug("[RES4LYF shim] restore rk_sampler_beta.torch failed", exc_info=True)
        # --- restore diffusion_model ---
        if diffusion_installed:
            try:
                if original_diffusion_present:
                    inner_ldm.__dict__["diffusion_model"] = original_diffusion_val
                else:
                    inner_ldm.__dict__.pop("diffusion_model", None)
            except Exception:
                logger.debug(
                    "[RES4LYF shim] cleanup diffusion_model failed",
                    exc_info=True,
                )
        # --- restore model_sampling ---
        try:
            if had_model_sampling:
                inner_ldm.model_sampling = original_model_sampling
            else:
                try:
                    delattr(inner_ldm, "model_sampling")
                except AttributeError:
                    pass
        except Exception:
            logger.debug(
                "[RES4LYF shim] cleanup: could not restore model_sampling",
                exc_info=True,
            )

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
D:\USERFILES\A1111\
├── modules\
│   ├── a1111_res4lyf_samplers.py   [new]
│   ├── a1111_res4lyf_shim.py         [new]
│   ├── initialize.py                 [modified: +10 lines]
│   └── RES4LYF\                    [unchanged, vendored]
├── ComfyUI-master\                 [unchanged, vendored]
└── md\
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
