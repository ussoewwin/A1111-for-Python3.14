# 🎉 MAJOR UPDATE: v3.0.0 - Python 3.14 Full Renewal!

**First release of `ussoewwin/A1111-for-Python3.14`.** The fork is now **Python 3.14–only**, with a renewed default CUDA stack (`torch==2.13.0+cu132` / FA2 **2.8.4**) and critical CPython 3.14 runtime fixes.

---

## Overview

**v3.0.0** is the first release of this fork as **`ussoewwin/A1111-for-Python3.14`**.

Through **v2.3.5**, the published surface was **`ussoewwin/A1111-for-Python3.12`**. From **v3.0.0**, the fork is a **Python 3.14–only** renewal: install docs, launch scripts, dependency pins, Flash-Attention 2, and runtime gates all assume **CPython 3.14**. Other Python versions are not supported.

Tag commit: `d9433e42`  
Repository: https://github.com/ussoewwin/A1111-for-Python3.14

---

## Highlights

| Area | v3.0.0 |
|------|--------|
| Python | **3.14 only** (`check_python_version` / README / `webui.sh`) |
| Default CUDA torch (Win/Linux) | `torch==2.13.0+cu132` + `torchvision==0.28.0+cu132` |
| Torchaudio | **Not** in the default first-install command (cu132 index has no matching torchaudio; audio paths lazy-import) |
| Flash-Attention 2 | **2.8.4** aligned to torch 2.13.0+cu132 |
| CUDA toolkit (Linux FA2 source build) | **13.2** + `nvcc` |
| NumPy / SciPy | `numpy==2.4.6` and `scipy==1.16.1` from **PyPI** (all platforms); `facexlib==0.3.0` enabled |
| ONNX Runtime | Windows: `onnxruntime-gpu` from **ort-cuda-13-nightly** on first launch; Linux: PyPI `onnxruntime-gpu`; macOS: `onnxruntime` |
| Requirements files | Renamed to `requirements_versions_py314*.txt` |
| Critical runtime fix | CPython 3.14 `shared.sd_model` LOAD_ATTR / `None` shadow |

---

## 1. Python 3.14–only platform

### What changed

- Install and platform docs state: **This repository supports Python 3.14 only.**
- `modules/launch_utils.py` rejects other majors/minors unless `--skip-python-version-check` is used (not recommended).
- `webui.sh` / `webui-macos-env.sh` defaults aligned to the 3.14 / torch 2.13 stack.
- Dependency requirement files renamed from `*_py312*` to `*_py314*` / `*_py314_windows*`.
- Distutils stub install retained for 3.14 (stdlib `distutils` removed upstream).

### Migration from A1111-for-Python3.12

1. Install **Python 3.14**.
2. Recreate `venv` (do not reuse a 3.12 venv).
3. Follow README Windows / Linux / macOS sections for torch + FA2.
4. Expect extension compatibility gaps; not all extensions are verified on 3.14.

---

## 2. Default first-install CUDA stack (Windows / Linux)

Default environment variables / command (overridable):

```text
TORCH_INDEX_URL = https://download.pytorch.org/whl/cu132
TORCH_COMMAND   = pip install torch==2.13.0+cu132 torchvision==0.28.0+cu132 --index-url <TORCH_INDEX_URL>
```

### Design goals

- **Same pin on Windows and Linux** so Linux FA2 source builds against the same torch ABI family as the Windows FA2 wheel.
- Match Flash-Attention 2 Windows artifact: `2.8.4+cu132torch2.13.0` (cp314).
- **Omit torchaudio** from the default install: the cu132 index does not ship a matching torchaudio for this pin. ComfyUI-master paths that need torchaudio use **lazy imports** so missing torchaudio does not break unrelated model loads.

### Override

Set `TORCH_COMMAND` / `TORCH_INDEX_URL` before launch if you need another CUDA / CPU build.

---

## 3. Flash-Attention 2 (2.8.4)

| Platform | Method |
|----------|--------|
| **Windows** | Hugging Face prebuilt wheel `flash_attn-2.8.4+cu132torch2.13.0` (cp314), via `FLASH_ATTN_PACKAGE` / launch_utils |
| **Linux** | PyPI `flash-attn==2.8.4` source build with `--no-build-isolation` against installed `torch==2.13.0+cu132`; needs **CUDA toolkit 13.2** and `nvcc` on `PATH` (often ~30 minutes) |
| **macOS** | FA2 skipped (MPS limitation) |

README Platform Support / Default Package Versions / Linux install steps were rewritten to this stack (not older FA2 2.9.x / cu130 pins).

---

## 4. CPython 3.14 `shared.sd_model` LOAD_ATTR / `None` shadow (critical)

### Symptom class

On CPython 3.14, after the first successful property access, specializing **`LOAD_ATTR`** could return **`None`** for `shared.sd_model` / `p.sd_model` even while `model_data` still held a live model. ControlNet, MultiDiffusion, Tiled VAE, and checkpoint switch paths then saw `sd_model is None` and failed or behaved inconsistently.

### Root cause (fork analysis)

If both:

1. `Shared.sd_model` as a **data descriptor** (property), and  
2. `modules.shared.__dict__['sd_model'] = None`  

exist, 3.14 attribute load specialization can prefer the dict slot and return the shadowed `None`.

### Countermeasure (high level)

- Do **not** bind `sd_model = None` into `modules.shared` module dict; keep annotation / Shared property only (`modules/shared.py`, `modules/shared_items.py`).
- Strengthen load / keep-old / Processing paths so callers read the live model from `model_data` / guarded accessors (`modules/sd_models.py`, `modules/processing.py`).
- Tiled VAE / related guards updated so they do not assume a stale `None` module attribute (`extensions-builtin/.../tilevae.py` and related).

Commit: `88646c18` — *fix: CPython 3.14 shared.sd_model LOAD_ATTR None shadow*

---

## 5. Checkpoint / VAE switch guards and LoRA safety

Part of the 3.14 defaults commit (`8588e0ca`):

- Guards around model / VAE switch so callers do not hit `NoneType` during checkpoint transitions.
- LoRA network path adjustments for safer behavior when the active model is mid-switch.
- Related touch-ups in textual inversion / sd_vae helpers.

---

## 6. NumPy / SciPy (`numpy==2.4.6`)

- Startup installs **`numpy==2.4.6`** from PyPI on **all platforms** (`modules/launch_utils.py`), then pins again after extension installers so drift cannot stick.
- Requirements: `requirements_versions_py314.txt` / `requirements_versions_py314_windows.txt` pin `numpy==2.4.6`.
- **SciPy** is `scipy==1.16.1` from PyPI, aligned to that NumPy pin (no Windows-only HF scipy wheel forced onto Linux).
- **`facexlib==0.3.0`** is installed from requirements so GFPGAN / CodeFormer face helpers and ControlNet PuLID preprocessors load on first install. NumPy stays on **2.4.6** because current numba rejects `numpy>=2.5` at import.
- Interim `whl/numpy-1.26.4-cp314-…` work (commit `15d8a90b`) was **never** the v3.0.0 install path; runtime installs NumPy from PyPI only. That interim wheel is removed from the tree so docs cannot point at a local 1.26.4 artifact as current.

## 6b. ONNX Runtime (InsightFace / ReActor / ADetailer)

- On **Windows**, first launch runs:
  `pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ onnxruntime-gpu`
  via `modules/launch_utils.py` (before extension installers), so InsightFace and ReActor do not fail with `No module named 'onnxruntime'`.
- Linux installs PyPI `onnxruntime-gpu`; macOS installs `onnxruntime`.
- ADetailer no longer force-installs the CPU `onnxruntime` package name (which conflicts with the `onnxruntime-gpu` distribution).

---

## 7. Vendored `ComfyUI-master` sync

`ComfyUI-master/` (used for RES4LYF / `comfy.*` runtime) was synced with upstream ComfyUI updates in `8588e0ca`, including newer models / nodes / API surfaces.

Follow-up in `3b873426`: **lazy torchaudio imports** in:

- `ComfyUI-master/comfy/text_encoders/gemma4.py`
- `ComfyUI-master/comfy/ldm/lightricks/vae/audio_vae.py`

so optional audio dependencies are not required for the default A1111 torch install.

---

## 8. Documentation

- README: Python 3.14-only, Platform Support table, Default Package Versions, Windows / Linux / macOS install aligned to torch 2.13.0+cu132 and FA2 2.8.4.
- Compatibility / Linux–macOS notes under `md/` updated for the same stack.
- `md/CHANGELOG.md`: header updated through **v3.0.0**; notes the transition from `A1111-for-Python3.12` to `A1111-for-Python3.14`.

---

## 9. Commits included in this tag (Python 3.14 renewal line)

```text
8588e0ca  feat: Python 3.14 defaults, model/VAE switch guards, sync ComfyUI-master
15d8a90b  feat: replace tracked numpy wheel cp312 with cp314
88646c18  fix: CPython 3.14 shared.sd_model LOAD_ATTR None shadow
3b873426  feat: default torch 2.13.0+cu132 and align Linux README/FA2
d9433e42  docs: add CHANGELOG v3.0.0 Python 3.14 renewal overview
```

Earlier v2.3.x work (RES4LYF, Forge tiled VAE, ControlNet bf16, LoRA logging, etc.) remains in history from the 3.12-era lineage; this release focuses on the **3.14 renewal** delta.

---

## 10. Suggested verify after upgrade

1. Fresh **Python 3.14** venv; confirm startup does not reject the interpreter.
2. Confirm installed torch prints `2.13.0+cu132` (or your override) and matching torchvision.
3. Confirm `numpy==2.4.6` (and `scipy==1.16.1`, `facexlib==0.3.0`) from PyPI — not a local `whl/numpy-1.26.4` install.
4. Confirm `import onnxruntime` works (Windows: `onnxruntime-gpu` from ort-cuda-13-nightly).
4. Windows: FA2 wheel installs; Linux: FA2 source build completes with toolkit 13.2.
5. Load SD1.5 / SDXL, run txt2img with ControlNet and/or MultiDiffusion if you use them — confirm no intermittent `sd_model is None`.
6. Switch checkpoints / VAE mid-session without LoRA / VAE `NoneType` crashes.
7. Optional: RES4LYF hybrid `*4h4s` still completes (shim from prior releases).

---

## Links

- Tag: https://github.com/ussoewwin/A1111-for-Python3.14/releases/tag/v3.0.0
- Changelog: https://github.com/ussoewwin/A1111-for-Python3.14/blob/main/md/CHANGELOG.md
- Prior 3.12 fork (historical): https://github.com/ussoewwin/A1111-for-Python3.12
