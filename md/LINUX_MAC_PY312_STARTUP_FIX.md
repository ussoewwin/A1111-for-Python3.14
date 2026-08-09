# Linux / Mac Python 3.12 Startup Failure Fix

> **Current fork target (supersedes this document's Python 3.12 wording):** Python **3.14** only. Runtime pins and OS branching live in `modules/launch_utils.py` and the root `README.md`. Today: FA2 Windows HF `2.8.4` cp314 / Linux `flash-attn==2.8.4`; **`numpy==2.5.1` and `scipy==1.16.1` from PyPI on all platforms** (not `numpy==1.26.4`, not a local `whl/numpy-1.26.4` install, not a Windows-only SciPy HF wheel). `webui.sh` default `python_cmd=python3.14`. Sections below that mention `numpy==1.26.4` / HF SciPy are the **historical design record** of the original Linux/Mac branching work — not the live install path.

## 1. The Original Problem

This repository targets Windows + Python 3.12 as its primary platform for the A1111 (AUTOMATIC1111 Stable Diffusion WebUI) fork. Its `PYTHON312_COMPATIBILITY.md` claims platform-aware support for Linux / Mac + Python 3.12 as well.

In reality, launching `webui.sh` on Linux / Mac with Python 3.12 would **fail during startup** due to multiple compounding issues:

| Symptom | Location | Result |
|---|---|---|
| `FileNotFoundError: requirements_versions_py314.txt` | `modules/launch_utils.py` — referenced in the non-Windows Py3.12 branch | Exception before any dependency install |
| `pip install <Windows wheel URL>` rejected on other platforms | FA2 install (`flash_attn-2.8.3+cu130torch2.10.0cxx11abiTRUE-cp312-cp312-win_amd64.whl`) | Linux/Mac pip rejects `win_amd64` platform tag, install fails |
| `pip install <Windows wheel URL> --no-deps --no-index` unavoidable | scipy install (`scipy-1.16.1-cp312-cp312-win_amd64.whl`) | `--no-index` forces direct URL download, always fails on non-Windows |
| `clip.py` auto-fix never triggers | `fix_clip_packaging_import()` hardcodes the Windows venv layout (`venv\Lib\site-packages\clip\clip.py`) | Linux/Mac venv is `venv/lib/python3.12/site-packages/...`; the `os.path.isfile(...)` check is False, so the fix silently skips |

## 2. Root Cause

### 2-1. OS Target Bias

The Python 3.12 compatibility work for A1111 prioritized Windows, and the auto-install / auto-fix logic was written **assuming Windows-specific wheel URLs and venv paths**.

### 2-2. Missing Platform Branching

`modules/launch_utils.py` had no `platform.system()` branching in its dependency install logic; it unconditionally executed Windows-targeted behavior.

### 2-3. `--no-index` Blocks the Escape Route

A plain `pip install <Windows wheel URL>` could have let pip detect the platform tag mismatch and skip on Linux/Mac. However, the scipy install used `--no-index` together with a direct URL, bypassing pip's platform check and **always attempting — always failing — the install** on non-Windows.

### 2-4. Missing Requirements File

`launch_utils.py` referenced `requirements_versions_py314.txt` for non-Windows Py3.12, but **the file did not exist in the repository**.

## 3. Fix Strategy

Overall approach:

- Introduce `platform.system()` branching at **three points** in `modules/launch_utils.py`.
- Preserve the Windows path as **byte-identical** (verified by runtime simulation that the emitted command strings do not differ by a single byte).
- Add Linux / Mac paths with platform-appropriate dependency acquisition.
- Create the missing requirements file.

| # | Point | Windows | Linux | Mac |
|---|---|---|---|---|
| A | FA2 install | HF prebuilt `torch2.10.0 cu130 win_amd64` wheel (unchanged) | `pip install flash-attn==2.8.3 --no-build-isolation` (PyPI source build, ~30 min, requires CUDA toolkit + nvcc + gcc) | skipped (`fa2_install_enabled=False`) — FA2 is CUDA-only, MPS backend cannot use it |
| B | scipy install (historical) | HF prebuilt `cp312 win_amd64` wheel (`--no-index`, then aligned to era `numpy==1.26.4`) | PyPI `scipy==1.16.1` (manylinux) | PyPI `scipy==1.16.1` (macosx) |
| C | `clip_py_path` | `venv\Lib\site-packages\clip\clip.py` (unchanged) | `venv/lib/python{M}.{N}/site-packages/clip/clip.py` | same as Linux |
| D | requirements file | `requirements_versions_py314_windows.txt` (existing) | `requirements_versions_py314.txt` (newly created by this fix) | same as Linux |

## 4. Modified Files

| File | Change | Commit |
|---|---|---|
| `modules/launch_utils.py` | FA2 platform branch (L360-372) / scipy platform branch (L526-540) / `clip_py_path` platform branch (L534-540) | `0346830` (FA2) + `12eeee0` (scipy + clip) |
| `requirements_versions_py314_windows.txt` | Bumped to `transformers==5.4.0` / `protobuf==7.34.1` (precondition for this fix) | `0346830` |
| `requirements_versions_py314.txt` (new) | Requirements for Linux / Mac Py3.12. Byte-identical to the Windows version (platform-specific packages are handled by the platform branching in `launch_utils.py`, so a shared file is sufficient) | `0346830` |

## 5. Code and What It Means

### 5-1. `flash_attn_package` Definition — Platform Branching (`modules/launch_utils.py`)

```python
# Flash-Attention 2 source is platform-specific:
#   Windows: prebuilt wheel (cu130 + torch 2.10)
#   Linux:   source build via PyPI (requires CUDA toolkit + nvcc, ~30min compile)
#   Mac:     skipped (FA2 requires CUDA; MPS backend cannot use it)
if platform.system() == "Windows":
    flash_attn_package = os.environ.get('FLASH_ATTN_PACKAGE', 'https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows/resolve/main/flash_attn-2.8.3%2Bcu130torch2.10.0cxx11abiTRUE-cp312-cp312-win_amd64.whl')
    fa2_install_enabled = True
elif platform.system() == "Linux":
    flash_attn_package = os.environ.get('FLASH_ATTN_PACKAGE', 'flash-attn==2.8.3')
    fa2_install_enabled = True
else:
    flash_attn_package = None
    fa2_install_enabled = False
```

**What it means:**

- **Windows** keeps fetching the self-built HuggingFace wheel (`cu130 + torch 2.10 + cxx11abiTRUE + cp312`) directly (~30 sec).
- **Linux** pulls the PyPI `flash-attn==2.8.3` sdist and compiles it locally with `--no-build-isolation`. A decision was made not to publish a Linux wheel; Linux users build from source (~30 min) even if it takes 30 minutes.
- **Mac** sets `fa2_install_enabled=False`, which causes the downstream install block to be skipped entirely. FA2 is a CUDA kernel — it cannot run on Metal / MPS.

### 5-2. FA2 Install Block — Platform Branching (`modules/launch_utils.py`)

```python
if fa2_install_enabled and not is_installed("flash_attn"):
    if platform.system() == "Linux":
        run_pip(f"install {flash_attn_package} --no-build-isolation", "flash_attn")
    else:
        run_pip(f"install {flash_attn_package}", "flash_attn")
    startup_timer.record("install flash_attn")
```

**What it means:**

- The outer `fa2_install_enabled` guard causes Mac (and any unknown platform) to skip the install entirely.
- Linux alone adds `--no-build-isolation`, which lets the source build reuse torch / packaging from the active venv as build dependencies. Without this flag, PEP 517's isolated build environment would not have torch and the build would fail.
- Windows falls through to `else` with no `--no-build-isolation` (not needed for a prebuilt wheel). The emitted install command is byte-identical to the pre-edit version.

### 5-3. `scipy_wheel` Install — Platform Branching (`modules/launch_utils.py`)

```python
# scipy install: platform-specific to match numpy 1.26.4 dtype layout
#   Windows: HuggingFace prebuilt cp312 win_amd64 wheel (forced via --no-index)
#   Linux/Mac: PyPI scipy==1.16.1 (manylinux/macosx wheel)
if platform.system() == "Windows":
    scipy_wheel = os.environ.get('SCIPY_WHEEL', 'https://huggingface.co/ussoewwin/scipy-1.16.1-cp312-cp312-win_amd64/resolve/main/scipy-1.16.1-cp312-cp312-win_amd64.whl')
    run(f'"{python}" -m pip uninstall scipy -y', "uninstalling scipy", "Couldn't uninstall scipy", live=False)
    run(f'"{python}" -m pip install {scipy_wheel} --no-deps --no-index', "Installing scipy", "Couldn't install scipy", live=False)
    print("[INFO] Installed scipy from HuggingFace to fix numpy dtype size incompatibility")
else:
    run(f'"{python}" -m pip uninstall scipy -y', "uninstalling scipy", "Couldn't uninstall scipy", live=False)
    run(f'"{python}" -m pip install --no-cache-dir scipy==1.16.1', "Installing scipy 1.16.1", "Couldn't install scipy 1.16.1", live=False)
    print("[INFO] Installed scipy 1.16.1 from PyPI")
```

**What it means:**

- **(Historical)** Windows forced the HF prebuilt wheel via `--no-index` while the repo still pinned `numpy==1.26.4`; that SciPy wheel matched that NumPy dtype layout.
- **(Historical)** Linux / Mac already pulled `scipy==1.16.1` from PyPI via platform tags.
- **Current (v3.0.0 / Python 3.14):** all platforms install `numpy==2.5.1` then `scipy==1.16.1` from PyPI (`modules/launch_utils.py`). No HF SciPy URL and no local NumPy 1.26.4 wheel on the install path.

### 5-4. `clip_py_path` — Platform Branching (`modules/launch_utils.py`)

```python
def fix_clip_packaging_import():
    if platform.system() == "Windows":
        clip_py_path = os.path.join(script_path, "venv", "Lib", "site-packages", "clip", "clip.py")
    else:
        py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        clip_py_path = os.path.join(script_path, "venv", "lib", py_ver, "site-packages", "clip", "clip.py")
    if os.path.isfile(clip_py_path):
        # (omitted: both fixes below are platform-agnostic)
        # Fix 1: "from pkg_resources import packaging" → "from packaging import version"
        # Fix 2: "packaging.version.parse"            → "version.parse"
```

**What it means:**

- openai/CLIP's `clip.py` uses `pkg_resources.packaging`, which Python 3.12 removed, so A1111 patches `clip.py` in place right after install.
- Windows venv layout: `venv\Lib\site-packages\...` (capital `L`).
- Linux / Mac venv layout: `venv/lib/python{major}.{minor}/site-packages/...` (lowercase `l` plus a versioned intermediate directory).
- With this branch, the auto-fix triggers on both platforms. The Windows path is not changed by a single character.

## 6. Windows Invariance Verification

The following were verified to ensure the **Windows path is exactly equivalent to the pre-edit version**:

1. **Logical equivalence**: When `platform.system() == "Windows"` is True, the branched code executes the same sequence of statements, in the same order, with the same variable values, line by line.
2. **Emitted command byte-identity**: The actual command strings passed to `run(...)` / `run_pip(...)` on Windows match the pre-edit version byte for byte (verified by Python simulation).
3. **`git diff` sanity check**: Across the two commits (`0346830`, `12eeee0`), `git diff` shows no unintended changes — every touched line is an intended platform branch.
4. **Size / sha256 check**: Byte size delta direction matched expectation for each commit (FA2: +769B, scipy+clip: +787B). SHA256 changed as expected.

## 7. Runtime Notes for Linux / Mac

### 7-1. Linux

- FA2 source build requires a **CUDA toolkit + nvcc + gcc**.
- Build time is roughly 30 minutes on typical hardware; longer for exotic CUDA architectures.
- The build can peak above 16 GB of memory; ensure swap / RAM is adequate.
- **Current** runtime stack: `numpy==2.5.1` / `scipy==1.16.1` / default torch `2.13.0+cu132` (see README). The older assumption (`numpy==1.26.4` / torch 2.10.0 + cu130) is historical only.

### 7-2. Mac

- FA2 is not installed on either Apple Silicon or Intel (skipped).
- Attention layers fall back to the existing paths in `modules/sd_hijack_optimizations.py` (`sub_quadratic_attention`, SDP, split attention, etc.). Operation is possible without FA2.
- For MPS-backend memory pressure, `sd_hijack_optimizations.py` contributes `SDP_ATTNBLOCK_MAX_SEQ` tuning, `empty_cache()` hooks, and `sub_quad_attention` fallback.

## 8. Related Commits (Chronological)

| Commit | Subject |
|---|---|
| `0346830` | `env: bump torch 2.10.0 / transformers 5.4.0 / protobuf 7.34.1, add FA2 platform branching (Win wheel / Linux source build / Mac skip)` |
| `29ba620` | `compat: drop transformers 4.x shims now that 5.4.0+ is baseline (fix quick-load AttributeError)` |
| `2b70c68` | `config: normalize v1-inference.yaml to 2-space indent, keep use_checkpoint: True for VRAM saving` |
| `f438e7d` | `optim: promote A1111 OOM prevention and FA2 direct-load optimizations to fork` |
| `12eeee0` | `env: branch scipy install and clip.py fix path for Linux/Mac (Windows byte-identical)` |

At the time this document was authored, the A1111 local working tree matched `origin/main` (GitHub fork) at `12eeee0`.
