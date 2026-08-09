# Python 3.12 Compatibility (1.03, Windows / Linux / macOS)

This fork of Stable Diffusion WebUI supports Python 3.12. Modifications in this document target Windows build-error avoidance and Linux / macOS startup-failure avoidance.

This document reflects the repository state at `1.03` (commit `b5b1dd8` and later). For the full Linux / macOS fix design see [`LINUX_MAC_PY312_STARTUP_FIX.md`](LINUX_MAC_PY312_STARTUP_FIX.md). For the FA2 direct-load design see [`FA2_direct_load_design.md`](FA2_direct_load_design.md).

## Modifications

### 1. Python version check
**File**: `modules/launch_utils.py` (`check_python_version()`)

- Python 3.12 is added to the allowed list on Windows, Linux, and macOS.
- Upstream A1111 does not include 3.12 in its allow-list, so `check_python_version()` is extended here.
- The `--skip-python-version-check` flag suppresses the warning.

### 2. Per-platform requirements file auto-selection
**File**: `modules/launch_utils.py`

Because this fork is Python-3.12-only, startup auto-selects a requirements file by OS:

| Environment | Requirements file |
|-------------|-------------------|
| Windows | `requirements_versions_py314_windows.txt` |
| Linux / macOS | `requirements_versions_py314.txt` |

Both files are currently byte-identical (shared baseline). Platform-specific behaviour is absorbed by the launch_utils branches described below.

### 3. scikit-image build avoidance (Windows)
- `requirements_versions_py314_windows.txt` pins `scikit-image>=0.22.0`.
- A version with a prebuilt wheel is used so that a Visual Studio C/C++ toolchain is not required.
- Aligned with `numpy==1.26.4` so the dtype layout matches the scipy wheel described below.

### 4. torch / CUDA stack
**File**: `modules/launch_utils.py`

- `torch==2.10.0` + `torchvision`
- `TORCH_INDEX_URL`: `https://download.pytorch.org/whl/cu130` (CUDA 13.0 family)
- Significant jump from the earlier CUDA 11.8 / torch 2.1.x baseline.

### 5. Flash-Attention 2 — per-platform install
**File**: `modules/launch_utils.py`

FA2 source is branched by OS:

| Environment | Install source | Notes |
|-------------|----------------|-------|
| Windows | Prebuilt wheel (HuggingFace `ussoewwin/Flash-Attention-2_for_Windows`, cu130 + torch 2.10.0, cxx11abiTRUE, cp312) | Immediate install |
| Linux | PyPI `flash-attn==2.8.3` built from source (`--no-build-isolation`) | Requires CUDA toolkit / `nvcc`; ~30 min |
| macOS | Skipped | FA2 requires CUDA; MPS backend cannot use it |

Override with the `FLASH_ATTN_PACKAGE` environment variable. For rationale see [`FA2_direct_load_design.md`](FA2_direct_load_design.md).

### 6. SciPy — per-platform install
**File**: `modules/launch_utils.py`

scipy is sourced differently per OS to keep the dtype layout aligned with `numpy==1.26.4`:

| Environment | Install source |
|-------------|----------------|
| Windows | HuggingFace `ussoewwin/scipy-1.16.1-cp312-cp312-win_amd64` wheel, forced via `--no-deps --no-index` |
| Linux / macOS | PyPI `scipy==1.16.1` (manylinux / macosx wheel) |

Override (Windows only) with the `SCIPY_WHEEL` environment variable.

### 7. clip.py auto-fix path branching
**File**: `modules/launch_utils.py` (`fix_clip_packaging_import()`)

A patch that removes OpenAI CLIP's `pkg_resources` dependency is applied to the `clip.py` file inside the virtual environment. The target path is switched per OS:

- Windows: `venv/Lib/site-packages/clip/clip.py`
- Linux / macOS: `venv/lib/pythonX.Y/site-packages/clip/clip.py` (major/minor resolved dynamically from the running interpreter)

### 8. transformers 5.4+ baseline / 4.x shim removal
**Files**: `requirements_versions_py314_windows.txt`, `requirements_versions_py314.txt`, `modules/sd_disable_initialization.py`

- Baseline raised to `transformers==5.4.0`.
- The transformers 4.x compatibility hack in `sd_disable_initialization.py` has been removed.
- This resolves the `AttributeError` that occurred during quick model loading.

### 9. protobuf v7 compatibility
**Files**: `requirements_versions_py314_windows.txt`, `requirements_versions_py314.txt`

- Pinned to `protobuf==7.34.1` (previously 4.x; before that 3.20.x — a large jump).
- Upstream A1111 does not assume Python 3.12, so following the newer protobuf line is necessary.

### 10. OOM mitigation and FA2 direct-load
- `use_checkpoint: True` in `configs/v1-inference.yaml` (gradient checkpointing) is retained to reduce VRAM usage.
- FA2 is enabled via direct kernel load instead of via xformers (see [`FA2_direct_load_design.md`](FA2_direct_load_design.md)).
- The xformers auto-install block has been removed.

## Installation

### On Python 3.12
1. Ensure Python 3.12 is installed.
2. Change into this folder.
3. Run `webui.sh` (Linux / macOS) or `webui.bat` (Windows).

### Windows notes
- No Visual Studio install is required.
- FA2 / scipy are pulled from prebuilt wheels automatically.
- scikit-image prefers a prebuilt wheel as well.

### Linux notes
- Building FA2 from source requires the CUDA toolkit (`nvcc`) and takes roughly 30 minutes.
- To avoid the build, point `FLASH_ATTN_PACKAGE` at an alternate wheel.

### macOS notes
- FA2 install is skipped automatically (the MPS backend is not CUDA-compatible).
- Other Python 3.12 adjustments are shared with Linux.

## Troubleshooting

### scikit-image build error (Windows)
Allow only prebuilt wheels manually:
```cmd
pip install "scikit-image>=0.22.0" --only-binary=all
```

### Other package errors
Delete and recreate the virtual environment:
```cmd
rmdir /s venv
```
Restart the Web UI; dependencies will be reinstalled.

### Bypassing the Python version check
Add `--skip-python-version-check` to the launch arguments.

## File layout
```
stable-diffusion-webui/
├── md/
│   ├── PYTHON312_COMPATIBILITY.md         # This document
│   ├── LINUX_MAC_PY312_STARTUP_FIX.md     # Linux / macOS startup fix design
│   ├── FA2_direct_load_design.md          # FA2 direct-load design
│   └── INCIDENT_2026-04-22_A1111_Cursor_git_disabled.md
├── requirements_versions_py314.txt         # Linux / macOS
├── requirements_versions_py314_windows.txt # Windows
├── modules/launch_utils.py                 # Branching logic described above
├── modules/sd_disable_initialization.py    # transformers 5.x compatibility
└── configs/v1-inference.yaml               # OOM mitigation (use_checkpoint: True)
```

## Related documents
- [`LINUX_MAC_PY312_STARTUP_FIX.md`](LINUX_MAC_PY312_STARTUP_FIX.md): root-cause analysis and fix for Linux / macOS + Python 3.12 startup failures (FA2 / scipy / clip.py branching).
- [`FA2_direct_load_design.md`](FA2_direct_load_design.md): design for loading Flash-Attention 2 directly, bypassing xformers.
- [`INCIDENT_2026-04-22_A1111_Cursor_git_disabled.md`](INCIDENT_2026-04-22_A1111_Cursor_git_disabled.md): record of the `.git` → `.git_disabled` rename incident.
- [`CHANGELOG.md`](CHANGELOG.md): release notes (v1.01 through v1.03).

## Notes
- Python 3.12 is still relatively new; some extensions may have compatibility issues.
- On Windows, prebuilt wheels are preferred to avoid requiring Visual Studio.
- Use `--skip-python-version-check` to bypass the version check if needed.

## Revision info
- Last updated: 2026-04-23
- Target: Stable Diffusion WebUI master branch (ussoewwin/A1111-for-Python3.12, 1.03 equivalent)
- Scope: Python 3.12 compatibility across Windows / Linux / macOS
