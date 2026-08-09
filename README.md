# Stable Diffusion web UI

A web interface for Stable Diffusion, implemented using the Gradio library.
## Key Features & Improvements

## 🎉 MAJOR UPDATE: v3.0.0 - Python 3.14 Full Renewal!

**This repository is Python 3.14–only** (`ussoewwin/A1111-for-Python3.14`). Default first-install CUDA stack: `torch==2.13.0+cu132` + `torchvision==0.28.0+cu132`, Flash-Attention 2 **2.8.4**, plus the CPython 3.14 `shared.sd_model` LOAD_ATTR fix.

For detailed technical explanation, see [v3.0.0 Release Notes](https://github.com/ussoewwin/A1111-for-Python3.14/releases/tag/v3.0.0)

## 🎉 MAJOR UPDATE: v2.0 - Pony and Illustrious (IL) SDXL Full Support Added!

For detailed technical explanation, see [v2.0 Release Notes](https://github.com/ussoewwin/A1111-for-Python3.12/releases/tag/v2.0)

### SDXL Pony/Illustrious Compatibility Fix (v2.0+)

**The first A1111 fork to fully support Pony and Illustrious SDXL models — including LoRA.**

For years, SDXL derivative models (Pony Diffusion, WAI Illustrious, etc.) were unreliable on A1111. Enthusiasts had to switch to ComfyUI or Forge to use these models properly.

**What works now:**
- ✅ Pony series — base generation + LoRA
- ✅ Illustrious series — base generation + LoRA
- ✅ Any SDXL model that previously crashed with `RuntimeError: attn_mask shape` or produced noise

### Built-in Extensions

The following popular extensions are built-in and ready to use out of the box:
- **ControlNet** (v1.1.455) — integrated at startup with automatic path migration
- **ADetailer** (v1.01.0) — face and person detailer with 7 models
- **FreeU** — effortless detail enhancement
- **WD14 Tagger** — automatic prompt generation from images
- **ReActor** (v0.7.1-b3) — face swap
- **Dynamic Thresholding** (CFG Scale fix)
- **Aspect Ratio** presets (built-in, no extension needed)
- **MultiDiffusion + Tiled VAE** — high-resolution upscaling

### RES4LYF Samplers

Native support for **[RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF)** samplers in the A1111 **Sampling method** dropdown (100+ advanced RES / Bongmath / hybrid RK methods), plus related schedule types such as `beta57` and `bong_tangent`. Generation runs through an A1111-side shim; the vendored RES4LYF sources under `modules/RES4LYF/` are left unmodified.

### Python 3.14 Native

Fully ported to Python 3.14. No `pkg_resources` hacks, no legacy compatibility layers. All dependency conflicts (NumPy, SciPy, `clip.py`) are handled automatically at startup.

### Flash-Attention 2 with Graceful Fallback

Direct Flash-Attention 2 support with staged fallback:
```
1. FA-2 (Flash-Attention 2.8.4, torch 2.13.0+cu132) — maximum speed
2. SDP (PyTorch scaled_dot_product_attention) — no extra deps
3. sub_quad (built-in) — universal fallback
```

Windows: prebuilt HF wheel (`2.8.4+cu132torch2.13.0` cp314). Linux: builds `flash-attn==2.8.4` from source against the same `torch==2.13.0+cu132` stack (CUDA toolkit 13.2 + `nvcc`). macOS skips FA2 (MPS limitation).


## Python Version Support

**This repository supports Python 3.14 only.**

Other Python versions are not supported. Please ensure you are using Python 3.14 before proceeding with installation.

**Note:** Not all extensions may be compatible with Python 3.14. Some extensions may require additional modifications or may not work correctly.

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Windows  | Fully supported | PyTorch `2.13.0+cu132`; FA2 prebuilt `2.8.4+cu132torch2.13.0`; NumPy / SciPy from PyPI; Insightface via pip |
| Linux    | Supported | Same PyTorch `2.13.0+cu132` / `torchvision==0.28.0+cu132`; FA2 source-builds `flash-attn==2.8.4` (CUDA toolkit **13.2** + `nvcc`, ~30 min) |
| macOS    | Supported (limited) | PyTorch `2.13.0` CPU/MPS (`torchvision==0.28.0`); Flash-Attention 2 is skipped (CUDA required; MPS cannot use FA2) |

All platform-specific handling is performed automatically by `modules/launch_utils.py` at startup. This fork supports **Python 3.14 only**. Linux / macOS branches are additive (FA2 path, venv `site-packages` layout for `clip.py` auto-fix).

## Default Package Versions

The following packages are installed automatically during initial setup (see `modules/launch_utils.py`):

- **PyTorch** (suggested manual install): `2.13.0+cu132` with matching `torchvision==0.28.0+cu132` (override with `TORCH_COMMAND` / `TORCH_INDEX_URL` as needed; matches FA2 `cu132torch2.13.0`)
- **Flash-Attention 2**:
  - Windows: `flash_attn` `2.8.4+cu132torch2.13.0` cp314 prebuilt wheel (HF)
  - Linux: `flash-attn==2.8.4` (PyPI source build against `torch==2.13.0+cu132`; CUDA toolkit 13.2 + `nvcc`; `--no-build-isolation`)
  - macOS: skipped
- **transformers**: 5.4.0
- **protobuf**: 7.34.1
- **scipy**: 1.16.1 (PyPI, all platforms)
- **numpy**: 2.4.6 (PyPI, all platforms; numba / facexlib compatible)
- **Gradio**: 3.41.2 (HF wheel with METADATA pins removed)

## Installation

Pick the section for your OS and follow the steps in order.

### Windows

**Toolchain prerequisites**
- No additional toolchain required (all wheels are prebuilt).

**Install steps**

1. Install Python 3.14.

2. Create and activate a virtual environment:
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

3. Upgrade pip:
   ```cmd
   python -m pip install --upgrade pip
   ```

4. Install PyTorch 2.13.0+cu132 (aligned with Flash-Attention 2):
   ```cmd
   pip install torch==2.13.0+cu132 torchvision==0.28.0+cu132 --index-url https://download.pytorch.org/whl/cu132
   ```

5. Install cross-platform Python deps:
   ```cmd
   pip install importlib_metadata onnx polygraphy coloredlogs flatbuffers packaging protobuf sympy
   ```

6. Install Triton (Windows prebuilt):
   ```cmd
   pip install triton-windows
   ```

7. ONNX Runtime GPU is installed automatically on first launch (`modules/launch_utils.py`) from PyPI:
   ```cmd
   pip install onnxruntime-gpu
   ```
   Manual install of the same command is only needed if you used `--skip-install`.

8. Install Insightface:
   ```cmd
   pip install insightface
   ```

9. Launch:
   ```cmd
   webui-user.bat
   ```

### Linux

**Toolchain prerequisites**
- CUDA toolkit 13.2 with `nvcc` on `PATH` (required to build Flash-Attention 2 from source; matches `torch==2.13.0+cu132`).
- Build toolchain (`gcc`, `g++`, `make`, Python headers).
- First startup spends ~30 minutes building FA2. To use an alternate prebuilt wheel instead:
  ```bash
  export FLASH_ATTN_PACKAGE=<url-or-wheel-path>
  ```

**Install steps**

1. Install Python 3.14.

2. Create and activate a virtual environment:
   ```bash
   python3.14 -m venv venv
   source venv/bin/activate
   ```

3. Upgrade pip:
   ```bash
   python -m pip install --upgrade pip
   ```

4. Install PyTorch 2.13.0+cu132 (aligned with Flash-Attention 2):
   ```bash
   pip install torch==2.13.0+cu132 torchvision==0.28.0+cu132 --index-url https://download.pytorch.org/whl/cu132
   ```

5. Install cross-platform Python deps:
   ```bash
   pip install importlib_metadata onnx polygraphy coloredlogs flatbuffers packaging protobuf sympy
   ```

6. Install Triton (stock PyPI manylinux wheel):
   ```bash
   pip install triton
   ```

7. ONNX Runtime GPU is installed automatically on first launch (`modules/launch_utils.py`):
   ```bash
   pip install onnxruntime-gpu
   ```
   Manual install is only needed if you used `--skip-install`.

8. Install Insightface:
   ```bash
   pip install insightface
   ```

9. Launch (`webui.sh` defaults to `python3.14`; override with `python_cmd` in `webui-user.sh` if needed):
   ```bash
   ./webui.sh
   ```

### macOS

**Toolchain prerequisites**
- Xcode Command Line Tools (`xcode-select --install`) for source builds.
- Flash-Attention 2 is skipped automatically (the MPS backend is not CUDA-compatible).
- Triton is not supported by upstream Triton on macOS; skip.

**Install steps**

1. Install Python 3.14.

2. Create and activate a virtual environment:
   ```bash
   python3.14 -m venv venv
   source venv/bin/activate
   ```

3. Upgrade pip:
   ```bash
   python -m pip install --upgrade pip
   ```

4. Install PyTorch 2.13.0 (CPU / MPS; version-aligned with the CUDA stack, FA2 skipped on macOS):
   ```bash
   pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
   ```

5. Install cross-platform Python deps:
   ```bash
   pip install importlib_metadata onnx polygraphy coloredlogs flatbuffers packaging protobuf sympy
   ```

6. Install ONNX Runtime (CPU, or CoreML on Apple Silicon):
   ```bash
   pip install onnxruntime
   # or, on Apple Silicon:
   pip install onnxruntime-coreml
   ```

7. Install Insightface:
   ```bash
   pip install insightface
   ```

8. Launch:
   ```bash
   ./webui.sh
   ```

## How Linux / macOS support works

`modules/launch_utils.py` branches by `platform.system()` at startup:

- **PyTorch default** (Windows / Linux CUDA): `torch==2.13.0+cu132` + `torchvision==0.28.0+cu132` via `TORCH_COMMAND` / `TORCH_INDEX_URL` (`https://download.pytorch.org/whl/cu132`). Same pin on first install for both platforms so Linux FA2 builds against the Windows FA2 stack.
- **Requirements file**: Windows → `requirements_versions_py314_windows.txt`; Linux / macOS → `requirements_versions_py314.txt`.
- **Flash-Attention 2**: Windows HF prebuilt `flash_attn-2.8.4+cu132torch2.13.0` cp314 / Linux PyPI `flash-attn==2.8.4` with `--no-build-isolation` (needs CUDA toolkit 13.2 + `nvcc`) / macOS skip.
- **NumPy / SciPy**: all platforms install `numpy==2.4.6` and `scipy==1.16.1` from PyPI (no Windows-only wheel forced onto Linux). NumPy is pinned to 2.4.6 so numba/facexlib work on first install.
- **ONNX Runtime**: Windows / Linux first launch installs PyPI `onnxruntime-gpu` (InsightFace / ReActor / ADetailer). macOS uses `onnxruntime`.
- **`clip.py` `pkg_resources` auto-fix**: `venv/Lib/...` on Windows; `venv/lib/pythonX.Y/...` on Linux / macOS (major / minor from the running interpreter).
- **Python gate**: `check_python_version()` allows **3.14 only** (`webui.sh` default `python_cmd` is `python3.14`).

Historical design notes (early Linux/Mac branching under the older 3.12 docs): [`md/LINUX_MAC_PY312_STARTUP_FIX.md`](md/LINUX_MAC_PY312_STARTUP_FIX.md), [`md/PYTHON312_COMPATIBILITY.md`](md/PYTHON312_COMPATIBILITY.md). Prefer this README and `launch_utils.py` for current pins.

## Changelog

See [`md/CHANGELOG.md`](md/CHANGELOG.md) for the full change history.

## License

This project is licensed under **AGPL-3.0** (GNU Affero General Public License v3.0).

### Licenses of Base Repositories

This project is built upon the following repositories, each with their respective licenses:

- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** — GPL-3.0 — The most powerful and modular diffusion model GUI, API and backend with a graph/nodes interface by [Comfy-Org](https://github.com/Comfy-Org).
- **[RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF)** — AGPL-3.0 - Commercial AI image generation service use requires separate permission and/or a commercial license from the copyright holder. — ComfyUI custom node for advanced RES/Bongmath samplers and schedulers.
- **[stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)** - AGPL-3.0
- **[ADetailer](https://github.com/Bing-su/adetailer)** - AGPL-3.0
- **[sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet)** - GPL-3.0
- **[multidiffusion-upscaler-for-automatic1111](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111)** - CC BY-NC-SA 4.0
- **[stable-diffusion-webui-wd14-tagger](https://github.com/picobyte/stable-diffusion-webui-wd14-tagger)** - Public Domain
- **[ControlAltAI-Nodes](https://github.com/gseth/ControlAltAI-Nodes)** - MIT
- **[sd-webui-freeu](https://github.com/ShinChaser/sd-webui-freeu)** — MIT
- **[sd-dynamic-thresholding](https://github.com/mcmonkeyprojects/sd-dynamic-thresholding)** — MIT

See [LICENSE](LICENSE) file for details.

## Documentation

For detailed features, installation instructions, and usage documentation, please refer to the official upstream repository:

https://github.com/AUTOMATIC1111/stable-diffusion-webui
