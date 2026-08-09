# Stable Diffusion web UI

A web interface for Stable Diffusion, implemented using the Gradio library.
## Key Features & Improvements

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
1. FA-2 (Flash-Attention 2.9.1) — maximum speed
2. SDP (PyTorch scaled_dot_product_attention) — no extra deps
3. sub_quad (built-in) — universal fallback
```

Prebuilt Windows wheels included. Linux builds from source automatically. macOS skips FA2 (MPS limitation).


## Python Version Support

**This repository supports Python 3.14 only.**

Other Python versions are not supported. Please ensure you are using Python 3.14 before proceeding with installation.

**Note:** Not all extensions may be compatible with Python 3.14. Some extensions may require additional modifications or may not work correctly.

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Windows  | Fully supported | Prebuilt wheels for FA2 / SciPy / NumPy; Insightface via pip |
| Linux    | Supported | Flash-Attention 2 is built from source (requires CUDA toolkit + `nvcc`, ~30 min) |
| macOS    | Supported (limited) | Flash-Attention 2 is skipped (CUDA required; MPS backend cannot use FA2) |

All platform-specific handling is performed automatically by `modules/launch_utils.py` at startup. Windows install flow is byte-identical to the pre-1.03 behaviour; Linux / macOS branches are additive.

## Default Package Versions

The following packages are installed automatically during initial setup:

- **PyTorch**: 2.11.0+cu130 (CUDA 13.0), with matching `torchvision==0.26.0+cu130` and `torchaudio==2.11.0+cu130`
- **Flash-Attention 2**:
  - Windows: `2.8.3+cu130torch2.10.0` (prebuilt wheel)
  - Linux: `flash-attn==2.8.3` (source build)
  - macOS: skipped
- **transformers**: 5.4.0+
- **protobuf**: 7.34.1
- **scipy**: 1.16.1
- **numpy**: 1.26.4

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

4. Install PyTorch 2.11.0+cu130:
   ```cmd
   pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 torchaudio==2.11.0+cu130 --index-url https://download.pytorch.org/whl/cu130
   ```

5. Install cross-platform Python deps:
   ```cmd
   pip install importlib_metadata onnx polygraphy coloredlogs flatbuffers packaging protobuf sympy
   ```

6. Install Triton (Windows prebuilt):
   ```cmd
   pip install triton-windows
   ```

7. Install ONNX Runtime GPU (Windows CUDA 13 nightly feed):
   ```cmd
   pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ onnxruntime-gpu
   ```

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
- CUDA toolkit 13.0 with `nvcc` on `PATH` (required to build Flash-Attention 2 from source).
- Build toolchain (`gcc`, `g++`, `make`, Python headers).
- First startup spends ~30 minutes building FA2. To use an alternate prebuilt wheel instead:
  ```bash
  export FLASH_ATTN_PACKAGE=<url-or-wheel-path>
  ```

**Install steps**

1. Install Python 3.14.

2. Create and activate a virtual environment:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

3. Upgrade pip:
   ```bash
   python -m pip install --upgrade pip
   ```

4. Install PyTorch 2.11.0+cu130:
   ```bash
   pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 torchaudio==2.11.0+cu130 --index-url https://download.pytorch.org/whl/cu130
   ```

5. Install cross-platform Python deps:
   ```bash
   pip install importlib_metadata onnx polygraphy coloredlogs flatbuffers packaging protobuf sympy
   ```

6. Install Triton (stock PyPI manylinux wheel):
   ```bash
   pip install triton
   ```

7. Install ONNX Runtime GPU:
   ```bash
   pip install onnxruntime-gpu
   ```

8. Install Insightface:
   ```bash
   pip install insightface
   ```

9. Launch:
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
   python3.12 -m venv venv
   source venv/bin/activate
   ```

3. Upgrade pip:
   ```bash
   python -m pip install --upgrade pip
   ```

4. Install PyTorch 2.11.0 (CPU / MPS):
   ```bash
   pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
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

- **Flash-Attention 2**: Windows wheel / Linux `--no-build-isolation` source build / macOS skip.
- **SciPy**: Windows HuggingFace prebuilt wheel / Linux + macOS PyPI `scipy==1.16.1`.
- **NumPy**: local Windows `whl/numpy-*.whl` is used when present (Windows only); otherwise NumPy is installed from PyPI at the pinned version.
- **clip.py `pkg_resources` auto-fix**: targets `venv/Lib/...` on Windows and `venv/lib/pythonX.Y/...` on Linux / macOS (major / minor resolved dynamically).

See [`md/LINUX_MAC_PY312_STARTUP_FIX.md`](md/LINUX_MAC_PY312_STARTUP_FIX.md) for the full fix design and [`md/PYTHON312_COMPATIBILITY.md`](md/PYTHON312_COMPATIBILITY.md) for the overall Python 3.14 compatibility notes.

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
