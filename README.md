# Stable Diffusion web UI

A web interface for Stable Diffusion, implemented using the Gradio library.
## Key Features & Improvements

## 🎉 MAJOR UPDATE: v3.0.0 - Python 3.14 Full Renewal!

**This repository is Python 3.14–only** (`ussoewwin/A1111-for-Python3.14`). Default first-install CUDA stack: `torch==2.15.0+cu134` + `torchvision==0.29.2+cu134`, Flash-Attention 2 **2.9.2.post2** on Windows / **2.8.3.post2** on Linux, plus the CPython 3.14 `shared.sd_model` LOAD_ATTR fix.

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
1. FA-2 (Flash-Attention 2.9.2.post2 on Windows / 2.8.3.post2 on Linux, torch 2.15.0+cu134) — maximum speed
2. SDP (PyTorch scaled_dot_product_attention) — no extra deps
3. sub_quad (built-in) — universal fallback
```

Windows: prebuilt HF wheel (`2.9.2.post2+cu134torch2.15.0` cp314), picked by GPU compute capability — `blackwell` for sm_100/120/121, `legacy` for sm_80/89/90. Linux: builds `flash-attn==2.8.3.post2` from source against the same `torch==2.15.0+cu134` stack (CUDA toolkit 13.4 + `nvcc`). macOS skips FA2 (MPS limitation).


### ConvRot INT8 Checkpoints (SD1.5 and SDXL) (v3.0.1+, runtime INT8 in v3.0.6)

Load SD1.5 and SDXL checkpoints quantized with [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization) **ConvRot INT8** directly in A1111. The quantized weights stay resident as int8 and the forward pass runs a true W8A8 ConvRot GEMM, fully isolated from all other model formats.

- **Runtime INT8 (v3.0.6)** - the weight stays int8 in VRAM for the whole session and the Conv2d / Linear forward runs a real low-precision GEMM: `comfy_kitchen.int8_linear` for Linear layers, and for Conv2d a rotated activation, im2col view, rowwise int8 quantization, `torch._int_mm` against the pre-rotated int8 weight, and a dequantization epilogue. Measured: SD1.5 holds **859.1 MB** of int8 weights against **1719.0 MB** for the fp16 equivalent (2.0x), and SDXL holds **2221.8 MB**. The earlier behavior (dequantize every int8 weight back to fp16 at load time) restored the full fp16 footprint and saved no VRAM; it has been replaced.
- **SD1.5 and SDXL** - SD1.5 (`modules/sd15_int8_convrot.py`) covers the UNet. SDXL (`modules/sdxl_int8_convrot.py`) covers the UNet **and** the quantized text encoders (CLIP-L and CLIP-G), across `nn.Linear`, `nn.Embedding` and the CLIP-G `out_proj` fused inside `nn.MultiheadAttention`.
- **LoRA keeps working** - the quantized modules keep their original `torch.nn.Conv2d` / `torch.nn.Linear` classes (the `weight` parameter is replaced in place and an instance-level `forward` is bound), so the LoRA extension's exact-type checks pass and `half()` / `cuda()` leave the int8 payload intact. LoRA deltas are read from the extension's loaded networks (read-only) and baked into the int8 weights through `calc_updown`.
- **ConvRot INT8** (`int8_tensorwise` + `convrot:true`) - plain / ConvRot Linear / ConvRot Conv2d scale layouts. For reverse-hybrid packs the FP16 checkpoint is the only input; per-layer trajectory-impact ranking packs the lowest-impact layers as FULL ConvRot INT8 while every other layer stays FP16.
- **Broken layers are handled explicitly** - a layer whose int8 payload carries no information (all-zero weight with a NaN weight scale; measured on CLIP-L layer 11 in published SDXL packs) is left on the float path with an explicit warning instead of propagating NaN through the text encoder.
- **SDXL ConvRot NVFP4 development is cancelled** (HSWQ loader [v3.6.1](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.6.1)): deterministic latent-trajectory measurements showed 4-bit precision fails to preserve trajectory fidelity on SDXL (final cosine mean below the 0.90 gate; legacy decoded SSIM was masking it). SDXL is supported strictly in ConvRot INT8 for any new quantization.
- **NVFP4 loading has been removed from A1111 as well.** Non-`int8_tensorwise` layers (e.g. `format=nvfp4`) are detected and reported with an explicit error (`[SDXL INT8 ConvRot] ERROR: ... is format=nvfp4 ...`) and left untouched (no silent corruption; int8 layers in the same file still load normally). Previously published NVFP4 / mixed-pack checkpoints therefore cannot be loaded; re-quantize them as ConvRot INT8 if needed.
- Strict no-op for checkpoints without `.comfy_quant` keys - SD1.5 / SD2 / SD3 / fp16 / fp8 are completely unaffected. The SD1.5 and SDXL gates are exclusive (`cond_stage_model.*` vs `conditioner.*`), so each checkpoint reaches exactly one code path.
- Supported checkpoints: only models quantized with [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)

Technical details: [v3.0.6 Release Notes](https://github.com/ussoewwin/A1111-for-Python3.14/releases/tag/v3.0.6), [SD1.5 runtime INT8 report](https://github.com/ussoewwin/A1111-for-Python3.14/blob/main/md/A1111_SD15_CONVROT_INT8_RUNTIME.md), [SDXL runtime INT8 report](https://github.com/ussoewwin/A1111-for-Python3.14/blob/main/md/A1111_SDXL_CONVROT_INT8_RUNTIME.md), [SDXL NVFP4 cancellation](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.6.1)

## Python Version Support

**This repository supports Python 3.14 only.**

Other Python versions are not supported. Please ensure you are using Python 3.14 before proceeding with installation.

**Note:** Not all extensions may be compatible with Python 3.14. Some extensions may require additional modifications or may not work correctly.

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Windows  | Fully supported | PyTorch `2.15.0+cu134`; FA2 prebuilt `2.9.2.post2+cu134torch2.15.0` (blackwell / legacy by GPU arch); NumPy / SciPy from PyPI; Insightface via pip |
| Linux    | Supported | Same PyTorch `2.15.0+cu134` / `torchvision==0.29.2+cu134`; FA2 source-builds `flash-attn==2.8.3.post2` (CUDA toolkit **13.4** + `nvcc`, ~30 min) |
| macOS    | Supported (limited) | PyTorch `2.15.0` CPU/MPS (`torchvision==0.29.1`); Flash-Attention 2 is skipped (CUDA required; MPS cannot use FA2) |

All platform-specific handling is performed automatically by `modules/launch_utils.py` at startup. This fork supports **Python 3.14 only**. Linux / macOS branches are additive (FA2 path, venv `site-packages` layout for `clip.py` auto-fix).

## Default Package Versions

The following packages are installed automatically during initial setup (see `modules/launch_utils.py`):

- **PyTorch** (suggested manual install): `2.15.0+cu134` with matching `torchvision==0.29.2+cu134` (override with `TORCH_COMMAND` / `TORCH_INDEX_URL` as needed; matches FA2 `cu134torch2.15.0`)
- **Flash-Attention 2**:
  - Windows: `flash_attn` `2.9.2.post2+cu134torch2.15.0` cp314 prebuilt wheel (HF), `blackwell` for compute capability major >= 10 (sm_100/120/121) and `legacy` otherwise (sm_80/89/90)
  - Linux: `flash-attn==2.8.3.post2` (PyPI source build against `torch==2.15.0+cu134`; CUDA toolkit 13.4 + `nvcc`; `--no-build-isolation`)
  - macOS: skipped
- **transformers**: >=5.6.2 (flattened CLIPTextModel compatibility; verified on 5.20.1)
- **protobuf**: 7.34.1
- **scipy**: 1.16.1 (PyPI, all platforms)
- **numpy**: 2.5.4 (PyPI, all platforms; numba / facexlib compatible)
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

4. Install PyTorch 2.15.0+cu134 (aligned with Flash-Attention 2):
   ```cmd
   pip install torch==2.15.0+cu134 torchvision==0.29.2+cu134 --index-url https://download.pytorch.org/whl/cu134
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
- CUDA toolkit 13.4 with `nvcc` on `PATH` (required to build Flash-Attention 2 from source; matches `torch==2.15.0+cu134`).
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

4. Install PyTorch 2.15.0+cu134 (aligned with Flash-Attention 2):
   ```bash
   pip install torch==2.15.0+cu134 torchvision==0.29.2+cu134 --index-url https://download.pytorch.org/whl/cu134
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

4. Install PyTorch 2.15.0 (CPU / MPS; version-aligned with the CUDA stack, FA2 skipped on macOS):
   ```bash
   pip install torch==2.15.0 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
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

- **PyTorch default** (Windows / Linux CUDA): `torch==2.15.0+cu134` + `torchvision==0.29.2+cu134` via `TORCH_COMMAND` / `TORCH_INDEX_URL` (`https://download.pytorch.org/whl/cu134`). Same pin on first install for both platforms so Linux FA2 builds against the Windows FA2 stack.
- **Requirements file**: Windows → `requirements_versions_py314_windows.txt`; Linux / macOS → `requirements_versions_py314.txt`.
- **Flash-Attention 2**: Windows HF prebuilt `flash_attn-2.9.2.post2+cu134torch2.15.0` cp314, wheel variant selected from the GPU compute capability (blackwell / legacy) / Linux PyPI `flash-attn==2.8.3.post2` with `--no-build-isolation` (needs CUDA toolkit 13.4 + `nvcc`) / macOS skip.
- **NumPy / SciPy**: all platforms install `numpy==2.5.4` and `scipy==1.16.1` from PyPI (no Windows-only wheel forced onto Linux). NumPy is pinned to 2.5.3 so numba/facexlib work on first install.
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
