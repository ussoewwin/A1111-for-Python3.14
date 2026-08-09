# Release Notes (v1.01 to v3.0.0)

This document contains release notes for versions v1.01 through v3.0.0. Through v2.3.5 the published surface was `ussoewwin/A1111-for-Python3.12`; from **v3.0.0** the fork continues as `ussoewwin/A1111-for-Python3.14`.

---

## v3.0.0

### 🎉 MAJOR UPDATE: v3.0.0 - Python 3.14 Full Renewal!

- **Summary**: Full renewal for **Python 3.14** — this fork now targets Python 3.14 only (`webui.sh` / install docs / runtime gate). Default first-install CUDA stack is `torch==2.13.0+cu132` and `torchvision==0.28.0+cu132` (torchaudio omitted; matches FA2 `2.8.4+cu132torch2.13.0`), with Windows FA2 prebuilt wheels and Linux FA2 source builds against CUDA toolkit 13.2. Includes the CPython 3.14 `shared.sd_model` LOAD_ATTR shadow fix, **`numpy==2.4.6` / `scipy==1.16.1` from PyPI** (all platforms; `facexlib==0.3.0` enabled; not the interim `1.26.4` local wheel), **first-install PyPI `onnxruntime-gpu`** (Windows / Linux; macOS `onnxruntime`), Gradio pins for 3.14, and README / Linux–macOS launch alignment to the same stack.
- **Release Note**: [v3.0.0 Release](https://github.com/ussoewwin/A1111-for-Python3.14/releases/tag/v3.0.0)

---

## v2.3.5

- **Fixed**: **RES4LYF hybrid `*4h4s` sampler IndexError** — `lawson45-gen-mod_4h4s` / `lawson45-gen-mod_4h4s_ode` crashed mid-run with `IndexError: index 4 is out of bounds for dimension 0 with size 4` in `rk_sampler_beta.py` when rotating `data_prev_`. A1111 shim (`modules/a1111_res4lyf_shim.py`) expands the hardcoded size-4 history buffers to 8 for the duration of a RES4LYF sample; `modules/RES4LYF/` is left unmodified (`302589c0`).
- **Docs**: README Key Features notes native RES4LYF sampler support; technical write-up for the IndexError fix (`4bdb17e6`, `54ac6e36`).
- **Summary**: RES4LYF `4h4s` hybrid sampler crash fix via A1111-side shim; README / docs for RES4LYF support.
- **Release Note**: [v2.3.5 Release](https://github.com/ussoewwin/A1111-for-Python3.12/releases/tag/v2.3.5)

---

## v2.3.4

- **Added**: **LoRA load logging** — when the active LoRA stack changes, `extensions-builtin/Lora/networks.py` prints per-file UNet/CLIP load lines (loaded key count, multiplier, skipped keys) to the console; duplicate prompts are suppressed via a fingerprint of loaded networks (`3798a083`).
- **Fixed**: **SDXL LoRA layer mapping** — SDXL upsampler conv keys map to `output_blocks_*_2_conv` (not SD1.x `type=1` on `up_blocks_0`); SDXL CLIP/OpenCLIP alias keys and Forge-style diffusers lookup paths so more LoRA keys resolve (`3798a083`).
- **Summary**: Console LoRA load diagnostics and SDXL-focused network layer mapping fixes.
- **Release Note**: [v2.3.4 Release](https://github.com/ussoewwin/A1111-for-Python3.12/releases/tag/v2.3.4)

---

## v2.3.3

- **Added**: **RES4LYF samplers and schedulers** — ComfyUI RES4LYF integration for A1111 via `modules/a1111_res4lyf_samplers.py` and `modules/a1111_res4lyf_shim.py`. Registers RES4LYF-derived sampling methods (including dynamic RK variants) plus schedulers `beta57` and `bong_tangent` in the native txt2img/img2img UI. Vendored `ComfyUI-master/` on `sys.path` for `comfy.*` runtime imports; `modules/RES4LYF/` is left unmodified.
- **Added**: Startup hook in `modules/initialize.py` — calls RES4LYF registration immediately after `sd_samplers.set_samplers()`.
- **Fixed**: **RES4LYF / A1111 sampler name collision** — when a RES4LYF sampler name matches a built-in A1111 sampler (e.g. `dpmpp_2m`, `euler`), the RES4LYF entry is skipped so standard samplers keep working (`a194ea56`).
- **Summary**: Native A1111 UI support for RES4LYF samplers and schedulers (Forge-style glue + execution shim).
- **Release Note**: [v2.3.3 Release](https://github.com/ussoewwin/A1111-for-Python3.12/releases/tag/v2.3.3)

---

## v2.3.2

- **Summary**: Documentation only — restored `md/A1111_Img2Img_Forge_Tiled_VAE_Integration.md` to the pre-v2.3.1 integration scope; added standalone Forge tiled VAE encode NaN fix write-up in `md/A1111_Forge_Tiled_VAE_Encode_NaN_Fix_Explained.md` (code unchanged from v2.3.1 commit `1e2f758a`).
- **Release Note**: [v2.3.2 Release](https://github.com/ussoewwin/A1111-for-Python3.12/releases/tag/v2.3.2)

---

## v2.3.1

- **Fixed**: **ControlNet per-tile cache after latent canvas rebuild** — when MultiDiffusion realigns the latent canvas (e.g. Forge tiled VAE encode `240×252` → `241×252`), `batched_bboxes` is recalculated but `control_tensor_batch` was still sized for the init-time tile count. Noise Inversion + ControlNet tile then crashed at step 0 with `IndexError: list index out of range` in `switch_controlnet_tensors`. `_rebuild_latent_canvas()` now calls `_rebuild_controlnet_tile_cache()` to rebuild per-tile hints from `org_control_tensor_batch` (`1cf51f90`).
- **Fixed**: **Forge tiled VAE encode NaN / narrow crash** — VAE encoder returns `floor(H/8)` latent rows but `tiled_scale` buffers use `round(H/8)`. On pass[1] wide tiles (e.g. `1024×256` with image `H=966`) end-align left row 0 as `out_div==0` → NaN; padding the encoder input overshot when `floor==round` (e.g. `H=954`) and crashed `narrow`. Last-tile output is now replicate-padded at the trailing edge in `tiled_scale_multidim` (`1e2f758a`).
- **Summary**: img2img MultiDiffusion + ControlNet tile + Noise Inversion — ControlNet tile cache after canvas realign; Forge tiled VAE encode NaN fix.
- **Release Note**: [v2.3.1 Release](https://github.com/ussoewwin/A1111-for-Python3.12/releases/tag/v2.3.1)

---

## v2.3

- **Added**: **Forge-parity tiled VAE encode/decode** — new `modules/forge_tiled_vae.py` patches SDXL `DiffusionEngine` and SD1.5/2.x `LatentDiffusion` `encode_first_stage` / `decode_first_stage` with Forge-style 3-pass tiled processing and a progress bar. When MultiDiffusion Tiled VAE is enabled in the UI, legacy `VAEHook` in `tilevae.py` is bypassed in favor of the Forge path (`modules/sd_models_xl.py` applies the patch on SDXL load).
- **Fixed**: **img2img bookend VRAM spikes** — full-resolution VAE encode at the start and decode at the end of img2img (MultiDiffusion + ControlNet tile + Noise Inversion) no longer run as monolithic full-res passes; tiled bookends cap per-tile VRAM on ~16GB GPUs.
- **Fixed**: **Forge tiled encode scale** — encode tile blending no longer uses `downscale=True`, which mis-mapped pixel→latent coordinates and shifted H/W relative to the MultiDiffusion latent canvas (`88d89476`).
- **Fixed**: **Noise Inversion / ControlNet hint sizing** — `abstractdiffusion.py` and `multidiffusion.py` align the latent canvas and ControlNet hints to the input latent size for Noise Inversion (`88d89476`).
- **Fixed**: **MultiDiffusion latent canvas alignment** — asymmetric `pixel_to_latent_h` (floor) / `pixel_to_latent_w` (ceil) replace uniform `// 8` sizing; canvas is rebuilt from the input latent when shapes differ; removed `org_func` full-UNet fallback on size mismatch that caused VRAM explosions (`24aefab9`).
- **Fixed**: **Forge VAE edge-tile NaN** — edge tiles in tiled encode/decode no longer produce NaNs when latent width is not a multiple of 8 (`24aefab9`).
- **Fixed**: **Noise Inversion noise/x canvas alignment** — `sample_img2img` aligns `noise` and `x` to `p.init_latent` shape before `renoise_mask` setup, fixing `RuntimeError: The size of tensor a (231) must match the size of tensor b (232)` when Forge tiled VAE uses ceil for latent width (`fd306900`).
- **Summary**: img2img 16GB VRAM stability — Forge Tiled VAE bookends + MultiDiffusion tiled UNet integration without UI setting changes.
- **Technical details**: Refer here for the full technical write-up: [A1111 Img2Img Forge Tiled VAE Integration](A1111_Img2Img_Forge_Tiled_VAE_Integration.md).
- **Release Note**: [v2.3 Release](https://github.com/ussoewwin/A1111-for-Python3.12/releases/tag/v2.3)

---

## v2.2

- **Fixed**: **ControlNet SDXL bf16 dtype mismatch with LoRA** — SDXL UNet in bf16 with ControlNet hooked and LoRAs loaded could crash at step 0 with `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16` in main UNet `time_embed` (`hook.py`). Timestep embeddings stayed float32 while LoRA-patched Linear weights were bf16; autocast is a no-op for bf16 on this fork. Aligned dtypes in `extensions-builtin/sd-webui-controlnet` (`hook.py`, `cldm.py`, `controlnet_model_guess.py`); `extensions-builtin/Lora/networks.py` unchanged.
- **Summary**: ControlNet Union Pro Max SDXL + multiple LoRAs + Hi-Res Fix on bf16; technical write-up in `md/A1111_ControlNet_SDXL_bf16_dtype_fix.md`.
- **Release Note**: [v2.2 Release](https://github.com/ussoewwin/A1111-for-Python3.12/releases/tag/v2.2)

---

## v2.1

- **Fixed**: **PCM / Turbo SDXL speed LoRA key mapping** — `convert_diffusers_name_to_compvis()` in `extensions-builtin/Lora/networks.py` now maps `lora_unet_up_blocks_*_upsamplers_0_conv` to `diffusion_model_output_blocks_{N}_2_conv` (upsampler type=2) instead of the wrong `_1_conv` slot. Kohaku-ss / Diffusers-format PCM LoRAs (e.g. `pcm_sdxl_normalcfg_16step_converted.safetensors`) no longer show `3/2364 unmatched keys` and apply correctly on A1111, matching Forge and ComfyUI behavior.
- **Summary**: PCM and Turbo-style SDXL speed LoRA support via upsampler CompVis mapping fix; technical write-up in `md/A1111_PCM_LoRA_Mapping_Fix.md`.
- **Release Note**: [v2.1 Release](https://github.com/ussoewwin/A1111-for-Python3.12/releases/tag/v2.1)

---

## v2.0

### MAJOR UPDATE: Pony and Illustrious (IL) SDXL — Full Support

**The first A1111 fork to fully support Pony and Illustrious SDXL models on native A1111 — including LoRA.**

For years, SDXL derivative models (Pony Diffusion, WAI Illustrious, etc.) were unreliable on A1111: noisy images, LoRAs that seemed to do nothing, and `RuntimeError: attn_mask shape` on load. Users often had to switch to ComfyUI or Forge. **v2.0 resolves the root cause and makes Pony / IL first-class on this fork.**

- **Added**: **Full Pony series support** — base txt2img / img2img and LoRA application verified on Pony-family SDXL checkpoints.
- **Added**: **Full Illustrious (IL) series support** — base generation and LoRA verified on WAI Illustrious and related IL checkpoints.
- **Fixed**: **SDXL CLIP-G `batch_first` compatibility for open_clip 3.1.0** — conditional NLD/LND permute in `repositories/generative-models/sgm/modules/encoders/modules.py` so A1111 matches ComfyUI/Forge text-encoding behavior when `open_clip` sets `batch_first=True` on `nn.MultiheadAttention`. This was the actual root cause of noise, broken LoRA effects, and load-time `attn_mask` shape errors on Pony / IL (not CLIP-L layer selection, not v-prediction, not UNet dtype hacks).
- **Fixed**: **Model load failures** — `RuntimeError` on CLIP-G load (attn_mask vs tensor shape mismatch) no longer occurs on affected checkpoints.
- **Fixed**: **LoRA effectiveness** — style and character LoRAs on Pony / IL checkpoints apply correctly after the CLIP-G fix (previously embeddings were wrong, so LoRAs appeared ineffective).
- **Removed**: **Obsolete CLIP-G workaround** — temporary `attn_mask` disable hack in `modules/sd_hijack_open_clip.py` (no longer needed after the `batch_first` fix).
- **Reverted**: **Filename-based v-prediction auto-detection** — removed the hack that mis-detected Pony / IL as v-pred models; these models use eps prediction, not v-prediction.
- **Verified**: Pony / IL base generation and LoRA on **Flash-Attention 2.9.1**, **PyTorch 2.12.1+cu132**, **open_clip 3.1.0**, **Python 3.12.10** (see environment table in technical doc).

- **Summary**: Major release — full Pony and Illustrious SDXL support (base + LoRA) via CLIP-G `batch_first` fix for open_clip 3.1.0; v-prediction filename hack reverted; documented in `md/A1111_SDXL_CLIP_Fix.md`.
- **Release Note**: [v2.0 Release](https://github.com/ussoewwin/A1111-for-Python3.12/releases/tag/v2.0)

---

## v1.15

- **Summary**: Simplified ADetailer face detection to a YOLO-only inference path. Removed the unused InsightFace hybrid detection branch (`hybrid_face_predict` and related overlap logic) that was not used at runtime; all models now call `ultralytics_predict()` directly.

---

## v1.14

- **Summary**: Fixed img2img CUDA OOM when MultiDiffusion + Tiled VAE is enabled. Removed the "tiny input" short-circuit in `tilevae.py` that skipped tiling and caused global VAE encoder attention to OOM. Forced `chunk_threshold=0` in all `sub_quad_attention` call sites so the "fits VRAM" fast path (which routes to a non-chunked `torch.bmm` on the full `(seq_len, seq_len)` matrix) is never taken. Hardened `memmon.py` with `try/except` around `cuda_mem_get_info` and `memory_stats` so the memory monitor survives sticky CUDA errors after an OOM.
- **Release Note**: [v1.14 Release](https://github.com/ussoewwin/A1111-for-Python3.12/releases/tag/1.14)

---

## v1.13

- **Summary**: Hardened Multidiffusion Tiled VAE attention fallback behavior to avoid SDPA-triggered OOM cascades, with recovery-oriented fallback sequencing and detailed technical documentation.
- **Release Note**: [v1.13 Release](https://github.com/ussoewwin/A1111-for-Python3.12/releases/tag/1.13)

---

## v1.12

- **Summary**: Unified SciPy install flow across all platforms to install `scipy==1.16.1` from PyPI, removing the Windows-specific HuggingFace wheel dependency that caused startup stalls.
- **Updated**: `modules/launch_utils.py` now uses a single cross-platform SciPy install path (`pip install --no-cache-dir scipy==1.16.1`).
- **Release Note**: [v1.12 Release](https://github.com/ussoewwin/A1111-for-Python3.12/releases/tag/1.12)

---

## v1.11

- **Added**: **ControlNet** (`sd-webui-controlnet`) is now vendored as a built-in extension under `extensions-builtin/sd-webui-controlnet` (no nested `.git`), same pattern as other built-ins.
- **Added**: On startup, if the extension still exists under `extensions/`, it is automatically moved to `extensions-builtin/` (`migrate_controlnet_to_builtin` in `modules/launch_utils.py`).
- **Added**: ControlNet runtime dependencies are integrated into main Python 3.12 requirement files (`requirements_versions_py312.txt` and `requirements_versions_py312_windows.txt`). Note: `mediapipe` is excluded per project policy (v1.07).
- **Removed**: `install.py` from built-in copy; dependencies are now managed by the main requirement files.

---

## v1.10

- **Added**: **Aspect Ratio selector** (`forge_aspect_ratio`) is added as a built-in extension under `extensions-builtin/forge_aspect_ratio`.
- **Added**: Built-in aspect ratio logic is adapted from **ControlAltAI-Nodes** for A1111 integration.
- **Updated**: Replaced previous AR implementation (`sd-webui-ar`) with the new built-in aspect ratio extension.

---

## v1.09

- **Added**: **FreeU** (`sd-webui-freeu`) is now vendored as a built-in extension under `extensions-builtin/sd-webui-freeu` (no nested `.git`), same pattern as other built-ins.
- **Added**: On startup, if the extension still exists under `extensions/`, it is automatically moved to `extensions-builtin/` (`migrate_freeu_to_builtin` in `modules/launch_utils.py`).
- **Note**: FreeU has no external Python package dependencies beyond stdlib; no entries added to requirement files.

---

## v1.08

*Tag range: `1.07`..`1.08`*

- **Added**: **sd-dynamic-thresholding** (`sd-dynamic-thresholding`) is now vendored as a built-in extension under `extensions-builtin/sd-dynamic-thresholding` (no nested `.git`), same pattern as other built-ins.
- **Added**: On startup, nested `.git` under the built-in copy is removed; if the extension still exists under `extensions/`, it is automatically moved to `extensions-builtin/` (`migrate_sd_dynamic_thresholding_to_builtin` in `modules/launch_utils.py`).
- **Updated**: `README.md` — license section lists the upstream repository and MIT license.

---

## v1.07

*Tag range: `1.06`..`1.07`*

- **Updated**: ADetailer face detection completely replaced — `mediapipe` is no longer used; all mediapipe-based models (`mediapipe_face_short`, `mediapipe_face_full`, `mediapipe_face_mesh`, `mediapipe_face_mesh_eyes_only`) now route through InsightFace.
- **Fixed**: Removed mediapipe `--no-deps` install logic from `modules/launch_utils.py` (no longer needed).
- **Fixed**: Cleaned up mediapipe-related comments from `requirements_versions_py312.txt` and `requirements_versions_py312_windows.txt`.

---

## v1.06

*Tag range: `1.05`..`1.06`*

- **Added**: **WD14-tagger** (`stable-diffusion-webui-wd14-tagger`) is now vendored as a built-in extension under `extensions-builtin/stable-diffusion-webui-wd14-tagger` (no nested `.git`), same pattern as other built-ins.
- **Updated**: WD14-tagger runtime dependencies are integrated into main Python 3.12 requirement files (`requirements_versions_py312.txt` and `requirements_versions_py312_windows.txt`) to ensure install-time consistency. Only packages not already covered by the main list are added (`deepdanbooru`, `jsonschema`, `opencv_contrib_python`).
- **Updated**: `extensions-builtin/stable-diffusion-webui-wd14-tagger/install.py` now skips its own pip install path when loaded from `extensions-builtin`, preventing duplicate dependency installation.

---

## v1.05

*Tag range: `1.04`..`1.05`*

- **Added**: **Multidiffusion upscaler** (`multidiffusion-upscaler-for-automatic1111`) is now vendored as a built-in extension under `extensions-builtin/multidiffusion-upscaler-for-automatic1111` (no nested `.git`), same pattern as other built-ins.
- **Added**: On startup, if the extension still exists under `extensions/`, it is automatically moved to `extensions-builtin/` and any nested `.git` under the built-in copy is removed (`migrate_multidiffusion_to_builtin` in `modules/launch_utils.py`).

---

## v1.04

*Tag range: `1.03`..`1.04` (2026-04-23)*

- **Added**: ADetailer is now vendored as a built-in extension under `extensions-builtin/adetailer` (no nested `.git`), so it ships with the main repository.
- **Updated**: ADetailer runtime dependencies are integrated into main Python 3.12 requirement files (`requirements_versions_py312.txt` and `requirements_versions_py312_windows.txt`) to ensure install-time consistency.
- **Updated**: `extensions-builtin/adetailer/install.py` now skips its own pip install path when loaded from `extensions-builtin`, preventing duplicate dependency installation.
- **Fixed**: Startup no longer crashes when the default `localizations` directory is missing (`modules/localization.py` now checks directory existence before listing).
- **Fixed**: Removed `mediapipe` from global requirements to resolve pip dependency conflict with pinned `protobuf==7.34.1` + `tensorflow==2.20.0`; mediapipe is now ensured via no-deps install path so ADetailer mediapipe detectors remain usable without downgrading protobuf.
- **Fixed**: [v1.07 backport] Replaced `mediapipe` with `insightface` for all ADetailer face detection; mediapipe dependency and `--no-deps` install logic completely removed from the main codebase.

---

## v1.03

*Tag range: `1.02`..`1.03` (2026-04-22 to 2026-04-23)*

- **Added**: Cross-platform Python 3.12 startup support — created `requirements_versions_py312.txt` for Linux / macOS (byte-identical to the Windows file), resolving the `FileNotFoundError` that blocked non-Windows Python 3.12 users.
- **Added**: Flash-Attention 2 platform branching in `modules/launch_utils.py` — Windows uses prebuilt wheel `flash_attn-2.8.3+cu130torch2.10.0`, Linux source-builds `flash-attn==2.8.3` via PyPI (`--no-build-isolation`, ~30 min, requires CUDA toolkit), macOS is skipped (FA2 requires CUDA).
- **Added**: SciPy platform branching — Windows uses HuggingFace prebuilt `scipy-1.16.1-cp312-cp312-win_amd64` wheel forced via `--no-deps --no-index`; Linux / macOS use PyPI `scipy==1.16.1`.
- **Added**: `clip.py` `pkg_resources` auto-fix path branched by platform — Windows `venv/Lib/...`, Linux / macOS `venv/lib/pythonX.Y/...` (dynamic `sys.version_info`).
- **Added**: OOM prevention + Flash-Attention 2 direct-load optimizations promoted from the A1111 working tree into the fork.
- **Added**: `md/FA2_direct_load_design.md` — English design document for direct Flash-Attention 2 kernel loading (bypassing xformers).
- **Added**: `md/LINUX_MAC_PY312_STARTUP_FIX.md` — English design document for the Linux / macOS startup fix.
- **Added**: `md/INCIDENT_2026-04-22_A1111_Cursor_git_disabled.md` — incident record for the `.git` → `.git_disabled` rename event.
- **Added**: `md/CHANGELOG.md` — per-release notes (this document) covering v1.01 through v1.03, linked from README.
- **Updated**: PyTorch **2.9.1+cu130 → 2.10.0+cu130**.
- **Updated**: Flash-Attention 2 wheel **`2.8.3+cu130torch2.9.1` → `2.8.3+cu130torch2.10.0`** (Windows).
- **Updated**: transformers baseline raised to **5.4.0+** (4.x shims dropped throughout).
- **Updated**: protobuf **4.25.2 → 7.34.1**.
- **Updated**: `configs/v1-inference.yaml` normalized to 2-space indentation while preserving `use_checkpoint: True` (gradient checkpointing, VRAM saving).
- **Updated**: `md/PYTHON312_COMPATIBILITY.md` rewritten for the 1.03 codebase, narrowed strictly to Python 3.12 scope, and translated to English.
- **Updated**: README rewritten for 1.03 — per-OS Installation sections (Windows / Linux / macOS), Default Package Versions table, How Linux / macOS support works, Changelog link.
- **Fixed**: `AttributeError` during quick model load caused by transformers ≥ 5 API changes (`sd_disable_initialization.py` shim removed, `import_hook.py` / `initialize.py` 4.x shims removed).
- **Removed**: Reference to non-existent `requirements_versions_py312_fallback.txt` from documentation.
- **Guarantee**: Windows install flow is byte-identical to pre-1.03 — all Linux / macOS branches are additive.

---

## v1.02

*Tagged commit: `80729aa` (2026-04-22)*

- **Added**: Self-contained installation — all repository dependencies (`ldm`, `sgm`, `k_diffusion`, `BLIP`, `assets`) vendored under `repositories/` so first-time setup performs **no external `git clone`** calls.
- **Fixed**: `RuntimeError: Couldn't fetch assets.` and `fatal: not a git repository` failures on restricted / offline environments.
- **Updated**: README release-notes section now links to GitHub Releases instead of inlined content.

---

## v1.01

*Tagged commit: `b456793` (2025-11-18)*

- **Added**: Direct Flash-Attention 2 kernel load path — FA2 is loaded directly without going through xformers.
- **Updated**: PyTorch baseline to **2.9.1+cu130**.
- **Updated**: Flash-Attention 2 pinned to **`2.8.3+cu130torch2.9.1`**.
- **Fixed**: transformers `GenerationMixin` import corrected for newer releases.
- **Removed**: `xformers` auto-install block (FA2 is loaded directly; xformers no longer required).
- **Removed**: Japanese text from README in favor of English.

---

## Related documents

- [`PYTHON312_COMPATIBILITY.md`](PYTHON312_COMPATIBILITY.md) — Python 3.12 compatibility overview.
- [`LINUX_MAC_PY312_STARTUP_FIX.md`](LINUX_MAC_PY312_STARTUP_FIX.md) — Linux / macOS startup fix design.
- [`FA2_direct_load_design.md`](FA2_direct_load_design.md) — FA2 direct kernel load design.
- [`INCIDENT_2026-04-22_A1111_Cursor_git_disabled.md`](INCIDENT_2026-04-22_A1111_Cursor_git_disabled.md) — `.git` rename incident record.
