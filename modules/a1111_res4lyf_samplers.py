"""
A1111 integration for RES4LYF samplers and schedulers.

Forge (Stable-Diffusion-WebUI-Forge-Nunchaku) 側の
`modules_forge/forge_res4lyf_samplers.py` を A1111 向けに書き直したもの。

Forge との主な差分:
- A1111 には ``sd_samplers.add_sampler()`` が無いため
  ``all_samplers`` を直接拡張し ``set_samplers()`` を再実行する
- ComfyUI ランタイム (``ComfyUI-master``) は ``modules/paths.py`` では
  sys.path に追加されないため、本モジュールが自前で追加する
- ``bong_tangent`` は Forge の ``sd_schedulers.py`` にはネイティブ実装があったが
  A1111 には無いため、``beta57`` と併せて本モジュール側で登録する
- ``comfy.samplers.beta_scheduler`` / RES4LYF ``sigmas.bong_tangent_scheduler`` は
  ComfyUI 由来のシグネチャなので、A1111 の
  ``(n, sigma_min, sigma_max, inner_model, device)`` にラップする
"""

from __future__ import annotations

import logging
import os
import sys
import types
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def _ensure_comfyui_on_path() -> Optional[str]:
    """``ComfyUI-master`` を sys.path に挿入。既にあれば何もしない。"""
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
    """Install the optional deps. Failures are non-fatal."""
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
    # comfy-kitchen / comfy-aimdo are runtime deps of the vendored ComfyUI-master
    # engine (comfy.samplers -> comfy.model_prefetch -> comfy_aimdo). ComfyUI-master
    # tracks upstream, so these are deliberately NOT pinned: always upgrade to the
    # latest release. Non-fatal: startup continues if the update fails.
    try:
        run_pip("install -U comfy-kitchen comfy-aimdo", "comfy-kitchen / comfy-aimdo")
        logger.info("[RES4LYF] Ensured latest comfy-kitchen / comfy-aimdo")
    except Exception as e:
        logger.warning(f"[RES4LYF] Failed to update comfy-kitchen / comfy-aimdo: {e}")


def _mock_comfyui_globals() -> None:
    """
    ``folder_paths`` と ``server.PromptServer`` を、
    ComfyUI 本体を起動していない状態でも import できるように差し込む。
    ``ComfyUI-master`` に本物の ``folder_paths.py`` があればそれを優先する。
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

    # A1111 の標準サンプラーが参照する ``k_diffusion.sampling.sample_<name>``
    # と衝突する名前は絶対に上書きしない。
    # ``RK_SAMPLER_NAMES_BETA_NO_FOLDERS`` には ``dpmpp_2m`` / ``dpmpp_3m`` /
    # ``dpmpp_2s`` / ``dpmpp_sde_2s`` / ``dpmpp_3s`` / ``euler`` / ``ddim`` 等、
    # A1111 の DPM++ 2M / Euler / DDIM の実体名と重なるエントリが含まれる。
    # これらを ``setattr(k_diffusion.sampling, "sample_<name>", ...)`` すると、
    # A1111 標準サンプラーが起動する ``getattr(k_diffusion.sampling, "sample_<name>")``
    # の返り値が RES4LYF のクロージャに差し替わり、標準サンプラーが破壊される。
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
            # 標準サンプラー名と衝突。上書き禁止。
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
    """``SamplerData.constructor`` 用のクロージャを返す。

    A1111 の ``KDiffusionSampler`` に渡す ``func`` を、
    RES4LYF が ComfyUI 互換 API を要求する箇所を吸収するための
    :func:`modules.a1111_res4lyf_shim.res4lyf_shim_context` で包む。
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

        # KDiffusionSampler は ``inspect.signature(self.func).parameters`` で
        # ``n`` / ``sigmas`` / ``sigma_min`` / ``sigma_max`` などを検査するため、
        # 元関数の署名情報を wrapper に引き継ぐ。
        functools.update_wrapper(wrapped_func, original_func)

        return sd_samplers_kdiffusion.KDiffusionSampler(wrapped_func, model)
    return constructor


def register_res4lyf_samplers() -> None:
    """RES4LYF のサンプラーを A1111 の ``all_samplers`` に追加する。"""
    try:
        _install_optional_deps()
        _mock_comfyui_globals()
        comfyui_path = _ensure_comfyui_on_path()
        if comfyui_path is None:
            logger.warning("[RES4LYF] Aborting sampler registration (ComfyUI-master missing)")
            return

        # RES4LYF 本体を import。ここで __init__.py 末尾の add_samplers() が走り、
        # comfy.k_diffusion.sampling.sample_* が生える。
        from modules import RES4LYF
        import comfy.k_diffusion.sampling as comfy_k_diffusion_sampling
        import k_diffusion.sampling

        # Forge fork の RES4LYF は beta/__init__.py で全 RK サンプラーを動的登録
        # するのに対し、A1111 に置いた upstream 版は 17 個しか登録しない。
        # ここで足りない分を A1111 側から追加登録する（RES4LYF 本体は無編集）。
        extra_added = _register_extra_rk_beta_samplers()
        if extra_added:
            logger.info(f"[RES4LYF] Added {len(extra_added)} extra RK samplers (Forge parity)")

        extra_samplers = getattr(RES4LYF, 'extra_samplers', {})
        if not extra_samplers:
            logger.info("[RES4LYF] No samplers found in extra_samplers")
            return

        # comfy.k_diffusion.sampling → k_diffusion.sampling へ関数コピー
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
    RES4LYF スケジューラーを A1111 の ``sd_schedulers`` に登録する。

    - ``beta57``: ComfyUI の ``beta_scheduler(model_sampling, steps, alpha=0.5, beta=0.7)`` を
      A1111 シグネチャ ``(n, sigma_min, sigma_max, inner_model, device)`` にラップ
    - ``bong_tangent``: RES4LYF ``sigmas.bong_tangent_scheduler(model_sampling, steps, ...)`` を
      同じくラップ
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
