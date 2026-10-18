"""Compatibility shims for transformers 5.6+ (ported from Forge-Nunchaku).

Applies lazily via a builtins.__import__ hook so no transformers import
happens at patch-install time:

  1. HybridCache -> DynamicCache alias (peft 0.21.x imports HybridCache at
     module import time; transformers 5.2+ removed the class).
  2. modeling_utils.no_init_weights -> local shim (context manager that
     no-ops torch.nn.init.*; transformers 5.x removed the helper while
     downstream code still calls it).
  3. PreTrainedModel.init_weights is no-op'ed inside the shim context
     (transformers 5.x calls it automatically after construction).

Important: apply() must never import transformers at module load time.
All patches run inside the import hook, after the real import succeeds.
"""
import builtins
import sys
from contextlib import contextmanager

_original_import = None
_no_init_weights_patched = False


@contextmanager
def _no_init_weights():
    """Shim compatible with transformers 4.x/5.x modeling_utils.no_init_weights().

    Disables torch.nn.init ops and PreTrainedModel.init_weights inside the
    context so weight initialisation is skipped."""
    import torch

    _skip = {
        "uniform_": torch.nn.init.uniform_,
        "normal_": torch.nn.init.normal_,
        "trunc_normal_": torch.nn.init.trunc_normal_,
        "constant_": torch.nn.init.constant_,
        "xavier_uniform_": torch.nn.init.xavier_uniform_,
        "xavier_normal_": torch.nn.init.xavier_normal_,
        "kaiming_uniform_": torch.nn.init.kaiming_uniform_,
        "kaiming_normal_": torch.nn.init.kaiming_normal_,
        "orthogonal_": torch.nn.init.orthogonal_,
        "sparse_": torch.nn.init.sparse_,
        "zeros_": torch.nn.init.zeros_,
        "ones_": torch.nn.init.ones_,
        "dirac_": torch.nn.init.dirac_,
        "eye_": torch.nn.init.eye_,
    }

    def _noop(*args, **kwargs):
        pass

    # Also patch PreTrainedModel.init_weights (transformers 5.x calls this
    # automatically during construction; the native no_init_weights does the
    # same).
    _original_init_weights = None
    try:
        from transformers.modeling_utils import PreTrainedModel
        _original_init_weights = PreTrainedModel.init_weights
        PreTrainedModel.init_weights = _noop
    except Exception:
        pass

    try:
        for name in _skip:
            setattr(torch.nn.init, name, _noop)
        yield
    finally:
        for name, fn in _skip.items():
            setattr(torch.nn.init, name, fn)
        if _original_init_weights is not None:
            PreTrainedModel.init_weights = _original_init_weights


def _patch_no_init_weights(mod):
    """Inject no_init_weights into modeling_utils if missing."""
    global _no_init_weights_patched
    if _no_init_weights_patched:
        return
    _no_init_weights_patched = True
    if not hasattr(mod, "no_init_weights"):
        mod.no_init_weights = _no_init_weights


def _hooked_import(name, *args, **kwargs):
    mod = _original_import(name, *args, **kwargs)


    # transformers 5.x replaces the module object in sys.modules after the
    # real import returns, so patch the final instance registered there.
    if name == "transformers":
        final = sys.modules.get(name)
        if final is not None and final is not mod:
            mod = final

    # --- HybridCache patch (peft imports it via from transformers import ...) ---
    if name == "transformers":
        fromlist = args[2] if len(args) > 2 else kwargs.get("fromlist", ())
        if fromlist and "HybridCache" in fromlist and not hasattr(mod, "HybridCache"):
            from transformers.cache_utils import DynamicCache
            mod.__dict__["HybridCache"] = DynamicCache

    # --- no_init_weights patch ---
    # `import transformers.modeling_utils` -> name="transformers.modeling_utils"
    if name == "transformers.modeling_utils":
        _patch_no_init_weights(mod)
    # `from transformers import modeling_utils` -> name="transformers",
    # fromlist=("modeling_utils",)
    elif name == "transformers":
        fromlist = args[2] if len(args) > 2 else kwargs.get("fromlist", ())
        if fromlist and "modeling_utils" in fromlist:
            mu = sys.modules.get("transformers.modeling_utils")
            if mu is not None:
                _patch_no_init_weights(mu)

    return mod


def apply():
    """Install the import hook. Must be called before any transformers import.

    This function never imports transformers itself."""
    global _original_import

    if _original_import is not None:
        return  # already applied

    _original_import = builtins.__import__
    builtins.__import__ = _hooked_import

# ---------------------------------------------------------------------------
# transformers 5.6+ flattened CLIPTextModel: state_dict key normalization
# ---------------------------------------------------------------------------
# transformers 5.6+ removed the intermediate `text_model` wrapper inside
# CLIPTextModel (flat layout: transformer.embeddings.* instead of
# transformer.text_model.embeddings.*). Checkpoints written for 4.x still
# carry `text_model.` prefixes, so a naive load_state_dict silently drops
# every CLIP weight (strict=False => random init remainders).

_CLIP_LEGACY_PREFIXES = (
    "transformer.text_model.",
    "transformer.",
    "text_model.",
)


def is_flat_clip_layout(model) -> bool:
    """True when the loaded model has the transformers 5.6+ flattened layout."""
    return hasattr(model, "embeddings") and not hasattr(model, "text_model")


def normalize_clip_state_dict_keys_for_flat_model(state_dict, model_prefixes):
    """Rewrite 4.x-style CLIP keys to 5.6+ flat keys, in place.

    model_prefixes: tuple of key prefixes that point at a CLIPTextModel
    (e.g. ("cond_stage_model.transformer.", "conditioner.embedders.0.transformer.")).
    Only keys starting with one of these prefixes are considered; the
    `transformer.` / `text_model.` wrapper segments after that prefix are
    stripped so the remaining key matches the flat model layout.
    Returns the number of renamed keys.
    """
    if not model_prefixes:
        return 0
    renamed = 0
    out = {}
    for k, v in state_dict.items():
        hit_prefix = None
        for pre in model_prefixes:
            if k.startswith(pre):
                hit_prefix = pre
                break
        if hit_prefix is None:
            out[k] = v
            continue
        tail = k[len(hit_prefix):]
        stripped = False
        for legacy in _CLIP_LEGACY_PREFIXES:
            if tail.startswith(legacy):
                tail = tail[len(legacy):]
                stripped = True
                break
        if stripped:
            out[hit_prefix + tail] = v
            renamed += 1
        else:
            out[k] = v
    if renamed:
        state_dict.clear()
        state_dict.update(out)
    return renamed

def is_flat_clip_layout_active() -> bool:
    """True when the installed transformers is 5.6+ (flat CLIPTextModel layout).

    Uses packaging metadata; never imports transformers."""
    try:
        from importlib.metadata import version as _v
        v = _v("transformers")
        major = int(v.split(".")[0])
        if major < 5:
            return False
        minor = int(v.split(".")[1]) if "." in v.split(".", 1)[1] else 0
        return (major, minor) >= (5, 6)
    except Exception:
        return False
