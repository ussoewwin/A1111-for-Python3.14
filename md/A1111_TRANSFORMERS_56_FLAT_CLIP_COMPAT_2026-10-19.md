# transformers 5.6+ Compatibility: The Flattened CLIPTextModel and the Import-Hook Shims

> **Date:** 2026-10-19
> **Commits:** `19580b0b` (compatibility layer), `f03110aa` (checkpoint-load normalization)
> **Applies to:** `transformers >= 5.6` (verified on 5.6.2 and 5.20.1). No-op on `transformers` 4.x / 5.4.x.
> **Scope:** SDXL (and SD1.5) text-encoder loading via HuggingFace `CLIPTextModel`, plus the `peft`/`diffusers` import chain.

---

## 1. The observed failures

### 1.1 Silent CLIP weight loss → horizontally smeared images (the core failure)

With `transformers >= 5.6`, loading an SDXL checkpoint whose state dict stores
the text encoder under the legacy key layout produced **no exception at load
time**, yet every generated image was severely corrupted: horizontal banding,
block artifacts, smeared colors, with only a faint humanoid silhouette
remaining. Concretely, for each SDXL checkpoint:

- The checkpoint carries the CLIP-L (HuggingFace `CLIPTextModel`) weights under
  keys such as `conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight`
  (197 tensors).
- The `CLIPTextModel` instance created by `transformers >= 5.6` expects
  `conditioner.embedders.0.transformer.encoder.layers.0.self_attn.q_proj.weight`
  — the `text_model.` wrapper segment no longer exists in the key space.
- `load_state_dict(state_dict, strict=False)` is used deliberately throughout
  the checkpoint loader (checkpoint layouts vary; unknown keys are expected),
  so **every** CLIP key was silently discarded and the encoder remained at its
  random initial values.

A direct reproduction of the mismatch, before the fix:

```text
missing: 196   unexpected: 197
missing sample:    ['embeddings.token_embedding.weight',
                    'embeddings.position_embedding.weight',
                    'encoder.layers.0.self_attn.k_proj.weight', ...]
unexpected sample: ['transformer.text_model.embeddings.position_embedding.weight',
                    'transformer.text_model.embeddings.position_ids',
                    'transformer.text_model.embeddings.token_embedding.weight', ...]
```

After the fix (Section 4.3 normalization applied):

```text
missing: 0   unexpected: 1
unexpected sample: ['embeddings.position_ids']   # buffer removed in transformers 5.20; benign
```

The corruption pattern itself is diagnostic. The UNet and VAE weights load
normally (their keys are untouched by the `transformers` refactor), but the
cross-attention conditioning vector is produced by a randomly initialized
CLIP. The denoiser is therefore driven by noise-shaped conditioning of the
correct *shape* (77×2048) but meaningless *content*, which collapses into the
characteristic horizontal-streaking artifact class.

### 1.2 Startup `AttributeError` inside the quick-load path

With `transformers` 5.20.1 installed, before the changes in Section 4, the
eager model-load path failed with:

```text
creating model quickly: AttributeError
  File "modules/sd_disable_initialization.py", line 135/146, in __enter__
    self.transformers_modeling_utils_load_pretrained_model = self.replace(
        transformers.modeling_utils.PreTrainedModel, '_load_pretrained_model', ...)
AttributeError: module 'transformers.utils.hub' has no attribute 'modeling_utils'
```

(An `... has no attribute 'utils'` variant was observed depending on which
dotted access resolved first.) The lookup target `transformers.modeling_utils`
is a submodule attribute of the package; the error message naming
`transformers.utils.hub` shows the dotted chain resolved to the wrong module
object. Section 2.3 explains why.

### 1.3 Latent breakage in the `peft` / `diffusers` import chain

Two APIs that downstream packages import were removed in the transformers 5.x
line:

- `HybridCache` — removed in 5.2.0. `peft`'s `get_prompt()` path contains
  `from transformers import HybridCache` (Gemma2/Gemma3 prefix-tuning branch).
  Any import/usage that reaches this line raises `ImportError`.
- `modeling_utils.no_init_weights` — removed in 5.x. Downstream code (and the
  checkpoint fast-load path) expects this context manager to exist.

These do not necessarily fire at startup on every configuration (some are lazy
imports inside functions), but any code path that does reach them crashes on
5.6+. The compatibility layer in Section 4.1 neutralizes both.

---

## 2. The essential root cause

### 2.1 The model-structure change behind every symptom

`transformers 5.6` removed the intermediate `text_model` submodule from
`CLIPTextModel`. The parameters are the same tensors with the same shapes; only
their **qualified names inside the module tree** changed:

```text
transformers 4.x – 5.5                transformers 5.6+
CLIPTextModel                         CLIPTextModel
└── text_model                        ├── embeddings
    ├── embeddings                    ├── encoder
    ├── encoder                           └── layers...
    └── final_layer_norm              └── final_layer_norm
```

State-dict keys follow mechanically:

| Component | key in 4.x–5.5 checkpoint | key expected by 5.6+ model |
|---|---|---|
| token embeddings | `text_model.embeddings.token_embedding.weight` | `embeddings.token_embedding.weight` |
| encoder layer | `text_model.encoder.layers.0.self_attn.q_proj.weight` | `encoder.layers.0.self_attn.q_proj.weight` |
| final norm | `text_model.final_layer_norm.weight` | `final_layer_norm.weight` |
| position buffer | `text_model.embeddings.position_ids` | *(removed in 5.20)* |

A checkpoint is a **frozen artifact**: it permanently stores the key layout of
the library version that wrote it. SDXL-family checkpoints in the wild store
the `text_model.` form. The loading application therefore must provide an
adapter from "frozen key names" to "current model structure". That is the
entire essence: **this is a key-layout migration problem at the checkpoint
boundary, and the application layer owns it.**

### 2.2 Why `strict=False` made it invisible

`strict=False` exists precisely because real checkpoints are heterogeneous.
Its cost is that a *total* mismatch of one subsystem (all 197 CLIP keys) is
indistinguishable, in control flow terms, from a *deliberate* subset skip. No
exception is raised; the only clue is the return-value report, which the
generic loader discards. Silent partial load is the dangerous failure mode,
and the horizontally smeared output images were its visible face.

### 2.3 Why the dotted attribute lookup broke at startup

In transformers 5.x the top-level `transformers` package is a `_LazyModule`:
submodules resolve on first attribute access, through internal registries. Two
consequences were measured:

1. Dotted chains like `transformers.modeling_utils` / `transformers.utils.hub`
   can resolve inconsistently while the package is still initializing or after
   the import system replaces the module object in `sys.modules` (the observed
   `has no attribute 'modeling_utils'`/`'utils'` variants).
2. A patch applied to the object returned by an import statement may hit a
   **different instance** than the one later registered in `sys.modules` — the
   patch must target the final instance.

The fix principle: resolve each needed submodule to a concrete object exactly
once (`importlib.import_module`) and hold that reference, instead of
re-deriving it through the lazy wrapper at use time.

### 2.4 Why pinning/downgrading is not a solution

Freezing `transformers` at 5.4.x defers the problem on a stack that keeps
moving under it (Python, PyTorch, `peft`, `diffusers`, model formats). The
upstream layout change is permanent; the checkpoint corpus in the wild uses the
legacy keys. The only forward-compatible answer is for this application to own
the key-mapping adapter — which is what this change implements.

---

## 3. Files created / modified

| File | Status | Role |
|---|---|---|
| `modules/transformers_compat.py` | **new** | Import-hook shims (`HybridCache`, `no_init_weights`, `init_weights` control) + flat-layout CLIP key normalization helpers |
| `modules/launch_utils.py` | modified | Install the import hook in `start()` before `webui` is imported |
| `modules/sd_models.py` | modified | Normalize CLIP checkpoint keys immediately before `load_model_weights` calls `model.load_state_dict(...)`; gated to transformers ≥ 5.6 |
| `modules/sd_disable_initialization.py` | modified | Resolve transformers submodules eagerly (immune to `_LazyModule` lookup instability) |
| `requirements_versions_py314.txt` | modified | `transformers==5.4.0` → `transformers>=5.6.2` |
| `requirements_versions_py314_windows.txt` | modified | same |

---

## 4. Full code

### 4.1 `modules/transformers_compat.py` (new, complete file)

```python
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
```

### 4.2 `modules/launch_utils.py` (diff, from commit `19580b0b`)

```diff
 def start():
     print(f"Launching {'API server' if '--nowebui' in sys.argv else 'Web UI'} with arguments: {shlex.join(sys.argv[1:])}")
+    # Install import hook so first "import transformers" gets 5.6+ shims applied
+    # (HybridCache alias for peft, no_init_weights shim). Hook never imports
+    # transformers at install time. Ported from Forge-Nunchaku.
+    from modules import transformers_compat
+    transformers_compat.apply()
     import webui
```

### 4.3 `modules/sd_models.py` (diff, from commit `f03110aa`)

```diff
     if getattr(model, 'is_sd1', False):
         cond = getattr(model, 'cond_stage_model', None)
         if cond is not None and hasattr(cond, 'transformer'):
             if sd_hijack_clip_utils.is_flat_clip_transformer(cond.transformer):
                 state_dict = sd_hijack_clip_utils.remap_sd1_clip_state_dict_for_flat_clip_text_model(state_dict)
+    # --- transformers 5.6+ flat CLIPTextModel: normalize checkpoint keys (isolated; no-op on 4.x/5.4) ---
+    try:
+        from modules import transformers_compat as _tfc
+        if _tfc.is_flat_clip_layout_active():
+            _prefixes = []
+            if getattr(model, 'is_sdxl', False) or hasattr(model, 'conditioner'):
+                _prefixes.append('conditioner.embedders.0.transformer.')
+                _prefixes.append('conditioner.embedders.1.transformer.')
+            elif hasattr(model, 'cond_stage_model') and hasattr(model.cond_stage_model, 'transformer'):
+                _prefixes.append('cond_stage_model.transformer.')
+            if _prefixes:
+                _n = _tfc.normalize_clip_state_dict_keys_for_flat_model(state_dict, tuple(_prefixes))
+                if _n:
+                    print(f'[TF5.6+] Normalized {_n} CLIP checkpoint keys to flattened layout')
+    except Exception as _e:
+        print(f'[TF5.6+] CLIP key normalization skipped: {_e}')
 
 
     model.load_state_dict(state_dict, strict=False)
     timer.record("apply weights to model")
```

### 4.4 `modules/sd_disable_initialization.py` (diff, from commit `19580b0b`)

```diff
+import importlib as _ilib
+
 import ldm.modules.encoders.modules
 import open_clip
 import torch
 import transformers.utils.hub

+# transformers 5.6+: resolve submodules eagerly (attribute lookup on the _LazyModule wrapper is unreliable during model load)
+_transformers = _ilib.import_module('transformers')
+_transformers_modeling_utils = _ilib.import_module('transformers.modeling_utils')
+_transformers_configuration_utils = _ilib.import_module('transformers.configuration_utils')
+_transformers_tokenization_utils_base = _ilib.import_module('transformers.tokenization_utils_base')
+
 from modules import shared
```

```diff
-            self.transformers_modeling_utils_load_pretrained_model = self.replace(transformers.modeling_utils.PreTrainedModel, '_load_pretrained_model', transformers_modeling_utils_load_pretrained_model)
-            self.transformers_tokenization_utils_base_cached_file = self.replace(transformers.tokenization_utils_base, 'cached_file', transformers_tokenization_utils_base_cached_file)
-            self.transformers_configuration_utils_cached_file = self.replace(transformers.configuration_utils, 'cached_file', transformers_configuration_utils_cached_file)
+            self.transformers_modeling_utils_load_pretrained_model = self.replace(_transformers_modeling_utils.PreTrainedModel, '_load_pretrained_model', transformers_modeling_utils_load_pretrained_model)
+            self.transformers_tokenization_utils_base_cached_file = self.replace(_transformers_tokenization_utils_base, 'cached_file', transformers_tokenization_utils_base_cached_file)
+            self.transformers_configuration_utils_cached_file = self.replace(_transformers_configuration_utils, 'cached_file', transformers_configuration_utils_cached_file)
```

### 4.5 Requirements pins

```diff
-transformers==5.4.0
+transformers>=5.6.2
```

(applied in both `requirements_versions_py314.txt` and
`requirements_versions_py314_windows.txt`)

---

## 5. What each piece means

### 5.1 The `builtins.__import__` hook (`transformers_compat.apply()`)

`apply()` replaces Python's import builtin with a wrapper **before anything
imports `transformers`**, and crucially *never imports `transformers` itself*.
This matters for two reasons:

- Importing `transformers` early (at install time) pulls its full dependency
  graph before the environment is ready and can crash startup silently. The
  hook is inert until the first real `import transformers` happens naturally.
- Patches must be in place **at the moment of the import**, not after —
  `from transformers import HybridCache` in `peft` fails during the import
  statement itself, so a post-hoc `transformers.HybridCache = DynamicCache`
  cannot save the caller that already raised.

Mechanics worth noting:

- `from X import Y` is executed by CPython as `__import__('X', ..., fromlist=('Y',))`
  followed by `_handle_fromlist`. By writing `mod.__dict__["HybridCache"] = DynamicCache`
  inside the wrapper — after the real import returns but before
  `_handle_fromlist` runs — the attribute exists in the module `__dict__`, so
  lookup succeeds without ever entering `_LazyModule.__getattr__` (which is
  exactly where the removed class would have failed resolution).
- The `if final is not None and final is not mod: mod = final` block handles
  the measured fact that the import system can register a *different* module
  instance in `sys.modules` than the one returned by the import statement;
  patches applied to the stale instance would be invisible. The rewrite is
  deliberately restricted to the exact name `"transformers"`: applying the
  same rewrite to arbitrary dotted names breaks parent-attribute binding for
  ordinary submodules (observed with `collections.abc`), so the scope is
  minimal on purpose.
- `no_init_weights` is re-injected into `transformers.modeling_utils` only if
  missing (`hasattr` guard), and the shim additionally no-ops
  `PreTrainedModel.init_weights` *inside* the context — because transformers
  5.x calls `init_weights()` automatically during construction, the 4.x-style
  "disable `torch.nn.init.*` only" shim was insufficient for a deterministic
  skip of weight initialization.

### 5.2 CLIP key normalization (`normalize_clip_state_dict_keys_for_flat_model`)

This is the actual fix for the corrupted images. For every key under a known
CLIP-owning prefix (`conditioner.embedders.0.transformer.` /
`conditioner.embedders.1.transformer.` for SDXL,
`cond_stage_model.transformer.` for SD1.5), the leading legacy wrapper segment
(`transformer.text_model.` → nothing; `transformer.` → nothing;
`text_model.` → nothing) is stripped once, longest-pattern first, so the key
matches the flat model layout.

- It rewrites only keys under the given prefixes — UNet/VAE/open_clip keys pass
  through untouched, so one pass over the whole checkpoint state dict is safe.
- It operates in place (`state_dict.clear(); update(out)`) so the caller's
  reference semantics stay identical for `load_state_dict`.
- Legacy-form keys are exactly the ones present in the wild checkpoint corpus;
  keys already in flat form do not match any legacy pattern and are left
  alone — the function is idempotent and compatible with checkpoints that
  happen to store the new layout.
- `embeddings.position_ids` is intentionally not dropped: `strict=False`
  tolerates it (1 harmless "unexpected key" measured), which keeps the loader
  lenient across transformers point-versions that removed the buffer.

### 5.3 Version gate (`is_flat_clip_layout_active`)

`importlib.metadata.version("transformers")` answers "is this ≥ 5.6?" purely
from package metadata — **it never imports transformers**, preserving the
"do not import early" invariant of the whole compatibility layer. On 4.x and
5.4.x the gate returns False and the normalization block is a literal no-op,
so the same code base runs correctly on both sides of the layout change.

### 5.4 Insertion point in `load_model_weights`

Normalization sits between the INT8 dequantization hook / SD1.5 remap and
`model.load_state_dict(state_dict, strict=False)` — the last moment where the
state dict is still editable and before the silent `strict=False` drop can
occur. It is wrapped in `try/except` with a printed warning: a normalization
failure must degrade to the previous behavior, never abort model loading.

### 5.5 Eager submodule resolution in `sd_disable_initialization`

The quick-load path patches internals of `transformers.modeling_utils`,
`transformers.configuration_utils`, `transformers.tokenization_utils_base`.
The former code reached those via dotted attribute chains on the lazy wrapper,
which is exactly what Section 2.3 shows to be unreliable during model load.
`importlib.import_module(...)` produces each concrete module object exactly
once and holds it in a module-level name; the patch targets are now stable
regardless of the wrapper's internal registry state.

### 5.6 Requirements pins

`transformers>=5.6.2` reflects that the supported baseline is now the flat
layout. Because the compatibility layer is version-gated (5.3), an environment
still carrying 5.4.x also works; the pin simply stops pinning the application
to a pre-migration library state.

---

## 6. Verification record (measured)

| Check | Result |
|---|---|
| Key-normalization unit test against `waiIllustriousSDXL_v170.safetensors` (CLIP-L embedder) | 197 keys renamed; `missing: 0, unexpected: 1 (position_ids)` — was `missing: 196, unexpected: 197` before |
| WebUI startup, transformers 5.20.1 | normal; console prints `[TF5.6+] Normalized 197 CLIP checkpoint keys to flattened layout` |
| Quick-load `AttributeError` (1.2) | resolved by 4.4 eager resolution |
| SDXL txt2img (512×512, steps 20, CFG 7, Euler a, seed 42) | coherent anime image; no horizontal smearing |
| SD1.5 txt2img, same environment | unaffected / normal |
| `peft 0.21.2` import under the hook; `from transformers import HybridCache` | succeeds, resolves to `DynamicCache` |
| `no_init_weights` present after `import transformers.modeling_utils` | yes (both `import x.y` and `from x import y` styles) |

## 7. Maintenance notes

- If a future transformers version renames another wrapper segment, add its
  prefix to `_CLIP_LEGACY_PREFIXES` (order: longest first) — no other change
  is required for CLIP loading.
- Keep `apply()` import-free. Any `import transformers` added at module top
  level of the compat file or its call sites defeats the lazy-hook design.
- Do not move the normalization after `load_state_dict` — at that point the
  mismatched weights are already gone.
- The `unexpected: 1 (embeddings.position_ids)` residue is expected and
  benign under `strict=False`; do not "fix" it by deleting checkpoint keys,
  which would break older layouts that still consume the buffer.
