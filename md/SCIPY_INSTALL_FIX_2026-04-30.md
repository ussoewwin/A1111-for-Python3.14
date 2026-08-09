# SciPy Install Fix (2026-04-30): Complete Technical Explanation

> **Current stack (supersedes the NumPy pin named in the historical code below):** startup installs **`numpy==2.5.1`** and **`scipy==1.16.1` from PyPI** on all platforms (`modules/launch_utils.py`). Comments and prose that say `numpy==1.26.4` below describe the **2026-04-30** commit state, not today's pin.

## Summary

This document explains, in full detail, the SciPy installation issue that caused startup to appear stuck, what file was changed, the exact code that was changed, and what the change means operationally.

- Commit: `18d4a4e`
- Scope: SciPy install logic in startup path
- Result: Unified SciPy install flow across Windows/Linux/macOS via PyPI
- Later (v3.0.0 / Python 3.14): NumPy pin moved to **`numpy==2.5.1`**; SciPy remains PyPI `1.16.1`

---

## 1) Error Cause

### Observed runtime symptom

During startup, output stopped at:

- `uninstalling scipy`
- `Installing scipy`

and did not proceed in a reasonable time.

### Root cause

On Windows, `modules/launch_utils.py` was configured to install SciPy from a HuggingFace wheel URL:

`https://huggingface.co/ussoewwin/scipy-1.16.1-cp312-cp312-win_amd64/resolve/main/scipy-1.16.1-cp312-cp312-win_amd64.whl`

That URL returned `401 Unauthorized` at runtime.  
Because startup depended on this external artifact, installation stalled/faulted at SciPy step.

In short: **the Windows path had a single-point dependency on a private/inaccessible remote wheel**.

---

## 2) File Modified

Only one source file was changed:

- `modules/launch_utils.py`

No other runtime module was modified for the SciPy fix itself.

---

## 3) All Code Modified (Exact)

Below is the exact old block and the exact new block, corresponding to commit `18d4a4e`.

### 3.1 Old code (before)

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

### 3.2 New code (after)

```python
# scipy install: unified across platforms via PyPI to match numpy 1.26.4 dtype layout
run(f'"{python}" -m pip uninstall scipy -y', "uninstalling scipy", "Couldn't uninstall scipy", live=False)
run(f'"{python}" -m pip install --no-cache-dir scipy==1.16.1', "Installing scipy 1.16.1", "Couldn't install scipy 1.16.1", live=False)
print("[INFO] Installed scipy 1.16.1 from PyPI")
```

### 3.3 What was removed

- Windows-only `if platform.system() == "Windows":` branch
- `SCIPY_WHEEL` env override path for HuggingFace wheel
- `--no-deps --no-index` install mode tied to remote wheel URL

### 3.4 What was kept

- Explicit uninstall before reinstall (`pip uninstall scipy -y`)
- Explicit SciPy pin (`scipy==1.16.1`)
- Compatibility intent comment for the then-current `numpy==1.26.4` dtype layout (later superseded by `numpy==2.5.1`)

---

## 4) Meaning of the Fix

### Operational meaning

The startup process now uses one deterministic rule for all supported platforms:

- uninstall existing SciPy
- install `scipy==1.16.1` from PyPI (`--no-cache-dir`)

This removes dependency on a potentially inaccessible private wheel host.

### Reliability impact

- **Before:** Windows startup could fail/hang if HuggingFace wheel URL was unauthorized or unavailable.
- **After:** Windows follows the same public package path as Linux/macOS.

### Maintainability impact

- Less branching and less platform-specific special handling
- Easier troubleshooting: one code path, one package source, one pinned version

### Compatibility intent

At commit `18d4a4e` the fix preserved that era's compatibility target:

- SciPy remained pinned to `1.16.1`
- The comment still named alignment with then-current `numpy==1.26.4` startup assumptions

**Current:** SciPy is still `1.16.1` from PyPI; NumPy is **`numpy==2.5.1`** from PyPI (all platforms). Do not treat `1.26.4` as the live pin.

---

## 5) Why this is safer than the previous logic

1. Removes opaque auth dependency (401 risk).
2. Removes hidden infra coupling to a single external private endpoint.
3. Aligns Windows behavior with Linux/macOS, reducing environment drift.
4. Keeps version pin stable, so behavior remains predictable.

---

## 6) Validation Checklist

After this fix, expected startup log pattern is:

- `uninstalling scipy`
- `Installing scipy 1.16.1`
- `[INFO] Installed scipy 1.16.1 from PyPI`

and then startup proceeds to the next steps without hanging on a HuggingFace wheel fetch.

