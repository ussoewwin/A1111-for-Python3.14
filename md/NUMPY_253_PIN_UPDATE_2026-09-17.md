# NumPy 2.5.3 Upgrade — Complete Technical Explanation

- Target: this repository (Python 3.14.7 / win-amd64 / venv)
- Date: 2026-09-17
- Goal: upgrade NumPy to the latest release **2.5.3** and align the version pins in the configuration/code to the same **2.5.3**

---

## ① Overview of the NumPy latest-version upgrade

### 1. Background (state before the change)

- The NumPy in the venv was **2.4.6**. The task was to update it to the latest release, **2.5.3**.
- The requirement files `requirements_versions_py314*.txt` were already updated to `numpy==2.5.3` in the working tree (HEAD is `numpy==2.4.6`).
- However, **`modules/launch_utils.py` forcibly pinned NumPy to `2.4.6` at every startup**, so even after raising the requirements the environment was reverted to 2.4.6 on each launch. That pin was the effective "setting", and it had to be raised together with the requirement files.

### 2. Latest versions (confirmed on PyPI)

| Package | Before | After | PyPI latest |
|---|---|---|---|
| numpy | 2.4.6 | **2.5.3** | 2.5.3 |
| numba | 0.66.0 | **0.67.0** | 0.67.0 |
| llvmlite | 0.48.0 | **0.49.0** | 0.49.0 |

`pip index versions` confirmed the latest NumPy is 2.5.3 and the latest numba is 0.67.0.

### 3. What was changed

1. **venv packages**
   - Installed `numpy==2.5.3` (`numpy-2.5.3-cp314-cp314-win_amd64.whl`).
   - Upgraded numba to the latest: `numba 0.66.0 → 0.67.0` (with `llvmlite 0.48.0 → 0.49.0`).
   - numba 0.67.0 requires `numpy<2.6,>=1.22`, so it **coexists with numpy 2.5.3**. When resolved together with numpy 2.5.3, this combination is selected automatically (`pip install --dry-run` shows `Would install llvmlite-0.49.0 numba-0.67.0 numpy-2.5.3`).
2. **Configuration side (in-code pin) aligned**
   - Changed the NumPy force-pin in `modules/launch_utils.py` from `2.4.6 → 2.5.3` (three install sites plus related messages/comments).

### 4. Verification results

- `numpy 2.5.3` / `numba 0.67.0` / `llvmlite 0.49.0` all import successfully.
- `facexlib 0.3.0` (the package that depends on numba) imports successfully.
- `python -m pip check`: the numba/NumPy conflict is resolved.
- `ast.parse` syntax check on `launch_utils.py` passes.
- No remaining `2.4.6` in the file (**0 occurrences**). No line-ending (EOL) changes.

> Note: The following two `pip check` items are **pre-existing** and unrelated to the NumPy upgrade:
> - `k-diffusion 0.0.16 → wandb not installed`
> - `pydantic-settings 2.15.0 → requires pydantic>=2.7.0 (currently 1.10.25)`

---

## ② Files created / modified

### Modified files

| File | Change |
|---|---|
| `modules/launch_utils.py` | NumPy force-pin changed `2.4.6 → 2.5.3` |
| `requirements_versions_py314.txt` | `numpy==2.4.6 → numpy==2.5.3` (updated in working tree) |
| `requirements_versions_py314_windows.txt` | `numpy==2.4.6 → numpy==2.5.3` (updated in working tree) |

### Created files

| File | Kind | Content |
|---|---|---|
| `md/NUMPY_253_PIN_UPDATE_2026-09-17.md` | Deliverable | This document |

### Packages updated in the venv

- `venv/Lib/site-packages/numpy` : `2.4.6 → 2.5.3`
- `venv/Lib/site-packages/numba` : `0.66.0 → 0.67.0`
- `venv/Lib/site-packages/llvmlite` : `0.48.0 → 0.49.0`

---

## ③ Full text of created / modified code

### 3-1. `modules/launch_utils.py` (modified, before → after)

**Before (HEAD)**

```python
    run_pip("install --upgrade packaging", "packaging")

    # Pin numpy==2.4.6 before -r (numba / facexlib compatible; not 2.5.x).
    # Install early so already-imported modules do not keep a stale numpy.
    run(f'"{python}" -m pip uninstall numpy -y', "uninstalling numpy", "Couldn't uninstall numpy", live=False)
    run(f'"{python}" -m pip install --no-cache-dir numpy==2.4.6', "Installing numpy 2.4.6", "Couldn't install numpy 2.4.6", live=False)
    print("[INFO] Installed numpy==2.4.6 from PyPI")
```

**After**

```python
    run_pip("install --upgrade packaging", "packaging")

    # Pin numpy==2.5.3 before -r.
    # Install early so already-imported modules do not keep a stale numpy.
    run(f'"{python}" -m pip uninstall numpy -y', "uninstalling numpy", "Couldn't uninstall numpy", live=False)
    run(f'"{python}" -m pip install --no-cache-dir numpy==2.5.3', "Installing numpy 2.5.3", "Couldn't install numpy 2.5.3", live=False)
    print("[INFO] Installed numpy==2.5.3 from PyPI")
```

**Before (HEAD)**

```python
    # scipy via PyPI to match numpy 2.4.6
    run(f'"{python}" -m pip uninstall scipy -y', "uninstalling scipy", "Couldn't uninstall scipy", live=False)
    run(f'"{python}" -m pip install --no-cache-dir scipy==1.16.1', "Installing scipy 1.16.1", "Couldn't install scipy 1.16.1", live=False)
    print("[INFO] Installed scipy 1.16.1 from PyPI")
```

**After**

```python
    # scipy via PyPI to match numpy 2.5.3
    run(f'"{python}" -m pip uninstall scipy -y', "uninstalling scipy", "Couldn't uninstall scipy", live=False)
    run(f'"{python}" -m pip install --no-cache-dir scipy==1.16.1', "Installing scipy 1.16.1", "Couldn't install scipy 1.16.1", live=False)
    print("[INFO] Installed scipy 1.16.1 from PyPI")
```

**Before (HEAD)**

```python
    if not args.skip_install:
        run_extensions_installers(settings_file=args.ui_settings_file)
        # Keep pin if an extension installer drifted numpy.
        run(f'"{python}" -m pip install --force-reinstall --no-deps --no-cache-dir numpy==2.4.6', "re-pin: numpy 2.4.6", "Couldn't install numpy 2.4.6", live=False)
        print("[INFO] Re-pinned numpy==2.4.6 after extensions")
```

**After**

```python
    if not args.skip_install:
        run_extensions_installers(settings_file=args.ui_settings_file)
        # Keep pin if an extension installer drifted numpy.
        run(f'"{python}" -m pip install --force-reinstall --no-deps --no-cache-dir numpy==2.5.3', "re-pin: numpy 2.5.3", "Couldn't install numpy 2.5.3", live=False)
        print("[INFO] Re-pinned numpy==2.5.3 after extensions")
```

**Reference: `git diff` (actual diff)**

```diff
@@ -730,11 +730,11 @@ def prepare_environment():
 
     run_pip("install --upgrade packaging", "packaging")
 
-    # Pin numpy==2.4.6 before -r (numba / facexlib compatible; not 2.5.x).
+    # Pin numpy==2.5.3 before -r.
     # Install early so already-imported modules do not keep a stale numpy.
     run(f'"{python}" -m pip uninstall numpy -y', "uninstalling numpy", "Couldn't uninstall numpy", live=False)
-    run(f'"{python}" -m pip install --no-cache-dir numpy==2.4.6', "Installing numpy 2.4.6", "Couldn't install numpy 2.4.6", live=False)
-    print("[INFO] Installed numpy==2.4.6 from PyPI")
+    run(f'"{python}" -m pip install --no-cache-dir numpy==2.5.3', "Installing numpy 2.5.3", "Couldn't install numpy 2.5.3", live=False)
+    print("[INFO] Installed numpy==2.5.3 from PyPI")
 
     # Gradio from HF (METADATA version pins removed)
     run_pip(f'install "{gradio_package}"', "gradio")
@@ -746,7 +746,7 @@ def prepare_environment():
         run_pip(f"install -U --upgrade-strategy only-if-needed -r \"{requirements_file}\"", "requirements")
         startup_timer.record("install requirements")
     
-    # scipy via PyPI to match numpy 2.4.6
+    # scipy via PyPI to match numpy 2.5.3
     run(f'"{python}" -m pip uninstall scipy -y', "uninstalling scipy", "Couldn't uninstall scipy", live=False)
     run(f'"{python}" -m pip install --no-cache-dir scipy==1.16.1', "Installing scipy 1.16.1", "Couldn't install scipy 1.16.1", live=False)
     print("[INFO] Installed scipy 1.16.1 from PyPI")
@@ -806,8 +806,8 @@ def prepare_environment():
     if not args.skip_install:
         run_extensions_installers(settings_file=args.ui_settings_file)
         # Keep pin if an extension installer drifted numpy.
-        run(f'"{python}" -m pip install --force-reinstall --no-deps --no-cache-dir numpy==2.4.6', "re-pin: numpy 2.4.6", "Couldn't install numpy 2.4.6", live=False)
-        print("[INFO] Re-pinned numpy==2.4.6 after extensions")
+        run(f'"{python}" -m pip install --force-reinstall --no-deps --no-cache-dir numpy==2.5.3', "re-pin: numpy 2.5.3", "Couldn't install numpy 2.5.3", live=False)
+        print("[INFO] Re-pinned numpy==2.5.3 after extensions")
 
     if args.update_check:
         version_check(commit)
```

### 3-2. `requirements_versions_py314.txt` / `requirements_versions_py314_windows.txt` (modified)

**Before (HEAD)**

```text
numpy==2.4.6
```

**After**

```text
numpy==2.5.3
```

### 3-3. Replacement script (local, not committed)

The pin strings in `modules/launch_utils.py` were replaced with a short local Python script (byte-safe: read as bytes, decode UTF-8, replace, write back, so line endings are preserved). It is kept outside the repository and is not part of the commit.

```python
P = "modules/launch_utils.py"

with open(P, "rb") as f:
    raw = f.read()
s = raw.decode("utf-8")

reps = [
    # 1) comment line: drop the now-false "not 2.5.x" rationale, pin 2.5.3
    (
        "    # Pin numpy==2.4.6 before -r (numba / facexlib compatible; not 2.5.x).",
        "    # Pin numpy==2.5.3 before -r.",
    ),
    # 2) exact version specifiers
    ("numpy==2.4.6", "numpy==2.5.3"),
    # 3) human-readable messages / comments that spell the version with a space
    ("numpy 2.4.6", "numpy 2.5.3"),
]

for old, new in reps:
    print(old, s.count(old))
    s = s.replace(old, new)

with open(P, "wb") as f:
    f.write(s.encode("utf-8"))

print("remaining 2.4.6:", [l for l in s.splitlines() if "2.4.6" in l])
```

### 3-4. Commands executed

```powershell
# 1) Inspect current state
& 'venv\Scripts\python.exe' -c "import sys, numpy; print(sys.version); print(numpy.__version__)"

# 2) Upgrade numpy to 2.5.3
& 'venv\Scripts\python.exe' -m pip install "numpy==2.5.3"

# 3) Upgrade numba to latest (resolved together with numpy)
& 'venv\Scripts\python.exe' -m pip install -U numba

# 4) Verify
& 'venv\Scripts\python.exe' -c "import numpy, numba, llvmlite; print(numpy.__version__, numba.__version__, llvmlite.__version__)"
& 'venv\Scripts\python.exe' -m pip check
```

---

## ④ What it means

### 1. The two layers of version control were aligned

- In this project, the NumPy version is decided in two places:
  1. Requirements (declarative dependency): `requirements_versions_py314*.txt`
  2. The effective pin in the startup script: `modules/launch_utils.py`
- `launch_utils.py` **uninstalls and reinstalls NumPy** at startup and, after the extension installers run, **re-pins** it. So even if requirements say 2.5.3, as long as this file said 2.4.6 the environment would **roll back to 2.4.6 on every launch**.
- Therefore, to make "NumPy 2.5.3" actually stick, both layers must be set to 2.5.3. That is what was done here.

### 2. Meaning of each code change

- `pip uninstall numpy -y` → `pip install --no-cache-dir numpy==2.5.3`
  - Fixes NumPy **before** requirements `-r` is applied. Installing it early prevents already-imported modules from holding a stale NumPy. `--no-cache-dir` avoids cache-driven inconsistencies.
- `# scipy via PyPI to match numpy 2.5.3`
  - scipy is also reinstalled from PyPI so the pairing with NumPy stays consistent. Only the comment was updated (`scipy==1.16.1` is unchanged).
- `--force-reinstall --no-deps numpy==2.5.3` (after extension installers)
  - A **safety net** so that extension installs cannot drift NumPy to another version; it re-fixes NumPy to 2.5.3 after extensions are installed.
- Resolving `numba` together with NumPy
  - numba depends tightly on NumPy (0.66.0 requires `numpy<2.5`; 0.67.0 requires `numpy<2.6`). Resolved together with NumPy 2.5.3, the newest satisfying numba 0.67.0 (with llvmlite 0.49.0) is chosen automatically. Upgrading NumPy alone leaves the old numba 0.66.0 in place and produces a conflict warning, so **upgrading them together** is the correct procedure.

### 3. Impact and risk

- numpy 2.5.3 coexists with numba 0.67.0, so facexlib (which depends on numba) and its face-restoration paths remain functional.
- The old comment "numba / facexlib compatible; not 2.5.x" no longer holds once 2.5.x is the target, so it was updated to match the facts.
- If anything goes wrong, restore your local pre-change copy of `modules/launch_utils.py` and reinstall the previous versions with `pip install "numpy==2.4.6" "numba==0.66.0" "llvmlite==0.48.0"`.

### 4. Rollback procedure

```powershell
# Restore modules/launch_utils.py from your local pre-change copy, then:

# Restore packages
& 'venv\Scripts\python.exe' -m pip install "numpy==2.4.6" "numba==0.66.0" "llvmlite==0.48.0"
```
