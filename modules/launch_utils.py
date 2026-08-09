# this scripts installs necessary requirements and launches main program in webui.py
import logging
import re
import subprocess
import os
import shutil
import sys
import importlib.util
import importlib.metadata
import platform
import json
import shlex
from functools import lru_cache

from modules import cmd_args, errors
from modules.paths_internal import script_path, extensions_dir
from modules.timer import startup_timer
from modules import logging_config



args, _ = cmd_args.parser.parse_known_args()
logging_config.setup_logging(args.loglevel)

python = sys.executable
git = os.environ.get('GIT', "git")
index_url = os.environ.get('INDEX_URL', "")
dir_repos = "repositories"

# Whether to default to printing command output
default_command_live = (os.environ.get('WEBUI_LAUNCH_LIVE_OUTPUT') == "1")

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')


def check_python_version():
    major = sys.version_info.major
    minor = sys.version_info.minor
    micro = sys.version_info.micro

    if not (major == 3 and minor == 14):
        import modules.errors

        modules.errors.print_error_explanation(f"""
INCOMPATIBLE PYTHON VERSION

This repository supports Python 3.14 only, but you have {major}.{minor}.{micro}.
Install Python 3.14 and recreate the "venv" folder in WebUI's directory.

You can download Python 3.14 from here: https://www.python.org/downloads/

Use --skip-python-version-check to suppress this warning.
""")


@lru_cache()
def commit_hash():
    try:
        return subprocess.check_output([git, "-C", script_path, "rev-parse", "HEAD"], shell=False, encoding='utf8').strip()
    except Exception:
        return "<none>"


@lru_cache()
def git_tag():
    try:
        return subprocess.check_output([git, "-C", script_path, "describe", "--tags"], shell=False, encoding='utf8').strip()
    except Exception:
        try:

            changelog_md = os.path.join(script_path, "CHANGELOG.md")
            with open(changelog_md, "r", encoding="utf-8") as file:
                line = next((line.strip() for line in file if line.strip()), "<none>")
                line = line.replace("## ", "")
                return line
        except Exception:
            return "<none>"


def run(command, desc=None, errdesc=None, custom_env=None, live: bool = default_command_live) -> str:
    if desc is not None:
        print(desc)

    run_kwargs = {
        "args": command,
        "shell": True,
        "env": os.environ if custom_env is None else custom_env,
        "encoding": 'utf8',
        "errors": 'ignore',
    }

    if not live:
        run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        error_bits = [
            f"{errdesc or 'Error running command'}.",
            f"Command: {command}",
            f"Error code: {result.returncode}",
        ]
        if result.stdout:
            error_bits.append(f"stdout: {result.stdout}")
        if result.stderr:
            error_bits.append(f"stderr: {result.stderr}")
        raise RuntimeError("\n".join(error_bits))

    return (result.stdout or "")


def is_installed(package):
    try:
        dist = importlib.metadata.distribution(package)
    except importlib.metadata.PackageNotFoundError:
        try:
            spec = importlib.util.find_spec(package)
        except ModuleNotFoundError:
            return False

        return spec is not None

    return dist is not None


def repo_dir(name):
    return os.path.join(script_path, dir_repos, name)


def run_pip(command, desc=None, live=default_command_live):
    if args.skip_install:
        return

    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    # Add --only-binary=all for packages that might have build issues
    # Check if command contains tokenizers or transformers
    is_tokenizers_related = ('tokenizers' in command or 'transformers' in command)
    only_binary_flag = ' --only-binary=all' if is_tokenizers_related else ''
    return run(f'"{python}" -m pip {command} --prefer-binary{only_binary_flag}{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}", live=live)


def _torch_stack_constraint_file() -> str | None:
    """
    Pin only installed torch (exact local version, e.g. 2.13.0+cu132).
    Prevents later pip -r / extension install.py from replacing CUDA builds with PyPI CPU wheels.
    """
    import importlib.metadata

    try:
        ver = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None

    path = os.path.join(script_path, "tmp", "torch_stack_constraints.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"torch=={ver}\n")
    return path


def _apply_torch_stack_pip_constraint() -> None:
    path = _torch_stack_constraint_file()
    if path is None:
        return
    os.environ["PIP_CONSTRAINT"] = path
    print(f"Torch pinned ({path}); pip will not replace torch")


def check_run_python(code: str) -> bool:
    result = subprocess.run([python, "-c", code], capture_output=True, shell=False)
    return result.returncode == 0


def git_fix_workspace(dir, name):
    run(f'"{git}" -C "{dir}" fetch --refetch --no-auto-gc', f"Fetching all contents for {name}", f"Couldn't fetch {name}", live=True)
    run(f'"{git}" -C "{dir}" gc --aggressive --prune=now', f"Pruning {name}", f"Couldn't prune {name}", live=True)
    return


def run_git(dir, name, command, desc=None, errdesc=None, custom_env=None, live: bool = default_command_live, autofix=True):
    try:
        return run(f'"{git}" -C "{dir}" {command}', desc=desc, errdesc=errdesc, custom_env=custom_env, live=live)
    except RuntimeError:
        if not autofix:
            raise

    print(f"{errdesc}, attempting autofix...")
    git_fix_workspace(dir, name)

    return run(f'"{git}" -C "{dir}" {command}', desc=desc, errdesc=errdesc, custom_env=custom_env, live=live)


def git_clone(url, dir, name, commithash=None):
    # TODO clone into temporary dir and move if successful

    if os.path.exists(dir):
        if commithash is None:
            return

        current_hash = run_git(dir, name, 'rev-parse HEAD', None, f"Couldn't determine {name}'s hash: {commithash}", live=False).strip()
        if current_hash == commithash:
            return

        if run_git(dir, name, 'config --get remote.origin.url', None, f"Couldn't determine {name}'s origin URL", live=False).strip() != url:
            run_git(dir, name, f'remote set-url origin "{url}"', None, f"Failed to set {name}'s origin URL", live=False)

        run_git(dir, name, 'fetch', f"Fetching updates for {name}...", f"Couldn't fetch {name}", autofix=False)

        run_git(dir, name, f'checkout {commithash}', f"Checking out commit for {name} with hash: {commithash}...", f"Couldn't checkout commit {commithash} for {name}", live=True)

        return

    try:
        run(f'"{git}" clone --config core.filemode=false "{url}" "{dir}"', f"Cloning {name} into {dir}...", f"Couldn't clone {name}", live=True)
    except RuntimeError:
        shutil.rmtree(dir, ignore_errors=True)
        raise

    if commithash is not None:
        run(f'"{git}" -C "{dir}" checkout {commithash}', None, "Couldn't checkout {name}'s hash: {commithash}")


def git_pull_recursive(dir):
    for subdir, _, _ in os.walk(dir):
        if os.path.exists(os.path.join(subdir, '.git')):
            try:
                output = subprocess.check_output([git, '-C', subdir, 'pull', '--autostash'])
                print(f"Pulled changes for repository in '{subdir}':\n{output.decode('utf-8').strip()}\n")
            except subprocess.CalledProcessError as e:
                print(f"Couldn't perform 'git pull' on repository in '{subdir}':\n{e.output.decode('utf-8').strip()}\n")


def version_check(commit):
    try:
        import requests
        commits = requests.get('https://api.github.com/repos/AUTOMATIC1111/stable-diffusion-webui/branches/master').json()
        if commit != "<none>" and commits['commit']['sha'] != commit:
            print("--------------------------------------------------------")
            print("| You are not up to date with the most recent release. |")
            print("| Consider running `git pull` to update.               |")
            print("--------------------------------------------------------")
        elif commits['commit']['sha'] == commit:
            print("You are up to date with the most recent release.")
        else:
            print("Not a git clone, can't perform version check.")
    except Exception as e:
        print("version check failed", e)


def run_extension_installer(extension_dir):
    path_installer = os.path.join(extension_dir, "install.py")
    if not os.path.isfile(path_installer):
        return

    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{script_path}{os.pathsep}{env.get('PYTHONPATH', '')}"

        stdout = run(f'"{python}" "{path_installer}"', errdesc=f"Error running install.py for extension {extension_dir}", custom_env=env).strip()
        if stdout:
            print(stdout)
    except Exception as e:
        errors.report(str(e))


def list_extensions(settings_file):
    settings = {}

    try:
        with open(settings_file, "r", encoding="utf8") as file:
            settings = json.load(file)
    except FileNotFoundError:
        pass
    except Exception:
        errors.report(f'\nCould not load settings\nThe config file "{settings_file}" is likely corrupted\nIt has been moved to the "tmp/config.json"\nReverting config to default\n\n''', exc_info=True)
        os.replace(settings_file, os.path.join(script_path, "tmp", "config.json"))

    disabled_extensions = set(settings.get('disabled_extensions', []))
    disable_all_extensions = settings.get('disable_all_extensions', 'none')

    if disable_all_extensions != 'none' or args.disable_extra_extensions or args.disable_all_extensions or not os.path.isdir(extensions_dir):
        return []

    return [x for x in os.listdir(extensions_dir) if x not in disabled_extensions]


def run_extensions_installers(settings_file):
    if not os.path.isdir(extensions_dir):
        return

    with startup_timer.subcategory("run extensions installers"):
        for dirname_extension in list_extensions(settings_file):
            logging.debug(f"Installing {dirname_extension}")

            path = os.path.join(extensions_dir, dirname_extension)

            if os.path.isdir(path):
                run_extension_installer(path)
                startup_timer.record(dirname_extension)


def migrate_multidiffusion_to_builtin():
    """Ensure multidiffusion-upscaler is treated as built-in extension."""
    ext_name = "multidiffusion-upscaler-for-automatic1111"
    src = os.path.join(extensions_dir, ext_name)
    dst_root = os.path.join(script_path, "extensions-builtin")
    dst = os.path.join(dst_root, ext_name)

    if not os.path.isdir(src):
        return

    try:
        os.makedirs(dst_root, exist_ok=True)

        if os.path.isdir(dst):
            print(f"[INFO] Built-in {ext_name} already exists; keeping built-in copy.")
        else:
            shutil.move(src, dst)
            print(f"[INFO] Migrated {ext_name} from extensions/ to extensions-builtin/.")

        git_path = os.path.join(dst, ".git")
        if os.path.isdir(git_path):
            shutil.rmtree(git_path, ignore_errors=True)
            print(f"[INFO] Removed nested .git from built-in {ext_name}.")
        elif os.path.isfile(git_path):
            os.remove(git_path)
            print(f"[INFO] Removed .git file marker from built-in {ext_name}.")
    except Exception as e:
        print(f"[WARNING] Failed to migrate {ext_name} to built-in: {e}")


def migrate_sd_dynamic_thresholding_to_builtin():
    """Ensure sd-dynamic-thresholding is treated as built-in extension."""
    ext_name = "sd-dynamic-thresholding"
    src = os.path.join(extensions_dir, ext_name)
    dst_root = os.path.join(script_path, "extensions-builtin")
    dst = os.path.join(dst_root, ext_name)

    if os.path.isdir(dst):
        git_path = os.path.join(dst, ".git")
        if os.path.isdir(git_path):
            shutil.rmtree(git_path, ignore_errors=True)
            print(f"[INFO] Removed nested .git from built-in {ext_name}.")
        elif os.path.isfile(git_path):
            os.remove(git_path)
            print(f"[INFO] Removed .git file marker from built-in {ext_name}.")

    if not os.path.isdir(src):
        return

    try:
        os.makedirs(dst_root, exist_ok=True)

        if os.path.isdir(dst):
            print(f"[INFO] Built-in {ext_name} already exists; keeping built-in copy.")
        else:
            shutil.move(src, dst)
            print(f"[INFO] Migrated {ext_name} from extensions/ to extensions-builtin/.")

        git_path = os.path.join(dst, ".git")
        if os.path.isdir(git_path):
            shutil.rmtree(git_path, ignore_errors=True)
            print(f"[INFO] Removed nested .git from built-in {ext_name}.")
        elif os.path.isfile(git_path):
            os.remove(git_path)
            print(f"[INFO] Removed .git file marker from built-in {ext_name}.")
    except Exception as e:
        print(f"[WARNING] Failed to migrate {ext_name} to built-in: {e}")


def migrate_freeu_to_builtin():
    """Ensure sd-webui-freeu is treated as built-in extension."""
    ext_name = "sd-webui-freeu"
    src = os.path.join(extensions_dir, ext_name)
    dst_root = os.path.join(script_path, "extensions-builtin")
    dst = os.path.join(dst_root, ext_name)

    if os.path.isdir(dst):
        git_path = os.path.join(dst, ".git")
        if os.path.isdir(git_path):
            shutil.rmtree(git_path, ignore_errors=True)
            print(f"[INFO] Removed nested .git from built-in {ext_name}.")
        elif os.path.isfile(git_path):
            os.remove(git_path)
            print(f"[INFO] Removed .git file marker from built-in {ext_name}.")

    if not os.path.isdir(src):
        return

    try:
        os.makedirs(dst_root, exist_ok=True)

        if os.path.isdir(dst):
            print(f"[INFO] Built-in {ext_name} already exists; keeping built-in copy.")
        else:
            shutil.move(src, dst)
            print(f"[INFO] Migrated {ext_name} from extensions/ to extensions-builtin/.")

        git_path = os.path.join(dst, ".git")
        if os.path.isdir(git_path):
            shutil.rmtree(git_path, ignore_errors=True)
            print(f"[INFO] Removed nested .git from built-in {ext_name}.")
        elif os.path.isfile(git_path):
            os.remove(git_path)
            print(f"[INFO] Removed .git file marker from built-in {ext_name}.")
    except Exception as e:
        print(f"[WARNING] Failed to migrate {ext_name} to built-in: {e}")


def migrate_controlnet_to_builtin():
    """Ensure sd-webui-controlnet is treated as built-in extension."""
    ext_name = "sd-webui-controlnet"
    src = os.path.join(extensions_dir, ext_name)
    dst_root = os.path.join(script_path, "extensions-builtin")
    dst = os.path.join(dst_root, ext_name)

    if os.path.isdir(dst):
        git_path = os.path.join(dst, ".git")
        if os.path.isdir(git_path):
            shutil.rmtree(git_path, ignore_errors=True)
            print(f"[INFO] Removed nested .git from built-in {ext_name}.")
        elif os.path.isfile(git_path):
            os.remove(git_path)
            print(f"[INFO] Removed .git file marker from built-in {ext_name}.")

    if not os.path.isdir(src):
        return

    try:
        os.makedirs(dst_root, exist_ok=True)

        if os.path.isdir(dst):
            print(f"[INFO] Built-in {ext_name} already exists; keeping built-in copy.")
        else:
            shutil.move(src, dst)
            print(f"[INFO] Migrated {ext_name} from extensions/ to extensions-builtin/.")

        git_path = os.path.join(dst, ".git")
        if os.path.isdir(git_path):
            shutil.rmtree(git_path, ignore_errors=True)
            print(f"[INFO] Removed nested .git from built-in {ext_name}.")
        elif os.path.isfile(git_path):
            os.remove(git_path)
            print(f"[INFO] Removed .git file marker from built-in {ext_name}.")
    except Exception as e:
        print(f"[WARNING] Failed to migrate {ext_name} to built-in: {e}")


re_requirement = re.compile(r"\s*([-_a-zA-Z0-9]+)\s*(?:==\s*([-+_.a-zA-Z0-9]+))?\s*")


def requirements_met(requirements_file):
    """
    Does a simple parse of a requirements.txt file to determine if all rerqirements in it
    are already installed. Returns True if so, False if not installed or parsing fails.
    """

    import importlib.metadata
    import packaging.version

    with open(requirements_file, "r", encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue

            m = re.match(re_requirement, line)
            if m is None:
                return False

            package = m.group(1).strip()
            version_required = (m.group(2) or "").strip()

            if version_required == "":
                continue

            try:
                version_installed = importlib.metadata.version(package)
            except Exception:
                return False

            if packaging.version.parse(version_required) != packaging.version.parse(version_installed):
                return False

    return True


def prepare_environment():
    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu130")
    torch_command = os.environ.get(
        'TORCH_COMMAND',
        f"pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 torchaudio==2.11.0+cu130 --index-url {torch_index_url}",
    )
    if args.use_ipex:
        if platform.system() == "Windows":
            # The "Nuullll/intel-extension-for-pytorch" wheels were built from IPEX source for Intel Arc GPU: https://github.com/intel/intel-extension-for-pytorch/tree/xpu-main
            # This is NOT an Intel official release so please use it at your own risk!!
            # See https://github.com/Nuullll/intel-extension-for-pytorch/releases/tag/v2.0.110%2Bxpu-master%2Bdll-bundle for details.
            #
            # Strengths (over official IPEX 2.0.110 windows release):
            #   - AOT build (for Arc GPU only) to eliminate JIT compilation overhead: https://github.com/intel/intel-extension-for-pytorch/issues/399
            #   - Bundles minimal oneAPI 2023.2 dependencies into the python wheels, so users don't need to install oneAPI for the whole system.
            #   - Provides a compatible torchvision wheel: https://github.com/intel/intel-extension-for-pytorch/issues/465
            # Limitation:
            #   - Only works for python 3.10
            url_prefix = "https://github.com/Nuullll/intel-extension-for-pytorch/releases/download/v2.0.110%2Bxpu-master%2Bdll-bundle"
            torch_command = os.environ.get('TORCH_COMMAND', f"pip install {url_prefix}/torch-2.0.0a0+gite9ebda2-cp310-cp310-win_amd64.whl {url_prefix}/torchvision-0.15.2a0+fa99a53-cp310-cp310-win_amd64.whl {url_prefix}/intel_extension_for_pytorch-2.0.110+gitc6ea20b-cp310-cp310-win_amd64.whl")
        else:
            # Using official IPEX release for linux since it's already an AOT build.
            # However, users still have to install oneAPI toolkit and activate oneAPI environment manually.
            # See https://intel.github.io/intel-extension-for-pytorch/index.html#installation for details.
            torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://pytorch-extension.intel.com/release-whl/stable/xpu/us/")
            torch_command = os.environ.get('TORCH_COMMAND', f"pip install torch==2.0.0a0 intel-extension-for-pytorch==2.0.110+gitba7f6c1 --extra-index-url {torch_index_url}")
    # Python 3.14 only
    if platform.system() == "Windows":
        requirements_file = os.environ.get('REQS_FILE', "requirements_versions_py314_windows.txt")
    else:
        requirements_file = os.environ.get('REQS_FILE', "requirements_versions_py314.txt")
    requirements_file_for_npu = os.environ.get('REQS_FILE_FOR_NPU', "requirements_npu.txt")
    # open_clip: required for SD2 / SDXL text encoders (not openai CLIP).
    # openai CLIP (`import clip`) is only needed by k-diffusion training metrics
    # (evaluation.CLIPFeatureExtractor). WebUI sampling does not import that path;
    # repositories/k-diffusion was patched so package import no longer pulls it in.
    openclip_package = os.environ.get('OPENCLIP_PACKAGE', "https://github.com/mlfoundations/open_clip/archive/bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b.zip")
    # Flash-Attention 2 source is platform-specific:
    #   Windows: prebuilt wheel (cu132 + torch 2.13, cp314)
    #   Linux:   source build via PyPI (requires CUDA toolkit + nvcc, ~30min compile)
    #   Mac:     skipped (FA2 requires CUDA; MPS backend cannot use it)
    if platform.system() == "Windows":
        flash_attn_package = os.environ.get('FLASH_ATTN_PACKAGE', 'https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows/resolve/main/flash_attn-2.8.4%2Bcu132torch2.13.0cxx11abiTRUE-cp314-cp314-win_amd64.whl')
        fa2_install_enabled = True
    elif platform.system() == "Linux":
        flash_attn_package = os.environ.get('FLASH_ATTN_PACKAGE', 'flash-attn==2.8.4')
        fa2_install_enabled = True
    else:
        flash_attn_package = None
        fa2_install_enabled = False
    distutils_wheel = os.environ.get(
        'DISTUTILS_WHEEL',
        'https://huggingface.co/ussoewwin/distutils/resolve/main/distutils-3.14.0-py3-none-any.whl',
    )
    # Gradio 3.41.2 HF wheel (METADATA numpy/pillow version pins removed).
    gradio_package = os.environ.get(
        'GRADIO_PACKAGE',
        'https://huggingface.co/ussoewwin/gradio3.41.2/resolve/main/gradio-3.41.2-py3-none-any.whl',
    )

    assets_repo = os.environ.get('ASSETS_REPO', "https://github.com/AUTOMATIC1111/stable-diffusion-webui-assets.git")
    stable_diffusion_repo = os.environ.get('STABLE_DIFFUSION_REPO', "https://github.com/Stability-AI/stablediffusion.git")
    stable_diffusion_xl_repo = os.environ.get('STABLE_DIFFUSION_XL_REPO', "https://github.com/Stability-AI/generative-models.git")
    k_diffusion_repo = os.environ.get('K_DIFFUSION_REPO', 'https://github.com/crowsonkb/k-diffusion.git')
    blip_repo = os.environ.get('BLIP_REPO', 'https://github.com/salesforce/BLIP.git')

    assets_commit_hash = os.environ.get('ASSETS_COMMIT_HASH', "6f7db241d2f8ba7457bac5ca9753331f0c266917")
    stable_diffusion_commit_hash = os.environ.get('STABLE_DIFFUSION_COMMIT_HASH', "cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf")
    stable_diffusion_xl_commit_hash = os.environ.get('STABLE_DIFFUSION_XL_COMMIT_HASH', "45c443b316737a4ab6e40413d7794a7f5657c19f")
    k_diffusion_commit_hash = os.environ.get('K_DIFFUSION_COMMIT_HASH', "ab527a9a6d347f364e3d185ba6d714e22d80cb3c")
    blip_commit_hash = os.environ.get('BLIP_COMMIT_HASH', "48211a1594f1321b00f14c9f7a5b4813144b2fb9")

    try:
        # the existence of this file is a signal to webui.sh/bat that webui needs to be restarted when it stops execution
        os.remove(os.path.join(script_path, "tmp", "restart"))
        os.environ.setdefault('SD_WEBUI_RESTARTING', '1')
    except OSError:
        pass

    if not args.skip_python_version_check:
        check_python_version()

    startup_timer.record("checks")
    migrate_multidiffusion_to_builtin()
    startup_timer.record("migrate multidiffusion builtin")
    migrate_sd_dynamic_thresholding_to_builtin()
    startup_timer.record("migrate sd-dynamic-thresholding builtin")
    migrate_freeu_to_builtin()
    startup_timer.record("migrate freeu builtin")
    migrate_controlnet_to_builtin()
    startup_timer.record("migrate controlnet builtin")

    commit = commit_hash()
    tag = git_tag()
    startup_timer.record("git version info")

    print(f"Python {sys.version}")
    print(f"Version: {tag}")
    print(f"Commit hash: {commit}")

    # Upgrade pip first
    run(f'"{python}" -m pip install --upgrade pip', "Upgrading pip", "Couldn't upgrade pip", live=False)
    
    # Ensure packaging is available for pkg_resources
    packaging_wheel = os.environ.get('PACKAGING_WHEEL', 'https://huggingface.co/ussoewwin/packaging-25.0-py3-none-any/resolve/main/packaging-25.0-py3-none-any.whl')
    if not is_installed("packaging"):
        run(f'"{python}" -m pip install {packaging_wheel} --no-deps --no-index', "Installing packaging", "Couldn't install packaging", live=False)

    # Install distutils stub (stdlib distutils removed; required on Python 3.14)
    try:
        from distutils.version import StrictVersion
    except ImportError:
        run(f'"{python}" -m pip install {distutils_wheel} --no-deps --no-index', "Installing distutils", "Couldn't install distutils", live=False)

    # xformers installation removed - install manually if needed
    # Copy xformers_fix files to xformers installation directory (only if xformers is installed)
    try:
        import xformers
        xformers_path = os.path.dirname(xformers.__file__)
        fmha_path = os.path.join(xformers_path, "ops", "fmha")
        xformers_fix_path = os.path.join(script_path, "xformers_fix")
        
        # Copy only these 2 files from xformers_fix to venv\Lib\site-packages\xformers\ops\fmha\
        files_to_copy = ["dispatch.py", "__init__.py"]
        
        if os.path.exists(xformers_fix_path) and os.path.exists(fmha_path):
            for file_name in files_to_copy:
                src_file = os.path.join(xformers_fix_path, file_name)
                dst_file = os.path.join(fmha_path, file_name)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
                    print(f"Copied {file_name} to xformers/ops/fmha/")
    except ImportError:
        # xformers is not installed, skip copying fix files
        pass
    except Exception as e:
        print(f"Warning: Failed to copy xformers_fix files: {e}")

    # PyTorch installation removed - install manually if needed

    # Check if torch is installed, but don't install it automatically
    if not is_installed("torch") or not is_installed("torchvision"):
        print("Warning: PyTorch is not installed. Please install it manually before running the web UI.")
        print(f"Suggested command: {torch_command}")

    # Pin installed CUDA torch before requirements / extension installers can pull PyPI CPU torch.
    if is_installed("torch"):
        _apply_torch_stack_pip_constraint()

    # Only test CUDA if torch is installed
    if is_installed("torch"):
        if args.use_ipex:
            args.skip_torch_cuda_test = True
        if not args.skip_torch_cuda_test and not check_run_python("import torch; assert torch.cuda.is_available()"):
            raise RuntimeError(
                'Torch is not able to use GPU; '
                'add --skip-torch-cuda-test to COMMANDLINE_ARGS variable to disable this check'
            )
        startup_timer.record("torch GPU test")

    # Ensure packaging is installed before clip (clip requires pkg_resources.packaging)
    packaging_wheel = os.environ.get('PACKAGING_WHEEL', 'https://huggingface.co/ussoewwin/packaging-25.0-py3-none-any/resolve/main/packaging-25.0-py3-none-any.whl')
    if not is_installed("packaging"):
        run(f'"{python}" -m pip install {packaging_wheel} --no-deps --no-index', "Installing packaging", "Couldn't install packaging", live=False)

    if not is_installed("open_clip"):
        run_pip(f"install {openclip_package}", "open_clip")
        startup_timer.record("install open_clip")

    if fa2_install_enabled and not is_installed("flash_attn"):
        if platform.system() == "Linux":
            run_pip(f"install {flash_attn_package} --no-build-isolation", "flash_attn")
        else:
            run_pip(f"install {flash_attn_package}", "flash_attn")
        startup_timer.record("install flash_attn")

    if not is_installed("ngrok") and args.ngrok:
        run_pip("install ngrok", "ngrok")
        startup_timer.record("install ngrok")

    os.makedirs(os.path.join(script_path, dir_repos), exist_ok=True)

    # External repository cloning/fetching is permanently disabled.
    # All required dependencies are vendored under `repositories/` in this repo.
    for _required_repo in ("stable-diffusion-webui-assets", "stable-diffusion-stability-ai", "generative-models", "k-diffusion", "BLIP"):
        if not os.path.isdir(repo_dir(_required_repo)):
            print(f"Warning: vendored repository missing: {repo_dir(_required_repo)}", file=sys.stderr)

    startup_timer.record("clone repositores")

    if not os.path.isfile(requirements_file):
        requirements_file = os.path.join(script_path, requirements_file)

    run_pip("install --upgrade packaging", "packaging")

    # Pin numpy==2.5.1 before -r (same premise as Stable-Diffusion-WebUI-Forge-Nunchaku).
    # Install early so already-imported modules do not keep a stale numpy.
    run(f'"{python}" -m pip uninstall numpy -y', "uninstalling numpy", "Couldn't uninstall numpy", live=False)
    run(f'"{python}" -m pip install --no-cache-dir numpy==2.5.1', "Installing numpy 2.5.1", "Couldn't install numpy 2.5.1", live=False)
    print("[INFO] Installed numpy==2.5.1 from PyPI")

    # Gradio from HF (METADATA version pins removed)
    run_pip(f'install "{gradio_package}"', "gradio")
    print(f"[INFO] Installed gradio from: {gradio_package}")
    startup_timer.record("install gradio")

    if not requirements_met(requirements_file):
        # only-if-needed: do not upgrade already-satisfied deps (esp. torch pulled by accelerate/etc.)
        run_pip(f"install -U --upgrade-strategy only-if-needed -r \"{requirements_file}\"", "requirements")
        startup_timer.record("install requirements")
    
    # scipy via PyPI to match numpy 2.5.1
    run(f'"{python}" -m pip uninstall scipy -y', "uninstalling scipy", "Couldn't uninstall scipy", live=False)
    run(f'"{python}" -m pip install --no-cache-dir scipy==1.16.1', "Installing scipy 1.16.1", "Couldn't install scipy 1.16.1", live=False)
    print("[INFO] Installed scipy 1.16.1 from PyPI")
    
    # Auto-fix clip.py packaging import (AFTER requirements installation)
    def fix_clip_packaging_import():
        if platform.system() == "Windows":
            clip_py_path = os.path.join(script_path, "venv", "Lib", "site-packages", "clip", "clip.py")
        else:
            py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
            clip_py_path = os.path.join(script_path, "venv", "lib", py_ver, "site-packages", "clip", "clip.py")
        if os.path.isfile(clip_py_path):
            try:
                with open(clip_py_path, "r", encoding="utf-8") as f:
                    content = f.read()

                if "from pkg_resources import packaging" in content:
                    content = content.replace("from pkg_resources import packaging", "from packaging import version")
                    print("[INFO] Fixed clip.py import: replaced pkg_resources with direct packaging import")

                if "packaging.version.parse" in content:
                    content = content.replace("packaging.version.parse", "version.parse")
                    print("[INFO] Fixed clip.py usage: replaced packaging.version.parse with version.parse")

                with open(clip_py_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print("[INFO] Successfully auto-fixed clip.py packaging imports")
            except Exception as e:
                print(f"[WARNING] Could not auto-fix clip.py: {e}")

    fix_clip_packaging_import()

    # Fix protobuf runtime_version import error
    try:
        from google.protobuf import runtime_version
    except ImportError:
        run_pip("install --force-reinstall --no-cache-dir protobuf==7.34.1", "protobuf", live=False)

    if not os.path.isfile(requirements_file_for_npu):
        requirements_file_for_npu = os.path.join(script_path, requirements_file_for_npu)

    if "torch_npu" in torch_command and not requirements_met(requirements_file_for_npu):
        run_pip(f"install -r \"{requirements_file_for_npu}\"", "requirements_for_npu")
        startup_timer.record("install requirements_for_npu")

    if not args.skip_install:
        run_extensions_installers(settings_file=args.ui_settings_file)
        # Keep Forge-aligned pin if an extension installer drifted numpy.
        run(f'"{python}" -m pip install --force-reinstall --no-deps --no-cache-dir numpy==2.5.1', "re-pin: numpy 2.5.1", "Couldn't install numpy 2.5.1", live=False)
        print("[INFO] Re-pinned numpy==2.5.1 after extensions")

    if args.update_check:
        version_check(commit)
        startup_timer.record("check version")

    if args.update_all_extensions:
        git_pull_recursive(extensions_dir)
        startup_timer.record("update extensions")

    if "--exit" in sys.argv:
        print("Exiting because of --exit argument")
        exit(0)


def configure_for_tests():
    if "--api" not in sys.argv:
        sys.argv.append("--api")
    if "--ckpt" not in sys.argv:
        sys.argv.append("--ckpt")
        sys.argv.append(os.path.join(script_path, "test/test_files/empty.pt"))
    if "--skip-torch-cuda-test" not in sys.argv:
        sys.argv.append("--skip-torch-cuda-test")
    if "--disable-nan-check" not in sys.argv:
        sys.argv.append("--disable-nan-check")

    os.environ['COMMANDLINE_ARGS'] = ""


def start():
    print(f"Launching {'API server' if '--nowebui' in sys.argv else 'Web UI'} with arguments: {shlex.join(sys.argv[1:])}")
    import webui
    if '--nowebui' in sys.argv:
        webui.api_only()
    else:
        webui.webui()


def dump_sysinfo():
    from modules import sysinfo
    import datetime

    text = sysinfo.get()
    filename = f"sysinfo-{datetime.datetime.utcnow().strftime('%Y-%m-%d-%H-%M')}.json"

    with open(filename, "w", encoding="utf8") as file:
        file.write(text)

    return filename
