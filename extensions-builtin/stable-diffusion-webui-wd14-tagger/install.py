"""Install requirements for WD14-tagger."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from importlib.metadata import version  # python >= 3.8

from packaging.version import parse

_WD14_ROOT = os.path.dirname(os.path.abspath(__file__))
_BUILTIN_SUFFIX = os.path.join("extensions-builtin", "stable-diffusion-webui-wd14-tagger")
_IS_BUILTIN = os.path.normpath(_WD14_ROOT).endswith(_BUILTIN_SUFFIX)

import_name = {"py-cpuinfo": "cpuinfo", "opencv-contrib-python": "cv2"}


def is_installed(
    package: str,
    min_version: str | None = None,
    max_version: str | None = None,
):
    name = import_name.get(package, package)
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return False

    if spec is None:
        return False

    if not min_version and not max_version:
        return True

    if not min_version:
        min_version = "0.0.0"
    if not max_version:
        max_version = "99999999.99999999.99999999"

    try:
        pkg_version = version(package)
        return parse(min_version) <= parse(pkg_version) <= parse(max_version)
    except Exception:
        return False


def run_pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", *args], check=True)


def install():
    if _IS_BUILTIN:
        return

    deps = [
        # Non-TF deps only. deepdanbooru + tensorflow install on demand when a
        # DeepDanbooru model is loaded (see tagger/interrogator.py).
        ("jsonschema", None, None),
        ("opencv-contrib-python", None, None),
    ]

    pkgs = []
    for pkg, low, high in deps:
        if not is_installed(pkg, low, high):
            if low and high:
                cmd = f"{pkg}>={low},<={high}"
            elif low:
                cmd = f"{pkg}>={low}"
            elif high:
                cmd = f"{pkg}<={high}"
            else:
                cmd = pkg
            pkgs.append(cmd)

    if pkgs:
        run_pip(*pkgs)


try:
    import launch

    skip_install = launch.args.skip_install
except Exception:
    skip_install = False

if not skip_install and not _IS_BUILTIN:
    install()
