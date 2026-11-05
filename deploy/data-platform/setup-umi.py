#!/usr/bin/env python3
"""Prepare an opt-in, pinned data-laundry checkout without changing platform services."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlsplit

PLATFORM = Path(__file__).resolve().parents[2]
LAUNDRY_COMMIT = "fd752156895c83ec75fde84796faea2513c626c7"
CONVERTER_COMMIT = "74b73a0d65b5940e539c260f91a4ff9c7528c194"
ROBOT = Path("src/laundry/umi_replay/assets/rmpflow/model/h10w_dvt2_chassis")


def clean_env(*roots: Path) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        env.pop(key, None)
    env.update(LAUNDRY_CUMOTION_NO_INSTALL="1", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    if roots:
        env["PYTHONPATH"] = os.pathsep.join(str(p) for p in roots)
    return env


def run(*args: str | Path, env=None, capture=False) -> str:
    result = subprocess.run(
        [str(a) for a in args], env=env or clean_env(), check=True, text=True, capture_output=capture
    )
    return result.stdout.strip() if capture else ""


def verify_source(root: Path, commit: str) -> None:
    if Path(run("git", "-C", root, "rev-parse", "--show-toplevel", capture=True)).resolve() != root:
        raise ValueError(f"Expected a dedicated Git checkout: {root}")
    if run("git", "-C", root, "rev-parse", "HEAD", capture=True) != commit:
        raise ValueError(f"Source must be pinned to {commit}: {root}")
    if run("git", "-C", root, "status", "--porcelain", "--untracked-files=all", capture=True):
        raise ValueError(f"Source has local changes; use a clean pinned checkout: {root}")


def verify_robot(root: Path) -> None:
    urdf = root / ROBOT / "H10W_DVT2_TekkenD_with_Camera_2.urdf"
    # Viser uses visual meshes; cuMotion uses the XRDF collision spheres. The
    # upstream URDF also names collision STL files that are not shipped or loaded.
    for mesh in ET.parse(urdf).findall(".//visual/geometry/mesh"):
        path = urdf.parent / mesh.attrib["filename"]
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing robot mesh: {path}")
        with path.open("rb") as stream:
            if stream.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"Fetch the actual Git LFS object: {path}")


def probe(python: Path, code: str, *roots: Path, runtime: Path | None = None) -> None:
    if not python.is_absolute() or not os.access(python, os.X_OK):
        raise ValueError(f"Interpreter is unavailable: {python}")
    env = clean_env(*roots)
    if runtime:
        env["DATA_PLATFORM_UMI_RUNTIME"] = str(runtime)
    run(python, "-c", code, env=env)


def check(runtime: Path, mode: str) -> None:
    config = json.loads(runtime.read_text())
    if config.get("laundry_commit") != LAUNDRY_COMMIT:
        raise ValueError("Configuration does not pin the supported data-laundry commit")
    laundry = Path(config["laundry_root"])
    if not laundry.is_absolute():
        raise ValueError("laundry_root must be absolute")
    verify_source(laundry.resolve(), LAUNDRY_COMMIT)
    verify_robot(laundry)
    imports = "import laundry.core.dataset, laundry.pipelines._cosmos_id.reporting, matplotlib"
    if mode == "viewer":
        probe(Path(config["python"]), imports + "; import laundry.umi_replay.viser", laundry / "src")
    else:
        operation = "umi." + ("replay" if mode == "replay" else "cosmos_id")
        probe(
            Path(config["host_python"]),
            "from lerobot.data_platform.precompute.umi.runtime import load_runtime; "
            f"load_runtime({operation!r}); import lerobot.data_platform.precompute.umi.worker; "
            "import laundry.core.dataset, laundry.pipelines._cosmos_id.bridge",
            PLATFORM,
            laundry / "src",
            runtime=runtime,
        )
        converter = Path(config["converter_root"]).resolve()
        verify_source(converter, CONVERTER_COMMIT)
        run(
            config["converter_python"],
            converter / "data_tool/convert_dual_hand_head_to_umi_v21.py",
            "--help",
            env=clean_env(converter / "src", Path(config["converter_lerobot_root"])),
            capture=True,
        )
        ET.parse(config["urdf_path"])
        if mode == "replay":
            probe(
                Path(config["cumotion_python"]),
                "import sys; assert sys.version_info[:2] == (3, 12); "
                "import _cumotion, cumotion, laundry.umi_replay.screen",
                laundry / "src",
            )
        else:
            probe(Path(config.get("report_python") or config["cumotion_python"]), imports, laundry / "src")
            probe(
                Path(config["model_python"]),
                "import json, os; from pathlib import Path; "
                "from laundry.pipelines._cosmos_id.prediction import resolve_prediction_artifacts; "
                "c=json.loads(Path(os.environ['DATA_PLATFORM_UMI_RUNTIME']).read_text()); "
                "paths=resolve_prediction_artifacts(Path(c['framework_root']), Path(c['checkpoint_path'])); "
                "assert all(p.is_file() for p in paths); "
                "from cosmos_framework.inference.umi_3view_29d_predictor import UMI3View29DPredictor",
                laundry / "src",
                runtime=runtime,
            )
    print(f"PASS: {mode} source, files and imports. No GPU inference or live service acceptance performed.")


def install(root: Path, repository: str, python: str) -> None:
    if not root.is_absolute() or root.resolve().is_relative_to(PLATFORM):
        raise ValueError("--root must be an absolute path outside the platform checkout")
    url = urlsplit(repository) if "://" in repository else None
    if repository.startswith("-") or (
        url and (url.password or url.query or (url.scheme in {"http", "https"} and url.username))
    ):
        raise ValueError("Use an SSH URL, credential-free repository URL, or local Git mirror")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    laundry = root / "laundry"
    run("git", "clone", "--no-checkout", "--", repository, laundry)
    run("git", "-C", laundry, "checkout", "--detach", LAUNDRY_COMMIT)
    verify_source(laundry, LAUNDRY_COMMIT)
    run("uv", "sync", "--project", laundry, "--python", python, "--frozen", "--no-dev", "--extra", "viser")
    interpreter = laundry / ".venv/bin/python"
    viewer = root / "viewer-runtime.json"
    viewer.write_text(
        json.dumps(
            {"laundry_commit": LAUNDRY_COMMIT, "python": str(interpreter), "laundry_root": str(laundry)},
            indent=2,
        )
        + "\n"
    )
    check(viewer, "viewer")
    run(laundry / ".venv/bin/laundry", "list")
    template = json.loads((PLATFORM / "docs/umi_runtime.example.json").read_text())
    for key in template:
        if key.endswith("_path") or key in (
            "converter_root",
            "converter_lerobot_root",
            "converter_python",
            "model_python",
            "framework_root",
        ):
            template[key] = "/REQUIRED/" + key
    template.update(
        host_python="/REQUIRED/platform_python",
        report_python=str(interpreter),
        laundry_root=str(laundry),
        cumotion_python=str(laundry / ".venv_cumotion/bin/python"),
        lock_dir="/REQUIRED/shared_gpu_lock_dir",
        gpus=[],
    )
    (root / "agent-runtime.template.json").write_text(json.dumps(template, indent=2) + "\n")
    receipt = {
        "laundry_commit": LAUNDRY_COMMIT,
        "lock_sha256": hashlib.sha256((laundry / "uv.lock").read_bytes()).hexdigest(),
        "python": run(interpreter, "--version", capture=True),
        "cpu_imports": "passed",
        "gpu_inference": "not_run",
        "services_changed": False,
    }
    (root / "installation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"CPU setup complete: {root}. Fill agent-runtime.template.json before enabling GPU operators.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("install", help="Fetch pinned source and install its locked CPU dependencies")
    setup.add_argument("--root", type=Path, required=True, help="New absolute directory outside the checkout")
    setup.add_argument(
        "--laundry-repository", required=True, help="Authorized upstream or Git mirror supplied by its owner"
    )
    setup.add_argument("--python", default="3.11", help="CPU Python interpreter/version understood by uv")
    verify = commands.add_parser(
        "check", help="Check source/files/imports without installing or running inference"
    )
    verify.add_argument("--runtime", type=Path, required=True)
    verify.add_argument("--mode", choices=("viewer", "replay", "cosmos"), required=True)
    args = parser.parse_args()
    try:
        if args.command == "install":
            install(args.root, args.laundry_repository, args.python)
        else:
            check(args.runtime.resolve(), args.mode)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError, ET.ParseError) as error:
        print(f"UMI setup failed: {error}", file=sys.stderr)
        print(
            "Existing directories are preserved. Fix configuration or choose a new installation root.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
