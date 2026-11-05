"""Node-owned runtime configuration and cancellable isolated UMI evaluation."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from lerobot.data_platform.operators import OperatorResult

LAUNDRY_COMMIT = "fd752156895c83ec75fde84796faea2513c626c7"


def load_runtime(operation):
    filename = os.environ.get("DATA_PLATFORM_UMI_RUNTIME")
    if not filename or not Path(filename).is_absolute():
        raise ValueError("Configure the node-owned DATA_PLATFORM_UMI_RUNTIME JSON")
    config = json.loads(Path(filename).read_text())
    if config.get("schema_version") != 1 or config.get("laundry_commit") != LAUNDRY_COMMIT:
        raise ValueError("UMI runtime must pin the supported laundry source version")
    required = [
        "host_python",
        "laundry_root",
        "lock_dir",
        "converter_python",
        "converter_root",
        "converter_lerobot_root",
        "urdf_path",
    ]
    if operation == "umi.replay":
        required += ["cumotion_python"]
    elif operation == "umi.cosmos_id":
        required += ["model_python", "framework_root", "checkpoint_path", "processor_path", "vae_path"]
    for key in required:
        path = Path(config.get(key, ""))
        if not path.is_absolute() or not path.exists():
            raise ValueError(f"UMI runtime component is unavailable: {key}")
        if key.endswith("python") and (not path.is_file() or not os.access(path, os.X_OK)):
            raise ValueError(f"UMI interpreter is not executable: {key}")
    gpus = config.get("gpus")
    if (
        not isinstance(gpus, list)
        or not gpus
        or len(gpus) > 8
        or any(type(g) is not int or g < 0 for g in gpus)
        or len(set(gpus)) != len(gpus)
    ):
        raise ValueError("UMI runtime requires an explicit GPU pool")
    timeout = config.get("timeout_seconds", 3600)
    if type(timeout) is not int or not 60 <= timeout <= 86400:
        raise ValueError("UMI runtime timeout must be 60–86400 seconds")
    config["timeout_seconds"] = timeout
    if not (Path(config["laundry_root"]) / "src/laundry/pipelines/_cosmos_id/worker.py").is_file():
        raise ValueError("Pinned laundry source is incomplete")
    return config


def runtime_unavailable(operation):
    try:
        load_runtime(operation)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return str(error)
    return None


def _run(context, parameters, operation, *, native_viewers=False):
    from lerobot.data_platform.operators.umi import validate_umi_info
    from lerobot.data_platform.precompute.umi.diagnostics import file_hash
    from lerobot.data_platform.temporal_caption_flora import FloraRuntime, acquire_gpu

    config = load_runtime(operation)
    if native_viewers and operation == "umi.cosmos_id":
        renderer = Path(config.get("report_python") or config.get("cumotion_python", ""))
        if not renderer.is_absolute() or not renderer.is_file() or not os.access(renderer, os.X_OK):
            raise ValueError("Configure a CPU report interpreter with laundry and matplotlib")
    work = Path(context.work_dir).resolve()
    roots = context.source_roots or (context.source_root,)
    for root in roots:
        validate_umi_info(json.loads((root / "meta/info.json").read_text()), operation)
    if any(
        work == p.resolve() or work.is_relative_to(p.resolve()) or p.resolve().is_relative_to(work)
        for p in roots
    ):
        raise ValueError("UMI runtime output must not overlap sources")
    if operation == "umi.cosmos_id":
        evidence = [p for name, p in context.artifacts.items() if name.endswith("umi_diagnostics.json")]
        if len(evidence) != 1:
            raise ValueError("Cosmos candidates require one matching UMI diagnosis artifact")
        report = json.loads(evidence[0].read_text())
        if (
            report.get("schema_version") != 1
            or Path(report["source"]).resolve() != context.source_root.resolve()
        ):
            raise ValueError("UMI diagnosis belongs to another dataset")
        from .diagnostics import data_files

        source = context.source_root.resolve()
        info = json.loads((source / "meta/info.json").read_text())
        expected = {
            str(p.relative_to(source))
            for p in [*data_files(source, info), *(p for p in (source / "meta").rglob("*") if p.is_file())]
        }
        if set(report["source_fingerprint"]) != expected:
            raise ValueError("UMI diagnosis does not bind every source file")
        for name, digest in report["source_fingerprint"].items():
            path = (context.source_root / name).resolve()
            if not path.is_relative_to(context.source_root.resolve()) or file_hash(path) != digest:
                raise ValueError("UMI diagnosis input changed")
    work.mkdir(parents=True, exist_ok=True)
    if native_viewers:
        from .native_cache import lookup

        cached = lookup(roots, parameters, operation, config)
        if cached:
            directory, provenance = cached
            receipt = json.loads((directory / "result.json").read_text())
            receipt["result_reuse"] = provenance
            context.emit(
                {"message": "Reusing fingerprint-verified UMI evidence; no model or replay computation"}
            )
            return stage_result(directory, work, receipt)
    # execute_registered publishes every file in bundle/. Keep evaluation datasets,
    # model requests and scratch files beside that bundle, then stage only evidence.
    directory = work.parent.parent / "umi-runtime-private"
    if any(
        directory == p.resolve()
        or directory.is_relative_to(p.resolve())
        or p.resolve().is_relative_to(directory)
        for p in roots
    ):
        raise ValueError("UMI scratch space must not overlap sources")
    directory.mkdir(exist_ok=False)
    request = {
        "schema_version": 1,
        "operation": operation,
        "source_roots": [str(p.resolve()) for p in roots],
        "output": str(directory),
        "parameters": parameters,
        "runtime": config,
        "native_viewers": native_viewers,
    }
    path = directory / "request.json"
    path.write_text(json.dumps(request, indent=2) + "\n")
    reserved = FloraRuntime(
        Path(config["host_python"]),
        Path(config["host_python"]),
        directory,
        directory,
        Path(config["lock_dir"]),
        gpus=tuple(config["gpus"]),
        stage_timeout_s=config["timeout_seconds"],
    )
    with acquire_gpu(
        reserved, lambda message: context.emit({"message": message.replace("FLORA", "UMI")})
    ) as gpu:
        env = os.environ.copy()
        package_root = str(Path(__file__).parents[4])
        env.update(
            CUDA_VISIBLE_DEVICES=str(gpu.gpu),
            PYTHONPATH=os.pathsep.join([package_root, str(Path(config["laundry_root"]) / "src")]),
            LAUNDRY_CUMOTION_NO_INSTALL="1",
        )
        started = time.monotonic()
        with (directory / "worker.log").open("w") as log:
            process = subprocess.Popen(
                [config["host_python"], "-u", "-m", "lerobot.data_platform.precompute.umi.worker", str(path)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                while process.poll() is None:
                    if context.check_cancelled:
                        context.check_cancelled()
                    if time.monotonic() - started > config["timeout_seconds"]:
                        raise TimeoutError("UMI runtime exceeded its configured timeout")
                    context.emit({"message": f"{operation}: isolated GPU worker running"})
                    time.sleep(2)
                if process.returncode:
                    raise RuntimeError(f"UMI evaluation worker failed; inspect {directory / 'worker.log'}")
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise
    receipt = json.loads((directory / "result.json").read_text())
    result = stage_result(directory, work, receipt)
    shutil.rmtree(directory)
    return result


def stage_result(directory, work, receipt):

    def public_artifact(path):
        relative = path.relative_to(directory)
        if not path.is_file() or path.is_symlink():
            return False
        if len(relative.parts) == 1:
            return path.name in {"result.json", "report.html", "worker.log", "source_fingerprints.json"}
        if relative.parts[0].startswith("replay_"):
            return path.name == "results.json"
        if relative.parts[0] == "native":
            return path.suffix in {".html", ".json", ".jsonl", ".npz", ".parquet", ".png"}
        return relative.parts[0] == "cosmos" and (
            path.name in {"response.json", "worker.log"}
            or path.name.startswith("episode_")
            and path.suffix == ".npz"
        )

    artifacts = []
    public = work / "runtime"
    for path in sorted(directory.rglob("*")):
        if public_artifact(path):
            destination = public / path.relative_to(directory)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            artifacts.append(
                {
                    "path": str(destination.relative_to(work)),
                    "kind": "report" if path.suffix == ".html" else "umi_evidence",
                    **(
                        {"kind": "umi_replay_viewer", "label": "Original UMI Replay"}
                        if path.name == "viewer.json" and path.parent.name.startswith("replay_")
                        else {}
                    ),
                    **(
                        {"label": "Original Cosmos ID viewer"} if path.name == "cosmos_id_viewer.html" else {}
                    ),
                }
            )
    return OperatorResult(summary=receipt, artifacts=tuple(artifacts))


def run_replay(context, parameters):
    return _run(context, parameters, "umi.replay")


def run_cosmos(context, parameters):
    return _run(context, parameters, "umi.cosmos_id")


def run_replay_native(context, parameters):
    return _run(context, parameters, "umi.replay", native_viewers=True)


def run_cosmos_native(context, parameters):
    return _run(context, parameters, "umi.cosmos_id", native_viewers=True)
