"""Durable gateway worker. A selected backend is never changed automatically."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from services.flora_api.errors import public_error
from services.flora_api.store import Store
from services.flora_router.app import KeyCatalog
from services.flora_router.capacity import Capacity
from services.flora_router.probe import configured_capacity
from services.flora_router.routes import Routes
from services.flora_router.transport import Backend, Multipart


class RetryLaterError(Exception):
    pass


class BackendRequestError(Exception):
    pass


def check_response(response):
    if response.status_code in (408, 429, 503):
        raise RetryLaterError("backend temporarily unavailable")
    if 400 <= response.status_code < 500:
        raise BackendRequestError(f"Backend rejected the request (HTTP {response.status_code})")
    response.raise_for_status()


def remote_capacity(command: list[str]) -> Capacity:
    result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=True)
    data = json.loads(result.stdout)
    if not data.get("ready"):
        raise RetryLaterError("capacity unavailable")
    return Capacity(
        tuple(data["free_gpus"]),
        tuple(data["active_job_ids"]),
        tuple(data["pending_gpu_job_ids"]),
        int(data["concurrency"]),
    )


def prepare_input(store: Store, job: dict, keys: KeyCatalog) -> dict:
    from lerobot.data_platform.operators.implementations.caption.flora import camera_roles

    directory = store.root / "jobs" / job["id"]
    manifest_path = directory / "forward.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    if job["kind"] == "video":
        settings = json.loads((directory / "input.json").read_text())
        manifest = {
            "instruction": settings["instruction"],
            "videos": {role: str(directory / f"{role}.mp4") for role in settings["roles"]},
        }
    else:
        from lerobot.data_platform.operators.implementations.caption.core import prepare
        from lerobot.data_platform.operators.implementations.caption.snapshot import (
            caption_source_fingerprint,
        )
        from services.flora_api.datasets import resolve_dataset

        settings = json.loads((store.root / "jobs" / job["parent_id"] / "input.json").read_text())
        source = resolve_dataset(
            store.root,
            settings["dataset_id"],
            job["owner"],
            json.loads(keys.read_text()),
            json.loads(Path(os.environ["FLORA_API_DATASETS"]).read_text()),
        )
        fingerprint = caption_source_fingerprint(source, store.root / "source-hashes", lambda _: None)
        bundle = directory / "bundle"
        if bundle.exists():
            shutil.rmtree(bundle)
        directory.mkdir(parents=True, exist_ok=True)
        prepare(source, job["episode_index"], bundle, export_camera_videos=True)
        meta = json.loads((bundle / "episode.json").read_text())
        roles = camera_roles(meta["cameras"])
        instruction = "; ".join(meta.get("task_hint", [])).strip()
        if not instruction:
            raise ValueError("Dataset episode has no task instruction")
        manifest = {
            "instruction": instruction,
            "videos": {
                role: str(bundle / "camera-videos" / f"{meta['cameras'].index(camera)}.mp4")
                for role, camera in roles.items()
            },
            "meta": meta,
            "source": str(source),
            "fingerprint": fingerprint,
        }
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest))
    temporary.replace(manifest_path)
    return manifest


def finish(store: Store, job: dict, manifest: dict, value: dict) -> None:
    from lerobot.data_platform.operators.implementations.caption.flora import validate_flora

    if "meta" in manifest:
        from lerobot.data_platform.operators.implementations.caption.snapshot import (
            caption_source_fingerprint,
        )

        meta = manifest["meta"]
        if value["frame_count"] != meta["frame_count"] or abs(value["fps"] - meta["fps"]) > 0.001:
            raise ValueError("Backend result timing differs from source episode")
        if (
            caption_source_fingerprint(Path(manifest["source"]), store.root / "source-hashes", lambda _: None)
            != manifest["fingerprint"]
        ):
            raise ValueError("Source dataset changed during inference")
        value.update(episode_index=meta["episode_index"], source_digest=meta.get("source_digest"))
    value["dataset_key"] = f"api/{job['id']}"
    validate_flora(value)
    output = store.root / "jobs" / job["id"] / "output"
    output.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value).encode()
    temporary = output / "caption.tmp"
    temporary.write_bytes(encoded)
    temporary.replace(output / "caption.json")
    receipt = output / "completed.tmp"
    receipt.write_text(json.dumps({"sha256": hashlib.sha256(encoded).hexdigest()}))
    receipt.replace(output / "completed.json")
    store.update(job["id"], status="completed", stage="completed")


def recover_completed(store: Store, job: dict) -> bool:
    from lerobot.data_platform.operators.implementations.caption.flora import validate_flora

    output = store.root / "jobs" / job["id"] / "output"
    try:
        data = (output / "caption.json").read_bytes()
        receipt = json.loads((output / "completed.json").read_text())
        if receipt["sha256"] != hashlib.sha256(data).hexdigest():
            return False
        validate_flora(json.loads(data))
    except (OSError, ValueError, KeyError):
        return False
    store.update(job["id"], status="completed", stage="completed")
    return True


def forward(store: Store, routes: Routes, job: dict, manifest: dict, assignment, backend: Backend) -> bool:
    with backend.session() as session:
        if assignment.remote_id is None:
            with Multipart(
                manifest["instruction"], {k: Path(v) for k, v in manifest["videos"].items()}
            ) as body:
                response = session.post(
                    backend.url + "/v1/flora/jobs",
                    data=body,
                    headers={
                        "Content-Type": f"multipart/form-data; boundary={body.boundary}",
                        "Idempotency-Key": f"router:{job['id']}",
                    },
                    timeout=(10, 120),
                )
            check_response(response)
            remote_id = response.json()["id"]
            routes.set_remote(job["id"], remote_id)
        else:
            remote_id = assignment.remote_id
        response = session.get(backend.url + f"/v1/flora/jobs/{remote_id}", timeout=(10, 30))
        if response.status_code == 404:
            raise RetryLaterError("assigned backend job unavailable; preserving identity")
        check_response(response)
        status = response.json()
        if status["status"] == "failed":
            store.update(
                job["id"],
                status="failed",
                stage=f"{backend.name} · {status['stage']}",
                error=status.get("error") or "Backend job failed",
            )
            return True
        if status["status"] != "completed":
            store.update(job["id"], status="running", stage=f"{backend.name} · {status['stage']}")
            return False
        response = session.get(backend.url + f"/v1/flora/jobs/{remote_id}/result", timeout=(10, 60))
        check_response(response)
        finish(store, job, manifest, response.json())
        return True


def execute(
    store: Store, routes: Routes, job: dict, keys: KeyCatalog, backends: dict, probe_command: list[str]
):
    if recover_completed(store, job):
        return
    try:
        assignment = routes.get(job["id"])
        if assignment is None and not any(
            record["id"] == job["owner"] and record.get("enabled", True)
            for record in json.loads(keys.read_text())
        ):
            raise ValueError("API key revoked before execution")
        if assignment is not None and not (store.root / "jobs" / job["id"] / "forward.json").is_file():
            raise ValueError("Assigned job is missing its original forwarding manifest")
        manifest = prepare_input(store, job, keys)
    except Exception as error:
        logging.exception("Input preparation failed for %s", job["id"])
        store.update(job["id"], status="failed", stage="preparing", error=public_error(error, None))
        return
    while True:
        try:
            assignment = routes.get(job["id"])
            if assignment is None:
                observations = {}
                for name, probe in (
                    ("A100", configured_capacity),
                    ("H100", lambda: remote_capacity(probe_command)),
                ):
                    try:
                        observations[name] = probe()
                    except Exception:
                        logging.warning("Capacity unavailable for %s", name)
                assignment = routes.select(job["id"], observations)
                if assignment is None:
                    store.update(
                        job["id"], status="running", stage="FLORA waiting for a free GPU across backends"
                    )
                    time.sleep(10)
                    continue
            if forward(store, routes, job, manifest, assignment, backends[assignment.backend]):
                return
        except (requests.RequestException, RetryLaterError, OSError) as error:
            logging.warning("Backend communication interrupted for %s (%s)", job["id"], type(error).__name__)
            store.update(job["id"], status="running", stage="FLORA reconnecting to assigned backend")
        except Exception as error:
            logging.exception("Gateway job %s failed", job["id"])
            store.update(
                job["id"], status="failed", stage="gateway validation", error=public_error(error, None)
            )
            return
        time.sleep(10)


def main():
    config = json.loads(Path(os.environ["FLORA_ROUTER_CONFIG"]).read_text())
    store = Store(Path(os.environ["FLORA_API_STATE"]))
    keys = KeyCatalog(Path(os.environ["FLORA_API_KEYS"]), Path(os.environ["FLORA_ROUTER_KEYS"]))
    backends = {
        name: Backend(
            name=name,
            url=item["url"].rstrip("/"),
            hostname=item["hostname"],
            ca=Path(item["ca"]),
            key_file=Path(item["key_file"]),
        )
        for name, item in config["backends"].items()
    }
    routes = Routes(store.root)
    with (store.root / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def heartbeat():
            while True:
                (store.root / "worker-heartbeat").touch()
                time.sleep(10)

        threading.Thread(target=heartbeat, daemon=True).start()
        with ThreadPoolExecutor(max_workers=8) as pool:
            pending = {
                pool.submit(execute, store, routes, job, keys, backends, config["probe_command"])
                for job in store.interrupted()
            }
            while True:
                pending = {future for future in pending if not future.done()}
                if len(pending) < 8:
                    job = store.claim()
                    if job:
                        pending.add(
                            pool.submit(execute, store, routes, job, keys, backends, config["probe_command"])
                        )
                        continue
                time.sleep(1)


if __name__ == "__main__":
    main()
