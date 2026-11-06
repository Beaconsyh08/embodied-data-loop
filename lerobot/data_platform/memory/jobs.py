"""Agent adapters: local raw-media bundles, small review publication and immutable export."""

from __future__ import annotations

import json
import os
import shutil

from lerobot.data_platform.ai_tools.qwen import QwenClient, dashscope_env_api_key
from lerobot.data_platform.artifacts.mem_export import digest as file_digest
from lerobot.data_platform.artifacts.mem_export import export
from lerobot.data_platform.memory.contracts import AnnotationConfig, digest
from lerobot.data_platform.memory.generation import generate, mosaic
from lerobot.data_platform.memory.sources import preview
from lerobot.data_platform.operators import OperatorResult
from lerobot.data_platform.runtime.supervisor import atomic_json


def capabilities():
    from lerobot.data_platform.memory.batch import parallel_episodes
    from lerobot.data_platform.runtime.resources import ExecutionResources

    return {
        "memory_annotation_protocol": 1,
        "memory_batch_protocol": 1,
        "memory_batch_max_parallel": parallel_episodes(ExecutionResources.resolve()),
        "memory_qwen_configured": bool(dashscope_env_api_key()),
        "memory_model": os.getenv("DATA_PLATFORM_MEMORY_MODEL", "qwen3-vl-plus"),
    }


def _selection(source, pin, artifact):
    return {
        "root": str(source),
        "dataset_key": pin["dataset_key"],
        "episode_index": pin["episode_index"],
        "flora_run": pin["flora_run"],
        "annotation": str(artifact),
        "cameras": pin["cameras"],
        "task": pin["task"],
        **pin["identity"],
        "source_files_sha256": pin["source_files_sha256"],
        "annotation_sha256": pin["annotation_sha256"],
    }


def execute(context, parameters):
    stop = context.check_cancelled or (lambda: None)
    input_path = context.artifacts.get("input.json")
    if input_path is None:
        raise ValueError("Use the authenticated Text memory workflow to prepare pinned inputs")
    request = json.loads(input_path.read_text())
    if request["request_id"] != parameters["request_id"] or request["dataset_key"] != context.dataset_key:
        raise ValueError("Memory request identity mismatch")
    if context.output_root is None:
        raise ValueError("A registered Agent data-disk output is required")
    context.output_root.mkdir(parents=True, exist_ok=True)
    if request["operation"] == "memory.export":
        return export_request(context, request, stop)
    if "episodes" in request:
        from lerobot.data_platform.memory.batch import execute_batch

        return execute_batch(context, request, annotate_episode)
    return annotate_episode(context, request)


def annotate_episode(context, request):
    stop = context.check_cancelled or (lambda: None)
    pin = request["pin"]
    if pin["config"]["model"] != capabilities()["memory_model"]:
        raise ValueError("Configured memory model changed; prepare a new request")
    artifact = context.artifacts["flora.json"]
    selection = _selection(context.source_root, pin, artifact)
    verified = preview(
        context.source_root, selection, AnnotationConfig(**pin["config"]), context.output_root / "hashes"
    )
    # Extra human checkpoints are allowed; all source/config/identity fields remain pinned.
    for field in ("checkpoints", "expected_requests"):
        verified[field] = pin[field]
    if verified != pin:
        raise ValueError("Source, timestamps, model or FLORA binding changed")
    workspace = context.output_root / digest(
        {k: v for k, v in pin.items() if k not in {"checkpoints", "expected_requests"}}
    )
    workspace.mkdir(parents=True, exist_ok=True)
    bundle = workspace / "bundle"
    if not bundle.exists():
        manifest = workspace / "selection.json"
        atomic_json(manifest, {"schema": "mem_export_selection/v1", "episodes": [selection]})
        export(manifest, bundle, causal=True, check_cancelled=stop)
    # Revalidate bundle contents before reuse; never trust a partially written cache.
    manifest = json.loads((bundle / "manifest.json").read_text())
    if any(file_digest(bundle / name) != expected for name, expected in manifest["files"].items()):
        raise ValueError("Cached memory bundle changed")
    output = workspace / "generations" / request["request_id"]
    previous = request.get("previous")
    if (output / "review.json").exists():
        previous = json.loads((output / "review.json").read_text())
    state = generate(
        bundle,
        pin,
        output,
        QwenClient(max_response_bytes=1024 * 1024),
        previous=previous,
        stop=stop,
        progress=context.emit,
    )
    state["generation_chain"] = list(
        dict.fromkeys([*(request.get("previous") or {}).get("generation_chain", []), request["request_id"]])
    )
    atomic_json(output / "review.json", state)
    # Source changes invalidate even a successfully completed model run.
    latest = preview(
        context.source_root, selection, AnnotationConfig(**pin["config"]), context.output_root / "hashes"
    )
    if any(latest[k] != pin[k] for k in latest if k not in {"checkpoints", "expected_requests"}):
        raise ValueError("Source changed during annotation")
    static = context.work_dir
    atomic_json(static / "review.json", state)
    shutil.copyfile(artifact, static / "flora.json")
    if (output / "requests").exists():
        shutil.copytree(output / "requests", static / "requests", dirs_exist_ok=True)
    episode = json.loads((bundle / "episodes.jsonl").read_text().splitlines()[0])
    frames = [json.loads(row) for row in (bundle / episode["frames_file"]).read_text().splitlines()]
    # Review media is only the sampled, causal three-view mosaic, never original raw videos.
    media = static / "frames"
    media.mkdir(exist_ok=True)
    for frame in sorted({f for e in state["entries"] for f in [e["frame"], *e["history_frames"]]}):
        stop()
        (media / f"{frame}.png").write_bytes(mosaic(bundle, frames[frame]["images"]))
    return OperatorResult(
        summary={
            "status": "draft",
            "checkpoints": len(state["entries"]),
            "needs_review": any(e["state"] != "complete" for e in state["entries"]),
            "model": pin["config"]["model"],
        },
        artifacts=tuple(
            {"path": p.relative_to(static).as_posix()} for p in sorted(static.rglob("*")) if p.is_file()
        ),
    )


def export_request(context, request, stop):
    from lerobot.data_platform.memory.review import review_payload

    selections = []
    states = request["states"]
    for index, state in enumerate(states):
        record = state["record"]
        review = record["review"]
        if review.get("status") != "approved" or review.get("payload_sha256") != digest(
            review_payload(record)
        ):
            raise ValueError("Training export requires current human approval")
        if any(e["state"] != "complete" for e in state["entries"]):
            raise ValueError("Stale or pending checkpoints cannot be exported")
        pin = state["pin"]
        artifact = context.artifacts[f"flora-{index}.json"]
        selection = _selection(context.source_root, pin, artifact)
        checked = preview(
            context.source_root, selection, AnnotationConfig(**pin["config"]), context.output_root / "hashes"
        )
        if checked["source_fingerprint"] != pin["source_fingerprint"]:
            raise ValueError("Export source changed")
        selections.append(selection)
    destination = context.output_root / "exports" / request["request_id"]
    if destination.exists():
        package = json.loads((destination / "package.json").read_text())
        if package["request_sha256"] != digest(request) or any(
            file_digest(destination / name) != expected for name, expected in package["files"].items()
        ):
            raise ValueError("Existing export does not match this immutable request")
        receipt = {
            "schema": "memory_export_receipt/v1",
            "output": str(destination),
            "episodes": len(states),
            "status": "awaiting_high_level_validation",
            "bundle_sha256": file_digest(destination / "bundle/manifest.json"),
        }
        atomic_json(context.work_dir / "export.json", receipt)
        return OperatorResult(summary=receipt, artifacts=({"path": "export.json"},))
    staging = destination.with_name("." + destination.name)
    staging.mkdir(parents=True)
    try:
        manifest = staging / "selection.json"
        atomic_json(manifest, {"schema": "mem_export_selection/v1", "episodes": selections})
        export(manifest, staging / "bundle", causal=True, check_cancelled=stop)
        fingerprint = file_digest(staging / "bundle/manifest.json")
        records = []
        assignments = {}
        for index, state in enumerate(states):
            import copy

            record = copy.deepcopy(state["record"])
            eid = f"episode-{index:06d}"
            record["episode_id"] = eid
            # Review identity changes only because the frozen export assigns package-local IDs.
            # Preserve source attestations in the signed-by-login revision attachments.
            record["review"]["payload_sha256"] = digest(review_payload(record))
            records.append(record)
            assignments[eid] = request["assignments"][str(state["pin"]["episode_index"])]
        atomic_json(
            staging / "ledger.json",
            {"schema": "mem_ledger/v1", "bundle_sha256": fingerprint, "episodes": records},
        )
        atomic_json(
            staging / "splits.json",
            {"bundle_sha256": fingerprint, "seed": request["seed"], "assignments": assignments},
        )
        atomic_json(
            staging / "audit.json",
            {
                "schema": "memory_export_audit/v1",
                "reviews": states,
                "revisions": request["revisions"],
                "training_validation": "pending mem_tools inspect/build/training-export",
            },
        )
        for index, state in enumerate(states):
            workspace = context.output_root / digest(
                {k: v for k, v in state["pin"].items() if k not in {"checkpoints", "expected_requests"}}
            )
            for generation in state.get("generation_chain", []):
                import uuid

                if str(uuid.UUID(generation)) != generation:
                    raise ValueError("Invalid generation identity")
                trace = workspace / "generations" / generation / "requests"
                if not trace.is_dir():
                    raise ValueError("Original request receipts are missing on the Agent")
                shutil.copytree(trace, staging / "audit" / f"episode-{index:06d}" / generation)
        manifest.unlink()
        atomic_json(
            staging / "package.json",
            {
                "schema": "memory_export_package/v1",
                "request_sha256": digest(request),
                "files": {
                    p.relative_to(staging).as_posix(): file_digest(p)
                    for p in sorted(staging.rglob("*"))
                    if p.is_file()
                },
            },
        )
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    receipt = {
        "schema": "memory_export_receipt/v1",
        "output": str(destination),
        "episodes": len(records),
        "status": "awaiting_high_level_validation",
        "bundle_sha256": fingerprint,
    }
    atomic_json(context.work_dir / "export.json", receipt)
    return OperatorResult(summary=receipt, artifacts=({"path": "export.json"},))


def complete(job, destination, ledger, result):
    """Bind publication to prepared inputs and current review revisions, after lease authentication."""
    from lerobot.data_platform.memory.review import MemoryReviewStore
    from lerobot.data_platform.runtime.jobs import JobConflictError

    prepared = ledger.repository.get("memory_requests", job["options"]["parameters"]["request_id"])
    if not prepared or prepared["location_id"] != job["location_id"]:
        raise ValueError("Memory publication has no pinned request")
    value = prepared["value"]
    reviews = MemoryReviewStore(ledger.repository)
    for reference in value.get("revisions", []):
        if reviews.get(reference["run"])["revision"] != reference["revision"]:
            raise JobConflictError("Review changed during execution; result must be prepared again")
    if job["operation"] == "memory.annotate":
        inputs = [(job["job_id"], value["pin"], destination / "static")] if "episodes" not in value else []
        if "episodes" in value:
            from lerobot.data_platform.memory.batch import episode_run

            batch = json.loads((destination / "static/batch.json").read_text())
            expected = {
                episode_run(value["request_id"], item["request_id"]): item["pin"]
                for item in value["episodes"]
            }
            entries = batch["episodes"]
            if (
                batch != result.get("summary")
                or len(entries) != len(expected)
                or {e["run"] for e in entries} != set(expected)
            ):
                raise ValueError("Published batch differs from the prepared episode selection")
            for entry in entries:
                run, pin = entry["run"], expected[entry["run"]]
                folder = destination / "static/episodes" / run
                if entry["episode_index"] != pin["episode_index"]:
                    raise ValueError("Published batch episode identity changed")
                if entry.get("has_review"):
                    inputs.append((run, pin, folder))
                elif entry["status"] != "error" or not (folder / "error.json").is_file():
                    raise ValueError("Batch episode has no review or failure receipt")
        states = []
        for run, pin, folder in inputs:
            state = json.loads((folder / "review.json").read_text())
            if state.get("pin") != pin or state.get("pin_digest") != digest(pin):
                raise ValueError("Published memory inputs differ from the pinned request")
            if state["record"]["review"] != {"status": "draft"}:
                raise ValueError("Model output cannot contain human approval")
            state["parents"] = value.get("revisions", [])
            states.append((run, state))
        for run, state in states:
            reviews.create(run, job["location_id"], state)
        result["review_url"] = f"/remote/{job['location_id']}/memory" + (
            f"?run={states[0][0]}" if states else ""
        )
    return result
