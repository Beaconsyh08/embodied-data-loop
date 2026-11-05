"""One dataset Job with bounded episode concurrency and isolated causal chains."""

from __future__ import annotations

import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace

from lerobot.data_platform.execution import StopRequestedError, atomic_json
from lerobot.data_platform.operators import OperatorResult


def parallel_episodes(resources=None):
    # Image-backed LeRobot inspection materializes one complete episode per worker.
    configured = int(os.getenv("DATA_PLATFORM_MEMORY_MAX_PARALLEL_EPISODES", "3"))
    if not 1 <= configured <= 32:
        raise ValueError("DATA_PLATFORM_MEMORY_MAX_PARALLEL_EPISODES must be between 1 and 32")
    return min(configured, max(1, resources.memory_limit // (8 * 1024**3))) if resources else configured


def episode_run(batch_request: str, episode_request: str) -> str:
    return str(uuid.uuid5(uuid.UUID(batch_request), str(uuid.UUID(episode_request))))


def execute_batch(context, request, annotate):
    items = request["episodes"]
    if not isinstance(items, list) or not 1 <= len(items) <= 10000:
        raise ValueError("Select a bounded dataset episode batch")
    indices = [item["pin"]["episode_index"] for item in items]
    if len(set(indices)) != len(indices):
        raise ValueError("A batch may contain each episode only once")
    runs = [episode_run(request["request_id"], item["request_id"]) for item in items]
    if len(set(runs)) != len(runs):
        raise ValueError("Duplicate episode request")
    stop = context.check_cancelled or (lambda: None)
    cancelled = threading.Event()
    lock = threading.Lock()
    states = [
        {"run": run, "episode_index": index, "status": "queued"}
        for run, index in zip(runs, indices, strict=True)
    ]

    def check():
        if cancelled.is_set():
            raise StopRequestedError("Dataset annotation cancelled")
        stop()

    def update(index, **changes):
        with lock:
            states[index].update(changes)
            finished = sum(s["status"] in {"draft", "error"} for s in states)
            atomic_json(context.work_dir / "batch.json", {"episodes": states})
            context.emit(
                {
                    "phase": "memory_annotation",
                    "current": finished,
                    "total": len(items),
                    "message": f"Episodes {finished}/{len(items)}; running {[s['episode_index'] for s in states if s['status'] == 'running']}; episode {indices[index]}: {states[index]['status']}",
                    "episodes": [dict(s) for s in states],
                }
            )

    def worker(index):
        check()
        item, run = items[index], runs[index]
        static = context.work_dir / "episodes" / run
        static.mkdir(parents=True, exist_ok=True)
        update(index, status="running")
        child = replace(
            context,
            work_dir=static,
            artifacts={"flora.json": context.artifacts[f"flora-{index}.json"]},
            check_cancelled=check,
            progress=lambda value: update(index, progress=value),
        )
        value = {
            **item,
            "request_id": run,
            "dataset_key": request["dataset_key"],
            "operation": "memory.annotate",
        }
        try:
            result = annotate(child, value)
        except StopRequestedError:
            cancelled.set()
            raise
        except Exception as exc:
            # Keep failures local to an episode; never invent a label or retry an invalid draft.
            error = {"status": "error", "error_type": type(exc).__name__}
            atomic_json(static / "error.json", error)
            update(index, **error)
        else:
            update(index, **result.summary, has_review=True)

    # Episode preparation includes decoding, so keep memory usage bounded as well as API requests.
    workers = min(parallel_episodes(context.resources), len(items))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, i) for i in range(len(items))]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            cancelled.set()
            for future in futures:
                future.cancel()
            raise
    check()
    summary = {
        "status": "draft",
        "episodes": states,
        "max_parallel_episodes": workers,
        "needs_review": any(s.get("needs_review") or s["status"] == "error" for s in states),
    }
    atomic_json(context.work_dir / "batch.json", summary)
    return OperatorResult(
        summary=summary,
        artifacts=tuple(
            {"path": path.relative_to(context.work_dir).as_posix()}
            for path in sorted(context.work_dir.rglob("*"))
            if path.is_file()
        ),
    )
