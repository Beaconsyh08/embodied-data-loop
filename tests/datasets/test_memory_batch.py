"""Bounded intra-Job concurrency; no live Qwen requests."""

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from lerobot.data_platform.memory.batch import execute_batch
from lerobot.data_platform.operators import ExecutionContext, OperatorResult
from lerobot.data_platform.runtime.supervisor import StopRequestedError


def request(count=4):
    return {
        "request_id": str(uuid.uuid4()),
        "dataset_key": "fixture",
        "episodes": [{"request_id": str(uuid.uuid4()), "pin": {"episode_index": i}} for i in range(count)],
    }


def context(tmp_path, count=4, stop=None):
    static = tmp_path / "static"
    static.mkdir()
    return ExecutionContext(
        source_root=tmp_path,
        work_dir=static,
        output_root=tmp_path / "output",
        check_cancelled=stop,
        artifacts={f"flora-{i}.json": tmp_path / f"{i}.json" for i in range(count)},
    )


def test_parallel_episode_chains_isolate_pending_and_failure(tmp_path):
    ctx = context(tmp_path)
    barrier = threading.Barrier(3)
    active = set()
    peak = []
    completed = []
    lock = threading.Lock()

    def annotate(child, value):
        index = value["pin"]["episode_index"]
        with lock:
            active.add(index)
            peak.append(len(active))
        if index < 3:
            barrier.wait(timeout=5)
        try:
            if index == 1:
                raise ValueError("synthetic preparation failure")
            for frame in range(2 if index else 1):
                completed.append((index, frame))
                child.emit({"current": frame + 1, "total": 2})
            return OperatorResult(
                summary={"status": "draft", "needs_review": index == 0, "checkpoints": 1 if index == 0 else 2}
            )
        finally:
            with lock:
                active.remove(index)

    result = execute_batch(ctx, request(), annotate)
    assert max(peak) == 3
    assert [frame for index, frame in completed if index == 0] == [0]
    assert [frame for index, frame in completed if index == 2] == [0, 1]
    assert [frame for index, frame in completed if index == 3] == [0, 1]
    assert result.summary["episodes"][1]["status"] == "error"
    assert result.summary["episodes"][0]["needs_review"]
    assert len({e["run"] for e in result.summary["episodes"]}) == 4


def test_cancel_batch_stops_active_chains_and_never_starts_queued_episode(tmp_path):
    started = threading.Barrier(4)
    cancelled = threading.Event()
    seen = []

    def stop():
        if cancelled.is_set():
            raise StopRequestedError("cancelled")

    ctx = context(tmp_path, stop=stop)

    def annotate(child, value):
        seen.append(value["pin"]["episode_index"])
        started.wait(timeout=5)
        assert cancelled.wait(timeout=5)
        child.check_cancelled()

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(execute_batch, ctx, request(), annotate)
        started.wait(timeout=5)
        cancelled.set()
        with pytest.raises(StopRequestedError):
            future.result(timeout=5)
    assert set(seen) == {0, 1, 2}


def test_duplicate_episode_rejected_before_execution(tmp_path):
    value = request(2)
    value["episodes"][1]["pin"]["episode_index"] = 0
    with pytest.raises(ValueError, match="only once"):
        execute_batch(context(tmp_path, 2), value, lambda *_: pytest.fail("must not execute"))


def test_memory_budget_caps_parallel_decoding():
    from types import SimpleNamespace

    from lerobot.data_platform.memory.batch import parallel_episodes

    assert parallel_episodes(SimpleNamespace(memory_limit=4 * 1024**3)) == 1
    assert parallel_episodes(SimpleNamespace(memory_limit=16 * 1024**3)) == 2
    assert parallel_episodes(SimpleNamespace(memory_limit=32 * 1024**3)) == 3


def test_configured_ten_way_api_concurrency_and_memory_bound(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from lerobot.data_platform.memory.batch import parallel_episodes

    monkeypatch.setenv("DATA_PLATFORM_MEMORY_MAX_PARALLEL_EPISODES", "10")
    assert parallel_episodes(SimpleNamespace(memory_limit=192 * 1024**3)) == 10
    assert parallel_episodes(SimpleNamespace(memory_limit=32 * 1024**3)) == 4
    ctx = context(tmp_path, 10)
    ctx.resources = SimpleNamespace(memory_limit=192 * 1024**3)
    barrier = threading.Barrier(10)
    active = 0
    peak = 0
    lock = threading.Lock()

    def model_call(child, value):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            barrier.wait(timeout=10)
        finally:
            with lock:
                active -= 1
        return OperatorResult(summary={"status": "draft", "needs_review": True, "checkpoints": 1})

    result = execute_batch(ctx, request(10), model_call)
    assert peak == 10 and result.summary["max_parallel_episodes"] == 10
    assert len(result.summary["episodes"]) == 10
    monkeypatch.setenv("DATA_PLATFORM_MEMORY_MAX_PARALLEL_EPISODES", "33")
    with pytest.raises(ValueError, match="between 1 and 32"):
        parallel_episodes()
