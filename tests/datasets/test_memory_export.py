"""Package-level compatibility fixtures; optional cross-repository high-level validation."""

import copy
import json
import os
import subprocess
import uuid
from pathlib import Path

import pytest

from lerobot.data_platform.memory.contracts import AnnotationConfig, digest
from lerobot.data_platform.memory.jobs import execute, export_request
from lerobot.data_platform.memory.review import review_payload
from lerobot.data_platform.memory.sources import preview, select_pilot
from lerobot.data_platform.operators import ExecutionContext
from lerobot.data_platform.runtime.supervisor import atomic_json
from tests.datasets.test_memory_annotation import Teacher, real_source


def approved_package(tmp_path, monkeypatch):
    source, selections = real_source(tmp_path, count=10)
    monkeypatch.setenv("DATA_PLATFORM_MEMORY_MODEL", "test-model")
    monkeypatch.setattr("lerobot.data_platform.memory.jobs.QwenClient", lambda **_: Teacher())
    states = []
    root = tmp_path / "data-disk"
    for index, selection in enumerate(selections):
        pin = preview(source, selection, AnnotationConfig("test-model"), tmp_path / "hashes")
        static = tmp_path / f"review-{index}"
        static.mkdir()
        generation = str(uuid.uuid4())
        input_path = tmp_path / f"input-{index}.json"
        atomic_json(
            input_path,
            {"request_id": generation, "operation": "memory.annotate", "dataset_key": "fixture", "pin": pin},
        )
        context = ExecutionContext(
            source_root=source,
            output_root=root,
            dataset_key="fixture",
            work_dir=static,
            artifacts={"input.json": input_path, "flora.json": Path(selection["annotation"])},
        )
        execute(context, {"request_id": generation})
        state = json.loads((static / "review.json").read_text())
        record = state["record"]
        record["review"] = {
            "status": "approved",
            "reviewer": "synthetic-reviewer-a",
            "second_reviewer": "synthetic-reviewer-b",
            "source_confirmed": True,
            "goal_confirmed": True,
            "causal_evidence_confirmed": True,
            "adjudicated": True,
            "payload_sha256": digest(review_payload(record)),
        }
        states.append(state)
    pilot = select_pilot([s["pin"] for s in states])
    static = tmp_path / "export-result"
    static.mkdir()
    context = ExecutionContext(
        source_root=source,
        output_root=root,
        dataset_key="fixture",
        work_dir=static,
        artifacts={f"flora-{i}.json": Path(s["annotation"]) for i, s in enumerate(selections)},
    )
    request = {
        "request_id": str(uuid.uuid4()),
        "states": states,
        "assignments": pilot["assignments"],
        "seed": 42,
        "revisions": [{"run": f"synthetic-{i}", "revision": 2} for i in range(10)],
    }
    result = export_request(context, request, lambda: None)
    return Path(result.summary["output"]), context, request


def test_export_immutable_and_draft_rejected(tmp_path, monkeypatch):
    output, context, request = approved_package(tmp_path, monkeypatch)
    assert len(json.loads((output / "ledger.json").read_text())["episodes"]) == 10
    assert len(list((output / "audit").rglob("draft.json"))) == 20
    again = export_request(context, request, lambda: None)
    assert Path(again.summary["output"]) == output
    changed = copy.deepcopy(request)
    changed["states"][0]["record"]["review"] = {"status": "draft"}
    with pytest.raises(ValueError, match="human approval"):
        export_request(context, changed, lambda: None)


@pytest.mark.skipif(
    not os.getenv("MEM_TOOLS_PYTHON"),
    reason="Set MEM_TOOLS_PYTHON and MEM_TOOLS_ROOT for cross-repository acceptance",
)
def test_high_level_inspect_build_training_export(tmp_path, monkeypatch):
    output, _, _ = approved_package(tmp_path, monkeypatch)
    python = os.environ["MEM_TOOLS_PYTHON"]
    cwd = os.environ["MEM_TOOLS_ROOT"]
    bundle = str(output / "bundle")
    commands = [
        ["inspect", "--bundle", bundle],
        [
            "build",
            "--bundle",
            bundle,
            "--ledger",
            str(output / "ledger.json"),
            "--splits",
            str(output / "splits.json"),
            "--output",
            str(tmp_path / "dataset"),
        ],
        [
            "training-export",
            "--bundle",
            bundle,
            "--dataset",
            str(tmp_path / "dataset"),
            "--condition",
            "summary",
            "--output",
            str(tmp_path / "training"),
        ],
    ]
    for command in commands:
        result = subprocess.run(
            [python, "-m", "mem_tools", *command], cwd=cwd, capture_output=True, text=True, timeout=90
        )
        assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run(
        [
            python,
            "-c",
            "from datasets import load_from_disk; import sys; d=load_from_disk(sys.argv[1]); assert set(d)=={'train','validation'}; assert set(d['train'].column_names)=={'image','task','previous_memory','prompt','answer','sample_id'}; print(d)",
            str(tmp_path / "training"),
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
