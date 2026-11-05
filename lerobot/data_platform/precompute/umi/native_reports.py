"""Portable data for the pinned laundry renderers, without publishing a repaired dataset."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def copy_evaluation_subset(root, destination, episodes):
    """Copy selected rows and metadata; no hard links back to the source."""
    from io import BytesIO

    import pyarrow as pa
    import pyarrow.parquet as pq
    from laundry.core.dataset import DatasetReader
    from PIL import Image

    reader = DatasetReader(root)
    destination = Path(destination)
    (destination / "meta").mkdir(parents=True, exist_ok=True)
    indices = set(episodes)
    records = [ep for ep in reader.episodes() if ep.index in indices]
    if {ep.index for ep in records} != indices:
        raise ValueError("Viewer selection contains unknown episodes")
    info = json.loads((Path(root) / "meta/info.json").read_text())
    # Upstream bounds episode IDs by total_episodes; preserve the original ID space.
    info.update(total_frames=sum(ep.length for ep in records), viewer_selected_episodes=sorted(indices))
    write_json(destination / "meta/info.json", info)
    shutil.copy2(Path(root) / "meta/tasks.jsonl", destination / "meta/tasks.jsonl")
    (destination / "meta/episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": ep.index, "length": ep.length, "tasks": ep.tasks}) + "\n"
            for ep in records
        )
    )
    for ep in records:
        path = destination / ep.path.relative_to(Path(root))
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(ep.path)
        for name, feature in info["features"].items():
            if feature.get("dtype") != "image" or name not in table.column_names:
                continue
            frames = []
            for cell in table[name].to_pylist():
                if not cell or not cell.get("bytes"):
                    raise ValueError("Native viewer requires embedded camera frames")
                encoded = BytesIO()
                with Image.open(BytesIO(cell["bytes"])) as frame:
                    frame.convert("RGB").save(encoded, format="JPEG", quality=90)
                frames.append({**cell, "bytes": encoded.getvalue()})
            field = table.schema.field(name)
            table = table.set_column(
                table.schema.get_field_index(name), field, pa.array(frames, type=field.type)
            )
        pq.write_table(table, path)
    write_json(
        destination / "VIEW_ONLY.json",
        {
            "evaluation_only": True,
            "training_approved": False,
            "camera_preview": {"encoding": "jpeg", "quality": 90, "dimensions": "unchanged"},
        },
    )
    return destination


def render_cosmos(root, folder, destination):
    """Render persisted candidates through laundry's original report and curve viewer."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from laundry.core.context import RunContext
    from laundry.core.dataset import DatasetReader
    from laundry.core.manifest import PipelineRecord, new_manifest
    from laundry.core.report import PipelineFigures, render_html_report
    from laundry.pipelines._cosmos_id.reporting import render_repair

    folder, destination = Path(folder), Path(destination)
    response = json.loads((folder / "response.json").read_text())
    indices = [int(key) for key in response["episodes"]]
    destination.mkdir(parents=True, exist_ok=True)
    candidate = copy_evaluation_subset(root, destination / "media", indices)
    reader = DatasetReader(root)
    candidate_reader = DatasetReader(candidate)
    operations = []
    for key, record in response["episodes"].items():
        index = int(key)
        with np.load(folder / f"episode_{index:06d}.npz", allow_pickle=False) as values:
            before, after = reader.episode(index).state(), values["state"]
            repaired = values["repaired"]
            if before.shape != after.shape or repaired.shape != before.shape:
                raise ValueError("Candidate shape differs from its source")
            if not np.array_equal(before[~repaired], after[~repaired], equal_nan=True):
                raise ValueError("Candidate changed observations outside its repair mask")
            path = candidate_reader.episode_path(index)
            table = pq.read_table(path)
            field = table.schema.field("state")
            table = table.set_column(
                table.schema.get_field_index("state"), field, pa.array(after.tolist(), type=field.type)
            )
            pq.write_table(table, path)
            operations.append(
                {
                    "episode": index,
                    "repaired_values": int(repaired.sum()),
                    "unresolved_values": int(values["unresolved"].sum()),
                    "gaps": record["gaps"],
                }
            )
    report = {
        "status": "partial" if any(op["unresolved_values"] for op in operations) else "complete",
        "candidate_only": True,
        "human_approvals": 0,
        "operations": operations,
    }
    ctx = RunContext(
        "cosmos-candidate", reader, SimpleNamespace(output_root=candidate, dropped_episodes=[]), destination
    )
    ctx.set_artifact("cosmos_id_report", report)
    figures = render_repair(ctx)
    viewer = destination / "cosmos_id_viewer.html"
    if viewer.exists():
        # Keep upstream's UI. Only supply its existing camera loader over authenticated HTTP.
        page = viewer.read_text()
        marker = "function initCameras() {\n  if (!IMAGES) return;"
        loader = """
  if (location.protocol === 'http:' || location.protocol === 'https:') {
    cams.source = async rel => {
      const response = await fetch(new URL('media/' + rel, location.href), {credentials: 'same-origin'});
      if (!response.ok) throw new Error('Camera data: HTTP ' + response.status);
      const data = await response.arrayBuffer();
      return {byteLength: data.byteLength, slice: (a, b) => data.slice(a, b)};
    };
  }
"""
        if page.count(marker) != 1:
            raise ValueError("Pinned laundry camera interface changed")
        viewer.write_text(page.replace(marker, marker + loader))
    manifest = new_manifest(
        "cosmos-candidate", "Original evaluation observations", "Candidate preview only", "umi"
    )
    manifest.pipelines = [
        PipelineRecord(
            name="umi-cosmos-id-repair",
            version="1.1.0",
            operations=operations,
            summary="模型候选，待人工审核；未发布修复数据集，人工批准为 0。",
        )
    ]
    (destination / "report.html").write_text(
        render_html_report(manifest, [], [PipelineFigures("umi-cosmos-id-repair", figures)])
    )
    manifest.write(destination / "manifest.json")
    write_json(destination / "viewer.json", {"schema_version": 1, "kind": "cosmos", "episodes": indices})


def install_replay_capture(screen, destination):
    """Capture the exact screening execution, including partial failures, once."""
    import numpy as np
    from laundry.umi_replay import replay as replay_data

    destination = Path(destination)
    original = screen.generate_replay
    replays = {}
    failures = []

    def generate(controller, reference, source, args, **kwargs):
        joints = []
        targets = {side: [] for side in controller.driven_sides}
        actuals = {side: [] for side in controller.driven_sides}
        previous_callback = kwargs.pop("on_frame", None)

        def capture(frame, q, actual, target):
            joints.append(q.copy())
            for side in targets:
                targets[side].append(target[side].copy())
                actuals[side].append(actual[side].copy())
            if previous_callback:
                previous_callback(frame, q, actual, target)

        diagnostics = kwargs.setdefault("diagnostics", {})
        failure = None
        try:
            return original(controller, reference, source, args, on_frame=capture, **kwargs)
        except Exception as error:
            failure = {"episode_index": source.episode_index, "error": str(error), "frames": len(joints)}
            failures.append(failure)
            raise
        finally:
            if joints:
                replays[source.episode_index] = {
                    "source": source,
                    "joints": np.stack(joints),
                    "targets": {side: np.stack(values) for side, values in targets.items()},
                    "actuals": {side: np.stack(values) for side, values in actuals.items()},
                    "errors": replay_data._captured_errors(actuals, targets),
                    "replay_setup": {**diagnostics, "partial_failure": failure},
                    "collision_events": [],
                }
                replay_data.save_replays(replays, destination)
            write_json(destination / "failures.json", failures)

    screen.generate_replay = generate

    def finish(root):
        screen.generate_replay = original
        if replays:
            copy_evaluation_subset(root, destination / "media", replays)
        write_json(
            destination / "viewer.json",
            {
                "schema_version": 1,
                "kind": "replay",
                "episodes": sorted(replays),
                "failures": failures,
            },
        )

    return finish


if __name__ == "__main__":
    import sys

    render_cosmos(*map(Path, sys.argv[1:]))
