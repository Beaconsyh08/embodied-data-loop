import io
import json

import pytest
from PIL import Image

from scripts import export_ee3_parallel as exporter


def test_reuse_raw_verified_images_and_reject_tampered_completed_export(tmp_path, monkeypatch):
    source = tmp_path / "ee3_20260929_2_grasp_from_wm"
    source.mkdir()
    (source / "data.bin").write_bytes(b"immutable raw source")
    annotation = tmp_path / "annotation.json"
    annotation.write_text(json.dumps({"result": {"segments": [{"start_frame": 0, "end_frame": 2}]}}))
    raw = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(raw, format="PNG")
    cameras = ["image", "left_wrist_image", "right_wrist_image"]
    rows = [{"timestamp": i / 30, **{c: {"bytes": raw.getvalue()} for c in cameras}} for i in range(2)]
    item = {
        "root": source,
        "videos": {},
        "rows": rows,
        "cameras": cameras,
        "fps": 30,
        "identity": {"lineage_group": "recording-0", "dataset_version_id": "snapshot", "episode_uid": "0"},
        "annotation_path": annotation,
        "annotation_sha256": exporter.digest(annotation),
        "artifact": json.loads(annotation.read_text()),
        "hashes": {"data.bin": exporter.digest(source / "data.bin")},
    }
    monkeypatch.setattr(exporter, "inspect_selection", lambda _: item)
    staging, audit = tmp_path / "staging", tmp_path / "audit"
    staging.mkdir()
    audit.mkdir()
    existing = staging / "media/episode-000000/camera-00/000000.png"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(raw.getvalue())
    selection = {"dataset_key": "ee3", "episode_index": 0, "task": "Transfer garments"}
    result = exporter.export_episode(0, selection, staging, audit)
    assert result["reused_images"] == 1 and result["episode"]["frame_count"] == 2
    assert exporter.export_episode(0, selection, staging, audit) == result
    existing.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Completed export files changed"):
        exporter.export_episode(0, selection, staging, audit)
    (audit / "episode-000000.json").unlink()
    Image.new("RGB", (8, 8), "blue").save(existing)
    with pytest.raises(ValueError, match="differs from raw source"):
        exporter.export_episode(0, selection, staging, audit)
