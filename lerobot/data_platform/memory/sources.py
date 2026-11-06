"""Read-only source audit and frozen episode selections; never consume state/action evidence."""

from __future__ import annotations

import random
from pathlib import Path

from lerobot.data_platform.artifacts.mem_export import inspect_selection
from lerobot.data_platform.memory.contracts import (
    AnnotationConfig,
    checkpoints,
    digest,
    provider_config_digest,
)
from lerobot.data_platform.operators.implementations.caption.flora import camera_roles, validate_flora
from lerobot.data_platform.operators.implementations.caption.snapshot import caption_source_fingerprint


def preview(source: Path, selection: dict, config: AnnotationConfig, cache: Path) -> dict:
    item = inspect_selection({**selection, "root": str(source)})
    artifact = item["artifact"]
    validate_flora(artifact)
    if any(len(segment["native_segment_indices"]) != 1 for segment in artifact["result"]["segments"]):
        raise ValueError("FLORA coverage contains gaps or overlapping captions")
    fingerprint = caption_source_fingerprint(source, cache)
    if artifact.get("source_fingerprint") != fingerprint:
        raise ValueError("FLORA source fingerprint is missing or does not match; cannot admit this episode")
    roles = camera_roles(item["cameras"])
    if set(roles) != {"main", "left", "right"} or item["cameras"] != [
        roles[r] for r in ("main", "left", "right")
    ]:
        raise ValueError("Camera order must be main, left wrist, right wrist")
    timestamps = [float(row["timestamp"]) for row in item["rows"]]
    starts = [s["start_frame"] for s in artifact["result"]["segments"]]
    schedule = checkpoints(timestamps, starts)
    return {
        "schema": "memory_input/v1",
        "provider_config_sha256": provider_config_digest(config.model),
        "dataset_key": selection["dataset_key"],
        "episode_index": selection["episode_index"],
        "flora_run": selection["flora_run"],
        "annotation_sha256": item["annotation_sha256"],
        "source_fingerprint": fingerprint,
        "source_files_sha256": item["hashes"],
        "cameras": item["cameras"],
        "task": selection["task"],
        "identity": item["identity"],
        "config": config.to_dict(),
        "frame_count": len(timestamps),
        "timestamps": timestamps,
        "checkpoints": schedule,
        "segment_starts": starts,
        "expected_requests": 2 * len(schedule),
        "duration_s": timestamps[-1] - timestamps[0],
    }


def select_pilot(eligible: list[dict], *, seed: int = 42, count: int = 10) -> dict:
    """Seeded duration strata, then whole-lineage group allocation. Unknown lineage is a gap."""
    rows = sorted(eligible, key=lambda r: (r["duration_s"], r["episode_index"]))
    if len({r["episode_index"] for r in rows}) != len(rows):
        raise ValueError("Select one pinned FLORA run per episode before pilot selection")
    rng = random.Random(seed)
    strata = [rows[i * len(rows) // 3 : (i + 1) * len(rows) // 3] for i in range(3)]
    for stratum in strata:
        rng.shuffle(stratum)
    chosen = []
    while len(chosen) < min(count, len(rows)):
        for stratum in strata:
            if stratum and len(chosen) < count:
                chosen.append(stratum.pop())
    groups = {}
    for row in chosen:
        group = row["identity"].get("lineage_group")
        if not group:
            raise ValueError("Recording lineage must be confirmed before freezing pilot splits")
        groups.setdefault(group, []).append(row["episode_index"])
    targets = {"train": 6, "calibration": 1, "validation": 2, "test": 1}
    # Exhaustive whole-group allocation for ten episodes, minimizing absolute target deviations.
    states = {(0, 0, 0, 0): {}}
    names = list(targets)
    for _group, ids in sorted(groups.items()):
        next_states = {}
        for counts, assignments in states.items():
            for i, name in enumerate(names):
                changed = list(counts)
                changed[i] += len(ids)
                next_states.setdefault(tuple(changed), {**assignments, **{str(e): name for e in ids}})
        states = next_states
    counts, assignments = min(
        states.items(),
        key=lambda pair: (sum(abs(pair[0][i] - targets[n]) for i, n in enumerate(names)), pair[0]),
    )
    result = {
        "schema": "memory_pilot/v1",
        "seed": seed,
        "selected": chosen,
        "assignments": assignments,
        "counts": dict(zip(names, counts, strict=True)),
        "target": targets,
        "shortfall": max(0, count - len(chosen)),
    }
    return {**result, "digest": digest(result)}


def read_preview(agent, read):
    """Bounded relay preflight. Caption payload is server-resolved, never a model input."""
    import tempfile

    from lerobot.data_platform.runtime.agent import _require_within

    source = _require_within(Path(read["root"]), agent.allowed_roots, "memory source")
    with tempfile.TemporaryDirectory(prefix="memory-preflight-", dir=agent.state_path.parent) as folder:
        path = Path(folder) / "flora.json"
        import base64

        path.write_bytes(base64.b64decode(read["artifact_base64"], validate=True))
        selection = {**read["selection"], "annotation": str(path), "dataset_key": read["dataset_key"]}
        return preview(
            source,
            selection,
            AnnotationConfig(**read["config"]),
            agent.state_path.parent / "caption-source-hashes",
        )
