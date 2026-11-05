"""Display metadata only; excluded from operator identity and execution snapshots."""

_COMMON = {
    "preprocess.split": ("Split", ["episode_range", "episodes", "exclude_episodes"]),
    "preprocess.merge": ("Merge", ["dimension_policy", "exclude_episodes"]),
    "curation.quality": ("Quality", ["checks", "episodes", "video_mode"]),
    "viewer.prepare": ("Viewer", ["episodes", "data_version"]),
    "curation.snapshot": ("Snapshot", []),
    "umi.diagnose": ("UMI Tracker diagnosis", ["episodes"]),
    "umi.compare": ("UMI trajectory comparison", ["episodes", "workers"]),
    "umi.replay": ("UMI robot replay", ["episodes", "motion_mode"]),
    "umi.cosmos_id": ("UMI Cosmos candidates", ["episodes", "targets"]),
}


def presentation(operator_id):
    title, common = _COMMON.get(operator_id, (operator_id, []))
    return {"title": title, "common_parameters": common}
