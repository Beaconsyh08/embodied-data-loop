"""Permission-scoped catalog of published captions and Agent scan inventories."""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import quote

from lerobot.data_platform.operators.implementations.caption.core import artifact_key

VARIANTS = {"coarse.json", "refined.json", "caption.json", "fusion_draft.json", "fusion_reviewed.json"}


def saved_run_count(root: Path) -> int:
    """Count saved runs, not variants or episodes; reject links outside the result tree."""
    if root.is_symlink() or not root.is_dir():
        return 0
    count = 0
    with os.scandir(root) as runs:
        for run in runs:
            if not run.is_dir(follow_symlinks=False) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run.name):
                continue
            with os.scandir(run.path) as files:
                count += any(item.name in VARIANTS and item.is_file(follow_symlinks=False) for item in files)
    return count


def caption_datasets(ctx, actor: dict | None, root: Path) -> tuple[list[dict], list[str]]:
    """Include Agent outputs without remote reads or exposing datasets outside the current grants."""
    store = ctx.control_plane_store
    locations = store.accessible_locations(actor["user_id"], summary=True) if store and actor else []
    nodes = {row["node_id"]: row for row in store.list_nodes()} if locations else {}
    candidates = [
        {
            "id": "remote:" + row["location_id"],
            "name": Path(row["root"]).name,
            "dataset_key": row["dataset_key"],
            "node": row["node_name"],
            "agent_runs": row.get("metadata", {}).get("caption_output_runs", 0),
            "agent_online": nodes.get(row["node_id"], {}).get("status") == "online",
            "review_url": f"/remote/{quote(row['location_id'], safe='')}/temporal-caption/review",
        }
        for row in locations
    ]
    catalog = getattr(ctx, "dataset_catalog", None)
    keys = catalog.visible_dataset_keys(actor) if catalog else getattr(ctx, "datasets_index", {})
    allowed = store.restricted_dataset_keys(actor["user_id"]) if store and actor else None
    remote_keys = {row["dataset_key"] for row in locations}
    for key in keys:
        key = "/".join(key) if isinstance(key, tuple) else key
        if key in remote_keys or (allowed is not None and key not in allowed):
            continue
        candidates.append(
            {
                "id": key,
                "name": key.rsplit("/", 1)[-1],
                "dataset_key": key,
                "node": "Local",
                "review_url": f"/{quote(key, safe='/')}/temporal-caption/review",
            }
        )
    rows, warnings = [], []
    for candidate in candidates:
        agent_runs = candidate.get("agent_runs", 0)
        if type(agent_runs) is not int or agent_runs < 0:
            agent_runs = 0
        try:
            count = saved_run_count(root / artifact_key(candidate["dataset_key"]))
        except OSError:
            warnings.append(f"Unable to read saved results for {candidate['name']}.")
            count = 0
        if count or agent_runs:
            rows.append({**candidate, "saved_runs": count, "agent_runs": agent_runs})
    return sorted(rows, key=lambda row: (row["name"].casefold(), row["node"], row["id"])), warnings
