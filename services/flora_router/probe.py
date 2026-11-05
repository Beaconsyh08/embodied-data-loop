"""Print only GPU and API capacity; used locally and by a restricted SSH command."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from services.flora_router.capacity import read_capacity


def configured_capacity(config_path: Path = Path("/etc/flora-router/capacity.json")):
    values = json.loads(config_path.read_text())
    return read_capacity(
        Path(values["runtime"]),
        Path(values["state"]),
        int(values["concurrency"]),
        extra_states=tuple(Path(value) for value in values.get("legacy_states", [])),
    )


def main():
    try:
        result = {"ready": True, **asdict(configured_capacity())}
    except Exception:
        # Driver exceptions and configuration paths must not cross the SSH boundary.
        result = {"ready": False}
    print(json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
