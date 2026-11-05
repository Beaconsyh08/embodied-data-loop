"""CPU-only entry point into the original laundry Viser UI with persisted replay arrays."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path


def main():
    bundle, laundry_root, port = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    parent = os.getppid()

    def watch_parent():
        while os.getppid() == parent:
            time.sleep(2)
        os._exit(0)

    threading.Thread(target=watch_parent, daemon=True).start()
    sys.path.insert(0, str(laundry_root / "src"))
    from laundry.umi_replay import replay, viser

    manifest = json.loads((bundle / "viewer.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "replay":
        raise ValueError("Unsupported replay viewer bundle")
    values = replay.load_replays(bundle, bundle / "media")
    if not values:
        raise ValueError("No executed frames are available for replay")
    # Upstream's renderer is unchanged; replace only its compute boundary with a cache read.
    # This dedicated process never imports the CUDA runtime or calls compute_replays.
    viser.compute_replays = lambda args: values
    args = viser.parse_args(
        [
            "--dataset",
            str(bundle / "media"),
            "--port",
            str(port),
            "--alignment",
            manifest["alignment"],
            "--motion-mode",
            manifest["motion_mode"],
        ]
    )
    viser.main(args)


if __name__ == "__main__":
    main()
