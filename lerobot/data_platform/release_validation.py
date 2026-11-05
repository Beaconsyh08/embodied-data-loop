"""Isolate caches for parallel release-validation workers."""

import os
import re
from pathlib import Path


def pytest_configure(config):
    root = os.environ.get("DATA_PLATFORM_VALIDATION_CACHE_ROOT")
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if not root or not worker:
        return
    if not re.fullmatch(r"gw[0-9]+", worker):
        raise ValueError("Invalid pytest worker identity")
    for key in ("HF_HOME", "HF_DATASETS_CACHE", "HF_LEROBOT_HOME", "UV_CACHE_DIR"):
        os.environ[key] = str(Path(root) / worker / key)
