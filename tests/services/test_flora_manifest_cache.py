"""Receipt persistence does not bypass the upstream validator or hide changed-asset failures."""

import hashlib
import json
from pathlib import Path

import pytest

from services.flora_api.manifest_cache import enable


def test_receipt_roundtrip_still_invokes_upstream(tmp_path):
    cache = tmp_path / "cache"
    model = tmp_path / "model"
    model.mkdir()
    key = str(model.resolve())
    value = ("manifest-hash", (("file", "sha256", ("resolved", 1, 2, 3, 4, 5)),))
    namespace = {"_VERIFIED_MANIFESTS": {}}
    calls = []

    def validate(path):
        calls.append(namespace["_VERIFIED_MANIFESTS"].get(key))
        namespace["_VERIFIED_MANIFESTS"][key] = value

    namespace["validate_manifest"] = validate
    enable(namespace, str(cache))
    namespace["validate_manifest"](model)
    assert calls == [None]
    # A new process has an empty in-memory registry; it must still call the original validator.
    namespace["_VERIFIED_MANIFESTS"].clear()
    namespace["validate_manifest"](model)
    assert calls == [None, value]


def test_upstream_failure_invalidates_receipt(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    key = str(model.resolve())
    receipt = cache / (hashlib.sha256(key.encode()).hexdigest() + ".json")
    receipt.write_text(json.dumps({"schema": 1, "root": key, "verified": ["previous", []]}))
    namespace = {"_VERIFIED_MANIFESTS": {}}

    def reject(path):
        raise RuntimeError("A model asset failed its integrity check")

    namespace["validate_manifest"] = reject
    enable(namespace, str(cache))
    with pytest.raises(RuntimeError, match="integrity"):
        namespace["validate_manifest"](model)
    assert not receipt.exists()
    assert key not in namespace["_VERIFIED_MANIFESTS"]


def test_malformed_receipt_falls_back_to_upstream(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    key = str(tmp_path.resolve())
    (cache / (hashlib.sha256(key.encode()).hexdigest() + ".json")).write_text("not json")
    calls = []
    namespace = {"_VERIFIED_MANIFESTS": {}, "validate_manifest": lambda path: calls.append(Path(path))}
    enable(namespace, str(cache))
    namespace["validate_manifest"](tmp_path)
    assert calls == [tmp_path]
