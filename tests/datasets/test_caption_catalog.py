"""Saved annotation discovery must be complete, scoped and independent of job ownership."""

from types import SimpleNamespace

import pytest

from lerobot.data_platform.caption_catalog import caption_datasets, saved_run_count
from lerobot.data_platform.routes.temporal_caption import register_temporal_caption_routes
from lerobot.data_platform.temporal_caption import artifact_key
from tests.datasets.test_analysis_metadata import _remote_context
from tests.datasets.test_control_plane import _bootstrap


def saved(root, key, run):
    path = root / artifact_key(key) / run
    path.mkdir(parents=True)
    for name in ("coarse.json", "refined.json"):
        (path / name).write_text("{}")
    return path


def test_catalog_all_saved_runs_not_variants_and_no_symlinks(tmp_path):
    root = tmp_path / "captions"
    for index in range(54):
        saved(root, "local/example", f"run-{index}")
    parent = root / artifact_key("local/example")
    (parent / "linked-run").symlink_to(parent / "run-0", target_is_directory=True)
    (parent / ".staging").mkdir()
    (parent / ".staging/caption.json").write_text("{}")
    assert saved_run_count(parent) == 54
    ctx = SimpleNamespace(control_plane_store=None, datasets_index={("local", "example"): {}})
    rows, warnings = caption_datasets(ctx, None, root)
    assert not warnings
    assert rows[0]["saved_runs"] == 54
    assert rows[0]["review_url"] == "/local/example/temporal-caption/review"
    assert saved_run_count(tmp_path / "missing") == 0


def test_catalog_requires_login_and_rechecks_dataset_grants(tmp_path, monkeypatch):
    app, store, location = _remote_context(tmp_path)
    root = tmp_path / "captions"
    monkeypatch.setenv("DATA_PLATFORM_TEMPORAL_CAPTION_ROOT", str(root))
    other = store.sync_locations(
        location["node_id"],
        [{**location, "root": str(tmp_path / "other"), "dataset_key": "node/other"}],
    )[0]
    saved(root, location["dataset_key"], "first")
    saved(root, other["dataset_key"], "second")
    ctx = SimpleNamespace(control_plane_store=store, datasets_index={})
    register_temporal_caption_routes(app, ctx)
    client = app.test_client()
    url = "/api/temporal-caption/datasets"
    assert client.get(url).status_code == 401
    assert _bootstrap(client).status_code == 200
    response = client.get(url)
    assert response.headers["Cache-Control"] == "private, no-store"
    assert len(response.json["datasets"]) == 2
    admin = store.authenticate_user("platform-admin", "strong-admin-password")[1]
    reader = store.register_user(username="scoped-reader", password="reader-password")
    store.set_mutation_locations(reader["user_id"], [location["location_id"]], actor=admin)
    client.post("/api/auth/logout", json={})
    client.post("/api/auth/login", json={"username": "scoped-reader", "password": "reader-password"})
    rows = client.get(url).json["datasets"]
    assert [row["id"] for row in rows] == ["remote:" + location["location_id"]]
    assert other["dataset_key"] not in str(rows)
    store.set_mutation_locations(reader["user_id"], [], actor=admin)
    assert client.get(url).json["datasets"] == []
    assert not store.list_jobs()


def test_agent_only_catalog_is_scoped_offline_and_does_not_read_remote_results(tmp_path, monkeypatch):
    app, store, location = _remote_context(tmp_path)
    root = tmp_path / "captions"
    monkeypatch.setenv("DATA_PLATFORM_TEMPORAL_CAPTION_ROOT", str(root))
    store.sync_locations(location["node_id"], [{**location, "metadata": {"caption_output_runs": 76}}])
    other = store.sync_locations(
        location["node_id"],
        [
            {
                **location,
                "root": str(tmp_path / "other"),
                "dataset_key": "node/other",
                "metadata": {"caption_output_runs": 80},
            }
        ],
    )[0]
    ctx = SimpleNamespace(control_plane_store=store, datasets_index={})
    register_temporal_caption_routes(app, ctx)
    monkeypatch.setattr(ctx.caption_read_relay, "request", lambda *a, **kw: pytest.fail("Remote read"))
    nodes = store.list_nodes()
    monkeypatch.setattr(store, "list_nodes", lambda: [{**n, "status": "offline"} for n in nodes])
    client = app.test_client()
    url = "/api/temporal-caption/datasets"
    assert client.get(url).status_code == 401
    assert _bootstrap(client).status_code == 200
    rows = client.get(url).json["datasets"]
    assert {r["agent_runs"] for r in rows} == {76, 80}
    assert all(r["saved_runs"] == 0 and r["agent_online"] is False for r in rows)
    saved(root, location["dataset_key"], "also-published")
    rows = client.get(url).json["datasets"]
    assert len(rows) == 2  # Published and Agent evidence share one dataset row.
    row = next(r for r in rows if r["id"] == "remote:" + location["location_id"])
    assert row["saved_runs"] == 1 and row["agent_runs"] == 76
    admin = store.authenticate_user("platform-admin", "strong-admin-password")[1]
    reader = store.register_user(username="agent-catalog-reader", password="reader-password")
    store.set_mutation_locations(reader["user_id"], [location["location_id"]], actor=admin)
    client.post("/api/auth/logout", json={})
    client.post("/api/auth/login", json={"username": "agent-catalog-reader", "password": "reader-password"})
    rows = client.get(url).json["datasets"]
    assert [r["id"] for r in rows] == ["remote:" + location["location_id"]]
    assert other["dataset_key"] not in str(rows)
    store.set_mutation_locations(reader["user_id"], [], actor=admin)
    assert client.get(url).json["datasets"] == []
    assert not store.list_jobs()


@pytest.mark.parametrize("count", [None, True, -1, "76", {}, 0])
def test_legacy_or_invalid_agent_count_does_not_create_a_catalog_entry(tmp_path, count):
    app, store, location = _remote_context(tmp_path)
    assert _bootstrap(app.test_client()).status_code == 200
    store.sync_locations(location["node_id"], [{**location, "metadata": {"caption_output_runs": count}}])
    actor = store.authenticate_user("platform-admin", "strong-admin-password")[1]
    ctx = SimpleNamespace(control_plane_store=store, datasets_index={})
    rows, _ = caption_datasets(ctx, actor, tmp_path / "captions")
    assert rows == []
