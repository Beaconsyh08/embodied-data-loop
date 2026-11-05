"""Persisted annotation hierarchy, full-dataset aggregation and scoped drill-down."""

from sqlalchemy import delete, func, select

from lerobot.data_platform.caption_job_groups import GROUP_PREFIX
from lerobot.data_platform.caption_job_storage import backfill_caption_parents
from lerobot.data_platform.control_plane import CaptionDatasetJob, CaptionJobMember, RemoteJob
from lerobot.data_platform.list_queries import ListPage
from tests.datasets.test_control_plane import _app
from tests.datasets.test_job_management import setup  # noqa: F401


def caption(store, owner, location, episode, *, status="done", skipped=False):
    job = store.create_job(
        location_id=location["location_id"],
        requested_by=owner["user_id"],
        operation="caption.annotate",
        options={"episode_index": episode, "scheme": "flora"},
    )
    with store.sessions.begin() as db:
        row = db.get(RemoteJob, job["job_id"])
        row.status = status
        row.result = (
            {"review_url": "/saved-result", "operator_disposition": {"status": "skipped"}}
            if skipped
            else {"review_url": "/saved-result"}
        )
    return job


def test_historical_migration_preserves_all_execution_records_and_is_repeatable(setup):  # noqa: F811
    store, _, owner, _, location, _ = setup
    jobs = [caption(store, owner, location, n, status="running" if n == 3 else "done") for n in range(4)]
    before = [store.get_job(j["job_id"]) for j in jobs]
    with store.sessions.begin() as db:
        db.execute(delete(CaptionJobMember))
        db.execute(delete(CaptionDatasetJob))
    assert backfill_caption_parents(store) == 4
    assert backfill_caption_parents(store) == 0
    with store.sessions() as db:
        assert db.scalar(select(func.count()).select_from(CaptionDatasetJob)) == 1
        assert db.scalar(select(func.count()).select_from(CaptionJobMember)) == 4
    assert [store.get_job(j["job_id"]) for j in jobs] == before


def test_dataset_group_precedes_pagination_and_counts_skips_without_hiding_failed_videos(setup):  # noqa: F811
    store, _, owner, node, location, create = setup
    for n in range(53):
        caption(store, owner, location, n)
    caption(store, owner, location, 53, status="error", skipped=True)
    other = store.sync_locations(node["node_id"], [{"root": "/data/other", "dataset_key": "other"}])[0]
    caption(store, owner, other, 0, status="error")
    ordinary = create()
    page = ListPage.parse({"page_size": 1}, scope={})
    seen = []
    while True:
        jobs, info = store.list_jobs(actor=owner, summary=True, page=page, filters={"group_caption": True})
        seen.extend(jobs)
        if not info["has_more"]:
            break
        page = ListPage.parse({"page_size": 1, "cursor": info["next_cursor"]}, scope={})
    assert len(seen) == 3
    group = next(j for j in seen if j["job_id"] == GROUP_PREFIX + location["location_id"])
    assert group["caption_counts"]["total"] == 54
    assert group["caption_counts"]["done"] == 53
    assert group["caption_counts"]["skipped"] == 1
    assert group["caption_counts"]["error"] == 0
    assert group["status"] == "done" and group["progress"]["progress"] == 100
    assert all(
        group[key].endswith("+00:00") for key in ("created_at", "updated_at", "started_at", "finished_at")
    )
    assert ordinary["job_id"] in {j["job_id"] for j in seen}
    failures = store.list_jobs(actor=owner, filters={"group_caption": True, "status": "error"})
    assert [j["location_id"] for j in failures] == [other["location_id"]]
    raw = store.list_jobs(actor=owner, filters={"location_id": location["location_id"], "caption_only": True})
    assert len(raw) == 54 and not any(j.get("caption_group") for j in raw)


def test_group_counts_obey_owner_and_dataset_access_and_admin_mine(setup):  # noqa: F811
    store, admin, owner, node, location, _ = setup
    caption(store, owner, location, 0)
    caption(store, admin, location, 1)
    other = store.sync_locations(node["node_id"], [{"root": "/data/other", "dataset_key": "other"}])[0]
    caption(store, owner, other, 0)
    store.set_mutation_locations(owner["user_id"], [location["location_id"]], actor=admin)
    rows = store.list_jobs(actor=owner, filters={"group_caption": True})
    assert len(rows) == 1 and rows[0]["caption_counts"]["total"] == 1
    rows = store.list_jobs(actor=admin, filters={"group_caption": True, "mine": True})
    assert len(rows) == 1 and rows[0]["caption_counts"]["total"] == 1
    rows = store.list_jobs(actor=admin, filters={"group_caption": True})
    assert sum(j["caption_counts"]["total"] for j in rows) == 3


def test_group_routes_do_not_decorate_or_execute_parent_and_keep_raw_api(setup, tmp_path):  # noqa: F811
    store, _, owner, _, location, _ = setup
    child = caption(store, owner, location, 0)
    client = _app(tmp_path, store).test_client()
    assert (
        client.post(
            "/api/auth/login", json={"username": "owner-user", "password": "long-test-password"}
        ).status_code
        == 200
    )
    response = client.get("/api/control/jobs?group_caption=1&summary=1").get_json()
    parent = response["jobs"][0]
    assert parent["caption_group"] and parent["available_actions"] == []
    assert (
        client.get("/api/control/caption-dataset-jobs/" + location["location_id"]).get_json()["job"][
            "caption_counts"
        ]["total"]
        == 1
    )
    assert client.get("/api/control/caption-dataset-jobs/missing").status_code == 404
    raw = client.get("/api/control/jobs?summary=1").get_json()["jobs"]
    assert [j["job_id"] for j in raw] == [child["job_id"]]


def test_dataset_jobs_real_browser(setup, tmp_path, monkeypatch):  # noqa: F811
    import os
    import threading
    from pathlib import Path

    import pytest
    from werkzeug.serving import make_server

    from tests.datasets.test_console_services import console_app

    executable = os.environ.get("DATA_PLATFORM_BROWSER_EXECUTABLE")
    if not executable:
        pytest.skip("Set DATA_PLATFORM_BROWSER_EXECUTABLE for browser acceptance")
    playwright = pytest.importorskip("playwright.sync_api")
    store, _, owner, _, location, _ = setup
    for n in range(27):
        caption(store, owner, location, n, status="error" if n == 0 else "done", skipped=n == 0)
    app = console_app(tmp_path, database_url=str(store.engine.url))
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with playwright.sync_playwright() as runtime:
            browser = runtime.chromium.launch(
                executable_path=executable, args=["--no-sandbox", "--disable-gpu"]
            )
            page = browser.new_page(viewport={"width": 1366, "height": 900})
            assets = os.environ.get("DATA_PLATFORM_BROWSER_ASSETS")
            if assets:
                for pattern, name in [
                    ("https://cdn.tailwindcss.com/**", "ui-tailwind.js"),
                    ("https://cdn.jsdelivr.net/npm/alpinejs@*/dist/cdn.min.js", "ui-alpine.js"),
                ]:
                    page.route(
                        pattern,
                        lambda route, request, name=name: route.fulfill(
                            path=str(Path(assets) / name), content_type="application/javascript"
                        ),
                    )
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            assert (
                page.request.post(
                    base + "/api/auth/login",
                    data={"username": "owner-user", "password": "long-test-password"},
                ).status
                == 200
            )
            page.goto(base + "/?page=runs")
            page.wait_for_function("window.Alpine && document.querySelector('[x-data]')._x_dataStack")
            page.get_by_role("button", name="Temporal Caption · Dataset job").first.click()
            panel = page.locator("[data-caption-children]:visible")
            playwright.expect(panel.locator("button", has_text="Episode")).to_have_count(25)
            assert "27 videos" in page.locator("body").inner_text()
            panel.get_by_role("button", name="More videos").click()
            playwright.expect(panel.locator("button", has_text="Episode")).to_have_count(27)
            panel.get_by_label("Video subtask status").select_option("error")
            playwright.expect(panel.locator("button", has_text="Episode")).to_have_count(1)
            assert "skipped" in panel.inner_text()
            panel.get_by_role("button", name="Episode 0 · flora").click()
            playwright.expect(page.get_by_role("button", name="Back to dataset job").first).to_be_visible()
            assert (
                page.get_by_role("button", name="Retry", exact=True).count() == 0
                or not page.get_by_role("button", name="Retry", exact=True).first.is_visible()
            )
            page.get_by_role("button", name="Back to dataset job").first.click()
            playwright.expect(panel).to_be_visible()
            page.reload()
            page.wait_for_function("window.Alpine && document.querySelector('[x-data]')._x_dataStack")
            playwright.expect(
                page.get_by_role("button", name="Temporal Caption · Dataset job").first
            ).to_be_visible()
            page.goto(base + "/control-plane#jobs")
            page.get_by_role("button", name="Video subtasks", exact=True).click()
            dialog = page.get_by_role("dialog", name="Dataset annotation job")
            playwright.expect(dialog).to_be_visible()
            assert "27 videos" in dialog.inner_text()
            dialog.get_by_role("button", name="More videos").click()
            playwright.expect(dialog.get_by_role("button", name="Details", exact=True)).to_have_count(27)
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
