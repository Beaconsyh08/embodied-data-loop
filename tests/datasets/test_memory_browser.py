"""Real Chrome interactions against an isolated synthetic Flask/Agent fixture."""

import os
import shutil
import threading

import pytest
from werkzeug.serving import make_server

from tests.datasets.test_memory_routes import execute, frozen_batch, memory_remote  # noqa: F401


def test_memory_browser_review_batch_and_export(memory_remote, tmp_path):  # noqa: F811
    playwright = pytest.importorskip("playwright.sync_api")
    executable = (
        os.getenv("DATA_PLATFORM_BROWSER_EXECUTABLE")
        or shutil.which("google-chrome")
        or shutil.which("chromium")
    )
    if not executable:
        pytest.skip("System Chrome is required")
    remote = memory_remote
    prepared, pilot = frozen_batch(remote)
    staged = remote.client.post(
        remote.base + "/batch",
        json={
            "request_ids": [p["request_id"] for p in prepared],
            "pilot_id": pilot["digest"],
        },
    ).get_json()
    parent, _ = execute(remote, staged["request_id"], tmp_path)
    runs = [e["run"] for e in remote.store.get_job(parent)["result"]["summary"]["episodes"]]
    remote.store.register_user(username="independent", password="synthetic-review-password", role="operator")
    server = make_server("127.0.0.1", 0, remote.app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with playwright.sync_playwright() as runtime:
            browser = runtime.chromium.launch(
                executable_path=executable, args=["--no-sandbox", "--disable-gpu"]
            )
            page = browser.new_page(viewport={"width": 1200, "height": 900})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            assert (
                page.request.post(
                    base + "/api/auth/login",
                    data={"username": "platform-admin", "password": "strong-admin-password"},
                ).status
                == 200
            )
            page.goto(base + f"/remote/{remote.location['location_id']}/memory?run={runs[0]}")
            playwright.expect(page.locator("#review")).to_be_visible()
            assert "cutoff=0" in page.locator("#current").get_attribute("src")
            assert page.locator("#history img").count() == 0
            page.locator("#checkpoint").select_option("1")
            playwright.expect(page.locator("#history img")).to_have_count(1)
            page.locator("#play").click()
            playwright.expect(page.locator("#playback")).to_be_visible()
            page.locator("#subtask").fill("Inspect the block")
            page.locator("#reason").fill("Synthetic browser correction")
            page.locator("#save").click()
            playwright.expect(page.locator("#revision")).to_contain_text("Revision 2")
            for run in runs:
                page.locator("#runs").select_option(run)
                page.locator("#open").click()
                playwright.expect(page.locator("#revision")).to_contain_text("draft")
                for name in ("source-confirmed", "goal-confirmed", "causal-confirmed"):
                    page.locator("#" + name).check()
                page.locator("#approve").click()
                playwright.expect(page.locator("#revision")).to_contain_text("approved")
            other = browser.new_page()
            assert (
                other.request.post(
                    base + "/api/auth/login",
                    data={"username": "independent", "password": "synthetic-review-password"},
                ).status
                == 200
            )
            other.goto(base + f"/remote/{remote.location['location_id']}/memory")
            for run in runs:
                other.locator("#runs").select_option(run)
                other.locator("#open").click()
                playwright.expect(other.locator("#review")).to_be_visible()
                for name in ("source-confirmed", "goal-confirmed", "causal-confirmed"):
                    other.locator("#" + name).check()
                other.locator("#independent").click()
                playwright.expect(other.locator("#status")).to_contain_text("Saved revision")
            page.locator("#task").fill("Move block")
            for episode in range(2):
                page.get_by_label(f"Recording group episode {episode}", exact=True).fill(
                    f"recording-{episode}"
                )
            page.locator("#audit").click()
            playwright.expect(page.locator("#status")).to_contain_text("2/2 episodes passed")
            page.locator("#freeze").click()
            playwright.expect(page.locator("#selection")).to_contain_text('"shortfall": 8')
            before = len(remote.store.list_jobs())
            page.locator("#submit").click()
            playwright.expect(page.locator("#status")).to_contain_text("Dataset Job submitted for 2 episodes")
            assert len(remote.store.list_jobs()) == before + 1
            page.locator("#export").click()
            playwright.expect(page.locator("#export-submit")).to_be_visible()
            page.locator("#export-submit").click()
            playwright.expect(page.locator("#status")).to_contain_text("Export submitted")
            page.locator("#runs").select_option(runs[0])
            page.locator("#open").click()
            page.locator("#extra").fill("0")
            page.locator("#add").click()
            playwright.expect(page.locator("#revision")).to_contain_text("draft")
            page.locator("#regenerate").click()
            playwright.expect(page.locator("#status")).to_contain_text("Regeneration submitted")
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth+1")
            page.screenshot(path=str(tmp_path / "memory-mobile.png"), full_page=True)
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
