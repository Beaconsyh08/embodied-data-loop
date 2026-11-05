"""Submit a local LeRobot directory or an authorized server dataset, then collect all episode results."""

import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from urllib.parse import quote

import requests


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("FLORA_API_URL"), required=False)
    parser.add_argument("--ca", default=os.environ.get("FLORA_API_CA"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path)
    source.add_argument("--server-dataset", help="Authorized alias returned by GET /v1/flora/datasets")
    parser.add_argument("--episodes", default="all", help="all or comma-separated indices")
    parser.add_argument("--output", type=Path, default=Path("flora-dataset-results"))
    args = parser.parse_args()
    if not args.url:
        parser.error("--url or FLORA_API_URL is required")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "submission.json"
    identity = {
        "url": args.url,
        "source": str(args.dataset.resolve()) if args.dataset else args.server_dataset,
        "episodes": args.episodes,
    }
    saved = (
        json.loads(checkpoint.read_text())
        if checkpoint.exists()
        else {
            **identity,
            "idempotency_key": uuid.uuid4().hex,
        }
    )
    if any(saved.get(k) != v for k, v in identity.items()):
        parser.error("This output directory belongs to another submission; use a new output directory")

    def save():
        temporary = checkpoint.with_suffix(".tmp")
        temporary.write_text(json.dumps(saved, indent=2))
        temporary.replace(checkpoint)

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {os.environ['FLORA_API_KEY']}"
    session.verify = args.ca or True
    base = args.url.rstrip("/")

    def call(method, path, **kwargs):
        response = session.request(method, base + path, timeout=(15, 3600), **kwargs)
        response.raise_for_status()
        return response.json()

    if args.dataset:
        if "dataset_id" not in saved:
            files = []
            root = args.dataset.resolve()
            for folder in ("meta", "data", "videos", "images"):
                for path in sorted((root / folder).rglob("*")):
                    if not path.is_file():
                        continue
                    if not path.resolve().is_relative_to(root):
                        raise ValueError("Dataset contains links outside its root")
                    name = path.relative_to(root).as_posix()
                    if any(part.startswith(".") for part in path.relative_to(root).parts):
                        continue
                    files.append({"path": name, "size": path.stat().st_size, "sha256": sha256(path)})
            saved["dataset_id"] = call("POST", "/v1/flora/datasets", json={"files": files})["id"]
            save()
        upload = call("GET", f"/v1/flora/datasets/{saved['dataset_id']}")
        for index, name in enumerate(upload["missing"]):
            print(f"Uploading {index + 1}/{len(upload['missing'])}: {name}", flush=True)
            with (args.dataset / name).open("rb") as data:
                call(
                    "PUT",
                    f"/v1/flora/datasets/{saved['dataset_id']}/files/{quote(name, safe='/')}",
                    data=data,
                )
        call("POST", f"/v1/flora/datasets/{saved['dataset_id']}/complete", json={})
    else:
        saved["dataset_id"] = (
            args.server_dataset
            if args.server_dataset.startswith("server:")
            else ("server:" + args.server_dataset)
        )
        save()
    if "job_id" not in saved:
        job = call(
            "POST",
            "/v1/flora/dataset-jobs",
            headers={"Idempotency-Key": saved["idempotency_key"]},
            json={
                "dataset_id": saved["dataset_id"],
                "episodes": "all"
                if args.episodes == "all"
                else [int(index) for index in args.episodes.split(",")],
            },
        )
        saved["job_id"] = job["id"]
        save()
    print(f"Dataset job: {saved['job_id']}", flush=True)
    previous = None
    while True:
        job = call("GET", f"/v1/flora/jobs/{saved['job_id']}")
        if job.get("counts") != previous:
            print(job["status"], job.get("counts", {}), flush=True)
            previous = job.get("counts")
        if job["status"] in {"completed", "failed"}:
            break
        time.sleep(10)
    after = -1
    failures = []
    while True:
        page = call("GET", f"/v1/flora/jobs/{saved['job_id']}/episodes?after={after}")
        if not page["jobs"]:
            break
        for child in page["jobs"]:
            target = args.output / f"episode_{child['episode_index']:06d}.json"
            if child["status"] == "completed":
                if not target.exists():
                    result = call("GET", f"/v1/flora/jobs/{child['id']}/result")
                    temporary = target.with_suffix(".tmp")
                    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                    temporary.replace(target)
            else:
                failures.append(child)
                print(
                    f"Episode {child['episode_index']} FAILED [{child['stage']}] "
                    f"(job {child['id']}): {child.get('error') or 'No diagnostic available'}",
                    flush=True,
                )
        after = page["next_after"]
    (args.output / "summary.json").write_text(json.dumps({"job": job, "failures": failures}, indent=2))
    print(f"Results saved to {args.output}; failed episodes: {len(failures)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
