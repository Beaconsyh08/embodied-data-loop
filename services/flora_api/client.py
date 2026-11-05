"""Upload synchronized MP4s, wait asynchronously, and save a FLORA JSON result."""

import argparse
import json
import os
import time
import uuid
from pathlib import Path

import requests


class MultipartUpload:
    """Stream a known-length multipart body without buffering the videos in memory."""

    def __init__(self, instruction, videos):
        self.boundary = uuid.uuid4().hex
        boundary = self.boundary
        self.parts = [
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="instruction"\r\n\r\n'.encode()
                + instruction.encode("utf-8")
                + b"\r\n",
                None,
            )
        ]
        for role, path in videos:
            header = (
                f'--{boundary}\r\nContent-Disposition: form-data; name="{role}_video"; '
                f'filename="{role}.mp4"\r\nContent-Type: video/mp4\r\n\r\n'
            ).encode()
            self.parts.append((header, path))
        self.end = f"--{boundary}--\r\n".encode()
        self.length = len(self.end) + sum(
            len(header) + (path.stat().st_size + 2 if path else 0) for header, path in self.parts
        )

    def __len__(self):
        return self.length

    def __iter__(self):
        for header, path in self.parts:
            yield header
            if path:
                with path.open("rb") as video:
                    while chunk := video.read(1024 * 1024):
                        yield chunk
                yield b"\r\n"
        yield self.end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("FLORA_API_URL"))
    parser.add_argument("--ca", default=os.environ.get("FLORA_API_CA"))
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--main", type=Path)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", type=Path, default=Path("flora-result.json"))
    parser.add_argument("--idempotency-key", default=uuid.uuid4().hex)
    args = parser.parse_args()
    if bool(args.left) != bool(args.right) or not (args.main or args.left):
        parser.error("Provide --main alone, or both --left and --right with optional --main")
    if not args.url:
        parser.error("--url or FLORA_API_URL is required")
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {os.environ['FLORA_API_KEY']}"
    session.verify = args.ca or True
    base = args.url.rstrip("/")
    print(f"Idempotency-Key: {args.idempotency_key}", flush=True)
    body = MultipartUpload(
        args.instruction,
        [
            (role, path)
            for role, path in (("left", args.left), ("right", args.right), ("main", args.main))
            if path
        ],
    )
    response = session.post(
        f"{base}/v1/flora/jobs",
        headers={
            "Idempotency-Key": args.idempotency_key,
            "Content-Type": f"multipart/form-data; boundary={body.boundary}",
        },
        data=body,
        timeout=(10, 3600),
    )
    response.raise_for_status()
    job_id = response.json()["id"]
    print(f"Job: {job_id}", flush=True)
    previous = None
    while True:
        response = session.get(f"{base}/v1/flora/jobs/{job_id}", timeout=30)
        response.raise_for_status()
        job = response.json()
        if job["stage"] != previous:
            print(job["status"], job["stage"], flush=True)
            previous = job["stage"]
        if job["status"] == "failed":
            raise RuntimeError(job["error"])
        if job["status"] == "completed":
            break
        time.sleep(10)
    response = session.get(f"{base}/v1/flora/jobs/{job_id}/result", timeout=60)
    response.raise_for_status()
    args.output.write_text(json.dumps(response.json(), ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
