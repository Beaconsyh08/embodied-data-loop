# Standalone FLORA HTTP API

This independently versioned service exposes existing offline FLORA models without an SSH account.
It does not replace or restart the Data Platform Server or Agent, change their protocol, or publish
platform Jobs. Consequently this service installation is not a platform/CLI upgrade. The existing
platform web confirmation and account scopes remain unchanged.

## Authentication and client setup

Every inference/data endpoint requires `Authorization: Bearer <API key>`. Keys are random, stored as
SHA256 digests, revocable without restart, and scoped to one owner. Status, uploads, and results are
private to that owner. Server-side datasets require explicit aliases granted to that key; arbitrary
filesystem paths and remote URLs are not accepted. `/healthz` is public and reports worker readiness.

The operator supplies the HTTPS origin, a key in a private file, and the service's public CA certificate.
Clients verify that certificate; do not use `verify=False`. Install `requests`, then set:

```bash
export FLORA_API_URL=https://YOUR_SERVER:8448
export FLORA_API_CA=/path/to/flora-api-ca.pem
export FLORA_API_KEY="$(cat /path/to/private-api-key)"
```

For a single standalone MP4, use the official native single-view path:

```bash
python services/flora_api/client.py \
  --main camera.mp4 --instruction 'Move the bowl to the right' \
  --output flora-result.json
```

For synchronized wrist MP4s (both left/right required, scene optional):

```bash
python services/flora_api/client.py \
  --left left.mp4 --right right.mp4 --instruction 'Fold and store clothes' \
  --output flora-result.json
```

The multipart API accepts `main_video` alone, or `left_video` plus `right_video` with optional
`main_video`. Missing wrist partners and duplicate file fields are rejected. Dataset jobs automatically
detect one camera of any name; two/three camera layouts retain the existing wrist naming rules.
Single views are not duplicated or padded. Inference does not require training statistics files.

Standalone videos must have matching frame counts/FPS, constant timestamps, and matching wrist heights.
API 1.0.6 removes fixed video upload byte, maximum duration, frame count and resolution caps. Videos still
need at least two frames, 1–60 FPS, and dimensions of at least 16x16. Validation runs asynchronously and
malformed videos fail the job. The client streams multipart videos with a Content-Length; existing clients
that buffer uploads in memory should upgrade. Model inputs remain visual-only.

## Whole datasets, from either location

```bash
# Upload a local LeRobot v2.1/v3.0 directory once, then process all episodes.
python services/flora_api/dataset_client.py \
  --dataset /path/to/lerobot-dataset --output ./flora-results

# Use an authorized dataset already on the A100; no data upload.
python services/flora_api/dataset_client.py \
  --server-dataset DATASET_ALIAS --output ./flora-results

# Optional subset; omit --episodes to process the whole dataset.
python services/flora_api/dataset_client.py \
  --server-dataset DATASET_ALIAS --episodes 0,1,2,3,4,5,6,7 --output ./flora-eight
```

The client streams each file without loading the dataset into RAM, verifies SHA256 at upload, and stores
an upload/submission checkpoint in the output directory. Repeating the same command resumes missing
files and polls the original job; it never repeats a completed inference. Use a new output directory
to explicitly start a new run. Uploaded datasets become immutable after completion. API 1.0.6 removes
the fixed per-file and total dataset byte caps and the adapter's episode duration/frame-count caps.
Manifests and episode lists remain limited to 100000 entries and JSON requests to 32 MiB. Uploads require
Content-Length, exact file sizes and SHA256 checks. Dataset upload admission checks sufficient disk space
with 100 GiB reserved headroom; single-video admission checks that at least 100 GiB is free before upload.
Source metadata, task instructions, and camera roles must satisfy the existing FLORA contract.
Results are written per episode plus `summary.json`.

## HTTP contract

| Endpoint | Behavior |
| --- | --- |
| `POST /v1/flora/jobs` | Multipart `main_video` alone, or both wrist videos with optional `main_video`, plus `instruction`; 202 with job `id` |
| `GET /v1/flora/jobs/{id}` | Status, stage, timestamps; dataset jobs include per-state counts |
| `GET /v1/flora/jobs/{id}/result` | Completed episode JSON, otherwise 409 |
| `GET /v1/flora/jobs/{id}/episodes?after=-1` | Up to 100 children, ordered by episode index; `next_after` cursor |
| `GET /v1/flora/datasets` | Authorized existing server dataset aliases |
| `POST /v1/flora/datasets` | JSON `files: [{path,size,sha256}]`; creates private upload ID |
| `PUT /v1/flora/datasets/{id}/files/{path}` | Stream exact file bytes with Content-Length; checksum validation |
| `GET /v1/flora/datasets/{id}` | Missing files and sealed state for resumption |
| `POST /v1/flora/datasets/{id}/complete` | Verify receipts and seal immutable dataset; send `{}` |
| `POST /v1/flora/dataset-jobs` | JSON `dataset_id`, optional `episodes: "all"` or integer list; 202 parent ID |

Dataset submission requires `Idempotency-Key`; single-episode submission supports it. Reusing a key
returns the original job, even with a different body; never reuse a key for a different request.
An episode's states are `uploading → queued → running → completed/failed`. A parent finishes only after
all children settle; any failed child makes the parent failed, while successful results remain available.

Since API 1.0.4, a failed episode retains its processing stage and a bounded, actionable `error`.
For example, a missing episode Parquet file returns `stage: "preparing"` and
`error: "FileNotFoundError: Missing dataset file: data/chunk-000/episode_000000.parquet"`.
API 1.0.5 does not require training statistics, so missing `episodes_stats.jsonl` no longer causes this error. Dataset parent status
includes the failed count and first episode error; `/episodes` and `summary.json` retain every failure.
The dataset client prints each failed episode's index, stage, job ID, and error before exiting with code 1.
Completed results still download. Dataset-relative missing filenames and input validation messages
are exposed; service paths, credentials and raw model/subprocess tracebacks remain private to operator logs.
An existing failed submission remains failed on client restart; resumption does not retry inference.

## Parallelism and platform coexistence

The worker dispatches up to eight independent episode pipelines to eight physical GPUs. It uses the
**same existing `gpu-N.lock` files and FLORA allocator as the platform**, checking free GPU memory before
allocation. A GPU stays exclusively locked across segmentation, captioning, terminal and dense reward.
API allocation tries GPU indices in reverse order to reduce collisions with platform allocation. A
busy GPU is skipped; if all are busy the task waits. This improves dataset throughput, not necessarily
single-episode latency: one episode still executes its stages on one GPU. Models retain native precision.

Segmentation and captioning split long videos into bounded chunks; rewards still process the entire
episode. Removing fixed input limits does not guarantee arbitrary workloads will complete: available disk,
RAM, GPU memory and the configured FLORA `stage_timeout_s` still apply, and long outputs can be large.
The probe and normalization have no fixed per-video deadline. Two simultaneous CPU preparations and bounded
encoder/model CPU threads limit preprocessing pressure.
The source fingerprint cache avoids rehashing unchanged dataset bytes for every episode.
Model verification also shares validated receipts between workers. An isolated copy of the caption
runtime wraps the original validator: every call still checks the manifest and file identity including
ctime; changed files receive a complete SHA256 check. It never changes weights, precision, or inference
inputs. The platform's original model code is unchanged.
API systemd
units have their own process groups, lower CPU/IO priority, memory limit, and dedicated writable state.
They cannot stop platform workers or write registered source datasets. Running both systems shares
finite GPU/CPU/disk bandwidth, so identical latency under full load is not guaranteed.

The platform already supports **Temporal Caption → FLORA → All episodes**. Eight-card operation also
requires its Agent slots, Caption running limit, node/user quotas, and configured GPU pool to permit
eight concurrent tasks. Browser requests may be submitted one by one while GPU execution is parallel.
The existing platform submission behavior still applies: leaving the page stops further submissions,
while accepted jobs continue. The standalone API accepts a complete dataset job server-side, so its
accepted children continue after the caller disconnects.

## Installation and operation

From a validated, committed source snapshot, unpack `services/flora_api` under a new immutable service
release directory and run as root (all arguments are absolute paths):

```bash
bash /opt/flora-api/releases/NEW_VERSION/services/flora_api/install.sh \
  /opt/flora-api/releases/NEW_VERSION \
  /opt/data-platform-agent/prod/releases/EXISTING_IMMUTABLE_RELEASE \
  /etc/data-platform/prod/flora-runtime-pool.json \
  PUBLIC_IP /DATA/disk0/flora-api
```

The isolated venv pins Gunicorn 23.0.0 and imports an independently retained, read-only hard-link snapshot
of the exact existing platform adapter dependencies through a `.pth` file. Removing an old platform
release does not remove this snapshot. Neither service may modify installed dependency files in place.
For API 1.0.6, use the matching updated platform adapter source; API service files alone do not remove
the shared adapter limits. Verify that the worker imports the intended adapter before switching services.
For API 1.0.5 single-view support, use the updated platform adapter (Server/Agent V0.3.28 or later).
An independently bundled adapter is also supported: retain the full `lerobot/` tree from the exact
validated source commit beside `services/`, put that immutable release directory before the retained
dependencies in the `.pth` file, and verify the imported adapter path before switching services.
Record both the adapter source commit and retained dependency release; an older library snapshot alone
does not prove that the new inference adapter is loaded.
The installer grants only model
read/traversal and shared GPU-lock access to `flora-api`; it does not change platform configuration.
The independent private CA and certificate cover the supplied IP. Distribute only `ca.pem`, never keys.

Register server dataset aliases in `/etc/flora-api/datasets.json`, grant the service read/traversal ACLs
for those datasets, and issue a distinct key per consumer:

```bash
cd /opt/flora-api/releases/NEW_VERSION
.venv/bin/python -m services.flora_api.admin \
  --keys /etc/flora-api/keys.json --id colleague \
  --dataset DATASET_ALIAS --output /root/colleague-api-key
systemctl enable --now flora-api-worker.service flora-api-web.service
```

The secret file is mode 0600. Send it through a private channel. Revoke by invoking the same admin command
with `--id colleague --revoke` (omit output/dataset). Claims recheck revocation and dataset authorization;
already running inference is not cancelled. Only SHA256 digests remain in the service key registry.

SQLite WAL persists parent/child jobs and atomic claims. A singleton worker lock prevents duplicate
dispatchers. Restart recovery publishes only outputs with a validated completion receipt; interrupted
inference is marked failed without automatic recomputation. Queued jobs survive. GPU processes are
contained by systemd and the existing adapter's subprocess/parent-death cleanup.

Inspect `systemctl status flora-api-{web,worker}` and their journal. Monitor free disk and certificate
expiry. Uploads/results are retained until operator cleanup; there is no automatic deletion. Inference
refuses new work below 100 GiB free. An operator may remove completed/failed job directories and unused
uploads only after confirming no active child references them; retain the SQLite audit state. Never
delete shared GPU lock files. Stop only the API units for API maintenance.

Acceptance must separately verify TLS from outside the host, wrong/missing/revoked keys, owner isolation,
upload resumption and checksums, both dataset source modes, real FLORA results, concurrent distinct GPUs,
and unchanged platform Agent PID/health. Tests alone are not proof of a deployed service.

## Optional unified GPU gateway

The separately installed FLORA Router 0.1.0 exposes the same authenticated standalone API contract
at its own HTTPS origin. Configure client URL, CA and authorized key for that origin; IDs and upload
checkpoints belong to the origin that created them and cannot be moved between origins. Existing
standalone API services, their jobs and the Data Platform Agent continue operating independently.

The router observes each configured backend's GPU free memory, process occupancy, shared FLORA GPU
locks, worker heartbeat and pending work. It reserves one eligible slot and records the backend before
forwarding. A job uses one GPU; several jobs may run in parallel when eligible slots exist. If no slot
is eligible it waits; unavailable telemetry is treated as unavailable capacity. Reservations account
for the backend's existing pending jobs. A GPU used by software outside the shared lock protocol could
become occupied after observation; capacity observations cannot prevent that external race.

The original backend assignment and remote job ID remain durable across gateway restarts and temporary
network failures. An uncertain submission is retried only on the same backend with a stable idempotency
key. A missing previously assigned job is not automatically resubmitted elsewhere. Dataset inputs are
prepared at the gateway and their camera streams are forwarded to the selected backend; large videos
therefore consume network bandwidth and backend staging storage. Source fingerprints are rechecked
before publishing episode results. This gateway does not transfer model weights.

A bounded legacy bridge can dispatch selected *still-queued* dataset children from the original API
queue to the H100 backend without changing their job/owner IDs or resubmitting completed work. It makes
one atomic claim at a time, retains the remote job ID and idempotency key across reconnects, verifies
source fingerprints and timing before writing the existing result and receipt, and uses the normal
GPU capacity probe and shared locks. Already-running legacy attempts stay with their original worker.
The temporary original-worker recovery guard excludes only bridge-owned attempts during a worker
restart; it leaves the original recovery behavior for every other job. The bridge exits after the
explicitly selected parents have no further queued children. Its cleanup removes the guard only after
the bridge is inactive, no selected children remain queued or bridge-owned, and the guard's checksum
matches its deployment receipt. Model weights are never copied.

Deploy it as isolated services and state, with restricted backend credentials, verified TLS, private
backend listeners, shared GPU lock directories, and a read-only capacity channel. The isolated worker
needs write access to the configured model-verification cache and the inference runtime home cache;
retain their existing locking when sharing them with another worker. Do not restart or replace existing
workers to deploy it. Validate both standalone video and dataset jobs, backend
unavailability, interrupted submissions, multi-job reservations, GPU activity, and result retrieval
before directing clients to it. Access keys and dataset aliases stay owner-scoped. Revoking an owner
prevents new claims and does not cancel an already assigned job. No Data Platform protocol or CLI
change is required for this standalone service.
