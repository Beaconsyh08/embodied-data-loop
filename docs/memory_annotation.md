# Text memory annotation (protocol 1)

This source implementation adds **Annotation → Text memory** for registered Agent datasets.
It produces high-level textual `subtask` and `memory` labels. It does not train a model or modify
low-level pi0.5, original LeRobot files, or saved FLORA results. Model checks and a completed Job
are drafts, never human approval. Installed release evidence and real-provider pilot acceptance
are recorded separately from this document.

## Source and causal input contract

For the explicitly authorized EE3 cleanup, `scripts/repair_ee3_caption_coverage.py` is a
bounded operational CLI, separate from memory annotation and training. It preserves exact,
non-overlapping original intervals and asks the configured multimodal Qwen to recaption gaps
and intervals whose boundaries must change. Input contains only the recorded task and up to
eight chronological target-frame mosaics (main/left/right), never original captions, rewards,
state or action. New artifacts retain the legacy `flora-1` container for readers, but label
the model `FLORA + <Qwen model> coverage repair` and carry
`flora_qwen_coverage_repair/v1` provenance. Their `native_segments` field holds explicitly
program-partitioned intervals, not new TempSeg predictions. Original annotations, original
segmentation and original reward provenance are retained; these are unapproved draft sources.
Single-frame targets and model uncertainty require human review. Structural coverage alone
does not establish semantic correctness or authorize training. The CLI has exact-input
receipts, source verification before/after execution, bounded retries, immutable completed
packages and an output-directory `CANCEL` marker. Use a fresh output revision when its source
manifest, prompt or provider changes. It does not create memory Jobs or approve labels.

Select one complete FLORA run for each full episode. Preflight verifies the dataset key, episode,
frame count, FPS, ordered timestamps, source file hashes and FLORA source fingerprint. Native
caption gaps or overlaps are ineligible. A legacy run without a matching source fingerprint is
not silently accepted. Camera names must resolve unambiguously to **main, left wrist, right wrist**.
Specify the total task independently of captions and confirm original recording lineage.

Checkpoints match `mem_tools`: first/last frame, 1 Hz, and FLORA segment starts, deduplicated in
frame order. Human reviewers can insert additional cutoffs. History contains at most eight
samples from `[t − 2 seconds, t)`, at 4 Hz; current `t` is supplied separately. Sampling floors
timestamps, clips to the existing episode prefix, and never pads from the future. Video decoding
uses PTS at or before each target, with a maximum gap of 1.01 frame periods. Review mosaics reject
positive decode offsets and cross-camera decode-error spread above 50 ms. Image-backed sources
use their shared row timestamp; this cannot establish physical synchronization absent sensor metadata.

Every image is an ordered horizontal mosaic with three 224 px letterboxed panels, exactly matching
the existing training-export image layout. The downstream model processor may resize that mosaic;
its final visual resolution still needs real-data acceptance.

Each cutoff makes two independent Qwen requests:

1. Draft: explicit task, current mosaic, chronological past mosaics, earlier observed facts,
   previous memory and previous subtask. Image/frame order is explicit. Facts retain IDs,
   confirmation frames, certainty and camera evidence.
2. Student-input check: **only** task, current mosaic, previous memory and the proposed
   `memory/subtask`. It receives no history images, fact ledger, earlier subtask or FLORA text.

Both payloads are built from an allowlist. Future captions, whole-episode summary/Reward,
state/action and file paths are excluded. FLORA text is an auxiliary human-review candidate;
even its auxiliary panel excludes spans ending after the selected cutoff.

Strict JSON/schema/causal validation failures are `invalid_response`, transport failures are
`request_error`, and uncertainty or unsupported labels are `needs_review`. The episode stops
at that cutoff; later memory is not generated. Original responses are retained rather than
silently repaired. Each request records its input hash, response (including provider usage),
attempt durations and sanitized transport failures. Retryable network errors, HTTP 429 and 5xx
use bounded exponential backoff, at most three attempts per request. Cancellation is checked
between requests, during backoff, image export and artifact publication.

## Execution and durability

`memory.annotate` and `memory.export` are registered operators using the existing Job lease,
attempt, cancellation, staging, checksum validation and final-acknowledgement machinery. They
require their dedicated authenticated form; generic Pipeline/assistant execution cannot bypass
source preparation or approval. One annotation Job covers the selected dataset batch, with bounded
episode workers inside the existing Agent execution. Set the Agent-only
`DATA_PLATFORM_MEMORY_MAX_PARALLEL_EPISODES` (default 3, range 1–32); the effective limit
also reserves at least 8 GiB of the execution memory budget per worker. Each episode keeps its own sequential
checkpoint chain, receipts, review identity and pause state. A pending or failed episode does not
stop other episodes. Dataset cancellation stops queued episodes and checks active requests at the
existing cancellation boundaries. One idempotent submission creates one Job. Previously submitted
single-episode Jobs remain readable and are never resubmitted during an upgrade.

The owning Agent must advertise `job_protocol=2`, `memory_annotation_protocol=1`, `memory_batch_protocol=1` for new dataset batches, the operation
version `1.0.0`, and, for annotation, `memory_qwen_configured=true`. Configure its existing Qwen
credential and `DATA_PLATFORM_MEMORY_MODEL` (default `qwen3-vl-plus`) in the protected Agent
environment. No credential or model endpoint is accepted from the browser. Endpoint/model
configuration is hashed into the source pin; set `DATA_PLATFORM_MEMORY_MODEL_REVISION` when a
provider deployment changes under the same model alias. Opaque upstream alias changes cannot
be detected from the alias alone; actual response metadata remains in receipts.

Raw RGB bundles, generation receipts and exports stay beneath the registered Viewer output's
parent, in `memory/<location_id>/`, checked against Agent writable roots and outside the source.
The supervisor preserves that durable location while separately staging attempt uploads. Only
sampled review mosaics, draft records, receipts and the selected FLORA copy are published to the
Server through existing operator-artifact transport. It does not retransmit original datasets.

Identical input prefixes and saved successful responses are reused after interruption. Data,
model/configuration, prompt or preceding memory changes invalidate affected inputs. A source
fingerprint is checked before and after inference. Successful exports are immutable and can
resume publication from verified package hashes without rerunning model inference.

## Human Review and permissions

Where a frozen training snapshot has been installed, **Text memory → View frozen training
annotations** opens a read-only review of the actual model-generated training ledger. Select an
episode, play or seek the matching three-view video, and use checkpoint buttons to inspect memory,
subtask and facts. The warning button seeks to the stopped checkpoint and displays the recorded
model-check issues. A stopped checkpoint and later frames have no accepted labels. The page does
not edit, approve, regenerate or export annotations and model checks are not human review.
In Temporal Caption Review, a source-matched episode also shows memory and subtask beside captions
at the playback position. The panel switches to the saved review problems at the stopped frame;
its **Full Review** link preserves the selected episode. If the selected caption source cannot be
matched, the memory panel is hidden rather than attaching labels to another recording. Desktop
panes sit side by side; smaller screens stack them.

The administrator prepares a snapshot with `scripts/build_memory_training_review.py`, supplying the
exact frozen ledger, coverage, original offline-verification file, selection manifest, bundle
`episodes.jsonl` and bundle `manifest.json`. It checks the verification and bundle hashes, record
payloads, checked prefixes, dataset version, camera order and caption identities before writing a
new output file. Install the verified snapshot as `snapshot.json` under the lifecycle store's
`memory-training-reviews/<location_id>/` directory using a controlled, atomic publication; retain
the input hashes, grant only the Server service account read access, audit the publication and never
overwrite a different snapshot. The Server checks the registered dataset
scope on each request and verifies the bound caption hash, source fingerprint, frame count and FPS
before returning an episode. Video stays in the existing Temporal Caption store and follows its
access checks. Snapshot creation is an offline preparation step, not a Job or human approval.

Viewer is read-only. Operator, Data Manager and Admin use the existing dataset access checks;
permission is checked at API access, submission and Agent claim. Review identities come from
logged-in user IDs, never a browser-supplied reviewer name.

Review shows current views, past-only playback, prior memory, causal fact ledger, proposed memory,
subtask and issues. Correct facts/labels with a reason, mark a checkpoint pending, or insert an
earlier checkpoint. Edits use an expected revision; concurrent stale writes return 409. Each
revision is immutable in the existing lifecycle repository. Editing clears episode approval and
marks later entries stale. Regeneration creates a new Job-linked revision with explicit parent
revision references, preserving the old records. Editing an input revision during execution
prevents its result from being acknowledged as a current publication.

Episode approval requires all scheduled checkpoints to be usable and current, with explicit
source, goal and causal-evidence confirmation. Test data additionally requires a distinct logged-in
reviewer and explicit adjudication. These are human attestations, not automatic verification of
semantic correctness. Actions use the existing operation audit system.

## Pilot and export

Audit **every registered episode in the same audit batch**, including failed or missing FLORA
results. Freeze from all eligible episodes, using seed 42 and duration strata. Select at most ten;
report a shortfall without substituting a dataset or admitting failed annotations. Recording
lineage must be supplied before freezing. Whole groups are assigned toward train/calibration/
validation/test = 6/1/2/1; preserve groups and report actual counts when the target is impossible.
This ten-episode partition checks the workflow, not statistical model performance.

Export preparation binds the exact approved revisions and frozen split. The Agent checks sources
again and writes a new package containing:

- `bundle/`: unchanged `mem_dataset_export/v1` interface, original RGB images and source metadata;
- `ledger.json`: `mem_ledger/v1`, with authenticated review attestations and canonical payload hashes;
- `splits.json`: fixed assignments bound to the exported bundle hash;
- `audit.json` and `audit/`: review revisions, history and student-input checks, request receipts;
- `package.json`: immutable package checksums. The published receipt reports its Agent-side location.

The platform process does not import the training environment. On the **same data host**, use
an isolated high-level-policy environment to validate the package (paths are placeholders):

```bash
python -m mem_tools inspect --bundle /data/memory/export/bundle
python -m mem_tools build --bundle /data/memory/export/bundle \
  --ledger /data/memory/export/ledger.json --splits /data/memory/export/splits.json \
  --output /data/memory/built
python -m mem_tools training-export --bundle /data/memory/export/bundle \
  --dataset /data/memory/built --condition summary --output /data/memory/training
```

The existing source-confirmation, complete gold coverage, training audit and independent-test
checks remain in force. Trainer data contains only train and validation and only
`image/task/previous_memory/prompt/answer/sample_id`. History clips, FLORA, facts and audit
attachments are never additional student inputs. Until these commands and required human reviews
pass, report `draft` or `awaiting_high_level_validation`, not training-ready. No training command
is part of this workflow.

## Acceptance and rollout

Synthetic regressions cover causal payload isolation, future PTS rejection, short prefixes,
uncertainty, invalid schemas, retry/cancellation, resume, source binding, revision conflicts,
scoped reads/submissions/claims, old execution credentials and immutable export. Real Chrome
acceptance covers selection/preflight, batch submission, causal playback, correction, independent
review, regeneration, export submission and 390 px layout. These tests use isolated synthetic
accounts/data and a fixed-response teacher; they do not demonstrate real Qwen label quality.

Run focused tests from the exact source worktree:

```bash
.venv/bin/python -m pytest tests/datasets/test_memory_annotation.py \
  tests/datasets/test_memory_routes.py tests/datasets/test_memory_export.py \
  tests/datasets/test_memory_browser.py -q
```

For cross-repository acceptance, set `MEM_TOOLS_PYTHON` to the high-level environment interpreter
and `MEM_TOOLS_ROOT` to its checkout, then run `test_memory_export.py`. Its synthetic ten-episode
case runs the existing `inspect/build/training-export` commands and checks the six trainer fields.
No high-level source modification or training-runtime dependency is required.

Upgrade Server and Agent together; older Agents remain unavailable for memory submissions.
The additive lifecycle records need no SQL schema migration. Preserve retained executors and
active attempts during rolling deployment. Publish and accept the independently versioned CLI
with the platform, retaining immutable old objects and advancing `latest` only after acceptance.
CLI `/memory` remains assistant account memory, distinct from dataset **Text memory**.

The EE3 real-provider pilot requires a separate explicit start confirmation. Before it, verify
its registered development location, source-matched complete FLORA runs, recording lineage,
Qwen configuration, checkpoint/request counts and disk space. Development software acceptance
and synthetic exports do not satisfy that real-provider gate. Production promotion continues
through the existing approval process.

Dataset batch compatibility: the Server rejects batch submission to older Agents before creating a Job.
The MEM export contracts and CLI 0.3.30 remain unchanged. Batch review identities resolve to
episode-specific artifacts under the parent Job; approval, revision checks and independent test
review apply separately to each episode. Regeneration creates a new batch for the affected episode.

### Opt-in AI review for a model-only training draft

`annotate_ee3_model_trial.py --ai-review-plan PLAN` records camera observations and per-claim assessments from a pinned cutoff. The plan binds each previous review by SHA-256, original TRAIN split, raw-data hash, annotation hash and checkpoint schedule. Use a new output directory and copy the prior request receipts to reuse the unchanged prefix. No existing review or source data is modified.

The current mosaic is split losslessly into main/left-wrist/right-wrist panels; the independent reviewer receives exactly these student pixels, task, previous memory and the proposed output. It receives no clip, fact ledger or prior rejection comments. The teacher may still use the causal history window. Invalid responses or unsupported claims remain explicit errors/needs_review; bounded revisions cannot bypass a failed review. Saved receipts support exact offline replay using `verify_ee3_model_trial.py`.

This is explicitly **AI-reviewed model-only experimental data**, with `human_approvals=0` and draft status. It does not satisfy authenticated human approval or independent test review. Calibration, validation and test splits are not changed. Training retains the existing explicit model-only opt-in and export validation. Inference comparisons are delivered as browsable per-checkpoint pages.
