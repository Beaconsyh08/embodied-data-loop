# Architecture upgrade completion review

This review covers the operator/pipeline implementation based on 当前公开源码, the source-cleanup baseline
当前公开源码, and the subsequent legacy console separation. It distinguishes delivered contracts from
hardware-dependent acceptance. Installed release identity and online evidence belong to the immutable
release directory and must be checked independently.

| Original requirement | Implementation and review evidence | Status |
| --- | --- | --- |
| Unified versioned operators, parameters, resources and lazy availability | `operators/`; `test_operators` checks discovery without model imports, pinned versions and generic dispatch | Implemented |
| Map all existing processing entry families; keep privileged mutations separate | Migration inventory in `data_platform_operators.md`; registered background-entry coverage includes compatibility submission services | Implemented with explicit compatibility adapters |
| Shared submission, stable execution snapshots, existing Job/Attempt admission | `processing/commands.py`, `console_submissions.py`, `service.py`, local executor and Agent; local execution and preprocessing route suites | Implemented |
| Sequential versioned pipelines, upstream dataset/evidence references, APIs | Planner/service/routes; immutable revision and input identity tests | Implemented |
| Durable coordination, review, cancellation, retry and publication acknowledgement | Coordinator/systemd plus concurrent tick, maintenance, restart, stale decision and retained-output tests | Implemented; representative local/remote flows accepted |
| Selective laundry absorption | Optional EEF spike strategy, C3 reuse, pose interpolation bound to reviewed recipe; v2/v3 and shared-shard preservation tests | Implemented; representative remote v3 repair accepted |
| Source modification permissions and compatibility | Original endpoint names and global guards retained; source operations excluded from automatic chains | Implemented |
| Extract registration, caches, submission and result association from old entry point | `services/`, `processing/console_submissions.py`, thin compatibility route modules; one shared runtime per server | Completed in console separation |
| Split browsing, operator configuration, workflow editing and run state | `templates/console/`, shared mounted pipeline editor, preserved form bindings; composed-script and browser tests | Completed in console separation |
| Discard stale responses on account/environment/dataset changes | Scoped read generations; late response/error tests; embedded editor disposal on context switch | Implemented |
| Confirmed missing-source retirement and owned-cache reclamation | Existing source cleanup protocol, leases and ownership proofs retained; catalog removal clears shared runtime caches | Implemented; local/remote acceptance required for each release |
| AI capability description, preview and execution interfaces | Operator and pipeline APIs | Implemented |
| MCP, conversational UI and autonomous planning | Explicitly excluded from this round by the original plan | Deferred by design |
| Every model/media/conversion/build operator exercised on its real execution runtime | Contract and compatibility suites exist; some model backends/Embedding and local MCAP runtime were unavailable at the `.67` baseline | Full real-runtime acceptance remains outstanding |
| Offline FLORA caption/reward pipeline | Native one-camera and existing two/three-camera layouts; statistics-independent inference metadata; existing caption Job/Attempt and publication contracts; isolated GPU environments; per-episode GPU pool leases (up to eight), reusable content hashes, streaming video preparation and raw-window disagreement review | Implemented; native A100 and local browser acceptance, online rollout separate |
| Standalone FLORA HTTP API | Separate HTTPS/key authentication, private SQLite dataset/episode jobs, streaming checksummed uploads, and retained adapter dependencies; no fixed media duration or byte-size cap, while resource and stage timeouts apply; shares existing GPU locks without creating platform Jobs | Implemented; independent deployment and real-model acceptance required; see [API contract](flora_http_api.md) |
| Background cross-task experience | Immutable plans/commands/final results cross-checked; comparable recipes and observed recovery changes include counterexamples; exact source quotes and rule revisions checked; explicit memory retained | Implemented; live usefulness acceptance required |
| Usage learning | `ai_tools/learning`, typed source references, private analysis ledger and independent proposal lineage; explicit consent, zero-default budget, manual strategy activation | Implemented; offline acceptance and online release evidence remain separate |
| General assistant chat | Account-only `general/account` scope with empty location; same durable Runner and account checks; only context and operator descriptions, no dataset lookup or execution tools | Implemented; dataset contexts remain separate and confirmations unchanged |
| Caption result navigation | Permission-scoped published-result inventory plus Agent scan-time file counts; searchable dataset switching, explicit episode/result URL identity and offline labels; no remote reads or jobs on catalog requests | Implemented; additive `caption_output_runs` location metadata requires updated Agent discovery; no database or Job protocol migration |
| Mixed evaluation Viewer | Opt-in per-episode camera and raw-signal metadata; Server and Agent capability gate preparation and claim; existing Viewer cache publication and permissions remain in force | Source implementation; deployment and online acceptance pending; direct FLORA jobs and general training are unsupported |
| Standalone caption assistant | Review/Annotation reuse the shared sidebar; registered dataset/episode references pass existing authorization, local aliases resolve server-side; review text, scores and video are not injected | Implemented; no new execution tool or Agent protocol |
| Text memory annotation | `memory/` pure causal contracts, Agent-only media/model execution, existing Jobs/artifact transport, lifecycle revision records and fixed MEM exports; source-bound read-only training snapshots reuse caption video and follow playback in Caption Review | Synthetic and browser acceptance; real-provider EE3 pilot and installed rollout are separate gates |
| Production promotion | Same immutable package requires development evidence and the existing production approval mechanism | Separate release step |

## Scope of the console completion

Compatibility adapters are intentional: preserving external endpoints does not require keeping registration,
cache management or worker submission inside `viewer.run_server`. Those responsibilities now delegate to
services. Episode rendering and privileged legacy source-edit implementations remain feature-specific and
retain their existing protections; the plan does not require replacing their algorithms.

Verification covers the original API/CLI tests, shared registry and result callbacks outside a Flask request,
independent console instances, composed JavaScript state, asynchronous response isolation, real-browser
form/navigation behavior, the embedded and standalone editors, mobile width and Viewer role restrictions.
The release validation set includes these regression modules. Optional browser tests additionally require
an installed Chrome and Playwright; their explicit run must be recorded with the release acceptance evidence.

The remaining model-runtime acceptance is not a missing console split. Passing CPU fixtures or advertising
an unavailable operator does not establish real model inference acceptance, and development installation
does not constitute production promotion.

Installed acceptance also exercises the original local cache-preparation endpoint. Local command snapshots
retain requested parameters, while execution binds outputs to supervisor staging and uses its worker limits.
Viewer artifacts enter the normal upload/acknowledgement protocol, and custom dataset output paths are
included in admission and path locks. Real-process regressions cover cache publication and local v2.1/v3.0
dataset transforms, checking that the final destination is absent until publication and source bytes remain
unchanged. A completed attempt awaiting upload is finalized from its retained output rather than recomputed.

## Shared strategy experiment extension (source implementation)

An optional administrator-controlled strategy service reuses the operator/Pipeline/Job architecture;
its worker generates constrained recipes, measures independent synthetic cases and requires human
activation. Request-aware selection binds at most two templates through the existing preview; ambiguity
falls back to ordinary planning. Per-case diagnostics and stratified development gates precede holdout;
exact template-set duplicates are not independent candidates. Five additive control tables plus nullable Pipeline attribution retain versions and evidence.
No new Agent protocol, model training, private-memory sharing or automatic business execution is added.
Deployment and real-provider effectiveness require separate acceptance; see [the contract](data_platform_strategies.md).
