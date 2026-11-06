"""Offline regressions for composed strategies; provider calls are forbidden."""

# ruff: noqa: F811

import copy
import json
import time
from dataclasses import asdict, replace

import pytest

from lerobot.data_platform.ai_tools.strategies import Strategies
from lerobot.data_platform.ai_tools.strategy_contracts import Measurement, validate_strategy
from lerobot.data_platform.ai_tools.strategy_diagnostics import PlanningTrace, normalized_plan
from lerobot.data_platform.ai_tools.strategy_evaluation import (
    assess_detailed,
    initial_cases,
    stratified_gate,
)
from lerobot.data_platform.ai_tools.strategy_models import StrategyExperiment
from lerobot.data_platform.ai_tools.strategy_selection import bind_template, request_intent, semantic_digest
from lerobot.data_platform.ai_tools.strategy_worker import Experiment, tick
from lerobot.data_platform.operators import get_operator
from lerobot.data_platform.runtime.jobs import JobConflictError
from tests.datasets.test_ai_assistant import assistant, call, draft  # noqa: F401
from tests.datasets.test_ai_assistant_runtime import processing_setup  # noqa: F401
from tests.datasets.test_ai_strategies import configured, version  # noqa: F401


@pytest.fixture(autouse=True)
def no_provider(monkeypatch):
    from lerobot.data_platform.ai_tools.qwen import QwenClient

    def forbidden(*args, **kwargs):
        raise AssertionError("This regression suite must never call a model provider")

    monkeypatch.setattr(QwenClient, "post_chat_completion", forbidden)


def chain(*operators):
    steps = []
    source = "source"
    for op in ("curation.snapshot", *operators):
        steps.append(
            {
                "operator_id": op,
                "version": get_operator(op).version,
                "input": source,
                "artifacts": [] if not steps else [0],
                "parameters": {},
            }
        )
        if get_operator(op).output_type == "dataset":
            source = len(steps) - 1
    return {"formats": ["v2.1", "v3.0"], "required_features": [], "steps": steps}


@pytest.fixture
def capable(configured):
    from lerobot.data_platform.control.store import ControlPlaneNode

    strategies, _, _, config = configured
    with strategies.store.sessions.begin() as session:
        node = session.get(ControlPlaneNode, config["node_id"])
        node.capabilities = {**node.capabilities, "quality_checks": ["s3"]}
    return configured


def leased(configured):
    strategies, owner, suite, _ = configured
    key = strategies.start(owner, strategies.dashboard(owner)["revision"], suite)["experiment_id"]
    with strategies.store.sessions.begin() as session:
        row = session.get(StrategyExperiment, key)
        row.lease_owner, row.lease_until = "offline", time.time() + 300
    return Experiment(strategies.assistant, key, "offline")


@pytest.mark.parametrize("case_id", ["convert_subset", "convert_check"])
def test_conversion_binding_uses_derived_input_and_original_snapshot(capable, case_id):
    configured = capable
    strategies, owner, _, config = configured
    case = next(c for c in initial_cases()["holdout"] if c["case_id"] == case_id)
    template = chain(*case["operations"])
    validate_strategy({"schema_version": 1, "templates": [template]})
    selected = strategies.for_request(
        owner,
        config["locations"]["v2.1"],
        {"version_id": "offline", "digest": "fixture", "templates": [template]},
        case["request"],
    )
    assert selected and len(selected["templates"]) == 1
    definition = selected["templates"][0]["definition"]
    steps = definition["steps"]
    assert steps[0]["input"] == "source"
    assert steps[-1]["input"] == steps[1]["step_id"]
    preview = strategies.manager.preview(owner, config["locations"]["v2.1"], definition)
    assert assess_detailed(case, {"proposal": preview}) == (True, True, "correct")
    wrong = copy.deepcopy(preview)
    wrong["definition"]["steps"][-1]["input"] = "source"
    assert assess_detailed(case, {"proposal": wrong}) == (False, True, "input_reference")
    assert strategies.store.list_jobs() == []


def test_offline_ablation_baseline_first_five_and_matched_binding(capable):
    """A structural ablation, not a claim about LLM accuracy or latency."""
    strategies, owner, _, config = capable
    legacy = [
        chain("preprocess.split"),
        chain("curation.quality", "preprocess.split"),
        chain("preprocess.split", "viewer.prepare"),
        chain("curation.quality"),
        chain("preprocess.convert_v3"),
    ]
    for case_id in ("convert_subset", "convert_check"):
        case = next(c for c in initial_cases()["holdout"] if c["case_id"] == case_id)
        templates = legacy + [chain(*case["operations"])]
        intent = request_intent(case["request"])
        # Baseline has no shared recipes; legacy first-five has no exact composed recipe.
        matches = []
        for template in templates[:5]:
            with pytest.raises(ValueError, match="order"):
                bind_template(template, intent)
        selected = strategies.for_request(
            owner,
            config["locations"]["v2.1"],
            {"version_id": "test", "digest": "test", "templates": templates},
            case["request"],
        )
        matches.extend(selected["templates"])
        assert len(matches) == 1
        assert "$request" not in json.dumps(matches)
    assert strategies.store.list_jobs() == []


@pytest.mark.parametrize(
    "message",
    [
        "Run Quality before converting to v3.",
        "Convert to v3 but do not split.",
        "Explain conversion to v3?",
        "Subset some episodes.",
        "Run S3 then pause for review.",
        "Convert to v3 then split 0:1 from the original source.",
        "Prepare Viewer max_frames 2.5.",
        "Prepare Viewer max_frames -2.",
        "Prepare Viewer max_frames 8 and max_frames 16.",
        "Convert to v3 twice.",
        "Run Quality with an unknown unit.",
        "Run Quality with checks [s3, unsupported].",
        "Prepare Viewer max_frames 8, max_frames unknown.",
        "Split 0:1 then repair jumps.",
    ],
)
def test_ambiguous_or_unsupported_requests_fall_back(message):
    assert request_intent(message) is None


def test_binding_preserves_explicit_parameters_defaults_and_order():
    intent = request_intent("Prepare Viewer max_frames 8 downsample 2; no videos.")
    bound = bind_template(chain("viewer.prepare"), intent)
    params = bound.definition["steps"][-1]["parameters"]
    assert params["max_frames"] == 8 and params["downsample"] == 2
    assert params["prepare_videos"] is False
    from lerobot.data_platform.operators.schema import validate_parameters

    assert params == validate_parameters(get_operator("viewer.prepare"), intent.parameters["viewer.prepare"])
    with pytest.raises(ValueError, match="order"):
        bind_template(
            chain("preprocess.split", "preprocess.convert_v3"), request_intent("Convert to v3 then take 0:1.")
        )
    wrong = chain("preprocess.convert_v3", "preprocess.split")
    wrong["steps"][-1]["input"] = "source"
    with pytest.raises(ValueError, match="dataflow"):
        bind_template(wrong, request_intent("Convert to v3 then take 0:1."))


def test_semantic_deduplication_ignores_order_and_repeated_templates():
    a, b = chain("preprocess.split"), chain("curation.quality")
    assert semantic_digest({"schema_version": 1, "templates": [a, b]}) == semantic_digest(
        {"schema_version": 1, "templates": [b, a, a]}
    )
    assert semantic_digest({"schema_version": 1, "templates": [a]}) != semantic_digest(
        {"schema_version": 1, "templates": [b]}
    )


def measures(cases, *, wrong=(), tokens=100, calls=4):
    return [
        Measurement(c["case_id"], n, c["case_id"] not in wrong, calls, tokens, 1)
        for c in cases
        for n in range(2)
    ]


def test_development_net_gain_cannot_hide_stable_composed_regression_or_cost():
    cases = initial_cases()["development"]
    baseline = measures(cases, wrong=("all_subset", "convert", "subset_first", "viewer_small"))
    candidate = measures(cases, wrong=("subset_check",), tokens=127)
    report = stratified_gate(cases, baseline, candidate)
    assert report["candidate"]["correct"] > report["baseline"]["correct"]
    assert not report["eligible"]
    assert {"stable_case_regression", "stratum_regression", "token_regression"} <= set(report["reasons"])
    assert report["strata"]["composed"] == {"denominator": 4, "baseline_correct": 4, "candidate_correct": 2}
    assert not stratified_gate([], baseline, candidate)["eligible"]
    assert stratified_gate(cases, baseline, measures(cases))["eligible"]
    slower = [replace(r, seconds=1.2) for r in measures(cases)]
    assert "latency_regression" in stratified_gate(cases, baseline, slower)["reasons"]


def test_development_gate_blocks_holdout_and_retains_current_binding(configured):
    previous = version(configured[0].assistant)
    worker = leased(configured)
    from lerobot.data_platform.ai_tools.strategy_models import StrategySuite

    with worker.store.sessions() as session:
        cases = session.get(StrategySuite, worker.suite_id).cases["development"]
    worker.status = "development"
    worker.state["candidates"] = ["candidate"]
    worker.state["measurements"] = {
        "development:baseline": [asdict(r) for r in measures(cases, wrong=("convert", "subset_first"))],
        "development:candidate": [asdict(r) for r in measures(cases, wrong=("subset_check",))],
    }
    assert worker.step() == "rejected"
    assert worker.state["reason"] == "development_gate_failed"
    assert "winner" not in worker.state and "holdout:baseline" not in worker.state["measurements"]
    assert not worker.assistant.client_factory().requests
    assert worker.strategies.dashboard(configured[1])["active_id"] == previous


def test_activation_recomputes_stratified_gate(assistant, monkeypatch):  # noqa: F811
    service, _, owner, _ = assistant
    monkeypatch.setenv("DATA_PLATFORM_STRATEGIES_ENABLED", "1")
    key = version(service, active=False)
    with service.store.sessions.begin() as session:
        from lerobot.data_platform.ai_tools.strategy_models import StrategyVersion

        row = session.get(StrategyExperiment, session.get(StrategyVersion, key).experiment_id)
        state = copy.deepcopy(row.state)
        state["development_gates"] = {key: {"eligible": True}}
        state["measurements"]["development:" + key][0]["safe"] = False
        row.state = state
    with pytest.raises(JobConflictError, match="Complete evaluation"):
        Strategies(service).activate(owner, 0, key)


def test_paired_interleaving_resume_and_all_failure_diagnostics(configured, monkeypatch):
    worker = leased(configured)
    sequence = []

    def evaluate(self, case, variant):
        sequence.append((case["case_id"], variant))
        return Measurement(case["case_id"], 0, False, 1, 10, 0.1), {}

    monkeypatch.setattr(Experiment, "evaluate", evaluate)
    variants = [("baseline", None), ("candidate", "candidate")]
    for _ in range(3):
        assert not worker.measurement_tick("development", variants)
    worker.persist()
    worker = Experiment(worker.assistant, worker.key, worker.token)
    for _ in range(77):
        assert not worker.measurement_tick("development", variants)
    assert worker.measurement_tick("development", variants)
    assert sequence[:4] == [
        ("subset_first", None),
        ("subset_first", "candidate"),
        ("subset_first", "candidate"),
        ("subset_first", None),
    ]
    assert len(sequence) == len(worker.state["diagnostics"]) == 80
    assert all(not d["correct"] for d in worker.state["diagnostics"].values())


def test_trace_does_not_store_private_text_paths_or_reasoning():
    trace = PlanningTrace()
    plan = draft()
    plan["name"] = plan["steps"][0]["step_id"] = "PRIVATE_CANARY"
    plan["steps"][0]["parameters"].update(path="/PRIVATE_CANARY", prompt="PRIVATE_CANARY")
    trace.preview(
        {
            "definition": plan,
            "can_execute": False,
            "blocked": [{"code": "PRIVATE_CANARY", "message": "PRIVATE_CANARY"}],
        }
    )
    trace.tool("PRIVATE_CANARY", {"operator_id": ["PRIVATE_CANARY"]}, "invalid_schema")
    trace.finish({"proposal": {"definition": plan, "can_execute": True}, "reasoning": "PRIVATE_CANARY"})
    assert "PRIVATE_CANARY" not in json.dumps(asdict(trace))
    assert trace.termination == "ready" and trace.previews[0]["blocked_codes"] == ["other"]
    assert normalized_plan({"steps": [{"operator_id": [], "input": {}, "parameters": {}}]})


def test_usage_cache_unknown_and_restart_does_not_resend(configured):
    worker = leased(configured)
    worker.trace = PlanningTrace()
    worker.state["current_case"] = "development:baseline:subset_first:0"
    fake = worker.assistant.client_factory()
    fake.responses = [
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 30},
            }
        },
        {"usage": {"prompt_tokens": 80, "completion_tokens": 10, "prompt_tokens_details": None}},
    ]
    for _ in range(2):
        worker.post_chat_completion({"max_tokens": 10, "messages": []})
    assert worker.state["usage"] == {
        "prompt_tokens": 180,
        "completion_tokens": 30,
        "cached_tokens": 30,
        "cache_usage_complete": False,
    }
    assert worker.state["diagnostics"][worker.state["current_case"]]["usage"] == worker.state["usage"]
    worker.state["inflight"] = "unknown"
    worker.persist()
    with worker.store.sessions.begin() as session:
        session.get(StrategyExperiment, worker.key).lease_until = 0
    assert tick(worker.assistant)
    with worker.store.sessions() as session:
        row = session.get(StrategyExperiment, worker.key)
        assert row.status == "incomplete"
        assert (
            row.state["diagnostics"][worker.state["current_case"]]["termination"] == "unknown_model_outcome"
        )
    assert len(fake.requests) == 2


def test_diagnostic_call_limit_invalid_schema_and_ready(configured):
    worker = leased(configured)
    fake = worker.assistant.client_factory()
    fake.responses = [
        call("preview_pipeline", {"definition": {"steps": []}, "summary": "test"}) for _ in range(8)
    ]
    case = initial_cases()["development"][0]
    worker.state["current_case"] = "failure"
    metric, _ = worker.evaluate(case, None)
    diagnostic = worker.state["diagnostics"]["failure"]
    assert not metric.correct and metric.calls == 8
    assert diagnostic["termination"] == "model_call_limit"
    assert len(diagnostic["previews"]) == 8
    assert all(t["outcome"] == "invalid_schema" for t in diagnostic["tools"])
    fake.responses = [
        call("describe_operator", {"operator_id": "preprocess.split"}),
        call("preview_pipeline", {"definition": draft(), "summary": "test"}),
    ]
    worker.state["current_case"] = "success"
    metric, _ = worker.evaluate(case, None)
    assert metric.correct and worker.state["diagnostics"]["success"]["termination"] == "ready"
    assert worker.store.list_jobs() == []


def test_unbalanced_initial_suite_cannot_start_paid_round(configured):
    from lerobot.data_platform.ai_tools.strategy_evaluation import install_suite

    strategies, owner, _, _ = configured
    old = install_suite(strategies.store)
    reviewed = strategies.review_suite(owner, strategies.dashboard(owner)["revision"], old)
    with pytest.raises(ValueError, match="matching nonzero"):
        strategies.start(owner, reviewed["revision"], old)
    assert not strategies.assistant.client_factory().requests


def test_equivalent_placeholders_and_format_order_are_not_new_candidates():
    a = chain("preprocess.split")
    b = copy.deepcopy(a)
    b["formats"].reverse()
    b["steps"][-1]["parameters"] = {"episode_range": "$request.episode_range"}
    assert semantic_digest({"schema_version": 1, "templates": [a]}) == semantic_digest(
        {"schema_version": 1, "templates": [b]}
    )


def test_selection_deduplicates_bound_plans_and_keeps_scope(capable, monkeypatch):
    strategies, owner, _, config = capable
    a = chain("preprocess.split")
    b = copy.deepcopy(a)
    b["steps"][-1]["parameters"] = {"episode_range": "$request.episode_range"}
    snapshot = {"version_id": "test", "digest": "test", "templates": [a, b, a]}
    location = config["locations"]["v2.1"]
    selected = strategies.for_request(owner, location, snapshot, "Take 0:1.")
    assert len(selected["templates"]) == 1
    assert strategies.for_request(owner, location, snapshot, "Ask which episodes to take.") is None
    viewer = strategies.store.register_user(
        username="read-only", password="test-password-strong", role="viewer"
    )
    assert strategies.for_request(viewer["user_id"], location, snapshot, "Take 0:1.") is None
    monkeypatch.setenv("DATA_PLATFORM_STRATEGIES_ENABLED", "1")
    key = version(strategies.assistant)
    from lerobot.data_platform.ai_tools.strategy_models import StrategyVersion
    from lerobot.data_platform.datasets.tasks.catalog import content_digest

    with strategies.store.sessions.begin() as session:
        row = session.get(StrategyVersion, key)
        spec = copy.deepcopy(row.specification)
        spec["templates"][0]["formats"] = ["v3.0"]
        row.specification, row.digest = spec, content_digest(spec)
    # A v3-only template is not a public applicable strategy for this v2 source.
    assert strategies.select(owner, location) is None
    assert strategies.store.list_jobs() == []


@pytest.mark.parametrize("disable", [False, True])
def test_bound_strategy_checkpoint_recovery_keeps_version_and_confirmation(assistant, monkeypatch, disable):
    from lerobot.data_platform.ai_tools.runner import AssistantRunner
    from lerobot.data_platform.ai_tools.runtime import generate
    from lerobot.data_platform.ai_tools.runtime_repository import RuntimeRepository
    from tests.datasets.test_ai_assistant_runtime import expire_lease

    service, fake, owner, value = assistant
    service.config = replace(service.config, backend="langgraph")
    monkeypatch.setenv("DATA_PLATFORM_STRATEGIES_ENABLED", "1")
    original = version(service)
    value = {**value, "message": "Take episode_range 0:1."}
    assert service.submit(owner, value)["status"] == "running"
    repository = RuntimeRepository(service.store)
    lease = repository.claim("original-worker")
    _, _, payload = repository.snapshot(lease)
    assert payload["strategy"]["bound"] and payload["strategy"]["version_id"] == original
    result = generate(service, repository, lease)
    assert result["strategy"]["version_id"] == original
    replacement = version(service)
    if disable:
        Strategies(service).disable(owner, 0, original)
    expire_lease(service, lease)
    assert AssistantRunner(service).tick()
    restored = service.turn(owner, value["request_id"])
    if disable:
        assert restored.get("error") and not restored.get("proposal")
    else:
        assert restored["proposal"]["can_execute"]
        assert restored["strategy"]["version_id"] == original
    assert Strategies(service).dashboard(owner)["active_id"] == replacement
    assert len(fake.requests) == 2
    assert service.store.list_jobs() == []
