"""Bounded synthetic evaluation evidence. Never retain prompts, responses, or reasoning."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from lerobot.data_platform.ai_tools.strategy_contracts import OPERATORS, PARAMETERS


def safe_parameters(op, parameters):
    result = {}
    if not isinstance(op, str) or not isinstance(parameters, dict):
        return result
    for key, value in parameters.items():
        if key not in PARAMETERS.get(op, set()):
            continue
        if (
            type(value) in (int, float, bool)
            and math.isfinite(value)
            or isinstance(value, str)
            and re.fullmatch(r"[0-9:, -]{1,80}|none|all|sample|full|off", value)
        ):
            result[key] = value
        elif key == "checks" and isinstance(value, list):
            result[key] = [
                x
                for x in value[:12]
                if isinstance(x, str) and x in {"s1", "s3", "eef", "c3", "control", "annotations", "legacy"}
            ]
    return result


def normalized_plan(definition):
    rows = definition.get("steps", []) if isinstance(definition, dict) else []
    rows = [r for r in rows[:12] if isinstance(r, dict)] if isinstance(rows, list) else []
    ids = {r.get("step_id"): f"s{i}" for i, r in enumerate(rows) if isinstance(r.get("step_id"), str)}
    result = []
    for i, row in enumerate(rows):
        op = row.get("operator_id")
        source = row.get("input", "source")
        artifacts = row.get("artifacts", [])
        result.append(
            {
                "step_id": f"s{i}",
                "operator_id": op if op in OPERATORS else "unsupported",
                "input": "source"
                if source == "source"
                else ids.get(source, "invalid")
                if isinstance(source, str)
                else "invalid",
                "artifacts": [
                    ids.get(a, "invalid") if isinstance(a, str) else "invalid" for a in artifacts[:12]
                ]
                if isinstance(artifacts, (list, tuple))
                else [],
                "parameters": safe_parameters(op, row.get("parameters", {})),
                "review_before": row.get("review_before") is True,
            }
        )
    return result


@dataclass
class PlanningTrace:
    selected_templates: list[str] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)
    previews: list[dict] = field(default_factory=list)
    final_plan: list[dict] = field(default_factory=list)
    termination: str = "not_started"
    usage: dict = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "cache_usage_complete": True,
        }
    )

    def tool(self, name, arguments, outcome):
        if len(self.tools) >= 32:
            return
        op = arguments.get("operator_id") if isinstance(arguments, dict) else None
        self.tools.append(
            {
                "tool": name
                if name in {"read_context", "describe_operator", "preview_pipeline", "prepare_action"}
                else "unsupported",
                "operator_id": op if op in OPERATORS else None,
                "outcome": outcome,
            }
        )

    def preview(self, preview):
        codes = {
            "invalid_input",
            "invalid_schema",
            "output_scope",
            "missing_capability",
            "unsupported_format",
            "invalid_parameters",
            "invalid_evidence_reference",
            "missing_evidence",
            "missing_snapshot",
            "missing_source_snapshot",
            "operator_unavailable",
            "permission_denied",
        }
        if len(self.previews) < 8:
            self.previews.append(
                {
                    "plan": normalized_plan(preview.get("definition")),
                    "can_execute": bool(preview.get("can_execute")),
                    "blocked_codes": [
                        x.get("code")
                        if isinstance(x.get("code"), str) and x.get("code") in codes
                        else "other"
                        for x in preview.get("blocked", [])[:12]
                        if isinstance(x, dict)
                    ],
                }
            )

    def finish(self, result):
        proposal = result.get("action") or result.get("proposal") or {}
        self.final_plan = normalized_plan(proposal.get("definition"))
        if proposal.get("can_execute"):
            self.termination = "ready"
        elif self.termination == "running":
            self.termination = "blocked" if proposal else "clarification"
