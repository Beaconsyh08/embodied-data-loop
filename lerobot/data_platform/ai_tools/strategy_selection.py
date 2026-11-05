"""Conservative request matching and deterministic recipe binding; never execution authority."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass

from lerobot.data_platform.ai_tools.strategy_contracts import PARAMETERS, validate_strategy
from lerobot.data_platform.operators import get_operator
from lerobot.data_platform.operators.schema import validate_parameters
from lerobot.data_platform.processing.planner import PipelineSpec, availability
from lerobot.data_platform.task_catalog import content_digest


@dataclass(frozen=True)
class RequestIntent:
    operations: tuple[str, ...]
    parameters: dict[str, dict]


@dataclass(frozen=True)
class BoundTemplate:
    template_id: str
    definition: dict


def semantic_digest(specification):
    """Template ordering and duplicates are not independent strategies."""
    templates = validate_strategy(specification)["templates"]
    normalized = []
    for template in templates:
        value = copy.deepcopy(template)
        value["formats"] = sorted(set(value["formats"]))
        value["required_features"] = sorted(set(value["required_features"]))
        for step in value["steps"]:
            # Binding always applies explicit user parameters/defaults regardless of placeholders.
            step["parameters"] = {}
            step["artifacts"] = sorted(set(step["artifacts"]))
        normalized.append(content_digest(value))
    return content_digest(sorted(set(normalized)))


def request_intent(message):
    """Only recognize explicit supported requests; ambiguity falls back to normal planning."""
    text = message.lower()
    if re.search(
        r"\b(explain|ask|unknown|not|don't|before|after|instead|unless|if|review|pause|threshold|units)\b|"
        r"解释|询问|未知|不要|不能|先别|之前|之后|如果|是否|能否|审核|暂停|阈值|单位|吗|？|\?",
        text,
    ):
        return None
    if re.search(r"\b(no|without|never)\b", re.sub(r"\bno videos\b", "", text)):
        return None
    patterns = {
        "preprocess.convert_v3": r"\bconvert\b|转换",
        "preprocess.split": r"\b(subset|split|take|select)\b|拆分|提取",
        "curation.quality": r"\b(quality|s3)\b|质检",
        "viewer.prepare": r"\bviewer\b|准备可视化",
    }
    matches = []
    for op, pattern in patterns.items():
        hits = list(re.finditer(pattern, text))
        if len({h.group(0) for h in hits}) != len(hits):
            return None
        if hits:
            matches.append((hits[0].start(), op))
    operations = tuple(op for _, op in sorted(matches))
    if not operations or re.search(
        r"\b(merge|delete|repair|smooth|interpolat\w*)\b|合并|删除|修复|平滑|插值", text
    ):
        return None
    # Repeated operations or reverse-order language require the ordinary planner.
    if re.search(r"\b(again|twice|repeat)\b|再次|两次|重复", text):
        return None
    if len(operations) > 1 and re.search(r"original|same source|原始|源数据", text):
        return None
    parameters = {op: {} for op in operations}
    if "preprocess.split" in operations:
        ranges = set(re.findall(r"(?<![\w:])(\d+:\d+)(?![\w:])", text))
        if len(ranges) != 1:
            return None
        parameters["preprocess.split"]["episode_range"] = ranges.pop()
    if "preprocess.convert_v3" in operations and not re.search(r"\bv3(?:\.0)?\b", text):
        return None
    if "curation.quality" in operations:
        checks = list(dict.fromkeys(re.findall(r"\b(s1|s3|eef|c3|control|annotations|legacy)\b", text)))
        explicit = re.search(r"checks\s*(?:exactly\s*)?(?:=|:)?\s*\[([^]]*)\]", text)
        if explicit and any(token.strip() not in checks for token in explicit.group(1).split(",")):
            return None
        if checks:
            parameters["curation.quality"]["checks"] = checks
    for op, allowed in PARAMETERS.items():
        if op not in parameters:
            continue
        for name in allowed - {"episode_range", "checks"}:
            values = re.findall(r"\b" + name + r"\s*(?:=|:)?\s*(true|false|\d+)\b(?!\.\d)", text)
            if len(set(values)) > 1 or len(values) != len(re.findall(r"\b" + name + r"\b", text)):
                return None
            if values:
                value = values[0]
                parameters[op][name] = {"true": True, "false": False}.get(value, value)
                if value.isdigit():
                    parameters[op][name] = int(value)
            elif name in text:
                return None
    # Common explicit media requests are safe only with an unambiguous boolean meaning.
    if "viewer.prepare" in parameters and re.search(r"\bno videos\b|不准备视频|不含视频", text):
        parameters["viewer.prepare"]["prepare_videos"] = False
    return RequestIntent(operations, parameters)


def bind_template(template, intent):
    operations = tuple(s["operator_id"] for s in template["steps"] if s["operator_id"] != "curation.snapshot")
    if operations != intent.operations:
        raise ValueError("Template does not preserve the requested operator order")
    steps = []
    latest_input = "source"
    for index, step in enumerate(template["steps"]):
        op = step["operator_id"]
        if op == "curation.snapshot":
            if step["input"] != "source":
                raise ValueError("Snapshot binds the original source")
        elif step["input"] != latest_input:
            raise ValueError("Template dataflow does not match the requested chain")
        if get_operator(op).output_type == "dataset":
            latest_input = index
        # Apply every explicit supported parameter, even if the recipe omitted its placeholder.
        parameters = validate_parameters(get_operator(op, step["version"]), intent.parameters.get(op, {}))
        source = step["input"]
        steps.append(
            {
                "step_id": f"strategy_{index}",
                "operator_id": op,
                "version": step["version"],
                "input": "source" if source == "source" else f"strategy_{source}",
                "artifacts": [f"strategy_{n}" for n in step["artifacts"]],
                "parameters": parameters,
            }
        )
    definition = PipelineSpec.from_dict({"name": "Matched workflow", "steps": steps}).to_dict()
    return BoundTemplate(content_digest(template), definition)


def select_bound_templates(specification, message, info, node, preview):
    intent = request_intent(message)
    if intent is None:
        return []
    result, seen = [], set()
    for template in validate_strategy(specification)["templates"]:
        if info.get("codebase_version") not in template["formats"] or not set(
            template["required_features"]
        ) <= set(info.get("features", {})):
            continue
        try:
            bound = bind_template(template, intent)
            definition_digest = content_digest(bound.definition)
            if definition_digest in seen:
                continue
            if any(
                availability(get_operator(s["operator_id"], s["version"]), node, info)
                for s in template["steps"]
            ):
                continue
            checked = preview(bound.definition)
            from lerobot.data_platform.ai_tools.repairs import bind_source_snapshot

            repaired, repairs = bind_source_snapshot(bound.definition, checked["blocked"])
            if repairs:
                checked = preview(repaired)
            if not checked["can_execute"]:
                continue
            result.append(BoundTemplate(bound.template_id, checked["definition"]))
            seen.add(definition_digest)
            if len(result) == 2:
                break
        except (ValueError, TypeError, KeyError):
            continue
    return result
