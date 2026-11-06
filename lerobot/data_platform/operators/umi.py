"""Lazy UMI operator descriptions; model configuration belongs to the execution node."""

from .contracts import OperatorSpec


def validate_umi_info(info, operation):
    features = info.get("features", {})
    expected = {"x", "y", "z", "qw", "qx", "qy", "qz"}
    native = all(
        len(features.get(p + "_quaternion_pose", {}).get("names") or []) == 7
        and set(features[p + "_quaternion_pose"]["names"]) == expected
        for p in ("left_arm", "right_arm", "head")
    )
    packed_names = [
        f"{p}_{v}"
        for p in ("left", "right", "head")
        for v in ("x", "y", "z", "qw", "qx", "qy", "qz", *(() if p == "head" else ("gripper",)))
    ]
    if not native and features.get("state", {}).get("names") != packed_names:
        raise ValueError("UMI requires three explicitly named xyz/wxyz poses or the named 23D EE state")
    if operation in {"umi.replay", "umi.cosmos_id"} and info.get("fps") != 30:
        raise ValueError("UMI replay/ID requires 30Hz data")
    if operation == "umi.compare" and (
        not native
        or not all(
            features.get(p + "_image", {}).get("dtype") == "image" for p in ("left_arm", "right_arm", "head")
        )
    ):
        raise ValueError("UMI comparison requires native Tracker poses and three embedded images")


def umi_operators(*, legacy=False):
    episodes = {
        "type": ["array", "null"],
        "items": {"type": "integer", "minimum": 0},
        "minItems": 1,
        "default": None,
    }
    common = {"episodes": episodes}
    definitions = (
        (
            "umi.diagnose",
            "diagnostics:run_diagnostics",
            "Diagnose recorded flags, rotation encodings and Tracker motion independently.",
            common,
            (),
            ("umi_diagnostics",),
            False,
        ),
        (
            "umi.compare",
            "comparison:run_comparison",
            "Compare frame-aligned native Tracker versions and export complete videos.",
            {**common, "workers": {"type": "integer", "minimum": 1, "maximum": 4, "default": 2}},
            (),
            ("umi_comparison",),
            False,
        ),
        (
            "umi.replay",
            "runtime:run_replay",
            "Report H10W robot tracking with fixed gravity alignment and optional baseline comparison.",
            {
                **common,
                "motion_mode": {"type": "string", "enum": ["base", "torso"], "default": "base"},
                "position_p95_cm": {"type": "number", "minimum": 0.001, "default": 7},
                "orientation_p95_deg": {"type": "number", "minimum": 0.001, "maximum": 180, "default": 10},
            },
            (),
            ("umi_replay_report",),
            False,
        ),
        (
            "umi.cosmos_id",
            "runtime:run_cosmos",
            "Generate reviewed Cosmos pose candidates without modifying the source dataset.",
            {
                **common,
                "targets": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1000,
                    "items": {
                        "type": "object",
                        "properties": {
                            "episode_index": {"type": "integer", "minimum": 0},
                            "part": {"type": "string", "enum": ["left", "right", "head"]},
                            "start_frame": {"type": "integer", "minimum": 1},
                            "end_frame": {"type": "integer", "minimum": 2},
                        },
                        "required": ["episode_index", "part", "start_frame", "end_frame"],
                        "additionalProperties": False,
                    },
                },
            },
            ("umi_diagnostics",),
            ("umi_cosmos_candidates",),
            True,
        ),
    )
    result = []
    for operator, entry, description, properties, requires, provides, review in definitions:
        native = not legacy and operator in {"umi.replay", "umi.cosmos_id"}
        if native:
            entry += "_native"
        result.append(
            OperatorSpec(
                operator,
                "1.1.0" if native else "1.0.0",
                description,
                "umi",
                "cache",
                entry_point="lerobot.data_platform.operators.implementations.umi." + entry,
                parallel=True,
                cancellable=True,
                output_type="artifact",
                required_modules=("numpy", "pyarrow", "scipy", "PIL"),
                resource_requirements=(
                    "cpu",
                    "memory",
                    "disk",
                    *(("gpu",) if operator in {"umi.replay", "umi.cosmos_id"} else ()),
                ),
                source_commit="fd752156895c83ec75fde84796faea2513c626c7",
                review_required=review,
                requires=requires,
                provides=provides,
                supported_formats=("v3.0",) if operator == "umi.compare" else ("v2.1", "v3.0"),
                parameter_schema_override={
                    "type": "object",
                    "properties": properties,
                    "required": ["targets"] if review else [],
                    "additionalProperties": False,
                },
            )
        )
    return result
