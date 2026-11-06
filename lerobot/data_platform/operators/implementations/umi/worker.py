"""Fixed file protocol for node-configured laundry/Cosmos/cuMotion workers."""

from __future__ import annotations

import html
import json
import os
import subprocess
import sys
from pathlib import Path


def apply_frozen_alignment(source, calibrated, head_pivot):
    """Apply the baseline world/hand/head transforms without refitting the candidate."""
    from dataclasses import replace

    import numpy as np

    slots = {name: index for index, name in enumerate(source.end_effector_names)}
    setup = calibrated.metadata
    world = np.asarray(setup["world_transform"])
    hand = np.asarray(setup["hand_world_transform"])
    mapped = world[None, None] @ source.transforms
    for name in ("leftHandEE", "rightHandEE"):
        mapped[:, slots[name]] = hand @ source.transforms[:, slots[name]]
    head = source.transforms[:, slots["headEE"]]
    rotations = mapped[:, slots["headEE"], :3, :3] @ np.asarray(setup["head_orientation_offset"])
    pivots = head[:, :3, 3] + head[:, :3, :3] @ head_pivot
    mapped[:, slots["headEE"], :3, :3] = rotations
    mapped[:, slots["headEE"], :3, 3] = pivots @ world[:3, :3].T + world[:3, 3] - rotations @ head_pivot
    metadata = {**source.metadata, **{k: v for k, v in setup.items() if k not in source.metadata}}
    metadata["calibration_policy"] = "frozen_from_baseline_episode"
    return replace(source, transforms=mapped, metadata=metadata)


def prepare_evaluation(source, destination, config):
    from lerobot.data_platform.operators.implementations.umi.diagnostics import file_hash

    info = json.loads((source / "meta/info.json").read_text())
    if info.get("codebase_version") == "v2.1":
        if info.get("fps") != 30:
            raise ValueError("UMI replay/ID evaluation requires 30Hz data")
        return source
    if info.get("codebase_version") != "v3.0" or info.get("fps") != 30:
        raise ValueError("UMI evaluation requires native v3 30Hz or converted v2.1")
    if config.get("calibration_profile") != "h10w_tracker_rx90_rz180":
        raise ValueError("An explicit H10W Tracker calibration profile is required for native conversion")
    converter_root = Path(config["converter_root"])
    converter = converter_root / "data_tool/convert_dual_hand_head_to_umi_v21.py"
    urdf = Path(config["urdf_path"])
    if not converter.is_file() or not urdf.is_file():
        raise ValueError("UMI conversion code or calibration URDF is unavailable")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(converter_root / "src"), config["converter_lerobot_root"]])
    command = [
        config["converter_python"],
        str(converter),
        "--src",
        str(source),
        "--dst",
        str(destination),
        "--repo-id",
        destination.name,
        "--urdf",
        str(urdf),
        "--include-head-motion",
        "--tracker-local-x-rotation-deg",
        "90",
        "--tracker-local-y-rotation-deg",
        "0",
        "--tracker-local-z-rotation-deg",
        "180",
        "--gripper-close-threshold-mm",
        "20",
    ]
    subprocess.run(command, env=env, check=True)
    receipt = {
        "evaluation_only": True,
        "calibration_profile": config["calibration_profile"],
        "converter_sha256": file_hash(converter),
        "urdf_sha256": file_hash(urdf),
        "gripper_threshold_confirmed": False,
        "original_validity_semantics": "unconfirmed",
        "source": str(source),
    }
    (destination / "EVALUATION_ONLY.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return destination


def cosmos(request, root, output):
    import numpy as np
    from laundry.core.dataset import DatasetReader
    from laundry.pipelines._cosmos_id.bridge import launch_worker, snapshot_targets
    from laundry.pipelines._cosmos_id.quality import post_repair_check

    config = request["runtime"]
    ds = DatasetReader(root)
    records = {ep.index: ep for ep in ds.episodes()}
    targets = []
    selected = request["parameters"].get("episodes")
    for target in request["parameters"]["targets"]:
        ep, start, stop = target["episode_index"], target["start_frame"], target["end_frame"]
        if ep not in records or (selected is not None and ep not in selected):
            raise ValueError("Repair target is outside selected episodes")
        if not 0 < start < stop < records[ep].length or records[ep].length < 49:
            raise ValueError("Cosmos candidates require complete 49-frame video and two interval anchors")
        targets.append(
            {"episode": ep, "part": target["part"], "frames": [start, stop - 1], "source": "state"}
        )
    snapshot = snapshot_targets(ds, targets)
    settings = {k: config[k] for k in ("framework_root", "checkpoint_path", "processor_path", "vae_path")}
    settings.update(
        contract_path=None,
        stats_path=None,
        num_steps=30,
        seed=0,
        guidance=1.0,
        caption=None,
        use_task_prompt=True,
    )
    folder = output / "cosmos"
    response = launch_worker(
        {
            "snapshot": snapshot,
            "settings": settings,
            "gripper_threshold": 0.5,
            "solver": {
                "translation_scale_m": 0.01,
                "rotation_scale_rad": float(np.pi / 180),
                "max_nfev": 200,
            },
        },
        folder,
        config["model_python"],
        config["timeout_seconds"],
    )
    repaired = unresolved = review = 0
    for key, record in response["episodes"].items():
        with np.load(folder / f"episode_{int(key):06d}.npz", allow_pickle=False) as values:
            state = values["state"]
            original = ds.episode(int(key)).state()
            if not np.array_equal(state[~values["repaired"]], original[~values["repaired"]], equal_nan=True):
                raise ValueError("Cosmos candidate changed unrequested observations")
            repaired += int(values["repaired"].sum())
            unresolved += int(values["unresolved"].sum())
            review += post_repair_check(state, record["gaps"])
    (folder / "response.json").write_text(json.dumps(response, indent=2, allow_nan=False) + "\n")
    if request.get("native_viewers"):
        subprocess.run(
            [
                config.get("report_python") or config["cumotion_python"],
                str(Path(__file__).with_name("native_reports.py")),
                str(root),
                str(folder),
                str(output / "native/cosmos"),
            ],
            check=True,
        )
    return {
        "status": "needs_review",
        "candidate_only": True,
        "human_approvals": 0,
        "repaired_values": repaired,
        "unresolved_values": unresolved,
        "gaps_needing_review": review,
        "model_contract_sha256": response["contract_sha256"],
        "normalization_sha256": response["stats_sha256"],
    }


def run_cumotion(request):
    import numpy as np
    from laundry.umi_replay import screen
    from laundry.umi_replay.alignment import HEAD_PIVOT_IN_HEAD, load_gravity_hand_reference
    from laundry.umi_replay.lerobot import (
        LEROBOT_END_EFFECTOR_NAMES,
        load_lerobot_cartesian_episode,
    )

    output = Path(request["output"])
    roots = [Path(p) for p in request["evaluation_roots"]]
    fixed = load_gravity_hand_reference(roots[0])
    screen._prepare_alignment = lambda args: setattr(args, "gravity_hand_reference", fixed)
    original_align = screen._align_episode

    def frozen_align(source, reference, mode, **kwargs):
        baseline = load_lerobot_cartesian_episode(
            roots[0], source.episode_index, end_effector_names=LEROBOT_END_EFFECTOR_NAMES
        )
        calibrated = original_align(baseline, reference, mode, **kwargs)
        return apply_frozen_alignment(source, calibrated, HEAD_PIVOT_IN_HEAD)

    screen._align_episode = frozen_align
    settings = request["parameters"]
    results = []
    for i, root in enumerate(roots):
        np.random.seed(0)
        command = [
            "--dataset",
            str(root),
            "--output",
            str(output / f"replay_{i}"),
            "--alignment",
            "gravity",
            "--motion-mode",
            settings.get("motion_mode", "base"),
            "--max-workers",
            "1",
            "--position-p95-cm",
            str(settings.get("position_p95_cm", 7)),
            "--orientation-p95-deg",
            str(settings.get("orientation_p95_deg", 10)),
        ]
        for ep in settings.get("episodes") or []:
            command += ["--episode", str(ep)]
        finish = None
        if request.get("native_viewers"):
            if __package__:
                from .native_reports import install_replay_capture
            else:
                from native_reports import install_replay_capture
            finish = install_replay_capture(screen, output / f"native/replay_{i}")
        try:
            result = screen.run(screen.parse_args(command))
        finally:
            if finish:
                finish(root)
        if request.get("native_viewers"):
            view_path = output / f"native/replay_{i}/viewer.json"
            view = json.loads(view_path.read_text())
            view.update(
                alignment="gravity",
                motion_mode=settings.get("motion_mode", "base"),
                status_rows=result["episodes"],
            )
            view_path.write_text(json.dumps(view, indent=2, allow_nan=False) + "\n")
        results.append(result)
    comparisons = []
    if len(roots) == 2:
        info = json.loads((roots[0] / "meta/info.json").read_text())
        for ep in settings.get("episodes") or range(info["total_episodes"]):
            a = load_lerobot_cartesian_episode(roots[0], ep, end_effector_names=LEROBOT_END_EFFECTOR_NAMES)
            b = load_lerobot_cartesian_episode(roots[1], ep, end_effector_names=LEROBOT_END_EFFECTOR_NAMES)
            same = (
                a.num_samples == b.num_samples
                and np.allclose(a.time, b.time, atol=1e-7, rtol=0)
                and np.allclose(a.transforms[0], b.transforms[0], atol=1e-6, rtol=0)
            )
            comparisons.append(
                {
                    "episode_index": int(ep),
                    "comparable_initialization": bool(same),
                    "reason": None if same else "initial_pose_or_time_changed",
                }
            )
    summary = {
        "status": "reported",
        "alignment": "gravity",
        "motion_mode": settings.get("motion_mode", "base"),
        "frozen_hand_reference": fixed.metadata(),
        "calibration_policy": "baseline_world_hand_head_transforms_and_torso_initial_q",
        "versions": results,
        "comparisons": comparisons,
        "physical_quality_approved": False,
    }
    (output / "result.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


def run_request(path):
    from lerobot.data_platform.operators.implementations.umi.diagnostics import source_fingerprint

    request = json.loads(path.read_text())
    if request["schema_version"] != 1 or request["operation"] not in {"umi.replay", "umi.cosmos_id"}:
        raise ValueError("Unsupported UMI request")
    output = Path(request["output"])
    config = request["runtime"]
    # This is a dedicated host worker. Its installed site-packages may be on
    # PYTHONPATH, but the 3.12/3.13 children must use their own binary dependencies.
    os.environ["PYTHONPATH"] = str(Path(config["laundry_root"]) / "src")
    os.environ.pop("PYTHONHOME", None)
    source_roots = [Path(p).resolve() for p in request["source_roots"]]
    fingerprints = {}
    for root in source_roots:
        fingerprints[str(root)] = source_fingerprint(root)
    roots = [prepare_evaluation(root, output / f"eval_{i}", config) for i, root in enumerate(source_roots)]
    if request["operation"] == "umi.cosmos_id":
        summary = cosmos(request, roots[0], output)
        (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    else:
        request["evaluation_roots"] = [str(p) for p in roots]
        replay_request = output / "replay_request.json"
        replay_request.write_text(json.dumps(request, indent=2))
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = ""
        subprocess.run(
            [config["cumotion_python"], str(Path(__file__)), "--cumotion", str(replay_request)],
            env=env,
            check=True,
        )
        summary = json.loads((output / "result.json").read_text())
    for name, files in fingerprints.items():
        if source_fingerprint(Path(name)) != files:
            raise ValueError("Source changed during UMI evaluation")
    (output / "source_fingerprints.json").write_text(json.dumps(fingerprints, indent=2) + "\n")
    (output / "report.html").write_text(
        "<!doctype html><meta charset=utf-8><title>UMI evaluation</title><h1>UMI 候选与回放结果</h1><p>模型候选、机器人跟随结果和人工验收分别记录。</p><pre>"
        + html.escape(json.dumps(summary, indent=2, ensure_ascii=False))
        + "</pre><a href=result.json>机器可读结果</a>"
    )


if __name__ == "__main__":
    if sys.argv[1] == "--cumotion":
        run_cumotion(json.loads(Path(sys.argv[2]).read_text()))
    else:
        run_request(Path(sys.argv[1]))
