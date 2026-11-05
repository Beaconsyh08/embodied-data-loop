"""DenseCaption-style temporal matching and explicit human semantic assessments.

Intervals use source seconds, half-open, with integer microsecond arithmetic.
No text similarity is substituted for factual caption quality.
"""

from __future__ import annotations

import math

WEIGHTS = {"action": 0.50, "identity": 0.25, "coordination": 0.20, "detail": 0.05}
PROTOCOL = "flora-human-pilot-v1"


def iou(a, b):
    a0, a1, b0, b1 = (round(x * 1_000_000) for x in (a["start_s"], a["end_s"], b["start_s"], b["end_s"]))
    overlap = max(0, min(a1, b1) - max(a0, b0))
    return overlap / (a1 - a0 + b1 - b0 - overlap)


def edges(pred, gold, threshold=0.5):
    return [(i, j) for i, p in enumerate(pred) for j, g in enumerate(gold) if iou(p, g) >= threshold]


def matching(pred, gold, threshold=0.5, scores=None):
    # Maximum-cardinality matching for segmentation; maximum-weight for joint.
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if not pred or not gold:
        return []
    weights = np.zeros((len(pred), len(gold)))
    eligible = set(edges(pred, gold, threshold))
    for i, j in eligible:
        weights[i, j] = 1 if scores is None else scores.get(f"{i}:{j}", 0)
    rows, cols = linear_sum_assignment(weights, maximize=True)
    return [
        (int(i), int(j)) for i, j in zip(rows, cols, strict=False) if (i, j) in eligible and weights[i, j] > 0
    ]


def segmentation(pred, gold, threshold):
    tp = len(matching(pred, gold, threshold))
    return {
        "f1": 2 * tp / (len(pred) + len(gold)) if pred or gold else None,
        "precision": tp / len(pred) if pred else 0,
        "recall": tp / len(gold) if gold else None,
        "tp": tp,
        "predicted": len(pred),
        "reference": len(gold),
    }


def caption_quality(reference, assessment):
    facts = reference["facts"]
    group_sizes = {g: sum(f["group"] == g for f in facts) for g in WEIGHTS}
    mass = {str(i): WEIGHTS[f["group"]] / group_sizes[f["group"]] for i, f in enumerate(facts)}
    r = sum(mass.values())
    m = sum(mass[i] for i in assessment["covered"])
    # Families are named by the reviewer; duplicates in a family are penalized once.
    families = {(e["group"], e["family"].strip().lower()) for e in assessment["errors"]}
    penalty = 1.25 * sum(WEIGHTS[group] for group, _ in families)
    return 2 * m / (r + m + penalty) if r else None


def dense_counts(prediction, reference, duration, fps):
    matrix = [[0] * 4 for _ in range(3)]  # fourth column: unavailable prediction
    missing_reference = 0
    for second in range(math.ceil(duration)):
        labels = [q["label"] for q in reference if q["start_s"] <= second < q["end_s"]]
        if not labels:
            missing_reference += 1
            continue
        truth = min(labels)  # failure takes priority over suboptimal over normal
        frame = min(int(second * fps), max(0, len(prediction) - 1))
        value = prediction[frame] if prediction else 0
        matrix[truth - 1][value - 1 if value in (1, 2, 3) else 3] += 1
    return matrix, missing_reference


def dense_metrics(matrix):
    n = sum(map(sum, matrix))
    if not n:
        return {
            "macro_f1": None,
            "balanced_accuracy": None,
            "bad_action_f1": None,
            "failure_recall": None,
            "normal_false_failure_rate": None,
            "samples": 0,
            "matrix": matrix,
        }
    f1s, recalls = [], []
    for i in range(3):
        tp, support = matrix[i][i], sum(matrix[i])
        guessed = sum(row[i] for row in matrix)
        f1s.append(2 * tp / (support + guessed) if support + guessed else 0)
        if support:
            recalls.append(tp / support)
    bad_tp = sum(matrix[i][j] for i in (0, 1) for j in (0, 1))
    bad_truth = sum(sum(matrix[i]) for i in (0, 1))
    bad_guessed = sum(row[0] + row[1] for row in matrix)
    return {
        "macro_f1": sum(f1s) / 3,
        "balanced_accuracy": sum(recalls) / len(recalls),
        "bad_action_f1": 2 * bad_tp / (bad_truth + bad_guessed) if bad_truth + bad_guessed else None,
        "failure_recall": matrix[0][0] / sum(matrix[0]) if sum(matrix[0]) else None,
        "normal_false_failure_rate": matrix[2][0] / sum(matrix[2]) if sum(matrix[2]) else None,
        "samples": n,
        "prediction_coverage": sum(sum(r[:3]) for r in matrix) / n,
        "matrix": matrix,
    }


def evaluate(records):
    result = {
        "protocol": PROTOCOL,
        "annotation_origin": "model_assisted_human_review",
        "selected": len(records),
        "approved": 0,
        "episodes": [],
        "notes": [
            "Pilot uses FLORA-assisted human references, not independently annotated benchmark gold.",
            "Segmentation and joint: episode macro average, no boundary snapping.",
            "Dense: source-time 1 Hz pooled samples; fixed three-class macro F1; balanced accuracy over supported classes.",
            "Bad-action F1 collapses labels 1 and 2; this is the explicit platform convention.",
            "Caption and joint require human factual assessments; text similarity is not a substitute.",
            "Oracle caption quality requires a separate prediction generated on frozen reference windows.",
        ],
    }
    matrix = [[0] * 4 for _ in range(3)]
    outcomes, segs, joint, caption = [], {".5": [], ".75": []}, [], []
    for record in records:
        if record["status"] != "approved":
            continue
        result["approved"] += 1
        ref, pred = record["reference"], record["prediction"]
        row = {"episode_index": record["episode_index"], "revision": record["revision"]}
        for threshold in (0.5, 0.75):
            key = str(threshold).replace("0.", ".")
            metric = segmentation(pred["segments"], ref["segments"], threshold)
            row["segmentation_" + key] = metric
            if metric["f1"] is not None:
                segs[key].append(metric["f1"])
        outcome = pred["outcome"]
        if outcome is not None:
            outcomes.append(abs(outcome - ref["outcome"]))
        row["outcome_absolute_error"] = abs(outcome - ref["outcome"]) if outcome is not None else None
        counts, missing = dense_counts(pred["dense"], ref["quality"], record["duration_s"], record["fps"])
        for i in range(3):
            for j in range(4):
                matrix[i][j] += counts[i][j]
        row["dense"] = dense_metrics(counts)
        row["dense_unlabeled_seconds"] = missing
        required = edges(pred["segments"], ref["segments"])
        assessments = record.get("assessments", {})
        row["caption_pairs_required"] = len(required)
        row["caption_pairs_reviewed"] = sum(f"{i}:{j}" in assessments for i, j in required)
        row["joint_f1"] = None
        if all(f"{i}:{j}" in assessments for i, j in required):
            scores = {
                f"{i}:{j}": caption_quality(ref["segments"][j], assessments[f"{i}:{j}"]) for i, j in required
            }
            pairs = matching(pred["segments"], ref["segments"], scores=scores)
            den = len(pred["segments"]) + len(ref["segments"])
            row["joint_f1"] = 2 * sum(scores[f"{i}:{j}"] for i, j in pairs) / den if den else None
            if row["joint_f1"] is not None:
                joint.append(row["joint_f1"])
        oracle = record.get("oracle")
        row["oracle_caption_quality"] = None
        if oracle and all(
            str(j) in record.get("oracle_assessments", {}) for j in range(len(ref["segments"]))
        ):
            values = [
                caption_quality(s, record["oracle_assessments"][str(j)])
                for j, s in enumerate(ref["segments"])
            ]
            if values:
                row["oracle_caption_quality"] = sum(values) / len(values)
                caption.append(row["oracle_caption_quality"])
        result["episodes"].append(row)
    approved = result["approved"]

    def metric(values):
        # Partial subsets are never silently called a complete evaluation.
        return {
            "value": sum(values) / len(values) if values and len(values) == approved else None,
            "evaluated_episodes": len(values),
            "reference_episodes": approved,
        }

    result.update(
        segmentation_f1_50=metric(segs[".5"]),
        segmentation_f1_75=metric(segs[".75"]),
        outcome_mae=metric(outcomes),
        joint_f1=metric(joint),
        oracle_caption_quality=metric(caption),
        dense=dense_metrics(matrix),
    )
    return result
