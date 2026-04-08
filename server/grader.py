"""
Grading utilities for the Annotation QA Environment.

Provides deterministic scoring for semantic annotation auditing based on:
- Spurious precision (remove fake boxes without deleting real ones)
- Class-label accuracy (for retained real annotations)
- Missing-flag quality (precision/recall balanced via F1)

Final task score is always clamped to the strict open interval (0, 1)
to satisfy Phase 2 validator constraints.
"""

from collections import Counter
from typing import Dict, List


# Phase 2 validator requires task scores to be strictly within (0, 1).
SCORE_EPSILON = 0.001


def _to_open_unit_interval(value: float) -> float:
    """Clamp any score to the strict open interval (0, 1)."""
    return min(1.0 - SCORE_EPSILON, max(SCORE_EPSILON, value))


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Compute Intersection over Union between two boxes.
    Boxes are [x, y, w, h] with values in 0.0–1.0.
    """
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    # Convert to (x1, y1, x2, y2)
    a_x1, a_y1, a_x2, a_y2 = ax, ay, ax + aw, ay + ah
    b_x1, b_y1, b_x2, b_y2 = bx, by, bx + bw, by + bh

    # Intersection
    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Union
    area_a = aw * ah
    area_b = bw * bh
    union_area = area_a + area_b - inter_area

    if union_area < 1e-8:
        return 0.0

    return inter_area / union_area


def compute_annotation_quality(
    annotations: List[Dict],
    gold_annotations: List[Dict],
) -> float:
    """
    Compute specific Semantic VLM visual QA testing metrics (0.0-1.0).
    Graded on:
    - Spurious Precision (35%): Did you remove fake boxes without destroying real ones?
    - Class Match Accuracy (35%): For existing valid boxes, did you change to the correct Gold label?
    - Missing Flag Recall (30%): Did you successfully use FLAG_MISSING for objects removed from the image?
    """
    if not gold_annotations:
        return 1.0 if not annotations else 0.5

    # 1. Spurious Precision
    gold_map = {a["id"]: a for a in gold_annotations}
    predictions_valid = [a for a in annotations if not a.get("class_label", "").startswith("missing_")]

    if not predictions_valid:
        precision = 0.0
    else:
        precision = sum(1 for a in predictions_valid if a["id"] in gold_map) / len(predictions_valid)
        
    # 2. Class Match Accuracy for valid boxes
    matched = [a for a in predictions_valid if a["id"] in gold_map]
    if not matched:
        class_acc = 0.0
    else:
        class_acc = sum(1 for a in matched if a.get("class_label", "") == gold_map[a["id"]].get("class_label", "")) / len(matched)
        
    # 3. Missing object flag quality (balanced precision/recall)
    expected_classes = [g.get("class_label", "") for g in gold_annotations]
    present_classes = [a.get("class_label", "") for a in annotations if a["id"] in gold_map and not a.get("class_label", "").startswith("missing_")]
    
    # Compute which classes are truly missing from current non-missing annotations.
    exp_counts = Counter(expected_classes)
    pres_counts = Counter(present_classes)
    
    actual_missing_counts: Counter[str] = Counter()
    for cls, count in exp_counts.items():
        missing_n = count - pres_counts.get(cls, 0)
        if missing_n > 0:
            actual_missing_counts[cls] = missing_n

    flagged_classes = [
        a.get("class_label", "").replace("missing_", "", 1)
        for a in annotations
        if a.get("class_label", "").startswith("missing_")
    ]
    flagged_counts: Counter[str] = Counter(flagged_classes)

    total_actual_missing = sum(actual_missing_counts.values())
    total_flagged = sum(flagged_counts.values())

    matched = 0
    for cls, count in actual_missing_counts.items():
        matched += min(count, flagged_counts.get(cls, 0))

    if total_actual_missing == 0:
        missing_recall = 1.0
    else:
        missing_recall = matched / total_actual_missing

    if total_flagged == 0:
        missing_precision = 1.0 if total_actual_missing == 0 else 0.0
    else:
        missing_precision = matched / total_flagged

    if missing_precision + missing_recall == 0:
        missing_f1 = 0.0
    else:
        missing_f1 = (2.0 * missing_precision * missing_recall) / (missing_precision + missing_recall)

    quality = 0.35 * class_acc + 0.35 * precision + 0.30 * missing_f1
    return max(0.0, min(1.0, quality))


def grade_episode(
    initial_annotations: List[Dict],
    final_annotations: List[Dict],
    gold_annotations: List[Dict],
) -> float:
    """
    Compute the episode grade (0.0–1.0).
    """
    initial_quality = compute_annotation_quality(initial_annotations, gold_annotations)
    final_quality = compute_annotation_quality(final_annotations, gold_annotations)

    max_improvement = 1.0 - initial_quality
    if max_improvement < 0.01:
        base_score = 1.0 if final_quality >= initial_quality - 0.01 else 0.5
        return round(_to_open_unit_interval(base_score), 4)

    improvement = final_quality - initial_quality
    score = improvement / max_improvement
    return round(_to_open_unit_interval(score), 4)


def compute_step_reward(
    old_annotations: List[Dict],
    new_annotations: List[Dict],
    gold_annotations: List[Dict],
    action_type: str,
) -> float:
    """
    Compute dense per-step reward based on quality delta.
    """
    old_quality = compute_annotation_quality(old_annotations, gold_annotations)
    new_quality = compute_annotation_quality(new_annotations, gold_annotations)
    delta = new_quality - old_quality
    reward = delta * 2.0  # quality improvement → reward
    reward -= 0.01  # step penalty
    if action_type == "submit":
        reward += 0.05
    return round(reward, 4)
