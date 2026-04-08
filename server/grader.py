"""
Grading utilities for the Annotation QA Environment.

Provides deterministic scoring (0.0-1.0) based on:
- IoU (Intersection over Union) of bounding boxes
- Class label accuracy
- Precision (penalizes spurious annotations)
- Recall (penalizes missed annotations)

Uses Hungarian matching to optimally pair predicted vs gold annotations.
"""

from typing import Dict, List, Tuple


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
    from collections import Counter

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
        
    # 3. Missing Object Flag Recall
    expected_classes = [g.get("class_label", "") for g in gold_annotations]
    present_classes = [a.get("class_label", "") for a in annotations if a["id"] in gold_map and not a.get("class_label", "").startswith("missing_")]
    
    # Calculate exact missing instances mathematically
    exp_counts = Counter(expected_classes)
    pres_counts = Counter(present_classes)
    
    actual_missing_classes = []
    for cls, count in exp_counts.items():
        if count > pres_counts.get(cls, 0):
            for _ in range(count - pres_counts.get(cls, 0)):
                actual_missing_classes.append(cls)
                
    if not actual_missing_classes:
        missing_acc = 1.0
    else:
        flagged_classes = [a.get("class_label", "").replace("missing_", "", 1) for a in annotations if a.get("class_label", "").startswith("missing_")]
        flagged_counts = Counter(flagged_classes)

        caught = 0
        for cls in actual_missing_classes:
            if flagged_counts.get(cls, 0) > 0:
                caught += 1
                flagged_counts[cls] -= 1
        missing_acc = caught / len(actual_missing_classes)
        
    quality = 0.35 * class_acc + 0.35 * precision + 0.30 * missing_acc
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
        return 1.0 if final_quality >= initial_quality - 0.01 else 0.5

    improvement = final_quality - initial_quality
    score = improvement / max_improvement
    return max(0.0, min(1.0, score))


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
