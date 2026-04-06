"""
Grading utilities for the Annotation QA Environment.

Provides deterministic scoring (0.0–1.0) based on:
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


def hungarian_match(
    pred_annotations: List[Dict],
    gold_annotations: List[Dict],
    iou_threshold: float = 0.3,
) -> List[Tuple[int, int, float]]:
    """
    Match predicted annotations to gold annotations using greedy best-IoU matching.
    Returns list of (pred_idx, gold_idx, iou) tuples.

    Uses a simple greedy approach (good enough for our scale) instead of
    scipy.optimize.linear_sum_assignment to avoid the scipy dependency.
    """
    if not pred_annotations or not gold_annotations:
        return []

    # Compute IoU matrix
    n_pred = len(pred_annotations)
    n_gold = len(gold_annotations)
    iou_matrix = []
    for i in range(n_pred):
        row = []
        for j in range(n_gold):
            iou = compute_iou(pred_annotations[i]["bbox"], gold_annotations[j]["bbox"])
            row.append(iou)
        iou_matrix.append(row)

    # Greedy matching: pick highest IoU pair iteratively
    matches = []
    used_pred = set()
    used_gold = set()

    # Flatten and sort all (pred_idx, gold_idx, iou) by IoU descending
    all_pairs = []
    for i in range(n_pred):
        for j in range(n_gold):
            if iou_matrix[i][j] >= iou_threshold:
                all_pairs.append((i, j, iou_matrix[i][j]))

    all_pairs.sort(key=lambda x: x[2], reverse=True)

    for pred_idx, gold_idx, iou in all_pairs:
        if pred_idx not in used_pred and gold_idx not in used_gold:
            matches.append((pred_idx, gold_idx, iou))
            used_pred.add(pred_idx)
            used_gold.add(gold_idx)

    return matches


def compute_annotation_quality(
    annotations: List[Dict],
    gold_annotations: List[Dict],
    iou_threshold: float = 0.3,
) -> float:
    """
    Compute overall annotation quality score (0.0–1.0).

    Combines:
    - Mean IoU of matched annotations (40%)
    - Class label accuracy on matched annotations (30%)
    - Precision: matched / total_predicted (15%)
    - Recall: matched / total_gold (15%)
    """
    if not gold_annotations:
        # No gold → quality is 1.0 if no predictions, else penalized
        return 1.0 if not annotations else 0.5

    matches = hungarian_match(annotations, gold_annotations, iou_threshold)

    n_pred = len(annotations)
    n_gold = len(gold_annotations)
    n_matched = len(matches)

    # Mean IoU of matched pairs
    if n_matched > 0:
        mean_iou = sum(iou for _, _, iou in matches) / n_matched
    else:
        mean_iou = 0.0

    # Class accuracy on matched pairs
    if n_matched > 0:
        class_correct = sum(
            1
            for pred_idx, gold_idx, _ in matches
            if annotations[pred_idx].get("class_label", "")
            == gold_annotations[gold_idx].get("class_label", "")
        )
        class_acc = class_correct / n_matched
    else:
        class_acc = 0.0

    # Precision and recall
    precision = n_matched / n_pred if n_pred > 0 else 0.0
    recall = n_matched / n_gold if n_gold > 0 else 0.0

    # Weighted composite
    quality = 0.40 * mean_iou + 0.30 * class_acc + 0.15 * precision + 0.15 * recall
    return max(0.0, min(1.0, quality))


def grade_episode(
    initial_annotations: List[Dict],
    final_annotations: List[Dict],
    gold_annotations: List[Dict],
) -> float:
    """
    Compute the episode grade (0.0–1.0).

    Score = improvement in annotation quality normalized by maximum possible
    improvement.
    """
    initial_quality = compute_annotation_quality(initial_annotations, gold_annotations)
    final_quality = compute_annotation_quality(final_annotations, gold_annotations)

    max_improvement = 1.0 - initial_quality
    if max_improvement < 0.01:
        # Already near-perfect, give full credit if not degraded
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

    # Scale delta to a reasonable reward range
    reward = delta * 2.0  # quality improvement → reward

    # Small step penalty to encourage efficiency
    reward -= 0.01

    # Bonus for submit action (completion)
    if action_type == "submit":
        reward += 0.05  # small bonus for actually submitting

    return round(reward, 4)
