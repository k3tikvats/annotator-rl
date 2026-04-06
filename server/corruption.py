"""
Annotation corruption strategies for the Annotation QA Environment.

Takes gold-standard annotations and systematically corrupts them to create
training data with known errors. The corruption is deterministic given a seed.

Corruption types by difficulty:
- Task 1 (Easy): Obvious bbox errors — expand, shift, delete, add spurious
- Task 2 (Medium): bbox + class errors — similar class confusion, boundary errors
- Task 3 (Hard): Cross-image inconsistencies + subtle errors
"""

import copy
import random
from typing import Dict, List, Tuple

# Class confusion maps — used for "similar class" corruption
SIMILAR_CLASSES: Dict[str, List[str]] = {
    "car": ["truck", "van"],
    "truck": ["car", "van"],
    "van": ["car", "truck"],
    "person": ["cyclist"],
    "cyclist": ["person"],
    "dog": ["cat"],
    "cat": ["dog"],
    "bicycle": ["motorcycle"],
    "motorcycle": ["bicycle"],
    "tree": ["bush"],
    "bush": ["tree"],
    "building": ["house"],
    "house": ["building"],
    "traffic_light": ["street_light"],
    "street_light": ["traffic_light"],
    "bench": ["chair"],
    "chair": ["bench"],
}

# Completely different classes for "wrong category" corruption
ALL_CLASSES = [
    "car", "truck", "person", "bicycle", "dog", "cat",
    "tree", "building", "traffic_light", "bench",
]


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def _clamp_bbox(bbox: List[float]) -> List[float]:
    """Ensure bbox stays within [0, 1] image bounds."""
    x, y, w, h = bbox
    x = _clamp(x)
    y = _clamp(y)
    w = _clamp(w, 0.02, 1.0 - x)
    h = _clamp(h, 0.02, 1.0 - y)
    return [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]


def expand_bbox(bbox: List[float], factor: float) -> List[float]:
    """Expand a bbox by a factor (e.g., 1.5 = 50% larger)."""
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    new_w, new_h = w * factor, h * factor
    new_x = cx - new_w / 2
    new_y = cy - new_h / 2
    return _clamp_bbox([new_x, new_y, new_w, new_h])


def shift_bbox(bbox: List[float], dx_frac: float, dy_frac: float) -> List[float]:
    """Shift a bbox by a fraction of its size."""
    x, y, w, h = bbox
    new_x = x + w * dx_frac
    new_y = y + h * dy_frac
    return _clamp_bbox([new_x, new_y, w, h])


def shrink_bbox(bbox: List[float], factor: float) -> List[float]:
    """Shrink a bbox (factor < 1.0)."""
    return expand_bbox(bbox, factor)


def generate_spurious_annotation(
    existing_bboxes: List[List[float]], rng: random.Random
) -> Dict:
    """Generate a random annotation that doesn't overlap much with existing ones."""
    for _ in range(20):  # try up to 20 times
        w = rng.uniform(0.05, 0.20)
        h = rng.uniform(0.05, 0.20)
        x = rng.uniform(0.0, 1.0 - w)
        y = rng.uniform(0.0, 1.0 - h)
        bbox = [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]

        # Check it doesn't overlap too much with existing
        from .grader import compute_iou

        max_iou = max(
            (compute_iou(bbox, eb) for eb in existing_bboxes), default=0.0
        )
        if max_iou < 0.3:
            cls = rng.choice(ALL_CLASSES)
            return {"bbox": bbox, "class_label": cls}

    # Fallback: place it anyway
    return {
        "bbox": [round(rng.uniform(0.0, 0.8), 4), round(rng.uniform(0.0, 0.8), 4), 0.1, 0.1],
        "class_label": rng.choice(ALL_CLASSES),
    }


def corrupt_annotations(
    gold_annotations: List[Dict],
    difficulty: str,
    seed: int,
) -> Tuple[List[Dict], List[str]]:
    """
    Corrupt gold annotations based on difficulty level.

    Returns:
        (corrupted_annotations, corruption_log)
        corruption_log: list of strings describing what was corrupted (for debugging)
    """
    rng = random.Random(seed)
    corrupted = copy.deepcopy(gold_annotations)
    log = []

    if difficulty == "easy":
        # Task 1: Obvious bbox errors only (no class changes)
        corruption_rate = 0.35
        n_corrupt = max(1, int(len(corrupted) * corruption_rate))
        indices = list(range(len(corrupted)))
        rng.shuffle(indices)
        corrupt_indices = indices[:n_corrupt]

        for idx in corrupt_indices:
            action = rng.choice(["expand", "shift", "shrink", "delete"])
            ann = corrupted[idx]

            if action == "expand":
                factor = rng.uniform(1.5, 2.5)
                ann["bbox"] = expand_bbox(ann["bbox"], factor)
                log.append(f"Expanded ann {ann['id']} by {factor:.1f}x")

            elif action == "shift":
                dx = rng.uniform(-0.4, 0.4)
                dy = rng.uniform(-0.4, 0.4)
                ann["bbox"] = shift_bbox(ann["bbox"], dx, dy)
                log.append(f"Shifted ann {ann['id']} by ({dx:.2f}, {dy:.2f})")

            elif action == "shrink":
                factor = rng.uniform(0.3, 0.6)
                ann["bbox"] = shrink_bbox(ann["bbox"], factor)
                log.append(f"Shrunk ann {ann['id']} by {factor:.1f}x")

            elif action == "delete":
                log.append(f"Deleted ann {ann['id']} ({ann['class_label']})")
                corrupted[idx] = None  # mark for removal

        # Remove deleted
        corrupted = [a for a in corrupted if a is not None]

        # Add 2-3 spurious annotations
        existing_bboxes = [a["bbox"] for a in corrupted]
        n_spurious = rng.randint(2, 3)
        next_id = max((a["id"] for a in corrupted), default=0) + 1
        for i in range(n_spurious):
            spur = generate_spurious_annotation(existing_bboxes, rng)
            spur["id"] = next_id + i
            corrupted.append(spur)
            existing_bboxes.append(spur["bbox"])
            log.append(f"Added spurious ann {spur['id']} ({spur['class_label']})")

    elif difficulty == "medium":
        # Task 2: bbox errors + class confusion
        corruption_rate = 0.30
        n_corrupt = max(2, int(len(corrupted) * corruption_rate))
        indices = list(range(len(corrupted)))
        rng.shuffle(indices)
        corrupt_indices = indices[:n_corrupt]

        for idx in corrupt_indices:
            action = rng.choice([
                "expand", "shift", "wrong_similar_class",
                "wrong_different_class", "delete",
            ])
            ann = corrupted[idx]

            if action == "expand":
                factor = rng.uniform(1.3, 2.0)
                ann["bbox"] = expand_bbox(ann["bbox"], factor)
                log.append(f"Expanded ann {ann['id']} by {factor:.1f}x")

            elif action == "shift":
                dx = rng.uniform(-0.3, 0.3)
                dy = rng.uniform(-0.3, 0.3)
                ann["bbox"] = shift_bbox(ann["bbox"], dx, dy)
                log.append(f"Shifted ann {ann['id']}")

            elif action == "wrong_similar_class":
                old_cls = ann["class_label"]
                similar = SIMILAR_CLASSES.get(old_cls, [])
                if similar:
                    new_cls = rng.choice(similar)
                    ann["class_label"] = new_cls
                    log.append(f"Changed ann {ann['id']} class: {old_cls} → {new_cls}")
                else:
                    # Fallback to a different class
                    candidates = [c for c in ALL_CLASSES if c != old_cls]
                    ann["class_label"] = rng.choice(candidates)
                    log.append(f"Changed ann {ann['id']} class: {old_cls} → {ann['class_label']}")

            elif action == "wrong_different_class":
                old_cls = ann["class_label"]
                candidates = [c for c in ALL_CLASSES if c != old_cls]
                ann["class_label"] = rng.choice(candidates)
                log.append(f"Changed ann {ann['id']} class: {old_cls} → {ann['class_label']} (wrong category)")

            elif action == "delete":
                log.append(f"Deleted ann {ann['id']} ({ann['class_label']})")
                corrupted[idx] = None

        corrupted = [a for a in corrupted if a is not None]

        # Add 3-4 spurious
        existing_bboxes = [a["bbox"] for a in corrupted]
        n_spurious = rng.randint(3, 4)
        next_id = max((a["id"] for a in corrupted), default=0) + 1
        for i in range(n_spurious):
            spur = generate_spurious_annotation(existing_bboxes, rng)
            spur["id"] = next_id + i
            corrupted.append(spur)
            existing_bboxes.append(spur["bbox"])
            log.append(f"Added spurious ann {spur['id']} ({spur['class_label']})")

    elif difficulty == "hard":
        # Task 3: Subtle errors + class confusion + some bbox
        corruption_rate = 0.25
        n_corrupt = max(2, int(len(corrupted) * corruption_rate))
        indices = list(range(len(corrupted)))
        rng.shuffle(indices)
        corrupt_indices = indices[:n_corrupt]

        for idx in corrupt_indices:
            action = rng.choice([
                "subtle_shift", "wrong_similar_class",
                "wrong_similar_class", "delete", "subtle_expand",
            ])
            ann = corrupted[idx]

            if action == "subtle_shift":
                dx = rng.uniform(-0.15, 0.15)
                dy = rng.uniform(-0.15, 0.15)
                ann["bbox"] = shift_bbox(ann["bbox"], dx, dy)
                log.append(f"Subtly shifted ann {ann['id']}")

            elif action == "subtle_expand":
                factor = rng.uniform(1.15, 1.4)
                ann["bbox"] = expand_bbox(ann["bbox"], factor)
                log.append(f"Subtly expanded ann {ann['id']}")

            elif action == "wrong_similar_class":
                old_cls = ann["class_label"]
                similar = SIMILAR_CLASSES.get(old_cls, [])
                if similar:
                    new_cls = rng.choice(similar)
                    ann["class_label"] = new_cls
                    log.append(f"Changed ann {ann['id']}: {old_cls} → {new_cls} (similar)")
                else:
                    candidates = [c for c in ALL_CLASSES if c != old_cls]
                    ann["class_label"] = rng.choice(candidates)
                    log.append(f"Changed ann {ann['id']}: {old_cls} → {ann['class_label']}")

            elif action == "delete":
                log.append(f"Deleted ann {ann['id']}")
                corrupted[idx] = None

        corrupted = [a for a in corrupted if a is not None]

        # Add 2-3 spurious
        existing_bboxes = [a["bbox"] for a in corrupted]
        n_spurious = rng.randint(2, 3)
        next_id = max((a["id"] for a in corrupted), default=0) + 1
        for i in range(n_spurious):
            spur = generate_spurious_annotation(existing_bboxes, rng)
            spur["id"] = next_id + i
            corrupted.append(spur)
            existing_bboxes.append(spur["bbox"])
            log.append(f"Added spurious ann {spur['id']}")

    return corrupted, log
