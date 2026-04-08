"""
Annotation corruption strategies for the Annotation QA Environment.

Takes gold-standard COCO annotations and systematically corrupts them to create
data with known errors. The corruption is deterministic given a seed.

Corruption types by difficulty:
- Task 1 (Easy): Obvious bbox errors — expand, shift, delete, add spurious
- Task 2 (Medium): bbox + class errors — similar class confusion, boundary errors
- Task 3 (Hard): Cross-image inconsistencies + subtle errors
"""

import copy
import random
from typing import Dict, List, Tuple

# ──────────────────────────────────────────────
# COCO 80 categories
# ──────────────────────────────────────────────

ALL_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# Class confusion maps — COCO-specific similar category pairs
SIMILAR_CLASSES: Dict[str, List[str]] = {
    "car": ["truck", "bus"],
    "truck": ["car", "bus"],
    "bus": ["truck", "car"],
    "motorcycle": ["bicycle"],
    "bicycle": ["motorcycle"],
    "dog": ["cat", "horse"],
    "cat": ["dog"],
    "horse": ["cow", "dog"],
    "cow": ["horse", "sheep"],
    "sheep": ["cow"],
    "elephant": ["bear"],
    "bear": ["elephant"],
    "zebra": ["giraffe", "horse"],
    "giraffe": ["zebra"],
    "bird": ["airplane", "kite"],
    "airplane": ["bird", "kite"],
    "chair": ["couch", "bench"],
    "couch": ["chair", "bed"],
    "bed": ["couch"],
    "bench": ["chair"],
    "dining table": ["bed"],
    "bottle": ["cup", "wine glass", "vase"],
    "cup": ["bottle", "wine glass", "bowl"],
    "wine glass": ["cup", "bottle"],
    "bowl": ["cup"],
    "fork": ["knife", "spoon"],
    "knife": ["fork", "spoon", "scissors"],
    "spoon": ["fork", "knife"],
    "scissors": ["knife"],
    "banana": ["hot dog"],
    "hot dog": ["banana", "sandwich"],
    "pizza": ["cake", "donut"],
    "donut": ["pizza", "cake", "apple", "orange"],
    "cake": ["pizza", "donut"],
    "apple": ["orange", "donut", "sports ball"],
    "orange": ["apple", "donut", "sports ball"],
    "sandwich": ["hot dog", "pizza"],
    "broccoli": ["potted plant"],
    "carrot": ["banana"],
    "potted plant": ["broccoli", "vase"],
    "tv": ["laptop", "microwave"],
    "laptop": ["tv", "keyboard"],
    "keyboard": ["laptop", "remote"],
    "remote": ["cell phone", "keyboard"],
    "cell phone": ["remote"],
    "mouse": ["remote"],
    "microwave": ["oven", "tv"],
    "oven": ["microwave", "refrigerator"],
    "toaster": ["microwave"],
    "refrigerator": ["oven"],
    "sink": ["toilet", "bowl"],
    "toilet": ["sink", "chair"],
    "book": ["laptop", "cell phone"],
    "clock": ["sports ball"],
    "vase": ["bottle", "cup"],
    "backpack": ["suitcase", "handbag"],
    "handbag": ["backpack", "suitcase"],
    "suitcase": ["backpack", "handbag"],
    "umbrella": ["kite"],
    "tie": ["person"],
    "frisbee": ["sports ball", "kite"],
    "sports ball": ["frisbee", "apple", "orange"],
    "kite": ["bird", "umbrella", "frisbee"],
    "baseball bat": ["tennis racket", "surfboard"],
    "baseball glove": ["backpack"],
    "skateboard": ["surfboard", "snowboard"],
    "surfboard": ["skateboard", "snowboard"],
    "snowboard": ["skateboard", "surfboard", "skis"],
    "skis": ["snowboard"],
    "teddy bear": ["person", "dog"],
    "hair drier": ["toothbrush"],
    "toothbrush": ["hair drier"],
    "person": ["teddy bear"],
    "train": ["bus", "truck"],
    "boat": ["surfboard"],
    "traffic light": ["fire hydrant", "parking meter", "stop sign"],
    "fire hydrant": ["traffic light", "parking meter"],
    "stop sign": ["traffic light", "parking meter"],
    "parking meter": ["fire hydrant", "stop sign"],
}


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
    Corrupt gold annotations conceptually (no geometry shifts) based on difficulty level.

    Difficulties:
    - "spurious": Adds 2-4 entirely fake boxes.
    - "classes": Swaps 30% of class labels (similar and different) + adds some spurious.
    - "missing": Deletes 15-20% of annotations completely. VLM must FLAG_MISSING.
    """
    rng = random.Random(seed)
    corrupted = copy.deepcopy(gold_annotations)
    log = []

    if difficulty == "spurious":
        # Task 1: Spurious removal only
        existing_bboxes = [a["bbox"] for a in corrupted]
        n_spurious = rng.randint(2, 4)
        next_id = max((a["id"] for a in corrupted), default=0) + 1
        for i in range(n_spurious):
            spur = generate_spurious_annotation(existing_bboxes, rng)
            spur["id"] = next_id + i
            corrupted.append(spur)
            existing_bboxes.append(spur["bbox"])
            log.append(f"Added spurious ann {spur['id']} ({spur['class_label']})")

    elif difficulty == "classes":
        # Task 2: Fix Classes
        corruption_rate = 0.30
        n_corrupt = max(2, int(len(corrupted) * corruption_rate))
        indices = list(range(len(corrupted)))
        rng.shuffle(indices)
        corrupt_indices = indices[:n_corrupt]

        for idx in corrupt_indices:
            action = rng.choice(["wrong_similar_class", "wrong_different_class"])
            ann = corrupted[idx]
            old_cls = ann["class_label"]

            if action == "wrong_similar_class":
                similar = SIMILAR_CLASSES.get(old_cls, [])
                if similar:
                    new_cls = rng.choice(similar)
                    ann["class_label"] = new_cls
                    log.append(f"Changed ann {ann['id']} class: {old_cls} → {new_cls} (similar)")
                else:
                    candidates = [c for c in ALL_CLASSES if c != old_cls]
                    ann["class_label"] = rng.choice(candidates)
                    log.append(f"Changed ann {ann['id']} class: {old_cls} → {ann['class_label']} (fallback)")

            elif action == "wrong_different_class":
                candidates = [c for c in ALL_CLASSES if c != old_cls]
                ann["class_label"] = rng.choice(candidates)
                log.append(f"Changed ann {ann['id']} class: {old_cls} → {ann['class_label']} (different)")

        # Add 1-2 spurious just to keep them on their toes
        existing_bboxes = [a["bbox"] for a in corrupted]
        n_spurious = rng.randint(1, 2)
        next_id = max((a["id"] for a in corrupted), default=0) + 1
        for i in range(n_spurious):
            spur = generate_spurious_annotation(existing_bboxes, rng)
            spur["id"] = next_id + i
            corrupted.append(spur)
            existing_bboxes.append(spur["bbox"])
            log.append(f"Added spurious ann {spur['id']} ({spur['class_label']})")

    elif difficulty == "missing":
        # Task 3: Missing items evaluation
        # Randomly delete 15-20% of annotations completely
        delete_rate = rng.uniform(0.15, 0.20)
        n_delete = max(1, int(len(corrupted) * delete_rate))
        indices = list(range(len(corrupted)))
        rng.shuffle(indices)
        delete_indices = indices[:n_delete]

        for idx in delete_indices:
            ann = corrupted[idx]
            log.append(f"Missing Obj Created: Removed ann {ann['id']} ({ann['class_label']})")
            corrupted[idx] = None
        
        corrupted = [a for a in corrupted if a is not None]

        # Also add a little bit of class confusion
        corruption_rate = 0.20
        n_corrupt = max(1, int(len(corrupted) * corruption_rate))
        remaining_indices = list(range(len(corrupted)))
        rng.shuffle(remaining_indices)
        for idx in remaining_indices[:n_corrupt]:
            ann = corrupted[idx]
            old_cls = ann["class_label"]
            candidates = [c for c in ALL_CLASSES if c != old_cls]
            ann["class_label"] = rng.choice(candidates)
            log.append(f"Changed class: {old_cls} -> {ann['class_label']}")

    return corrupted, log
