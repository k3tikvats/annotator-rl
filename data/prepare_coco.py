"""
COCO val2017 Dataset Preprocessor for Annotation QA Environment.

Downloads instances_val2017.json from COCO, selects 500 images with diverse
annotations, normalizes bboxes to [0,1], and outputs pre-processed JSON files
for all 3 tasks.

Run this LOCALLY once — the output JSON files are committed to the repo.
Docker never needs to download COCO.

Usage:
    python -m data.prepare_coco
"""

import json
import os
import random
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ──────────────────────────────────────────────
# COCO category ID → name mapping (80 categories)
# ──────────────────────────────────────────────

COCO_CATEGORIES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup",
    48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana",
    53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot",
    58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair",
    63: "couch", 64: "potted plant", 65: "bed", 67: "dining table",
    70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote",
    76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven",
    80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock",
    86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier",
    90: "toothbrush",
}

COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
COCO_ANNOTATIONS_DIRECT_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
COCO_IMAGE_URL_TEMPLATE = "http://images.cocodataset.org/val2017/{:012d}.jpg"


def download_coco_annotations(cache_dir: Path) -> Dict:
    """Download and cache COCO val2017 annotations."""
    cache_file = cache_dir / "instances_val2017.json"

    if cache_file.exists():
        print(f"  Using cached annotations: {cache_file}")
        with open(cache_file, "r") as f:
            return json.load(f)

    # Try direct JSON download from a mirror / HF dataset
    print("  Downloading COCO val2017 annotations...")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Download the zip and extract
    zip_path = cache_dir / "annotations_trainval2017.zip"
    try:
        # Try HuggingFace mirror first (faster, no zip)
        hf_url = "https://huggingface.co/datasets/merve/coco/resolve/main/annotations/instances_val2017.json"
        print(f"  Trying HuggingFace mirror: {hf_url}")
        urllib.request.urlretrieve(hf_url, str(cache_file))
        print(f"  Downloaded to {cache_file}")
    except Exception as e:
        print(f"  HF mirror failed ({e}), trying COCO website...")
        # Fallback: download zip from COCO
        urllib.request.urlretrieve(COCO_ANNOTATIONS_URL, str(zip_path))
        import zipfile
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            # Extract just instances_val2017.json
            zf.extract("annotations/instances_val2017.json", str(cache_dir))
        # Move to expected location
        extracted = cache_dir / "annotations" / "instances_val2017.json"
        extracted.rename(cache_file)
        (cache_dir / "annotations").rmdir()
        zip_path.unlink()

    with open(cache_file, "r") as f:
        return json.load(f)


def select_diverse_images(
    coco_data: Dict,
    n_images: int = 500,
    min_annotations: int = 3,
    max_annotations: int = 15,
    seed: int = 42,
) -> List[Dict]:
    """
    Select diverse images from COCO val2017.

    Criteria:
    - At least `min_annotations` and at most `max_annotations` objects
    - Skip crowd annotations (iscrowd=1)
    - Prefer diversity in categories
    """
    rng = random.Random(seed)

    # Build image_id → annotations mapping
    img_anns: Dict[int, List[Dict]] = {}
    for ann in coco_data["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        if ann["category_id"] not in COCO_CATEGORIES:
            continue
        img_id = ann["image_id"]
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)

    # Build image_id → image info mapping
    img_info: Dict[int, Dict] = {}
    for img in coco_data["images"]:
        img_info[img["id"]] = img

    # Filter by annotation count
    candidates = []
    for img_id, anns in img_anns.items():
        if min_annotations <= len(anns) <= max_annotations:
            if img_id in img_info:
                candidates.append((img_id, anns))

    print(f"  Found {len(candidates)} candidate images with {min_annotations}-{max_annotations} annotations")

    # Shuffle and select
    rng.shuffle(candidates)

    # Prefer category diversity: score each image by unique categories
    candidates.sort(
        key=lambda x: len(set(a["category_id"] for a in x[1])),
        reverse=True,
    )

    selected = candidates[:n_images]
    rng.shuffle(selected)  # re-shuffle after diversity sort

    print(f"  Selected {len(selected)} images")
    return selected, img_info


def normalize_bbox(
    bbox: List[float], img_width: int, img_height: int
) -> List[float]:
    """Convert COCO [x_min, y_min, width, height] (pixels) → normalized [x, y, w, h] (0-1)."""
    x, y, w, h = bbox
    return [
        round(x / img_width, 4),
        round(y / img_height, 4),
        round(w / img_width, 4),
        round(h / img_height, 4),
    ]


def build_scene_description(objects: List[Dict], img_info: Dict) -> str:
    """Build a natural language scene description from COCO annotations."""
    # Count objects by class
    class_counts: Dict[str, int] = {}
    for obj in objects:
        cls = obj["class_label"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    # Build description
    parts = []
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        if count == 1:
            parts.append(f"a {cls}")
        else:
            parts.append(f"{count} {cls}s" if not cls.endswith("s") else f"{count} {cls}")

    scene_text = (
        f"A scene ({img_info.get('width', '?')}×{img_info.get('height', '?')} pixels) "
        f"containing {len(objects)} annotated objects: "
        + ", ".join(parts) + ". "
    )

    # Add spatial descriptions for each object
    obj_descs = []
    for obj in objects:
        bbox = obj["bbox"]
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        # Determine spatial position
        if cy < 0.33:
            v_pos = "top"
        elif cy < 0.66:
            v_pos = "middle"
        else:
            v_pos = "bottom"
        if cx < 0.33:
            h_pos = "left"
        elif cx < 0.66:
            h_pos = "center"
        else:
            h_pos = "right"
        position = f"{v_pos}-{h_pos}"
        obj["position"] = position

        obj_descs.append(
            f"{obj['class_label']} at {position} "
            f"(bbox: x={bbox[0]:.3f}, y={bbox[1]:.3f}, w={bbox[2]:.3f}, h={bbox[3]:.3f})"
        )

    scene_text += "Objects: " + "; ".join(obj_descs) + "."
    return scene_text


def convert_image_to_sample(
    img_id: int,
    anns: List[Dict],
    img_info_map: Dict[int, Dict],
    scene_id: str,
) -> Dict[str, Any]:
    """Convert a COCO image + annotations into our environment's sample format."""
    info = img_info_map[img_id]
    w, h = info["width"], info["height"]

    objects = []
    gold_annotations = []

    for i, ann in enumerate(anns):
        cat_name = COCO_CATEGORIES[ann["category_id"]]
        norm_bbox = normalize_bbox(ann["bbox"], w, h)

        obj = {
            "id": i,
            "class_label": cat_name,
            "position": "",  # filled by build_scene_description
            "bbox": norm_bbox,
        }
        objects.append(obj)

        gold_annotations.append({
            "id": i,
            "bbox": norm_bbox,
            "class_label": cat_name,
        })

    scene_description = build_scene_description(objects, info)
    image_url = COCO_IMAGE_URL_TEMPLATE.format(img_id)

    return {
        "scene_id": scene_id,
        "scene_type": "coco_val2017",
        "image_id": img_id,
        "image_url": image_url,
        "image_width": w,
        "image_height": h,
        "scene_description": scene_description,
        "objects": objects,
        "gold_annotations": gold_annotations,
    }


def generate_all_tasks(output_dir: str) -> None:
    """Generate dataset for all 3 tasks from COCO val2017."""
    output_path = Path(output_dir)
    cache_dir = Path(__file__).parent / ".cache"

    print("=== COCO val2017 Dataset Preparation ===")
    print()

    # Step 1: Download annotations
    print("Step 1: Loading COCO annotations...")
    coco_data = download_coco_annotations(cache_dir)
    print(f"  Loaded {len(coco_data['annotations'])} annotations, "
          f"{len(coco_data['images'])} images, "
          f"{len(coco_data['categories'])} categories")
    print()

    # Step 2: Select 500 diverse images
    print("Step 2: Selecting 500 diverse images...")
    selected, img_info_map = select_diverse_images(coco_data, n_images=500, seed=42)
    print()

    # Step 3: Split into tasks
    # Task 1: 250 images (easy — bbox corruption only)
    # Task 2: 150 images (medium — bbox + class errors)
    # Task 3: 100 images in batches of 5 (hard — subtle errors)
    task1_images = selected[:250]
    task2_images = selected[250:400]
    task3_images = selected[400:500]

    # Task 1: Spurious Removal (Easy)
    print("Step 3a: Generating Task 1 (remove_spurious) — 250 images...")
    task1_data = []
    for idx, (img_id, anns) in enumerate(task1_images):
        sample = convert_image_to_sample(
            img_id, anns, img_info_map,
            scene_id=f"remove_spurious_{idx:03d}",
        )
        sample["task_id"] = "remove_spurious"
        sample["difficulty"] = "spurious"
        sample["seed"] = 1000 + idx
        task1_data.append(sample)

    task1_dir = output_path / "task1_remove_spurious"
    task1_dir.mkdir(parents=True, exist_ok=True)
    with open(task1_dir / "samples.json", "w") as f:
        json.dump(task1_data, f, indent=2)
    print(f"  → {len(task1_data)} samples written to {task1_dir}")

    # Task 2: Fix Classes (Medium)
    print("Step 3b: Generating Task 2 (fix_classes) — 150 images...")
    task2_data = []
    for idx, (img_id, anns) in enumerate(task2_images):
        sample = convert_image_to_sample(
            img_id, anns, img_info_map,
            scene_id=f"fix_classes_{idx:03d}",
        )
        sample["task_id"] = "fix_classes"
        sample["difficulty"] = "classes"
        sample["seed"] = 2000 + idx
        task2_data.append(sample)

    task2_dir = output_path / "task2_fix_classes"
    task2_dir.mkdir(parents=True, exist_ok=True)
    with open(task2_dir / "samples.json", "w") as f:
        json.dump(task2_data, f, indent=2)
    print(f"  → {len(task2_data)} samples written to {task2_dir}")

    # Task 3: Find Missing (Hard)
    print("Step 3c: Generating Task 3 (find_missing) — 100 images...")
    task3_data = []
    for idx, (img_id, anns) in enumerate(task3_images):
        sample = convert_image_to_sample(
            img_id, anns, img_info_map,
            scene_id=f"find_missing_{idx:03d}",
        )
        sample["task_id"] = "find_missing"
        sample["difficulty"] = "missing"
        sample["seed"] = 3000 + idx
        task3_data.append(sample)

    task3_dir = output_path / "task3_find_missing"
    task3_dir.mkdir(parents=True, exist_ok=True)
    with open(task3_dir / "samples.json", "w") as f:
        json.dump(task3_data, f, indent=2)
    print(f"  → {len(task3_data)} samples written to {task3_dir}")

    print()
    print("=== Done! ===")

    # Report sizes
    total_size = 0
    for task_dir_name in ["task1_remove_spurious", "task2_fix_classes", "task3_find_missing"]:
        fpath = output_path / task_dir_name / "samples.json"
        size = fpath.stat().st_size
        total_size += size
        print(f"  {task_dir_name}/samples.json: {size / 1024:.1f} KB")
    print(f"  Total: {total_size / 1024:.1f} KB ({total_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    tasks_dir = script_dir / "tasks"
    generate_all_tasks(str(tasks_dir))
