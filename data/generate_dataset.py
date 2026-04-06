"""
Synthetic dataset generator for the Annotation QA Environment.

Generates scene descriptions + gold annotations without requiring any external
dataset (COCO, VOC, etc.). Everything is self-contained and deterministic.

WHY NOT USE COCO IMAGES?
========================
The COCO dataset would NOT work within the hackathon's resource constraints:

1. STORAGE: COCO train2017 is ~18GB of images alone. The Docker container must
   run on HF Spaces free tier (16GB RAM, 2 vCPU). Just loading the images into
   the container would exceed the storage budget.

2. MEMORY: Serving base64-encoded images in observations would consume ~1-5MB
   per step. With concurrent WebSocket sessions, memory would spike past 8GB
   instantly.

3. DOCKER BUILD: The Dockerfile must build within the 600s timeout in the
   pre-validation script. Downloading 18GB of COCO images during Docker build
   would timeout.

4. LLM COMPATIBILITY: The inference script uses text-only OpenAI API clients
   (e.g., Qwen2.5-72B-Instruct). Passing raw images would require a VLM
   (vision-language model), which is NOT guaranteed in the evaluation pipeline.
   The hackathon's evaluation uses "standard Open LLM agent (e.g. Nemotron 3
   Super)" which is text-only.

5. REPRODUCIBILITY: COCO images introduce non-determinism via JPEG compression
   artifacts and OCR variations. Our synthetic scenes are 100% deterministic.

OUR APPROACH:
- Generate synthetic scenes as structured JSON + natural language descriptions
- Objects have known classes and precise bounding boxes
- The agent reasons about spatial relationships purely through text
- Total dataset is <1MB — fits easily in the Docker image
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

# Object classes and their typical size ranges (normalized)
OBJECT_CLASSES = {
    "car": {"w_range": (0.10, 0.25), "h_range": (0.08, 0.15)},
    "truck": {"w_range": (0.15, 0.30), "h_range": (0.10, 0.18)},
    "person": {"w_range": (0.04, 0.08), "h_range": (0.10, 0.25)},
    "bicycle": {"w_range": (0.06, 0.12), "h_range": (0.06, 0.12)},
    "dog": {"w_range": (0.05, 0.10), "h_range": (0.04, 0.08)},
    "cat": {"w_range": (0.04, 0.08), "h_range": (0.04, 0.07)},
    "tree": {"w_range": (0.08, 0.15), "h_range": (0.15, 0.35)},
    "building": {"w_range": (0.15, 0.35), "h_range": (0.20, 0.45)},
    "traffic_light": {"w_range": (0.02, 0.04), "h_range": (0.06, 0.10)},
    "bench": {"w_range": (0.08, 0.15), "h_range": (0.05, 0.08)},
}

SCENE_TEMPLATES = [
    {
        "name": "urban_street",
        "description": "A busy urban street scene with vehicles, pedestrians, and city infrastructure.",
        "typical_objects": ["car", "truck", "person", "bicycle", "traffic_light", "building", "tree", "bench"],
        "min_objects": 5,
        "max_objects": 10,
    },
    {
        "name": "park",
        "description": "A peaceful park setting with trees, benches, and people walking their pets.",
        "typical_objects": ["person", "dog", "cat", "tree", "bench", "bicycle"],
        "min_objects": 4,
        "max_objects": 8,
    },
    {
        "name": "parking_lot",
        "description": "A parking lot with various vehicles and some pedestrians.",
        "typical_objects": ["car", "truck", "person", "bicycle", "building"],
        "min_objects": 5,
        "max_objects": 12,
    },
    {
        "name": "residential_area",
        "description": "A quiet residential neighborhood with houses, trees, and occasional pedestrians.",
        "typical_objects": ["building", "tree", "person", "car", "dog", "cat", "bench"],
        "min_objects": 4,
        "max_objects": 9,
    },
    {
        "name": "intersection",
        "description": "A road intersection with traffic lights, vehicles, and crossing pedestrians.",
        "typical_objects": ["car", "truck", "person", "traffic_light", "bicycle", "building"],
        "min_objects": 6,
        "max_objects": 11,
    },
]

SPATIAL_POSITIONS = [
    "top-left", "top-center", "top-right",
    "middle-left", "center", "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
]


def _position_to_region(position: str) -> tuple:
    """Map spatial position name to approximate (x_center, y_center) range."""
    mapping = {
        "top-left": (0.1, 0.3, 0.1, 0.3),
        "top-center": (0.35, 0.65, 0.1, 0.3),
        "top-right": (0.7, 0.9, 0.1, 0.3),
        "middle-left": (0.1, 0.3, 0.35, 0.65),
        "center": (0.35, 0.65, 0.35, 0.65),
        "middle-right": (0.7, 0.9, 0.35, 0.65),
        "bottom-left": (0.1, 0.3, 0.7, 0.9),
        "bottom-center": (0.35, 0.65, 0.7, 0.9),
        "bottom-right": (0.7, 0.9, 0.7, 0.9),
    }
    return mapping.get(position, (0.3, 0.7, 0.3, 0.7))


def generate_scene(
    rng: random.Random, scene_id: str, n_objects: int = None
) -> Dict[str, Any]:
    """Generate a single synthetic scene with objects and gold annotations."""
    template = rng.choice(SCENE_TEMPLATES)

    if n_objects is None:
        n_objects = rng.randint(template["min_objects"], template["max_objects"])

    objects = []
    annotations = []
    used_positions = []

    for i in range(n_objects):
        cls = rng.choice(template["typical_objects"])
        size_spec = OBJECT_CLASSES[cls]

        # Pick a position that doesn't overlap too much
        position = rng.choice(SPATIAL_POSITIONS)
        x_lo, x_hi, y_lo, y_hi = _position_to_region(position)

        w = rng.uniform(*size_spec["w_range"])
        h = rng.uniform(*size_spec["h_range"])

        # Place object center within the position region
        cx = rng.uniform(x_lo, x_hi)
        cy = rng.uniform(y_lo, y_hi)
        x = max(0.0, cx - w / 2)
        y = max(0.0, cy - h / 2)

        # Clamp to image bounds
        x = min(x, 1.0 - w)
        y = min(y, 1.0 - h)

        bbox = [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]

        objects.append({
            "id": i,
            "class_label": cls,
            "position": position,
            "bbox": bbox,
        })

        annotations.append({
            "id": i,
            "bbox": bbox,
            "class_label": cls,
        })

    # Build natural language description
    obj_descriptions = []
    for obj in objects:
        obj_descriptions.append(
            f"a {obj['class_label']} at {obj['position']} "
            f"(bbox: x={obj['bbox'][0]:.2f}, y={obj['bbox'][1]:.2f}, "
            f"w={obj['bbox'][2]:.2f}, h={obj['bbox'][3]:.2f})"
        )

    scene_text = (
        f"{template['description']} "
        f"The scene contains {len(objects)} objects: "
        + "; ".join(obj_descriptions)
        + "."
    )

    return {
        "scene_id": scene_id,
        "scene_type": template["name"],
        "scene_description": scene_text,
        "objects": objects,
        "gold_annotations": annotations,
    }


def generate_task_data(
    task_id: str,
    difficulty: str,
    n_samples: int,
    base_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate all samples for a given task."""
    samples = []

    for i in range(n_samples):
        rng = random.Random(base_seed + i)
        scene = generate_scene(rng, f"{task_id}_sample_{i:03d}")
        scene["task_id"] = task_id
        scene["difficulty"] = difficulty
        scene["seed"] = base_seed + i
        samples.append(scene)

    return samples


def generate_all_tasks(output_dir: str) -> None:
    """Generate dataset for all 3 tasks and save to disk."""
    output_path = Path(output_dir)

    # Task 1: Fix Bounding Boxes (Easy) — 50 samples
    task1_data = generate_task_data(
        task_id="fix_bboxes",
        difficulty="easy",
        n_samples=50,
        base_seed=1000,
    )
    task1_dir = output_path / "task1_fix_bboxes"
    task1_dir.mkdir(parents=True, exist_ok=True)
    with open(task1_dir / "samples.json", "w") as f:
        json.dump(task1_data, f, indent=2)
    print(f"  Task 1 (fix_bboxes): {len(task1_data)} samples → {task1_dir}")

    # Task 2: Fix Classes + Bboxes (Medium) — 30 samples
    task2_data = generate_task_data(
        task_id="fix_classes",
        difficulty="medium",
        n_samples=30,
        base_seed=2000,
    )
    task2_dir = output_path / "task2_fix_classes"
    task2_dir.mkdir(parents=True, exist_ok=True)
    with open(task2_dir / "samples.json", "w") as f:
        json.dump(task2_data, f, indent=2)
    print(f"  Task 2 (fix_classes): {len(task2_data)} samples → {task2_dir}")

    # Task 3: Batch Consistency Audit (Hard) — 10 batches of 5 scenes
    task3_data = []
    for batch_idx in range(10):
        batch_rng = random.Random(3000 + batch_idx * 100)
        batch_scenes = []
        for scene_idx in range(5):
            scene = generate_scene(
                batch_rng,
                f"batch_audit_batch{batch_idx:02d}_scene{scene_idx:02d}",
            )
            scene["batch_id"] = batch_idx
            scene["task_id"] = "batch_audit"
            scene["difficulty"] = "hard"
            scene["seed"] = 3000 + batch_idx * 100 + scene_idx
            batch_scenes.append(scene)
        task3_data.append({
            "batch_id": batch_idx,
            "scenes": batch_scenes,
        })

    task3_dir = output_path / "task3_batch_audit"
    task3_dir.mkdir(parents=True, exist_ok=True)
    with open(task3_dir / "samples.json", "w") as f:
        json.dump(task3_data, f, indent=2)
    print(f"  Task 3 (batch_audit): {len(task3_data)} batches × 5 scenes → {task3_dir}")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    tasks_dir = script_dir / "tasks"
    print("Generating Annotation QA dataset...")
    generate_all_tasks(str(tasks_dir))
    print("Done!")
