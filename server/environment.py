"""
Annotation QA Environment — Core Environment Logic.

Implements the OpenEnv 3-method interface:
- reset(task_id) → Observation
- step(action) → Observation
- state → State

The agent reviews intentionally-flawed annotations on synthetic scenes
and must correct bounding boxes, fix class labels, add missing annotations,
or remove spurious ones. Dense reward is provided at every step.
"""

import copy
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # Fallback for standalone
    pass

try:
    from ..models import (
        Annotation,
        AnnotationQAAction,
        AnnotationQAObservation,
        AnnotationQAState,
    )
except ImportError:
    from models import (
        Annotation,
        AnnotationQAAction,
        AnnotationQAObservation,
        AnnotationQAState,
    )
from .corruption import ALL_CLASSES, corrupt_annotations
from .grader import (
    compute_annotation_quality,
    compute_step_reward,
    grade_episode,
)


# ──────────────────────────────────────────────
# Task definitions
# ──────────────────────────────────────────────

TASK_CONFIGS = {
    "fix_bboxes": {
        "description": (
            "Fix bounding box errors in the annotations. Some boxes are too large, "
            "shifted to the wrong position, too small, or completely missing. "
            "There may also be spurious annotations that don't correspond to any object. "
            "Adjust bounding boxes, remove spurious annotations, and add any missing ones."
        ),
        "difficulty": "easy",
        "max_steps": 15,
        "data_file": "task1_fix_bboxes/samples.json",
    },
    "fix_classes": {
        "description": (
            "Fix both bounding box AND class label errors. Some annotations have the "
            "wrong class label (e.g., a 'car' labeled as 'truck', or a 'dog' labeled as 'cat'). "
            "Additionally, some bounding boxes are wrong. Fix class labels, adjust bounding "
            "boxes, remove spurious annotations, and add missing ones."
        ),
        "difficulty": "medium",
        "max_steps": 20,
        "data_file": "task2_fix_classes/samples.json",
    },
    "batch_audit": {
        "description": (
            "Perform a batch consistency audit across multiple scenes. Fix annotation "
            "errors including subtle bounding box shifts, similar-class confusions "
            "(car vs truck, dog vs cat), missing annotations, and spurious annotations. "
            "Errors are more subtle than in previous tasks."
        ),
        "difficulty": "hard",
        "max_steps": 30,
        "data_file": "task3_batch_audit/samples.json",
    },
}


class AnnotationQAEnvironment:
    """
    Annotation QA Environment following the OpenEnv pattern.

    The agent reviews synthetic scene annotations that contain intentional
    errors and must correct them through a series of actions.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = AnnotationQAState()
        self._gold_annotations: List[Dict] = []
        self._initial_annotations: List[Dict] = []
        self._current_annotations: List[Dict] = []
        self._scene_data: Dict[str, Any] = {}
        self._task_config: Dict[str, Any] = {}
        self._corrections_made: int = 0
        self._done: bool = False
        self._data_cache: Dict[str, Any] = {}
        self._next_ann_id: int = 0

        # Load data directory
        self._data_dir = Path(__file__).parent.parent / "data" / "tasks"

    def _load_task_data(self, task_id: str) -> List[Dict]:
        """Load and cache task data from disk."""
        if task_id in self._data_cache:
            return self._data_cache[task_id]

        config = TASK_CONFIGS[task_id]
        data_file = self._data_dir / config["data_file"]

        if not data_file.exists():
            # Generate data on-the-fly if not pre-generated
            try:
                from ..data.generate_dataset import generate_all_tasks
            except ImportError:
                from data.generate_dataset import generate_all_tasks
            generate_all_tasks(str(self._data_dir))

        with open(data_file, "r") as f:
            data = json.load(f)

        self._data_cache[task_id] = data
        return data

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> AnnotationQAObservation:
        """
        Start a new episode.

        Args:
            seed: Random seed for reproducibility
            episode_id: Optional episode ID
            task: Task ID — one of "fix_bboxes", "fix_classes", "batch_audit"
        """
        task_id = task or kwargs.get("task_id", "fix_bboxes")
        if task_id not in TASK_CONFIGS:
            task_id = "fix_bboxes"

        self._task_config = TASK_CONFIGS[task_id]
        data = self._load_task_data(task_id)

        # Select a random sample
        rng = random.Random(seed) if seed is not None else random.Random()

        if task_id == "batch_audit":
            # For batch audit, pick a random batch and use its first scene
            batch = rng.choice(data)
            scene = batch["scenes"][0]
            sample_seed = scene.get("seed", rng.randint(0, 99999))
        else:
            scene = rng.choice(data)
            sample_seed = scene.get("seed", rng.randint(0, 99999))

        # Store gold annotations
        self._gold_annotations = copy.deepcopy(scene["gold_annotations"])
        self._scene_data = scene

        # Create corrupted annotations
        corrupted, corruption_log = corrupt_annotations(
            self._gold_annotations,
            self._task_config["difficulty"],
            sample_seed,
        )
        self._initial_annotations = copy.deepcopy(corrupted)
        self._current_annotations = copy.deepcopy(corrupted)
        self._corrections_made = 0
        self._done = False

        # Track next annotation ID
        self._next_ann_id = max((a["id"] for a in self._current_annotations), default=-1) + 1

        # Compute initial quality
        initial_quality = compute_annotation_quality(
            self._initial_annotations, self._gold_annotations
        )

        self._state = AnnotationQAState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            sample_id=scene.get("scene_id", "unknown"),
            initial_quality=round(initial_quality, 4),
            current_quality=round(initial_quality, 4),
            corrections_made=0,
        )

        return self._build_observation(
            reward=None,
            message=(
                f"Review the annotations for this {scene.get('scene_type', 'scene')}. "
                f"There are {len(self._current_annotations)} annotations. "
                f"Some may have incorrect bounding boxes, wrong class labels, "
                f"or be entirely spurious. Some objects may be missing annotations. "
                f"You have {self._task_config['max_steps']} steps to fix them."
            ),
        )

    def step(
        self,
        action: AnnotationQAAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> AnnotationQAObservation:
        """Execute a correction action and return updated observation with reward."""
        if self._done:
            return self._build_observation(
                reward=0.0,
                message="Episode is already done. Call reset() to start a new episode.",
            )

        self._state.step_count += 1
        error_msg = None

        # Save pre-action state for reward computation
        old_annotations = copy.deepcopy(self._current_annotations)

        # Process action
        try:
            if action.action_type == "adjust_bbox":
                error_msg = self._handle_adjust_bbox(action)
            elif action.action_type == "change_class":
                error_msg = self._handle_change_class(action)
            elif action.action_type == "add_annotation":
                error_msg = self._handle_add_annotation(action)
            elif action.action_type == "remove_annotation":
                error_msg = self._handle_remove_annotation(action)
            elif action.action_type == "submit":
                return self._handle_submit()
            else:
                error_msg = f"Unknown action_type: {action.action_type}"
        except Exception as e:
            error_msg = f"Error processing action: {str(e)}"

        if error_msg is None:
            self._corrections_made += 1
            self._state.corrections_made = self._corrections_made

        # Compute reward
        reward = compute_step_reward(
            old_annotations,
            self._current_annotations,
            self._gold_annotations,
            action.action_type,
        )

        # Update quality tracking
        current_quality = compute_annotation_quality(
            self._current_annotations, self._gold_annotations
        )
        self._state.current_quality = round(current_quality, 4)

        # Check if max steps reached
        if self._state.step_count >= self._task_config["max_steps"]:
            self._done = True
            final_score = grade_episode(
                self._initial_annotations,
                self._current_annotations,
                self._gold_annotations,
            )
            return self._build_observation(
                reward=final_score,
                message=f"Max steps reached. Final score: {final_score:.3f}",
                error=error_msg,
            )

        return self._build_observation(
            reward=reward,
            message=(
                f"{'Error: ' + error_msg if error_msg else 'Correction applied.'} "
                f"Quality: {current_quality:.3f} "
                f"(was {self._state.initial_quality:.3f}). "
                f"Steps remaining: {self._task_config['max_steps'] - self._state.step_count}"
            ),
            error=error_msg,
        )

    @property
    def state(self) -> AnnotationQAState:
        """Get current episode state."""
        return self._state

    def close(self) -> None:
        """Clean up environment resources."""
        pass

    # ──────────────────────────────────────────
    # Action handlers
    # ──────────────────────────────────────────

    def _handle_adjust_bbox(self, action: AnnotationQAAction) -> Optional[str]:
        """Adjust the bounding box of an existing annotation."""
        if action.annotation_id is None:
            return "annotation_id is required for adjust_bbox"
        if action.new_bbox is None:
            return "new_bbox is required for adjust_bbox"
        if len(action.new_bbox) != 4:
            return "new_bbox must have exactly 4 values [x, y, w, h]"

        ann = self._find_annotation(action.annotation_id)
        if ann is None:
            return f"Annotation {action.annotation_id} not found"

        # Validate bbox values
        for v in action.new_bbox:
            if not (0.0 <= v <= 1.0):
                return "All bbox values must be between 0.0 and 1.0"

        ann["bbox"] = [round(v, 4) for v in action.new_bbox]
        return None

    def _handle_change_class(self, action: AnnotationQAAction) -> Optional[str]:
        """Change the class label of an existing annotation."""
        if action.annotation_id is None:
            return "annotation_id is required for change_class"
        if action.new_class is None:
            return "new_class is required for change_class"
        if action.new_class not in ALL_CLASSES:
            return f"Invalid class '{action.new_class}'. Valid: {ALL_CLASSES}"

        ann = self._find_annotation(action.annotation_id)
        if ann is None:
            return f"Annotation {action.annotation_id} not found"

        ann["class_label"] = action.new_class
        return None

    def _handle_add_annotation(self, action: AnnotationQAAction) -> Optional[str]:
        """Add a new annotation."""
        if action.new_bbox is None:
            return "new_bbox is required for add_annotation"
        if action.new_class is None:
            return "new_class is required for add_annotation"
        if len(action.new_bbox) != 4:
            return "new_bbox must have exactly 4 values [x, y, w, h]"
        if action.new_class not in ALL_CLASSES:
            return f"Invalid class '{action.new_class}'. Valid: {ALL_CLASSES}"

        for v in action.new_bbox:
            if not (0.0 <= v <= 1.0):
                return "All bbox values must be between 0.0 and 1.0"

        new_ann = {
            "id": self._next_ann_id,
            "bbox": [round(v, 4) for v in action.new_bbox],
            "class_label": action.new_class,
        }
        self._current_annotations.append(new_ann)
        self._next_ann_id += 1
        return None

    def _handle_remove_annotation(self, action: AnnotationQAAction) -> Optional[str]:
        """Remove an annotation."""
        if action.annotation_id is None:
            return "annotation_id is required for remove_annotation"

        idx = self._find_annotation_index(action.annotation_id)
        if idx is None:
            return f"Annotation {action.annotation_id} not found"

        self._current_annotations.pop(idx)
        return None

    def _handle_submit(self) -> AnnotationQAObservation:
        """Submit corrections and compute final grade."""
        self._done = True
        final_score = grade_episode(
            self._initial_annotations,
            self._current_annotations,
            self._gold_annotations,
        )

        return self._build_observation(
            reward=final_score,
            message=(
                f"Corrections submitted! "
                f"Final score: {final_score:.3f}. "
                f"Quality went from {self._state.initial_quality:.3f} "
                f"to {self._state.current_quality:.3f} over "
                f"{self._state.step_count} steps."
            ),
        )

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def _find_annotation(self, ann_id: int) -> Optional[Dict]:
        for ann in self._current_annotations:
            if ann["id"] == ann_id:
                return ann
        return None

    def _find_annotation_index(self, ann_id: int) -> Optional[int]:
        for i, ann in enumerate(self._current_annotations):
            if ann["id"] == ann_id:
                return i
        return None

    def _build_observation(
        self,
        reward: Optional[float],
        message: str,
        error: Optional[str] = None,
    ) -> AnnotationQAObservation:
        """Build an observation from current state."""
        return AnnotationQAObservation(
            done=self._done,
            reward=reward,
            scene_description=self._scene_data.get("scene_description", ""),
            scene_objects=[
                {
                    "id": obj["id"],
                    "class_label": obj["class_label"],
                    "position": obj["position"],
                    "bbox": obj["bbox"],
                }
                for obj in self._scene_data.get("objects", [])
            ],
            annotations=[
                Annotation(
                    id=ann["id"],
                    bbox=ann["bbox"],
                    class_label=ann["class_label"],
                )
                for ann in self._current_annotations
            ],
            available_classes=ALL_CLASSES,
            task_id=self._state.task_id,
            task_description=self._task_config.get("description", ""),
            corrections_made=self._corrections_made,
            step_count=self._state.step_count,
            max_steps=self._task_config.get("max_steps", 20),
            message=message,
            last_action_error=error,
        )
