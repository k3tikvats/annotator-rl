"""
Annotation QA Environment — Type-Safe Models.

Defines the API contract for the Annotation QA Environment:
- AnnotationQAAction: What corrections the agent can make
- AnnotationQAObservation: What the agent sees (scene + annotations)
- AnnotationQAState: Episode metadata

The agent reviews intentionally-flawed annotations on synthetic scenes
and must fix bounding boxes, correct class labels, add missing annotations,
or remove spurious ones.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Annotation data structure
# ──────────────────────────────────────────────

class Annotation(BaseModel):
    """A single annotation: bounding box + class label."""
    id: int
    bbox: List[float] = Field(
        ...,
        description="Bounding box as [x, y, w, h] normalized to 0.0–1.0",
        min_length=4,
        max_length=4,
    )
    class_label: str = Field(..., description="Object class label, e.g. 'car', 'person'")


# ──────────────────────────────────────────────
# Action
# ──────────────────────────────────────────────

class AnnotationQAAction(BaseModel):
    """
    An action the agent can take to correct annotations.

    action_type determines which fields are required:
    - "adjust_bbox": requires annotation_id, new_bbox
    - "change_class": requires annotation_id, new_class
    - "add_annotation": requires new_bbox, new_class
    - "remove_annotation": requires annotation_id
    - "submit": no extra fields needed (finalizes episode)
    """
    action_type: Literal[
        "adjust_bbox",
        "change_class",
        "add_annotation",
        "remove_annotation",
        "submit",
    ]
    annotation_id: Optional[int] = Field(
        None, description="ID of the annotation to modify"
    )
    new_bbox: Optional[List[float]] = Field(
        None,
        description="New bounding box [x, y, w, h] in 0.0–1.0",
        min_length=4,
        max_length=4,
    )
    new_class: Optional[str] = Field(
        None, description="New class label"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ──────────────────────────────────────────────
# Observation
# ──────────────────────────────────────────────

class AnnotationQAObservation(BaseModel):
    """
    What the agent sees after each step.

    Includes the scene description, current annotations (some may be wrong),
    available classes, and progress info.
    """
    done: bool = False
    reward: Optional[float] = None

    # Scene information
    scene_description: str = Field(
        "", description="Natural-language description of the scene"
    )
    scene_objects: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ground-truth object list with positions (visible to agent as scene context)",
    )

    # Current annotations (may contain errors)
    annotations: List[Annotation] = Field(
        default_factory=list,
        description="Current annotations the agent should review/fix",
    )

    # Task context
    available_classes: List[str] = Field(
        default_factory=list,
        description="Valid class labels for this task",
    )
    task_id: str = ""
    task_description: str = ""

    # Progress
    corrections_made: int = 0
    step_count: int = 0
    max_steps: int = 20

    # Feedback
    message: str = ""
    last_action_error: Optional[str] = None


# ──────────────────────────────────────────────
# State
# ──────────────────────────────────────────────

class AnnotationQAState(BaseModel):
    """Episode metadata — internal state tracked by the environment."""
    episode_id: Optional[str] = None
    step_count: int = 0
    task_id: str = ""
    sample_id: str = ""
    initial_quality: float = 0.0
    current_quality: float = 0.0
    corrections_made: int = 0
