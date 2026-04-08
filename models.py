"""
Annotation QA Environment — Type-Safe Models.

Defines the API contract for the Annotation QA Environment:
- AnnotationQAAction: What corrections the agent can make
- AnnotationQAObservation: What the agent sees (image + annotations)
- AnnotationQAState: Episode metadata

The agent reviews intentionally-flawed annotations on real COCO val2017 images
and performs semantic QA actions: remove spurious annotations, correct class
labels, and flag missing objects. A VLM (Vision-Language Model) is used to
visually inspect the images.
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
    - "flag_missing": requires missing_class
    - "submit": no extra fields needed (finalizes episode)
    """
    action_type: Literal[
        "adjust_bbox",
        "change_class",
        "remove_annotation",
        "add_annotation",
        "submit",
        "flag_missing",
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
    missing_class: Optional[str] = Field(
        None, description="Class of an object that was missing bounding boxes"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ──────────────────────────────────────────────
# Observation
# ──────────────────────────────────────────────

class AnnotationQAObservation(BaseModel):
    """
    What the agent sees after each step.

    Includes the image URL, scene description, current annotations (some may
    be wrong), available classes, and progress info. The VLM agent uses the
    image_url to visually inspect the scene.
    """
    done: bool = False
    reward: Optional[float] = None

    # Image information (real COCO val2017)
    image_url: Optional[str] = Field(
        None, description="Public URL to the COCO val2017 image"
    )
    image_width: int = Field(0, description="Image width in pixels")
    image_height: int = Field(0, description="Image height in pixels")

    # Scene information
    scene_description: str = Field(
        "", description="Natural-language description of the scene and its objects"
    )
    scene_objects: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Optional debug field; empty by default to avoid leaking ground-truth labels",
    )

    # Current annotations (may contain errors)
    annotations: List[Annotation] = Field(
        default_factory=list,
        description="Current annotations the agent should review/fix",
    )

    # Task context
    available_classes: List[str] = Field(
        default_factory=list,
        description="Valid class labels for this task (COCO 80 categories)",
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
