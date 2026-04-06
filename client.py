"""
Annotation QA Environment Client.

Provides the client for connecting to an Annotation QA Environment server.
"""

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from .models import (
    Annotation,
    AnnotationQAAction,
    AnnotationQAObservation,
    AnnotationQAState,
)


class AnnotationQAEnv(EnvClient[AnnotationQAAction, AnnotationQAObservation, AnnotationQAState]):
    """
    Client for the Annotation QA Environment.

    Example:
        >>> with AnnotationQAEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset(task="fix_bboxes")
        ...     print(result.observation.annotations)
        ...     result = env.step(AnnotationQAAction(
        ...         action_type="adjust_bbox",
        ...         annotation_id=0,
        ...         new_bbox=[0.1, 0.2, 0.15, 0.1],
        ...     ))
        ...     print(result.reward)
    """

    def _step_payload(self, action: AnnotationQAAction) -> dict:
        """Convert action to wire format."""
        payload = {"action_type": action.action_type}
        if action.annotation_id is not None:
            payload["annotation_id"] = action.annotation_id
        if action.new_bbox is not None:
            payload["new_bbox"] = action.new_bbox
        if action.new_class is not None:
            payload["new_class"] = action.new_class
        return payload

    def _parse_result(self, payload: dict) -> StepResult:
        """Parse server response into typed StepResult."""
        obs_data = payload.get("observation", payload)

        annotations = []
        for ann_data in obs_data.get("annotations", []):
            annotations.append(Annotation(
                id=ann_data.get("id", 0),
                bbox=ann_data.get("bbox", [0, 0, 0, 0]),
                class_label=ann_data.get("class_label", ""),
            ))

        observation = AnnotationQAObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            scene_description=obs_data.get("scene_description", ""),
            scene_objects=obs_data.get("scene_objects", []),
            annotations=annotations,
            available_classes=obs_data.get("available_classes", []),
            task_id=obs_data.get("task_id", ""),
            task_description=obs_data.get("task_description", ""),
            corrections_made=obs_data.get("corrections_made", 0),
            step_count=obs_data.get("step_count", 0),
            max_steps=obs_data.get("max_steps", 20),
            message=obs_data.get("message", ""),
            last_action_error=obs_data.get("last_action_error"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> AnnotationQAState:
        """Parse state response."""
        return AnnotationQAState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            sample_id=payload.get("sample_id", ""),
            initial_quality=payload.get("initial_quality", 0.0),
            current_quality=payload.get("current_quality", 0.0),
            corrections_made=payload.get("corrections_made", 0),
        )
