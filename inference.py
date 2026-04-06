"""
Inference Script — Annotation QA Environment
=============================================
MANDATORY
- Before submitting, ensure the following variables are defined:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the root
- Participants must use OpenAI Client for all LLM calls
- Participants must emit structured stdout logs strictly following [START],
  [STEP], and [END] format

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotation_qa_env.models import AnnotationQAAction, AnnotationQAObservation
from annotation_qa_env.server.environment import AnnotationQAEnvironment

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "annotation_qa_env"
TASKS = ["fix_bboxes", "fix_classes", "batch_audit"]
MAX_STEPS_PER_TASK = {"fix_bboxes": 15, "fix_classes": 20, "batch_audit": 30}
TEMPERATURE = 0.3
MAX_TOKENS = 500
SUCCESS_SCORE_THRESHOLD = 0.1

SYSTEM_PROMPT = textwrap.dedent("""
You are an AI annotation quality reviewer. You examine synthetic scene
annotations and fix errors in bounding boxes and class labels.

You will receive:
1. A scene description with objects and their true positions
2. Current annotations (some may have errors)
3. Available classes

Your job: Compare annotations against the scene description and fix errors.

AVAILABLE ACTIONS (respond with valid JSON):
- {"action_type": "adjust_bbox", "annotation_id": <id>, "new_bbox": [x, y, w, h]}
- {"action_type": "change_class", "annotation_id": <id>, "new_class": "<class>"}
- {"action_type": "add_annotation", "new_bbox": [x, y, w, h], "new_class": "<class>"}
- {"action_type": "remove_annotation", "annotation_id": <id>}
- {"action_type": "submit"}

All bbox values are normalized to 0.0–1.0.

STRATEGY:
1. Compare each annotation's bbox against the scene objects' bboxes
2. Check if class labels match the scene objects
3. Look for spurious annotations that don't match any scene object
4. Look for scene objects that have no annotation
5. Fix errors one at a time, then submit

RESPOND WITH ONLY A SINGLE JSON ACTION, no explanation.
""").strip()


# ──────────────────────────────────────────────
# Logging helpers (exact format from problem statement)
# ──────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ──────────────────────────────────────────────
# Prompt building
# ──────────────────────────────────────────────

def build_user_prompt(obs: AnnotationQAObservation) -> str:
    """Build the user prompt from the observation."""
    # Format scene objects
    scene_desc = obs.scene_description

    # Format current annotations
    ann_lines = []
    for ann in obs.annotations:
        ann_lines.append(
            f"  ID={ann.id}: class='{ann.class_label}', "
            f"bbox=[{ann.bbox[0]:.3f}, {ann.bbox[1]:.3f}, {ann.bbox[2]:.3f}, {ann.bbox[3]:.3f}]"
        )
    annotations_str = "\n".join(ann_lines) if ann_lines else "  (none)"

    # Format scene ground truth objects
    obj_lines = []
    for obj in obs.scene_objects:
        bbox = obj.get("bbox", [0, 0, 0, 0])
        obj_lines.append(
            f"  {obj['class_label']} at {obj.get('position', '?')}: "
            f"bbox=[{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]"
        )
    objects_str = "\n".join(obj_lines) if obj_lines else "  (none)"

    prompt = f"""Task: {obs.task_description}
Step {obs.step_count}/{obs.max_steps} | Corrections made: {obs.corrections_made}
Feedback: {obs.message}

SCENE OBJECTS (ground truth):
{objects_str}

CURRENT ANNOTATIONS (may have errors):
{annotations_str}

AVAILABLE CLASSES: {', '.join(obs.available_classes)}

Compare annotations against scene objects. Find and fix ONE error, or submit if all are correct.
Respond with a single JSON action."""

    return prompt


def parse_llm_response(response_text: str) -> AnnotationQAAction:
    """Parse the LLM's JSON response into an action."""
    # Try to extract JSON from the response
    text = response_text.strip()

    # Handle common LLM formatting issues
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        import re
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                # Fallback: submit
                return AnnotationQAAction(action_type="submit")
        else:
            return AnnotationQAAction(action_type="submit")

    return AnnotationQAAction(
        action_type=data.get("action_type", "submit"),
        annotation_id=data.get("annotation_id"),
        new_bbox=data.get("new_bbox"),
        new_class=data.get("new_class"),
    )


# ──────────────────────────────────────────────
# LLM interaction
# ──────────────────────────────────────────────

def get_model_action(
    client: OpenAI,
    obs: AnnotationQAObservation,
) -> AnnotationQAAction:
    """Query the LLM for the next action."""
    user_prompt = build_user_prompt(obs)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_llm_response(text)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return AnnotationQAAction(action_type="submit")


# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────

def run_task(client: OpenAI, env: AnnotationQAEnvironment, task_name: str) -> float:
    """Run a single task and return the score."""
    max_steps = MAX_STEPS_PER_TASK.get(task_name, 20)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment with the specific task
        obs = env.reset(task=task_name, seed=42)
        last_reward = 0.0

        for step in range(1, max_steps + 1):
            if obs.done:
                break

            # Get action from LLM
            action = get_model_action(client, obs)
            action_str = f"{action.action_type}"
            if action.annotation_id is not None:
                action_str += f"(id={action.annotation_id})"

            # Execute action
            obs = env.step(action)

            reward = obs.reward if obs.reward is not None else 0.0
            done = obs.done
            error = obs.last_action_error

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        # Compute final score: use the last reward (which is the grader score on submit/timeout)
        if rewards:
            score = rewards[-1]  # Last reward is the final grade
        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} error: {exc}", flush=True)
        score = 0.0
        success = False

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


def main() -> None:
    """Run inference on all 3 tasks."""
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = AnnotationQAEnvironment()

    total_score = 0.0
    for task_name in TASKS:
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task_name}", flush=True)
        print(f"{'='*60}", flush=True)
        score = run_task(client, env, task_name)
        total_score += score
        print(f"Task {task_name} score: {score:.3f}\n", flush=True)

    avg_score = total_score / len(TASKS)
    print(f"\n{'='*60}", flush=True)
    print(f"Average score across {len(TASKS)} tasks: {avg_score:.3f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
