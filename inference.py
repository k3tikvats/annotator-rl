"""
Inference Script — Annotation QA Environment (VLM Edition with Spatial Overlay)
==========================================================
MANDATORY
- Before submitting, ensure the following variables are defined:
    API_BASE_URL   The API endpoint for the VLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- STDOUT MUST EXACTLY follow [START], [STEP], and [END] formats.

VLM & VISUAL SPATIAL OVERLAY
- Uses Qwen/Qwen3-VL-8B-Instruct (or any VLM) via OpenAI-compatible API
- To solve the problem of VLMs struggling with raw float coordinates, we
  intercept the image at every step and use Pillow to draw the current bounding  
  boxes, their IDs, and a faint coordinate grid directly onto the image bytes.
- The VLM receives a literal marked-up image ("Set-of-Mark" style prompting).
"""

import base64
import io
import json
import os
import sys
import textwrap
import urllib.request
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from annotation_qa_env.models import AnnotationQAAction, AnnotationQAObservation
    from annotation_qa_env.server.environment import AnnotationQAEnvironment
except ImportError:
    from models import AnnotationQAAction, AnnotationQAObservation
    from server.environment import AnnotationQAEnvironment

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")

BENCHMARK = "annotation_qa_env"
TASKS = ["fix_bboxes", "fix_classes", "batch_audit"]
MAX_STEPS_PER_TASK = {"fix_bboxes": 15, "fix_classes": 20, "batch_audit": 30}
TEMPERATURE = 0.2
MAX_TOKENS = 500
SUCCESS_SCORE_THRESHOLD = 0.1

# Raw Image cache: Store downloaded PIL images so we can redraw them per-step without downloading.
_raw_image_cache = {}

SYSTEM_PROMPT = textwrap.dedent("""
You are an AI annotation quality reviewer. You have EXCELLENT spatial awareness because you are evaluating a MARKED-UP image.

You will receive an image with:
1. A red grid over it. The lines represent normalized X and Y coordinates from 0.0 to 1.0 (in increments of 0.1).
2. Thick brightly-colored bounding boxes drawn over the objects.
3. A large text label above each box indicating `[ID: <num> | <class_label>]`.

Your task is to visually inspect these boxes:
- Check if the box matches its class label.
- Check if the box tightly bounds the object without covering too much empty space.
- Look out for boxes that cover completely empty space or background (Spurious).
- Look out for visible objects that are missing a box.

AVAILABLE ACTIONS (respond with valid JSON only):
- To fix a bad box, output: {"action_type": "adjust_bbox", "annotation_id": <id>, "new_bbox": [x, y, w, h]}
- To fix a wrong class label, output: {"action_type": "change_class", "annotation_id": <id>, "new_class": "<class>"}
- To add a missing object, output: {"action_type": "add_annotation", "new_bbox": [x, y, w, h], "new_class": "<class>"}
- To delete a spurious box, output: {"action_type": "remove_annotation", "annotation_id": <id>}
- To submit (when everything looks perfect), output: {"action_type": "submit"}

All spatial coordinates you output (for add_annotation or adjust_bbox) MUST be in the `[x, y, w, h]` format normalized to 0.0-1.0. 
Use the red grid lines to estimate these coordinates.

Focus exclusively on what you literally SEE drawn on the image compared to the underlying photo. Fix one error at a time, prioritizing the most obvious ones.

OUTPUT A SINGLE JSON OBJECT AND NOTHING ELSE.
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
# Image Overlays ("Set-of-Mark")
# ──────────────────────────────────────────────

def get_base_image(image_url: str, max_dim: int = 768):
    """Download and resize an image from URL, caching the PIL Image in memory."""
    from PIL import Image

    if image_url in _raw_image_cache:
        return _raw_image_cache[image_url]

    try:
        req = urllib.request.Request(image_url, headers={"User-Agent": "AnnotationQA/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            img_bytes = resp.read()

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        _raw_image_cache[image_url] = img
        return img
    except Exception as e:
        print(f"[DEBUG] Failed to fetch image {image_url}: {e}", flush=True)
        return None


def fetch_annotated_image_as_base64(obs: AnnotationQAObservation, debug_save: bool = False) -> str:
    """
    Downloads raw image, draws coordinate grid and all bounding boxes with IDs,
    and returns a base64 encoded jpeg.
    """
    try:
        from PIL import ImageDraw, ImageFont
    except ImportError:
        return "" # Should have Pillow installed via reqs

    img = get_base_image(obs.image_url)
    if img is None:
        return ""

    # Make a fresh copy for this step
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    w, h = canvas.size

    # Try to load a reasonable font size
    try:
        # Windows typically has arial.ttf, Linux has DejaVuSans, fallback to default if not found
        fontsize = max(12, int(h * 0.025))
        try:
            font = ImageFont.truetype("arial.ttf", fontsize)
        except OSError:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", fontsize)
            except OSError:
                font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # 1. Draw the Faint Coordinate Grid
    # Draw lines every 0.1 normalized unit
    grid_color = (255, 0, 0, 80) # Semi-transparent red
    text_color = (255, 0, 0, 180)
    
    for i in range(1, 10):
        val = i / 10.0
        # Vertical line (X-axis demarcations)
        x_px = int(val * w)
        draw.line([(x_px, 0), (x_px, h)], fill=grid_color, width=1)
        draw.text((x_px + 2, 5), f"{val:.1f}", fill=text_color, font=font)
        
        # Horizontal line (Y-axis demarcations)
        y_px = int(val * h)
        draw.line([(0, y_px), (w, y_px)], fill=grid_color, width=1)
        draw.text((5, y_px + 2), f"{val:.1f}", fill=text_color, font=font)

    # 2. Draw the Current Annotations
    colors = [
        (0, 255, 0, 255),    # Green
        (255, 165, 0, 255),  # Orange
        (0, 255, 255, 255),  # Cyan
        (255, 0, 255, 255),  # Magenta
        (255, 255, 0, 255),  # Yellow
    ]

    for ann in obs.annotations:
        color = colors[ann.id % len(colors)]
        x_norm, y_norm, w_norm, h_norm = ann.bbox
        
        x0 = int(x_norm * w)
        y0 = int(y_norm * h)
        x1 = int((x_norm + w_norm) * w)
        y1 = int((y_norm + h_norm) * h)
        
        # Draw thick box
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        
        # Draw label ribbon
        label_text = f" ID:{ann.id} | {ann.class_label} "
        
        # Use simple text bbox to size ribbon
        # Newer Pillow: font.getbbox()
        try:
            bbox = font.getbbox(label_text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = 50, 10
            
        bg_rect = [x0, max(0, y0 - text_h - 4), x0 + text_w, y0]
        draw.rectangle(bg_rect, fill=color)
        draw.text((x0, max(0, y0 - text_h - 4)), label_text, fill=(0,0,0,255), font=font)

    # Optional saving for debugging
    if debug_save:
        canvas.save("debug_overlay_test.jpg")

    # Encode
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ──────────────────────────────────────────────
# Prompt building (multimodal)
# ──────────────────────────────────────────────

def build_user_content(obs: AnnotationQAObservation) -> list:
    content_blocks = []

    # 1. Visually Marked-up Image
    if obs.image_url:
        # We only save debug on the first step of the first sequence 
        # (This is just a local test mechanism)
        save_debug = (obs.step_count == 0)
        
        b64 = fetch_annotated_image_as_base64(obs, debug_save=save_debug)
        if b64:
            content_blocks.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                },
            })

    # 2. Textual Fallback / Metadata
    text = f"""Task: {obs.task_description}
Step {obs.step_count}/{obs.max_steps}
Feedback from last action: {obs.message}

Please find the image attached. YOU CAN SEE BOUNDING BOXES AND IDS ALREADY DRAWN ON IT!
The numeric grid lines will help you estimate coordinates in normalized format (0.0 to 1.0).

Look at each box sequentially. If a box has a wrong label, change it. If it doesn't wrap the object tightly or is completely spurious, remove it or adjust it.
Available COCO classes: {', '.join(obs.available_classes[:20])}... ({len(obs.available_classes)} total).

Respond with a valid JSON action fixing ONE error, or submit if perfect."""

    content_blocks.append({
        "type": "text",
        "text": text,
    })

    return content_blocks


def parse_llm_response(response_text: str) -> AnnotationQAAction:
    text = response_text.strip()
    if text.startswith("```json"): text = text[7:]
    if text.startswith("```"): text = text[3:]
    if text.endswith("```"): text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                return AnnotationQAAction(action_type="submit")
        else:
            return AnnotationQAAction(action_type="submit")

    # Aggressive sanitization
    action_type = data.get("action_type", "submit")
    if action_type not in ["adjust_bbox", "change_class", "add_annotation", "remove_annotation", "submit"]:
        action_type = "submit"

    ann_id = data.get("annotation_id")
    if isinstance(ann_id, str):
        import re
        match = re.search(r'\d+', ann_id)
        ann_id = int(match.group()) if match else None
    elif isinstance(ann_id, (int, float)):
        ann_id = int(ann_id)

    new_bbox = data.get("new_bbox")
    if new_bbox is not None and isinstance(new_bbox, list):
        clean_bbox = []
        for val in new_bbox:
            try:
                clean_bbox.append(float(val))
            except (ValueError, TypeError):
                clean_bbox.append(0.0)
        # Pad or truncate to exactly 4 items
        clean_bbox = (clean_bbox + [0.0, 0.0, 0.0, 0.0])[:4]
        new_bbox = clean_bbox

    return AnnotationQAAction(
        action_type=action_type,
        annotation_id=ann_id,
        new_bbox=new_bbox,
        new_class=str(data.get("new_class")) if data.get("new_class") else None,
    )


# ──────────────────────────────────────────────
# LLM interaction
# ──────────────────────────────────────────────

def get_model_action(client: OpenAI, obs: AnnotationQAObservation) -> AnnotationQAAction:
    user_content = build_user_content(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=TEMPERATURE,
            max_tokens=500,
            stream=False,
        )
        return parse_llm_response(completion.choices[0].message.content or "")
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return AnnotationQAAction(action_type="submit")


# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────

def run_task(client: OpenAI, env: AnnotationQAEnvironment, task_name: str) -> float:
    max_steps = MAX_STEPS_PER_TASK.get(task_name, 20)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task=task_name, seed=42)

        for step in range(1, max_steps + 1):
            if obs.done:
                break

            action = get_model_action(client, obs)
            
            action_str = f"{action.action_type}"
            if action.annotation_id is not None:
                action_str += f"(id={action.annotation_id})"
                
            obs = env.step(action)
            reward = obs.reward if obs.reward is not None else 0.0
            
            rewards.append(reward)
            steps_taken = step
            log_step(step, action_str, reward, obs.done, obs.last_action_error)

            if obs.done:
                break

        if rewards: score = rewards[-1]
        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} error: {exc}", flush=True)

    log_end(success, steps_taken, score, rewards)
    return score


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = AnnotationQAEnvironment()

    total_score = 0.0
    for task_name in TASKS:
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task_name} (VLM: {MODEL_NAME})", flush=True)
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
