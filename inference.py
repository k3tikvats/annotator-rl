"""
Inference Script — Annotation QA Environment (72B One-Shot VQA + Set-of-Mark)
==========================================================
MANDATORY
- Before submitting, ensure the following variables are defined:
    API_BASE_URL   The API endpoint for the VLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- STDOUT MUST EXACTLY follow [START], [STEP], and [END] formats.

72B ONE-SHOT VQA APPROACH
- Uses Qwen2.5-VL-72B-Instruct for incredibly high spatial accuracy.
- To bypass rigid API rate limits and token costs, the script makes EXACTLY 
  ONE API CALL per image. 
- The VLM acts as a visual reviewer, grading every single box in text format.
- The Python loop then mechanically executes those parsed actions.
"""

import base64
import io
import os
import re
import sys
import textwrap
import urllib.request
from typing import List, Optional

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

# We test OPENAI_API_KEY natively per spec requirement, falling back to HF_TOKEN for Serverless Inference.
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-VL-72B-Instruct")

BENCHMARK = "annotation_qa_env"
TASKS = ["remove_spurious", "fix_classes", "find_missing"]
MAX_STEPS_PER_TASK = {"remove_spurious": 15, "fix_classes": 20, "find_missing": 30}
TEMPERATURE = 0.2
MAX_TOKENS = 1500
SUCCESS_SCORE_THRESHOLD = 0.1
SCORE_EPSILON = 0.001

DEFAULT_FALLBACK_SCORE = 0.001

# Raw Image cache
_raw_image_cache = {}

SYSTEM_PROMPT = textwrap.dedent("""
You are a highly precise AI visual inspector reviewing annotated datasets.
You will be provided an image containing multiple drawn objects.
Every object has a thick colored bounding box and a distinct label showing `[ID: <number> | <class_label>]`.

Your task is to analyze EVERY SINGLE box drawn on the image systematically and check for errors, policy violations, incorrect attributes, or completely missing background objects.

IF the box tightly binds the object, the label is exactly correct, and it does not violate any safety policies, its status is KEEP.

You MUST respond strictly with a line-by-line list grading every single ID you see on the screen.
You may also append FLAG_MISSING commands at the very end of your list for objects that the annotator forgot to draw a box around.

Use EXACTLY this format and nothing else:

ID <number>: KEEP
ID <number>: CHANGE_CLASS <new_correct_class_name>
ID <number>: REMOVE
ID <number>: FLAG_SAFETY
ID <number>: CHANGE_ATTRIBUTE <new_attribute_name>
FLAG_MISSING: <missing_class_name>

Example Output:
ID 0: KEEP
ID 1: CHANGE_CLASS truck
ID 2: REMOVE
ID 3: FLAG_SAFETY
ID 14: KEEP
ID 15: CHANGE_ATTRIBUTE red skateboard
FLAG_MISSING: person
FLAG_MISSING: bicycle

Do NOT Output any other text, no intro, no json, no explanation. Just the list.
""").strip()

# ──────────────────────────────────────────────
# Logging helpers
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
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def clamp_open_score(score: float) -> float:
    """Clamp scores to the strict open interval (0, 1)."""
    return min(1.0 - SCORE_EPSILON, max(SCORE_EPSILON, score))


# ──────────────────────────────────────────────
# Image Overlays
# ──────────────────────────────────────────────

def get_base_image(image_url: str, max_dim: int = 768):
    from PIL import Image

    if image_url in _raw_image_cache:
        return _raw_image_cache[image_url]

    try:
        req = urllib.request.Request(image_url, headers={"User-Agent": "AnnotationQA/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            img_bytes = resp.read()

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
        # For 72B VQA, higher resolution is better. Scale proportionally.
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
    try:
        from PIL import ImageDraw, ImageFont
    except ImportError:
        return "" 

    img = get_base_image(obs.image_url)
    if img is None:
        return ""

    canvas = img.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    w, h = canvas.size

    try:
        fontsize = max(14, int(h * 0.03))
        try:
            font = ImageFont.truetype("arial.ttf", fontsize)
        except OSError:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", fontsize)
            except OSError:
                font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

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
        
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        
        label_text = f" ID:{ann.id} | {ann.class_label} "
        try:
            bbox = font.getbbox(label_text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = 60, 15
            
        bg_rect = [x0, max(0, y0 - text_h - 4), x0 + text_w, y0]
        draw.rectangle(bg_rect, fill=color)
        draw.text((x0, max(0, y0 - text_h - 4)), label_text, fill=(0,0,0,255), font=font)

    if debug_save:
        canvas.save("debug_overlay_test.jpg")

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ──────────────────────────────────────────────
# Prompt building
# ──────────────────────────────────────────────

def build_user_content(obs: AnnotationQAObservation) -> list:
    content_blocks = []

    if obs.image_url:
        save_debug = (obs.step_count == 0)
        b64 = fetch_annotated_image_as_base64(obs, debug_save=save_debug)
        if b64:
            content_blocks.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                },
            })

    # Prepare an inventory list of existing IDs so the VLM knows what needs checking
    inventory = [f"ID {a.id}: {a.class_label}" for a in obs.annotations]

    text = f"""Please analyze this image. The bounding boxes are clearly drawn with their current labels.
All valid standard COCO Classes are supported.

Here is the inventory of boxes on screen you MUST review:
{ chr(10).join(inventory) }

Provide your final line-by-line grading of every ID now:
"""
    content_blocks.append({
        "type": "text",
        "text": text,
    })

    return content_blocks


def parse_vqa_actions(response_text: str) -> List[AnnotationQAAction]:
    """Parse the line-by-line plain text output into distinct discrete actions."""
    text = response_text.strip()
    actions = []
    
    # regex match for "ID X: CHANGE_CLASS dog" or "ID Y: REMOVE"
    lines = text.split('\n')
    for line in lines:
        line = line.strip()

        # 1. Check for FLAG_MISSING (which doesn't have an ID)
        match_missing = re.search(r'FLAG_MISSING:\s*(.+)', line, re.IGNORECASE)
        if match_missing:
            m_class = match_missing.group(1).strip().lower()
            actions.append(AnnotationQAAction(
                action_type="flag_missing",
                missing_class=m_class
            ))
            continue

        # 2. Check for ID-based commands
        match = re.search(r'ID\s*(\d+)[:\-\s]+(.+)', line, re.IGNORECASE)
        if not match:
            continue
            
        ann_id = int(match.group(1))
        instruction = match.group(2).strip().upper()
        
        if instruction.startswith("REMOVE"):
            actions.append(AnnotationQAAction(
                action_type="remove_annotation",
                annotation_id=ann_id
            ))
        elif instruction.startswith("CHANGE_CLASS") or instruction.startswith("CHANGE"):
            parts = instruction.split()
            if len(parts) > 1:
                new_class = " ".join(parts[1:]).lower()
                actions.append(AnnotationQAAction(
                    action_type="change_class",
                    annotation_id=ann_id,
                    new_class=new_class
                ))
        elif instruction.startswith("FLAG_SAFETY"):
            actions.append(AnnotationQAAction(
                action_type="flag_safety",
                annotation_id=ann_id
            ))
        elif instruction.startswith("CHANGE_ATTRIBUTE"):
            parts = instruction.split()
            if len(parts) > 1:
                new_attr = " ".join(parts[1:]).lower()
                actions.append(AnnotationQAAction(
                    action_type="change_attribute",
                    annotation_id=ann_id,
                    new_attribute=new_attr
                ))
                
    return actions


# ──────────────────────────────────────────────
# Execution logic
# ──────────────────────────────────────────────

def get_vqa_actions(client: OpenAI, obs: AnnotationQAObservation) -> List[AnnotationQAAction]:
    user_content = build_user_content(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        response_text = completion.choices[0].message.content or ""
        print(f"[DEBUG] VLM Output:\n{response_text}\n", flush=True)
        return parse_vqa_actions(response_text)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return []


def run_task(client: OpenAI, env: AnnotationQAEnvironment, task_name: str) -> float:
    global _raw_image_cache
    _raw_image_cache = {} 

    obs = env.reset(task=task_name, seed=42)
    max_steps = MAX_STEPS_PER_TASK.get(task_name, 20)
    rewards: List[float] = []
    steps_taken = 0
    score = DEFAULT_FALLBACK_SCORE
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # 1. ONE-SHOT VISUAL INSPECTION
        # The script makes exactly ONE api call to grade the image
        actions_to_take = get_vqa_actions(client, obs)

        # 2. LOCAL SEQUENTIAL EXECUTION
        # Loop through actions independently locally
        for action in actions_to_take:
            if obs.done or steps_taken >= max_steps:
                break
                
            steps_taken += 1
            action_str = f"{action.action_type}("
            if action.annotation_id is not None:
                action_str += f"id={action.annotation_id}"
            if action.new_class:
                action_str += f" cls={action.new_class}"
            if action.new_attribute:
                action_str += f" attr={action.new_attribute}"
            if action.missing_class:
                action_str += f" missing={action.missing_class}"
            action_str += ")"

            obs = env.step(action)
            reward = obs.reward if obs.reward is not None else 0.0
            rewards.append(reward)
            
            log_step(steps_taken, action_str, reward, obs.done, obs.last_action_error)

        # 3. SUBMIT
        if not obs.done and steps_taken < max_steps:
            steps_taken += 1
            obs = env.step(AnnotationQAAction(action_type="submit"))
            reward = obs.reward if obs.reward is not None else 0.0
            rewards.append(reward)
            log_step(steps_taken, "submit", reward, obs.done, obs.last_action_error)

        if rewards:
            score = rewards[-1]

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} error: {exc}", flush=True)

    score = clamp_open_score(score)
    success = score >= SUCCESS_SCORE_THRESHOLD

    log_end(success, steps_taken, score, rewards)
    return score


def main() -> None:
    env = AnnotationQAEnvironment()

    if not API_KEY:
        print("[DEBUG] Missing OPENAI_API_KEY/HF_TOKEN. Falling back to minimal score mode.", flush=True)
        client = None
    else:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY, timeout=600.0)
        except Exception as exc:
            print(f"[DEBUG] OpenAI client initialization failed: {exc}", flush=True)
            client = None

    total_score = 0.0
    for task_name in TASKS:
        if client is None:
            # Preserve required START/END logging shape even without model credentials.
            log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
            score = clamp_open_score(DEFAULT_FALLBACK_SCORE)
            log_end(False, 0, score, [score])
        else:
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
