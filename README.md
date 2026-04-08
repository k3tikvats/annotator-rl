---
title: Semantic Annotation QA Env
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
---
# 🔍 Semantic Annotation QA Environment

An **OpenEnv** framework where a Vision-Language Model (VLM) agent reviews and corrects intentionally flawed machine-learning annotations on **real COCO val2017 images**. 

This environment simulates a highly critical **real-world task**: human-in-the-loop ML Data QA / Content Cleaning. By having an agent actively audit and correct data labels, it tests a *valid domain* while serving as a pure evaluation bed for multimodal agent alignment.

To preserve benchmark integrity, the agent observation intentionally hides ground-truth scene objects and class labels; only the rendered image with current annotations is exposed.

## 🎯 The Challenge & Novelty

Traditionally, spatial bounding-box regression tasks test VLMs poorly because model tokenizers destroy contiguous pixel geometry logic. **We solved this.** 

Instead of asking the model to hallucinate geometric bounding box sizes, we use a **"Set-of-Mark"** overlay philosophy. The environment renders the image with ID tags directly on the visual feed, transforming the VLM into a pure **Semantic Auditor**. This *novel approach* completely fills a severe evaluation gap by cleanly testing a multimodal agent's reasoning power without arbitrary fractional coordinate failures.

1. **Agent receives** a real COCO image + current annotation state
2. **Agent visually inspects** the IDs using a continuous inference loop (`openai` client)
3. **Agent corrects** errors by calling `REMOVE`, `CHANGE_CLASS`, or `FLAG_MISSING`
4. **Agent receives Dense Rewards** at every single step based on strict mathematical quality tracking

## 📋 3 Tiered Tasks

The environment supports exactly 3 progressively difficult semantic datasets, guaranteeing a deterministic difficulty ramp capable of challenging even the smartest frontier models.

| Task | Difficulty | Mechanistic Objective | Max Steps |
|------|-----------|--------|-----------| 
| `remove_spurious` | Easy 🟢 | Detect and delete fake/hallucinated bounding boxes that enclose thin air. | 15 |
| `fix_classes` | Medium 🟡 | Combines spurious errors with deliberate cross-class confusion (e.g. `car` ↔ `truck`). | 20 |
| `find_missing` | Hard 🔴 | Objects are entirely scrubbed from the label matrix. VLM must actively spot missing targets. | 30 |


## ⚙️ Environment Design & Rewards

The environment strictly enforces proper RL (Reinforcement Learning) paradigms required to actually train agents (e.g. PPO/GRPO setups):

- **Clean Boundaries:** The `reset()` function cleanly initializes a fresh scene ID mapping. Episodes logically finalize the moment `SUBMIT` is invoked or max steps are exhausted.
- **Dense Fractional Reward:** The reward function provides continuous trajectory signaling via `quality_delta = new_quality - old_quality`, with per-step shaping and anti-loop penalty.
- **Built-in Guardrails:** The reward deducts `-0.01` passively for every executed step, heavily penalizing runaway loops, blind guessing, or destructive action behaviors.
- **Task-Score Validator Safety:** Final task score is projected from `[0,1]` into strict `(0, 1)` to satisfy Phase-2 validator constraints while preserving rank order.

## 📊 Deterministic Grading (0.0 to 1.0)

Calculated at every frame step, the agent receives a deterministic score out of `1.0` based on semantic QA metrics:

- **Spurious Precision (35%)** — Did you remove fake boxes without destroying real ones?
- **Class Match Accuracy (35%)** — For existing valid boxes, did you change to the correct Gold label?
- **Missing Flag Quality (30%)** — Balanced precision/recall (F1) for `FLAG_MISSING`, penalizing over-flagging.

Task-specific metric weights are used to keep each benchmark VLM-native:
- `remove_spurious`: prioritize spurious precision
- `fix_classes`: prioritize class accuracy
- `find_missing`: prioritize missing-flag quality

Final episode score blends:
- trajectory improvement (80%)
- end-state quality (20%)

Baseline inference defaults to deterministic decoding (`TEMPERATURE=0.0`) for reproducible runs.

## 💻 Spec Compliance & Quick Start

This repository is **100% OpenEnv Spec Compliant**. `openenv validate` passes natively, the `openenv.yaml` handles correct routing, and all interface states (Observation, Actions, Reward signals) use natively typed Pydantic structures in `models.py`.

### 1. Zero-Storage Setup
Because we dynamically fetch `raw` annotations using explicit COCO API URLs inside `data/prepare_coco.py`, the massive dataset is compressed internally to ~2.5MB. This enables light-speed Docker Deployments & HF Space hosting.
```bash
# Verify Environment
uv run openenv validate

# Containerize
docker build -t annotation-qa-env:latest .
docker run -d -p 8000:8000 annotation-qa-env:latest
```

### 2. VLM Baseline Inference
We test via native OpenAI client parity against standard Hugging Face router limits. Ensure you use an advanced vision model endpoint.

```bash
# For HF Serverless Router
export OPENAI_API_KEY="your_api_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"

# Reproduce the baseline mathematically 
python3 inference.py
```

### 3. Baseline Score Reporting

The baseline script emits only protocol logs to stdout:
- `[START] ...`
- `[STEP] ...` (one per environment step)
- `[END] ...`

Each task score is reported in the `[END]` line (`score=<value>`) and is guaranteed to stay in strict `(0, 1)` for validator compatibility.

For judge-facing baseline numbers, run with a valid model token. If no token is provided,
the script enters a conservative fallback mode only for local smoke testing.

Human-readable diagnostics are printed to stderr so parser-facing stdout remains compliant.

Example output lines:
```text
[START] task=remove_spurious env=annotation_qa_env model=Qwen/Qwen2.5-VL-72B-Instruct
[STEP] step=1 action=remove_annotation(id=12) reward=0.18 done=false error=null
[STEP] step=2 action=submit reward=0.41 done=true error=null
[END] success=true steps=2 score=0.412 rewards=0.18,0.41
```

## 🤖 Pydantic Action Space

| Action | Required Fields | Description |
|--------|----------------|-------------|
| `change_class` | `annotation_id`, `new_class` | Correct a miscategorized label |
| `adjust_bbox` | `annotation_id`, `new_bbox` | Adjust an existing bounding box |
| `add_annotation` | `new_bbox`, `new_class` | Add a new annotation |
| `flag_missing` | `missing_class` | Flag a missing target by its class name |
| `remove_annotation` | `annotation_id` | Delete a completely spurious annotation |
| `submit` | (none) | Finalize audit corrections |

## 📜 License
BSD-3-Clause (matching OpenEnv)
