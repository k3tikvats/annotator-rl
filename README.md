---
title: Annotation QA Env
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
---
# 🔍 Annotation QA Environment

An **OpenEnv** environment where an AI agent reviews and corrects intentionally-flawed ML annotations on synthetic scenes. Built for the [Meta OpenEnv × SST Hackathon](https://github.com/meta-pytorch/OpenEnv).

## 🎯 The Challenge

Real-world ML training data is noisy. Annotation teams make mistakes — bounding boxes drift, class labels get swapped, objects get missed. This environment simulates that review pipeline:

1. **Agent receives** a scene description + current annotations (some are wrong)
2. **Agent identifies** errors by comparing annotations to scene objects
3. **Agent corrects** errors through bbox adjustments, class changes, additions, and removals
4. **Agent submits** and receives a score based on annotation quality improvement

## 📋 Tasks (3 Difficulty Levels)

| Task | Difficulty | Errors | Max Steps |
|------|-----------|--------|-----------|
| `fix_bboxes` | Easy | Bbox expansion, shifting, shrinking, spurious, missing | 15 |
| `fix_classes` | Medium | Bbox errors + class label confusion (car↔truck, dog↔cat) | 20 |
| `batch_audit` | Hard | Subtle bbox shifts + similar-class confusion + cross-batch issues | 30 |

## 🏗️ Architecture

```
annotation_qa_env/
├── models.py              ← Action, Observation, State (Pydantic)
├── client.py              ← EnvClient for WebSocket interaction
├── inference.py           ← Baseline LLM agent (OpenAI client)
├── server/
│   ├── environment.py     ← Core game logic (reset, step, state)
│   ├── grader.py          ← IoU-based deterministic grading
│   ├── corruption.py      ← Annotation corruption strategies
│   ├── app.py             ← FastAPI server
│   └── Dockerfile         ← Container definition
└── data/
    └── generate_dataset.py ← Synthetic scene generator
```

## 🚀 Quick Start

### Install & Run Locally
```bash
cd annotation_qa_env
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Use the Client
```python
from annotation_qa_env import AnnotationQAEnv, AnnotationQAAction

with AnnotationQAEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task="fix_bboxes")
    print(result.observation.annotations)

    result = env.step(AnnotationQAAction(
        action_type="adjust_bbox",
        annotation_id=0,
        new_bbox=[0.1, 0.2, 0.15, 0.1],
    ))
    print(f"Reward: {result.reward}")
```

### Docker
```bash
docker build -t annotation-qa-env:latest -f server/Dockerfile .
docker run -d -p 8000:8000 annotation-qa-env:latest
```

### Deploy to HF Spaces
```bash
openenv push --repo-id username/annotation-qa-env
```

## 📊 Grading

The grading function is **deterministic** and returns scores in `[0.0, 1.0]`:

```
Score = (final_quality - initial_quality) / (1.0 - initial_quality)
```

Where `quality` is a weighted composite of:
- **Mean IoU** (40%) — How well do predicted bboxes overlap with gold?
- **Class Accuracy** (30%) — Are class labels correct?
- **Precision** (15%) — Are there spurious annotations?
- **Recall** (15%) — Are there missing annotations?

## 🤖 Actions

| Action | Required Fields | Description |
|--------|----------------|-------------|
| `adjust_bbox` | `annotation_id`, `new_bbox` | Fix a bounding box |
| `change_class` | `annotation_id`, `new_class` | Fix a class label |
| `add_annotation` | `new_bbox`, `new_class` | Add a missing annotation |
| `remove_annotation` | `annotation_id` | Remove a spurious annotation |
| `submit` | (none) | Finalize corrections |

## 📦 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model for inference |
| `HF_TOKEN` | — | API key |

## 🔬 Why Synthetic Scenes?

We use programmatic scene descriptions instead of real COCO images because:

1. **Docker size**: COCO train2017 is ~18GB — exceeds container limits
2. **Memory**: Base64 images in observations would spike past 8GB RAM
3. **LLM text-only**: Evaluation uses text-only LLMs (no vision models)
4. **Determinism**: Same seed = same data = reproducible scores
5. **Zero setup**: No dataset download — everything is self-contained

The annotation QA task is fundamentally about **spatial + categorical reasoning**, which text captures fully.

## 📜 License

BSD-3-Clause (matching OpenEnv)
