# RISE
Official implementation of "Who Should Evolve? Uncertainty-Aware Role Bottleneck Inference for Multi-Agent LLM Training" with PyTorch.
RISE converts step-level trajectory evidence into role-level update decisions for role-partitioned multi-step reasoning pipelines.

## Structure

```text
RISE/
  rise/                 # RISE public interface
  scripts/              # Execution entry points
  configs/              # Relative-path experiment configuration
  examples/             # Small runnable JSONL example
  requirements.txt
```

## Installation

```bash
cd RISE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```bash
cd RISE
PYTHONPATH=. python scripts/run_rise_train.py \
  --input examples/toy_qa.jsonl \
  --output outputs/rise_train_summary.json \
  --delta 0.03

PYTHONPATH=. python scripts/run_rise_eval.py \
  --input examples/toy_qa.jsonl \
  --output outputs/rise_eval_summary.json \
  --delta 0.03

PYTHONPATH=. python scripts/run_rise_pipeline.py \
  --config configs/rise_hotpotqa.yaml
```

## Input Format

Each input line is a JSON object:

```json
{
  "id": "example-id",
  "question": "question text",
  "prediction": "model answer",
  "answer": "gold answer",
  "outcome": "success or failure",
  "steps": [
    {"step_id": 1, "role": "orch", "text": "planning step", "score": 0.2},
    {"step_id": 5, "role": "ret", "text": "retrieval step", "score": -0.4},
    {"step_id": 8, "role": "syn", "text": "final answer step", "score": -0.2}
  ]
}
```

Positive step scores support reinforcement. Negative step scores support repair.
The default role-step contract is:

```text
orch: steps 1, 4
ret:  steps 2, 3, 5
syn:  steps 6, 7, 8
```

