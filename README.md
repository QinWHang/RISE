# RISE
Official implementation of "Who Should Evolve? Uncertainty-Aware Role Bottleneck Inference for Multi-Agent LLM Training" with PyTorch.
RISE converts step-level trajectory evidence into role-level update decisions for role-partitioned multi-step reasoning pipelines.

## OpenRLHF Integration Points

RISE is not a standalone trainer in this snapshot. It uses OpenRLHF components directly:

- `openrlhf.models.actor.Actor` for role policies.
- `openrlhf.models.loss.DPOLoss` for repair-side preference optimization.
- `openrlhf.utils.get_strategy` and `openrlhf.utils.deepspeed.DeepspeedStrategy` for DeepSpeed-backed training.
- `DeepSpeedRoleSelectiveUpdateWrapper` to freeze non-selected roles and step only the selected role.
  
## Structure

```text
RISE_CODE_FULL/
  openrlhf/      OpenRLHF actor, loss, DeepSpeed strategy, trainer, dataset, and utility modules
  scripts/       Experiment, calibration, evaluation, and preprocessing entry points
  configs/       Method and judge configuration files
  prompts/       Judge and role prompt templates
  requirements.txt/
```
## Installation

```bash
cd RISE
pip install -r requirements.txt
```

## Main RISE Method Path

The main experiment entrypoint is:

```bash
python scripts/train_openrlhf_experiment.py
```

The formal runner builds the role-partitioned QA pipeline, creates role policies, performs routing and attribution, and dispatches updates through either local debug training or the OpenRLHF-backed experiment trainer.
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

