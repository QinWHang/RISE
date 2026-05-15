from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import RISEConfig, config_to_role_steps
from .data import load_jsonl, normalize_records, write_json
from .evaluation import RISEEvaluator
from .pipeline import RISEPipeline
from .trainer import RISETrainer


class RISERunner:
    def __init__(self, config: RISEConfig) -> None:
        self.config = config
        self.pipeline = RISEPipeline(role_steps=config_to_role_steps(config))

    def train(self) -> dict[str, Any]:
        records = load_jsonl(Path(self.config.train_jsonl))
        trajectories = normalize_records(records)
        trainer = RISETrainer(
            delta=self.config.delta,
            pipeline=self.pipeline,
            positive_threshold=self.config.positive_threshold,
            negative_threshold=self.config.negative_threshold,
        )
        return trainer.fit(trajectories)

    def evaluate(self) -> dict[str, Any]:
        records = load_jsonl(Path(self.config.eval_jsonl))
        evaluator = RISEEvaluator(
            delta=self.config.delta,
            pipeline=self.pipeline,
            positive_threshold=self.config.positive_threshold,
            negative_threshold=self.config.negative_threshold,
        )
        return evaluator.evaluate(records)

    def run(self) -> dict[str, Any]:
        train_summary = self.train()
        eval_summary = self.evaluate()
        payload = {"train": train_summary, "eval": eval_summary}
        write_json(Path(self.config.output_dir) / "rise_run_summary.json", payload)
        return payload
