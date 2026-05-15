from __future__ import annotations

from dataclasses import dataclass

from .pipeline import RISETrajectory


@dataclass(frozen=True)
class RISEStepEvidence:
    positive_steps: tuple[int, ...]
    negative_steps: tuple[int, ...]
    propagated_steps: tuple[int, ...]
    semantic_divergence_steps: tuple[int, ...]


class RISEEvidenceExtractor:
    def __init__(self, positive_threshold: float = 0.05, negative_threshold: float = -0.05) -> None:
        self.positive_threshold = float(positive_threshold)
        self.negative_threshold = float(negative_threshold)

    def extract(self, trajectory: RISETrajectory) -> RISEStepEvidence:
        positive: list[int] = []
        negative: list[int] = []
        propagated: list[int] = []
        semantic: list[int] = []
        for step in trajectory.steps:
            if step.score >= self.positive_threshold:
                positive.append(step.step_id)
            if step.score <= self.negative_threshold:
                negative.append(step.step_id)
            if bool(step.metadata.get("propagated_error", False)):
                propagated.append(step.step_id)
            if bool(step.metadata.get("semantic_divergence", False)) or step.score <= self.negative_threshold:
                semantic.append(step.step_id)
        return RISEStepEvidence(
            positive_steps=tuple(sorted(set(positive))),
            negative_steps=tuple(sorted(set(negative))),
            propagated_steps=tuple(sorted(set(propagated))),
            semantic_divergence_steps=tuple(sorted(set(semantic))),
        )
