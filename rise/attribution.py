from __future__ import annotations

from dataclasses import dataclass

from .evidence import RISEStepEvidence
from .pipeline import RISEPipeline, RISETrajectory
from .posterior import beta_posterior_strength


@dataclass(frozen=True)
class RISEEvidence:
    positive_steps: tuple[int, ...] = ()
    negative_steps: tuple[int, ...] = ()
    propagated_steps: tuple[int, ...] = ()


@dataclass(frozen=True)
class RISERoleScore:
    role: str
    positive_count: int
    negative_count: int
    total_steps: int
    positive_strength: float
    negative_strength: float


class RISEAttributor:
    def __init__(self, pipeline: RISEPipeline | None = None) -> None:
        self.pipeline = pipeline or RISEPipeline()

    def extract_evidence(self, trajectory: RISETrajectory) -> RISEEvidence:
        positive = []
        negative = []
        propagated = []
        for step in trajectory.steps:
            if step.score > 0.0:
                positive.append(step.step_id)
            elif step.score < 0.0:
                negative.append(step.step_id)
            if bool(step.metadata.get("propagated_error", False)):
                propagated.append(step.step_id)
        return RISEEvidence(
            positive_steps=tuple(sorted(set(positive))),
            negative_steps=tuple(sorted(set(negative))),
            propagated_steps=tuple(sorted(set(propagated))),
        )

    def score_roles(self, trajectory: RISETrajectory, evidence: RISEEvidence | RISEStepEvidence | None = None) -> dict[str, RISERoleScore]:
        trajectory = self.pipeline.normalize_steps(trajectory)
        if evidence is None:
            evidence = self.extract_evidence(trajectory)
        positive_steps = set(evidence.positive_steps)
        negative_steps = set(evidence.negative_steps)
        scores: dict[str, RISERoleScore] = {}
        for role, step_ids in self.pipeline.role_steps.items():
            role_step_set = set(step_ids)
            positive_count = len(role_step_set & positive_steps)
            negative_count = len(role_step_set & negative_steps)
            total_steps = len([step for step in trajectory.steps if step.role == role])
            total_steps = max(total_steps, len(step_ids), positive_count + negative_count)
            positive_strength = beta_posterior_strength(positive_count, total_steps).lower_confidence
            negative_strength = beta_posterior_strength(negative_count, total_steps).lower_confidence
            scores[role] = RISERoleScore(
                role=role,
                positive_count=positive_count,
                negative_count=negative_count,
                total_steps=total_steps,
                positive_strength=positive_strength,
                negative_strength=negative_strength,
            )
        return scores

    def implicated_roles(self, trajectory: RISETrajectory, evidence: RISEEvidence | RISEStepEvidence | None = None) -> set[str]:
        trajectory = self.pipeline.normalize_steps(trajectory)
        if evidence is None:
            evidence = self.extract_evidence(trajectory)
        implicated_steps = set(evidence.negative_steps) | set(evidence.propagated_steps)
        roles = {
            self.pipeline.role_for_step(step_id)
            for step_id in implicated_steps
        }
        return {role for role in roles if role is not None}
