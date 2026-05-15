from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


RISE_ROLE_STEPS: dict[str, tuple[int, ...]] = {
    "orch": (1, 4),
    "ret": (2, 3, 5),
    "syn": (6, 7, 8),
}


@dataclass(frozen=True)
class RISEStep:
    step_id: int
    role: str
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RISETrajectory:
    trajectory_id: str
    question: str
    prediction: str
    answer: str
    steps: list[RISEStep]
    outcome: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "RISETrajectory":
        steps = [
            RISEStep(
                step_id=int(step.get("step_id", step.get("step_idx", idx + 1))),
                role=str(step.get("role", step.get("role_id", "unknown"))),
                text=str(step.get("text", step.get("content", ""))),
                score=float(step.get("score", 0.0)),
                metadata=dict(step.get("metadata", {})),
            )
            for idx, step in enumerate(record.get("steps", []))
        ]
        return cls(
            trajectory_id=str(record.get("trajectory_id", record.get("id", ""))),
            question=str(record.get("question", "")),
            prediction=str(record.get("prediction", record.get("final_answer", ""))),
            answer=str(record.get("answer", record.get("gold_answer", ""))),
            steps=steps,
            outcome=str(record.get("outcome", "unknown")),
            metadata=dict(record.get("metadata", {})),
        )


class RISEPipeline:
    def __init__(self, role_steps: dict[str, tuple[int, ...]] | None = None) -> None:
        self.role_steps = role_steps or RISE_ROLE_STEPS
        self.step_to_role = {
            step_id: role for role, step_ids in self.role_steps.items() for step_id in step_ids
        }

    def role_for_step(self, step_id: int) -> str | None:
        return self.step_to_role.get(int(step_id))

    def normalize_steps(self, trajectory: RISETrajectory) -> RISETrajectory:
        normalized: list[RISEStep] = []
        for step in trajectory.steps:
            role = step.role if step.role != "unknown" else self.role_for_step(step.step_id)
            normalized.append(
                RISEStep(
                    step_id=step.step_id,
                    role=role or "unknown",
                    text=step.text,
                    score=step.score,
                    metadata=step.metadata,
                )
            )
        trajectory.steps = normalized
        return trajectory
