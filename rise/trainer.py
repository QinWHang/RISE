from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .attribution import RISEAttributor
from .evidence import RISEEvidenceExtractor
from .pipeline import RISEPipeline, RISETrajectory
from .responsibility import RISEResponsibilityRefiner
from .routing import RISERouter
from .update import RISEUpdater


@dataclass(frozen=True)
class RISETrainEvent:
    trajectory_id: str
    action: str
    selected_role: str | None
    update_role: str | None
    active: bool
    reason: str
    responsibility_reason: str


class RISETrainer:
    def __init__(
        self,
        delta: float = 0.03,
        pipeline: RISEPipeline | None = None,
        positive_threshold: float = 0.05,
        negative_threshold: float = -0.05,
    ) -> None:
        self.pipeline = pipeline or RISEPipeline()
        self.extractor = RISEEvidenceExtractor(
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
        )
        self.attributor = RISEAttributor(self.pipeline)
        self.router = RISERouter(delta=delta)
        self.refiner = RISEResponsibilityRefiner(self.pipeline)
        self.updater = RISEUpdater()

    def step(self, trajectory: RISETrajectory) -> RISETrainEvent:
        trajectory = self.pipeline.normalize_steps(trajectory)
        evidence = self.extractor.extract(trajectory)
        scores = self.attributor.score_roles(trajectory, evidence)
        route = self.router.route(scores, trajectory.outcome)
        refined_route, view = self.refiner.refine(route, evidence)
        update_plan = self.updater.plan(refined_route)
        return RISETrainEvent(
            trajectory_id=trajectory.trajectory_id,
            action=refined_route.action,
            selected_role=route.role,
            update_role=update_plan.role,
            active=update_plan.active,
            reason=refined_route.reason,
            responsibility_reason=view.reason,
        )

    def fit(self, trajectories: list[RISETrajectory]) -> dict[str, Any]:
        events = [self.step(trajectory) for trajectory in trajectories]
        active = sum(1 for event in events if event.active)
        freeze = sum(1 for event in events if event.action == "freeze_all")
        role_counts: dict[str, int] = {}
        for event in events:
            if event.update_role:
                role_counts[event.update_role] = role_counts.get(event.update_role, 0) + 1
        return {
            "trajectory_count": len(events),
            "active_updates": active,
            "freeze_count": freeze,
            "active_update_rate": active / len(events) if events else 0.0,
            "update_role_counts": role_counts,
            "events": [asdict(event) for event in events],
        }
