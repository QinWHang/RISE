from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .attribution import RISEAttributor
from .evidence import RISEEvidenceExtractor
from .metrics import score_answer
from .pipeline import RISEPipeline, RISETrajectory
from .responsibility import RISEResponsibilityRefiner
from .routing import RISERouter
from .update import RISEUpdater


class RISEEvaluator:
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

    def evaluate_record(self, record: dict[str, Any]) -> dict[str, Any]:
        trajectory = self.pipeline.normalize_steps(RISETrajectory.from_record(record))
        evidence = self.extractor.extract(trajectory)
        scores = self.attributor.score_roles(trajectory, evidence)
        decision = self.router.route(scores, trajectory.outcome)
        refined_decision, responsibility_view = self.refiner.refine(decision, evidence)
        update_plan = self.updater.plan(refined_decision)
        metrics = score_answer(trajectory.prediction, trajectory.answer)
        implicated = sorted(self.attributor.implicated_roles(trajectory, evidence))
        return {
            "trajectory_id": trajectory.trajectory_id,
            "f1": metrics.f1,
            "em": metrics.em,
            "outcome": trajectory.outcome,
            "implicated_roles": implicated,
            "implicated_role_count": len(implicated),
            "route": asdict(decision),
            "refined_route": asdict(refined_decision),
            "responsibility": asdict(responsibility_view),
            "update_plan": asdict(update_plan),
            "role_scores": {role: asdict(score) for role, score in scores.items()},
        }

    def evaluate(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        rows = [self.evaluate_record(record) for record in records]
        count = len(rows)
        active_updates = sum(1 for row in rows if row["update_plan"]["active"])
        f1 = sum(float(row["f1"]) for row in rows) / count if count else 0.0
        em = sum(float(row["em"]) for row in rows) / count if count else 0.0
        cardinality = {"1-role": 0, "2-role": 0, "3-role": 0, "other": 0}
        for row in rows:
            role_count = int(row["implicated_role_count"])
            if role_count == 1:
                cardinality["1-role"] += 1
            elif role_count == 2:
                cardinality["2-role"] += 1
            elif role_count >= 3:
                cardinality["3-role"] += 1
            else:
                cardinality["other"] += 1
        return {
            "count": count,
            "f1": f1,
            "em": em,
            "active_update_rate": active_updates / count if count else 0.0,
            "cardinality": cardinality,
            "records": rows,
        }
