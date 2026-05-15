from __future__ import annotations

from dataclasses import dataclass

from .losses import loss_spec_for_action
from .routing import RISERouteDecision


@dataclass(frozen=True)
class RISEUpdatePlan:
    action: str
    role: str | None
    loss_name: str | None
    active: bool


class RISEUpdater:
    def plan(self, decision: RISERouteDecision) -> RISEUpdatePlan:
        loss_spec = loss_spec_for_action(decision.action, decision.role)
        return RISEUpdatePlan(
            action=decision.action,
            role=decision.role if loss_spec is not None else None,
            loss_name=loss_spec.name if loss_spec is not None else None,
            active=loss_spec is not None,
        )
