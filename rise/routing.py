from __future__ import annotations

from dataclasses import dataclass

from .attribution import RISERoleScore
from .posterior import top2_margin


@dataclass(frozen=True)
class RISERouteDecision:
    action: str
    role: str | None
    margin: float
    reason: str


class RISERouter:
    def __init__(self, delta: float = 0.03) -> None:
        self.delta = float(delta)

    def route(self, scores: dict[str, RISERoleScore], outcome: str) -> RISERouteDecision:
        if outcome not in {"success", "failure"}:
            return RISERouteDecision(action="freeze_all", role=None, margin=0.0, reason="neutral_or_unknown_outcome")
        field = "positive_strength" if outcome == "success" else "negative_strength"
        values = {role: float(getattr(score, field)) for role, score in scores.items()}
        top1, top2, margin = top2_margin(values)
        if top1 is None or margin < self.delta:
            return RISERouteDecision(action="freeze_all", role=None, margin=margin, reason="top2_margin_below_delta")
        action = "reinforce" if outcome == "success" else "repair"
        return RISERouteDecision(action=action, role=top1, margin=margin, reason="margin_supported_target")
