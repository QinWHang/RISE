from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RISELossSpec:
    name: str
    role: str
    weight: float
    objective: str


def reinforce_loss_spec(role: str, weight: float = 1.0) -> RISELossSpec:
    return RISELossSpec(name="rise_reinforce", role=role, weight=float(weight), objective="reinforce_selected_role")


def repair_loss_spec(role: str, weight: float = 1.0) -> RISELossSpec:
    return RISELossSpec(name="rise_repair", role=role, weight=float(weight), objective="anchored_repair_selected_role")


def loss_spec_for_action(action: str, role: str | None) -> RISELossSpec | None:
    if role is None or action == "freeze_all":
        return None
    if action == "reinforce":
        return reinforce_loss_spec(role)
    if action == "repair":
        return repair_loss_spec(role)
    return None
