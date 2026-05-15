from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .update import RISEUpdatePlan


@dataclass(frozen=True)
class RISEBackendStatus:
    name: str
    available: bool
    trainable: bool
    detail: str


class RISETrainingBackend:
    name = "public"

    def status(self) -> RISEBackendStatus:
        return RISEBackendStatus(
            name=self.name,
            available=True,
            trainable=False,
            detail="public update-plan backend",
        )

    def apply_update(self, plan: RISEUpdatePlan, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "backend": self.name,
            "active": bool(plan.active),
            "role": plan.role,
            "loss": plan.loss_name,
            "applied": False,
        }


class OpenRLHFRoleBackend(RISETrainingBackend):
    name = "openrlhf"

    def __init__(self) -> None:
        self._available = self._check_available()

    @staticmethod
    def _check_available() -> bool:
        try:
            import openrlhf  # noqa: F401
        except Exception:
            return False
        return True

    def status(self) -> RISEBackendStatus:
        return RISEBackendStatus(
            name=self.name,
            available=self._available,
            trainable=self._available,
            detail="optional OpenRLHF-compatible role-policy backend",
        )

    def apply_update(self, plan: RISEUpdatePlan, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        status = self.status()
        return {
            "backend": self.name,
            "active": bool(plan.active),
            "role": plan.role,
            "loss": plan.loss_name,
            "applied": False,
            "status": asdict(status),
        }


def build_training_backend(name: str = "public") -> RISETrainingBackend:
    normalized = str(name or "public").lower()
    if normalized in {"public", "plan", "dry"}:
        return RISETrainingBackend()
    if normalized in {"openrlhf", "openrlhf_role"}:
        return OpenRLHFRoleBackend()
    return RISETrainingBackend()
