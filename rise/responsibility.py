from __future__ import annotations

from dataclasses import dataclass

from .evidence import RISEStepEvidence
from .pipeline import RISEPipeline
from .role_graph import RISERoleGraph
from .routing import RISERouteDecision


@dataclass(frozen=True)
class RISEResponsibilityView:
    symptom_role: str | None
    bottleneck_role: str | None
    implicated_roles: tuple[str, ...]
    propagation_chain: tuple[str, ...]
    reason: str


class RISEResponsibilityRefiner:
    def __init__(self, pipeline: RISEPipeline | None = None) -> None:
        self.pipeline = pipeline or RISEPipeline()
        self.graph = RISERoleGraph(role_steps=self.pipeline.role_steps, edges=(("orch", "ret"), ("ret", "syn"), ("orch", "syn")))

    def _roles_for_steps(self, steps: tuple[int, ...]) -> list[str]:
        roles: list[str] = []
        for step_id in steps:
            role = self.pipeline.role_for_step(step_id)
            if role and role not in roles:
                roles.append(role)
        return roles

    def build_view(self, route: RISERouteDecision, evidence: RISEStepEvidence) -> RISEResponsibilityView:
        negative_roles = self._roles_for_steps(evidence.negative_steps)
        propagated_roles = self._roles_for_steps(evidence.propagated_steps)
        semantic_roles = self._roles_for_steps(evidence.semantic_divergence_steps)
        symptom_role = route.role or (negative_roles[-1] if negative_roles else None)
        bottleneck_role = symptom_role
        chain: tuple[str, ...] = (symptom_role,) if symptom_role else ()
        reason = "local_target"
        if route.action == "repair" and symptom_role is not None:
            candidates = [role for role in semantic_roles or negative_roles if role in self.graph.upstream_roles(symptom_role)]
            supported = [role for role in candidates if role in negative_roles or role in propagated_roles or role in semantic_roles]
            if supported:
                order = {role: idx for idx, role in enumerate(self.pipeline.role_steps)}
                bottleneck_role = sorted(supported, key=lambda role: order.get(role, 10**6))[0]
                chain = self.graph.propagation_chain(bottleneck_role, symptom_role) or (bottleneck_role, symptom_role)
                reason = "upstream_supported"
            elif symptom_role:
                chain = (symptom_role,)
        implicated = tuple(dict.fromkeys([*negative_roles, *propagated_roles, *semantic_roles]))
        return RISEResponsibilityView(
            symptom_role=symptom_role,
            bottleneck_role=bottleneck_role,
            implicated_roles=implicated,
            propagation_chain=chain,
            reason=reason,
        )

    def refine(self, route: RISERouteDecision, evidence: RISEStepEvidence) -> tuple[RISERouteDecision, RISEResponsibilityView]:
        view = self.build_view(route, evidence)
        if route.action != "repair" or route.role is None:
            return route, view
        if view.bottleneck_role is None or view.bottleneck_role == route.role:
            return route, view
        refined = RISERouteDecision(
            action=route.action,
            role=view.bottleneck_role,
            margin=route.margin,
            reason=f"{route.reason}:refined_to_upstream",
        )
        return refined, view
