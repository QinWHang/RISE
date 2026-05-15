from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RISERoleGraph:
    role_steps: dict[str, tuple[int, ...]]
    edges: tuple[tuple[str, str], ...]

    @classmethod
    def canonical(cls) -> "RISERoleGraph":
        return cls(
            role_steps={"orch": (1, 4), "ret": (2, 3, 5), "syn": (6, 7, 8)},
            edges=(("orch", "ret"), ("ret", "syn"), ("orch", "syn")),
        )

    def role_for_step(self, step_id: int) -> str | None:
        for role, steps in self.role_steps.items():
            if int(step_id) in steps:
                return role
        return None

    def upstream_roles(self, role: str) -> tuple[str, ...]:
        upstream: set[str] = set()
        changed = True
        while changed:
            changed = False
            for src, dst in self.edges:
                if dst == role or dst in upstream:
                    if src not in upstream:
                        upstream.add(src)
                        changed = True
        return tuple(sorted(upstream))

    def downstream_roles(self, role: str) -> tuple[str, ...]:
        downstream: set[str] = set()
        changed = True
        while changed:
            changed = False
            for src, dst in self.edges:
                if src == role or src in downstream:
                    if dst not in downstream:
                        downstream.add(dst)
                        changed = True
        return tuple(sorted(downstream))

    def propagation_chain(self, source: str, target: str) -> tuple[str, ...]:
        if source == target:
            return (source,)
        agenda: list[tuple[str, tuple[str, ...]]] = [(source, (source,))]
        visited: set[str] = set()
        while agenda:
            role, path = agenda.pop(0)
            if role in visited:
                continue
            visited.add(role)
            for src, dst in self.edges:
                if src != role:
                    continue
                next_path = path + (dst,)
                if dst == target:
                    return next_path
                agenda.append((dst, next_path))
        return ()
