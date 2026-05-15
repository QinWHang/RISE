from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RISEPosteriorStrength:
    mean: float
    std: float
    lower_confidence: float


def beta_posterior_strength(successes: int, trials: int, z: float = 1.0) -> RISEPosteriorStrength:
    alpha = 1.0 + max(0, int(successes))
    beta = 1.0 + max(0, int(trials) - max(0, int(successes)))
    total = alpha + beta
    mean = alpha / total
    variance = (alpha * beta) / ((total * total) * (total + 1.0))
    std = math.sqrt(max(variance, 0.0))
    return RISEPosteriorStrength(mean=mean, std=std, lower_confidence=mean - float(z) * std)


def top2_margin(values: dict[str, float]) -> tuple[str | None, str | None, float]:
    if not values:
        return None, None, 0.0
    ranked = sorted(values.items(), key=lambda item: item[1], reverse=True)
    top1_role, top1_value = ranked[0]
    if len(ranked) == 1:
        return top1_role, None, float("inf")
    top2_role, top2_value = ranked[1]
    return top1_role, top2_role, float(top1_value - top2_value)
