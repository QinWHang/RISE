"""RISE: Role Inference for Selective Evolution."""

from .attribution import RISEAttributor, RISEEvidence, RISERoleScore
from .backends import OpenRLHFRoleBackend, RISEBackendStatus, RISETrainingBackend, build_training_backend
from .config import RISEConfig, config_to_role_steps, load_config
from .data import load_jsonl, normalize_records, validate_record, write_json
from .evidence import RISEEvidenceExtractor, RISEStepEvidence
from .losses import RISELossSpec, loss_spec_for_action, reinforce_loss_spec, repair_loss_spec
from .metrics import RISEQAMetrics, score_answer
from .posterior import RISEPosteriorStrength, beta_posterior_strength
from .pipeline import RISEPipeline, RISETrajectory, RISEStep
from .responsibility import RISEResponsibilityRefiner, RISEResponsibilityView
from .role_graph import RISERoleGraph
from .routing import RISERouteDecision, RISERouter
from .runner import RISERunner
from .trainer import RISETrainer, RISETrainEvent
from .update import RISEUpdatePlan, RISEUpdater

__all__ = [
    "RISEAttributor",
    "OpenRLHFRoleBackend",
    "RISEBackendStatus",
    "RISEConfig",
    "RISETrainingBackend",
    "RISEEvidence",
    "RISEEvidenceExtractor",
    "RISEStepEvidence",
    "RISERoleScore",
    "config_to_role_steps",
    "load_config",
    "load_jsonl",
    "normalize_records",
    "validate_record",
    "write_json",
    "RISELossSpec",
    "loss_spec_for_action",
    "reinforce_loss_spec",
    "repair_loss_spec",
    "RISEQAMetrics",
    "score_answer",
    "RISEPosteriorStrength",
    "beta_posterior_strength",
    "RISEPipeline",
    "RISETrajectory",
    "RISEStep",
    "RISEResponsibilityRefiner",
    "RISEResponsibilityView",
    "RISERoleGraph",
    "RISERouteDecision",
    "RISERouter",
    "RISERunner",
    "RISETrainer",
    "RISETrainEvent",
    "RISEUpdatePlan",
    "RISEUpdater",
    "build_training_backend",
]
