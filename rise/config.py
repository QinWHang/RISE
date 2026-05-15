from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RISEConfig:
    method: str = "RISE"
    dataset: str = "custom"
    model: str = "local"
    delta: float = 0.03
    role_step_contract: dict[str, list[int]] = field(
        default_factory=lambda: {"orch": [1, 4], "ret": [2, 3, 5], "syn": [6, 7, 8]}
    )
    positive_threshold: float = 0.05
    negative_threshold: float = -0.05
    training_backend: str = "public"
    train_jsonl: str = "examples/toy_qa.jsonl"
    eval_jsonl: str = "examples/toy_qa.jsonl"
    output_dir: str = "outputs"


def load_config(path: Path) -> RISEConfig:
    payload = _load_mapping(path)
    evidence = payload.get("evidence", {}) or {}
    training = payload.get("training", {}) or {}
    paths = payload.get("paths", {}) or {}
    return RISEConfig(
        method=str(payload.get("method", "RISE")),
        dataset=str(payload.get("dataset", "custom")),
        model=str(payload.get("model", "local")),
        delta=float(payload.get("delta", 0.03)),
        role_step_contract={
            str(role): [int(step) for step in steps]
            for role, steps in (payload.get("role_step_contract") or {}).items()
        }
        or {"orch": [1, 4], "ret": [2, 3, 5], "syn": [6, 7, 8]},
        positive_threshold=float(evidence.get("positive_threshold", 0.05)),
        negative_threshold=float(evidence.get("negative_threshold", -0.05)),
        training_backend=str(training.get("backend", payload.get("training_backend", "public"))),
        train_jsonl=str(paths.get("train_jsonl", "examples/toy_qa.jsonl")),
        eval_jsonl=str(paths.get("eval_jsonl", "examples/toy_qa.jsonl")),
        output_dir=str(paths.get("output_dir", "outputs")),
    )


def config_to_role_steps(config: RISEConfig) -> dict[str, tuple[int, ...]]:
    return {role: tuple(step_ids) for role, step_ids in config.role_step_contract.items()}


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            key, value = _split_key_value(line)
            if value == "":
                current = {}
                root[key] = current
            else:
                root[key] = _parse_scalar(value)
                current = None
        elif current is not None:
            key, value = _split_key_value(line.strip())
            current[key] = _parse_scalar(value)
    return root


def _split_key_value(line: str) -> tuple[str, str]:
    if ":" not in line:
        raise ValueError(f"Invalid config line: {line}")
    key, value = line.split(":", 1)
    return key.strip(), value.strip()


def _parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        items = [item.strip() for item in value[1:-1].split(",") if item.strip()]
        return [_parse_scalar(item) for item in items]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")
