from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .pipeline import RISETrajectory


REQUIRED_RECORD_FIELDS = ("question", "answer", "steps")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return records


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_record(record: dict[str, Any]) -> None:
    absent_fields = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
    if absent_fields:
        raise ValueError(f"Record does not contain required fields: {absent_fields}")
    if not isinstance(record["steps"], list):
        raise ValueError("Record field 'steps' must be a list.")


def normalize_records(records: Iterable[dict[str, Any]]) -> list[RISETrajectory]:
    trajectories: list[RISETrajectory] = []
    for record in records:
        validate_record(record)
        trajectories.append(RISETrajectory.from_record(record))
    return trajectories


def trajectory_to_dict(trajectory: RISETrajectory) -> dict[str, Any]:
    return asdict(trajectory)
