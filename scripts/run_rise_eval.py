#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rise.evaluation import RISEEvaluator


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the public RISE evaluation interface.")
    parser.add_argument("--input", type=Path, default=Path("examples/toy_qa.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/rise_eval_summary.json"))
    parser.add_argument("--delta", type=float, default=0.03)
    args = parser.parse_args()

    evaluator = RISEEvaluator(delta=args.delta)
    summary = evaluator.evaluate(load_jsonl(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("count", "f1", "em", "active_update_rate", "cardinality")}, indent=2))


if __name__ == "__main__":
    main()
