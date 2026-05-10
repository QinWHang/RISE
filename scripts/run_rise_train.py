#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rise.data import load_jsonl, normalize_records, write_json
from rise.trainer import RISETrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RISE selective evolution loop on JSONL trajectories.")
    parser.add_argument("--input", type=Path, default=Path("examples/toy_qa.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/rise_train_summary.json"))
    parser.add_argument("--delta", type=float, default=0.03)
    args = parser.parse_args()

    trajectories = normalize_records(load_jsonl(args.input))
    trainer = RISETrainer(delta=args.delta)
    summary = trainer.fit(trajectories)
    write_json(args.output, summary)
    print(json.dumps({key: summary[key] for key in ("trajectory_count", "active_updates", "freeze_count", "active_update_rate", "update_role_counts")}, indent=2))


if __name__ == "__main__":
    main()
