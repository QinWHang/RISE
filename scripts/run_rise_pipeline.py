#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rise.config import load_config
from rise.runner import RISERunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full RISE method framework from a YAML config.")
    parser.add_argument("--config", type=Path, default=Path("configs/rise_hotpotqa.yaml"))
    args = parser.parse_args()
    runner = RISERunner(load_config(args.config))
    payload = runner.run()
    brief = {
        "train_active_update_rate": payload["train"]["active_update_rate"],
        "eval_f1": payload["eval"]["f1"],
        "eval_em": payload["eval"]["em"],
        "eval_active_update_rate": payload["eval"]["active_update_rate"],
    }
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
