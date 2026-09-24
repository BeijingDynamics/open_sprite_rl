#!/usr/bin/env python3
"""Require a G72 candidate to pass both independent Isaac evaluation seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def select(primary: dict, confirmation: dict) -> dict:
    if not primary.get("motor_contract_finalized"):
        raise ValueError("primary summary is not motor-contract finalized")
    if not confirmation.get("motor_contract_finalized"):
        raise ValueError("confirmation summary is not motor-contract finalized")
    primary_seed = int(primary["evaluation_seed"])
    confirmation_seed = int(confirmation["evaluation_seed"])
    if primary_seed == confirmation_seed:
        raise ValueError("evaluation seeds must be independent")

    primary_passing = {int(value) for value in primary["passing_candidates"]}
    confirmation_passing = {int(value) for value in confirmation["passing_candidates"]}
    passing_both = [
        int(value)
        for value in primary["ranked_candidates"]
        if int(value) in primary_passing and int(value) in confirmation_passing
    ]
    candidates = sorted(
        {int(value) for value in primary["ranked_candidates"]}
        | {int(value) for value in confirmation["ranked_candidates"]}
    )
    return {
        "schema": "sprite0825_g72_dualseed_selection_v1",
        "qualified": bool(passing_both),
        "evaluation_seeds": [primary_seed, confirmation_seed],
        "winner_iteration": passing_both[0] if passing_both else None,
        "passing_both": passing_both,
        "candidates": {
            str(iteration): {
                "primary_pass": iteration in primary_passing,
                "confirmation_pass": iteration in confirmation_passing,
                "primary_failed_gates": primary["rows"].get(str(iteration), {}).get(
                    "failed_gates", []
                ),
                "confirmation_failed_gates": confirmation["rows"].get(
                    str(iteration), {}
                ).get("failed_gates", []),
            }
            for iteration in candidates
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--confirmation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--winner", required=True, type=Path)
    args = parser.parse_args()

    result = select(load(args.primary), load(args.confirmation))
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["qualified"]:
        raise SystemExit("No G72 checkpoint passed both Isaac evaluation seeds")
    args.winner.write_text(f'{result["winner_iteration"]}\n', encoding="utf-8")
    print(
        f'G72_DUALSEED_WINNER model_{result["winner_iteration"]}.pt '
        f'seeds={result["evaluation_seeds"]}'
    )


if __name__ == "__main__":
    main()
