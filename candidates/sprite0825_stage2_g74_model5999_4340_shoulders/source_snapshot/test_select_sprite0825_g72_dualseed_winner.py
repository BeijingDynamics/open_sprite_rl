#!/usr/bin/env python3
from select_sprite0825_g72_dualseed_winner import select


def summary(seed: int, passing: list[int]) -> dict:
    ranked = [2500, 5000, 7500]
    return {
        "evaluation_seed": seed,
        "motor_contract_finalized": True,
        "passing_candidates": passing,
        "ranked_candidates": ranked,
        "rows": {
            str(value): {"failed_gates": [] if value in passing else ["gate"]}
            for value in ranked
        },
    }


def main() -> None:
    result = select(summary(303, [2500, 5000]), summary(404, [5000, 7500]))
    assert result["qualified"]
    assert result["winner_iteration"] == 5000
    assert result["passing_both"] == [5000]

    rejected = select(summary(303, [2500]), summary(404, [7500]))
    assert not rejected["qualified"]
    assert rejected["winner_iteration"] is None

    try:
        select(summary(303, [2500]), summary(303, [2500]))
    except ValueError as exc:
        assert "independent" in str(exc)
    else:
        raise AssertionError("same-seed selection was accepted")

    print("G72_DUALSEED_SELECTION_TEST_PASS")


if __name__ == "__main__":
    main()
