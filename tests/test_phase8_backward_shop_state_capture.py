from __future__ import annotations

from scripts import phase8_backward_shop_state_capture as script


def test_record_priority_prefers_wins_and_last_late_shops() -> None:
    records = [
        {"seed": "early-loss", "terminal_won": False, "terminal_ante": 8, "shops_from_terminal": 0},
        {"seed": "win-older-shop", "terminal_won": True, "terminal_ante": 8, "shops_from_terminal": 1},
        {"seed": "win-last-shop", "terminal_won": True, "terminal_ante": 8, "shops_from_terminal": 0},
        {"seed": "win-earlier-ante", "terminal_won": True, "terminal_ante": 7, "shops_from_terminal": 0},
    ]

    ordered = sorted(records, key=script._record_priority_key)

    assert [record["seed"] for record in ordered] == [
        "win-last-shop",
        "win-older-shop",
        "win-earlier-ante",
        "early-loss",
    ]


def test_parse_action_types_csv_dedupes_in_order() -> None:
    parsed = script._parse_action_types_csv("buy,OPEN_PACK,buy,end_shop")

    assert tuple(action.value for action in parsed or ()) == ("buy", "open_pack", "end_shop")


def test_trajectory_qualifies_can_target_near_wins() -> None:
    assert script._trajectory_qualifies(
        has_shops=True,
        final_ante=8,
        won=False,
        min_final_ante=8,
        require_win=False,
        exclude_wins=True,
    )
    assert not script._trajectory_qualifies(
        has_shops=True,
        final_ante=8,
        won=True,
        min_final_ante=8,
        require_win=False,
        exclude_wins=True,
    )
    assert not script._trajectory_qualifies(
        has_shops=True,
        final_ante=7,
        won=False,
        min_final_ante=8,
        require_win=False,
        exclude_wins=True,
    )


def test_trajectory_qualifies_can_target_wins() -> None:
    assert script._trajectory_qualifies(
        has_shops=True,
        final_ante=8,
        won=True,
        min_final_ante=8,
        require_win=True,
        exclude_wins=False,
    )
    assert not script._trajectory_qualifies(
        has_shops=True,
        final_ante=8,
        won=False,
        min_final_ante=8,
        require_win=True,
        exclude_wins=False,
    )
