from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import context  # noqa: F401
from balatro_ai.search.replay_diff import diff_replays


def state_detail(
    *,
    phase: str = "selecting_hand",
    ante: int = 1,
    blind: str = "Small Blind",
    current_score: int = 0,
    required_score: int = 500,
    hands_remaining: int = 4,
    discards_remaining: int = 3,
    money: int = 4,
    deck_size: int = 10,
    hand: list[str] | None = None,
    known_deck: list[str] | None = None,
    jokers: list[dict[str, object]] | None = None,
    consumables: list[object] | None = None,
    vouchers: list[object] | None = None,
    shop: list[dict[str, object]] | None = None,
    voucher_shop: list[dict[str, object]] | None = None,
    booster_packs: list[dict[str, object]] | None = None,
    pack: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "phase": phase,
        "ante": ante,
        "blind": blind,
        "booster_packs": booster_packs or [],
        "consumables": consumables or [],
        "current_score": current_score,
        "hands_remaining": hands_remaining,
        "discards_remaining": discards_remaining,
        "money": money,
        "deck_size": deck_size,
        "hand": [{"rank": card[:-1], "suit": card[-1], "name": card} for card in (hand or [])],
        "hand_levels": {},
        "hands": {},
        "jokers": jokers or [],
        "known_deck": [{"rank": card[:-1], "suit": card[-1], "name": card} for card in (known_deck or [])],
        "pack": pack or [],
        "required_score": required_score,
        "shop": shop or [],
        "voucher_shop": voucher_shop or [],
        "vouchers": vouchers or [],
    }


class ReplayDiffTests(unittest.TestCase):
    def test_diff_replays_reports_exact_play_and_discard_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            rows = [
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=0,
                        required_score=500,
                        hands_remaining=4,
                        discards_remaining=3,
                        deck_size=10,
                        hand=["AS", "AH", "KD", "2C"],
                    ),
                    "chosen_action": {"type": "play_hand", "card_indices": [0, 1]},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=64,
                        required_score=500,
                        hands_remaining=3,
                        discards_remaining=3,
                        deck_size=8,
                        hand=["KD", "2C", "QS", "QC"],
                    ),
                    "chosen_action": {"type": "discard", "card_indices": [1]},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=64,
                        required_score=500,
                        hands_remaining=3,
                        discards_remaining=2,
                        deck_size=7,
                        hand=["KD", "QS", "QC", "JD"],
                    ),
                    "chosen_action": {"type": "play_hand", "card_indices": [0]},
                },
            ]
            replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = diff_replays((Path(directory),))

        self.assertEqual(summary.files_scanned, 1)
        self.assertEqual(summary.compared_transitions, 2)
        self.assertEqual(summary.exact_counts["play_hand"], 1)
        self.assertEqual(summary.exact_counts["discard"], 1)
        self.assertEqual(summary.field_mismatches, {})
        self.assertIn("play_hand: exact 1/1", summary.to_text())

    def test_diff_replays_reports_mismatched_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            rows = [
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=0,
                        required_score=500,
                        hands_remaining=4,
                        deck_size=10,
                        hand=["AS", "AH", "KD", "2C"],
                    ),
                    "chosen_action": {"type": "play_hand", "card_indices": [0, 1]},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=99,
                        required_score=500,
                        hands_remaining=3,
                        deck_size=8,
                        hand=["KD", "2C", "QS", "QC"],
                    ),
                    "chosen_action": {"type": "play_hand", "card_indices": [0]},
                },
            ]
            replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = diff_replays((replay_path,))

        self.assertEqual(summary.compared_transitions, 1)
        self.assertEqual(summary.exact_counts["play_hand"], 0)
        self.assertEqual(summary.field_mismatches["current_score"], 1)
        self.assertEqual(summary.action_field_mismatches["play_hand.current_score"], 1)
        self.assertEqual(summary.mismatch_groups["play_hand:current_score"], 1)
        self.assertEqual(summary.to_json_dict()["field_mismatches"], {"current_score": 1})
        example = summary.to_json_dict()["examples"][0]
        self.assertEqual(
            example["mismatch_details"],
            [{"field": "current_score", "simulated": 64, "actual": 99}],
        )
        self.assertIn("current_score: simulated=64 actual=99", summary.to_text())

    def test_diff_replays_reports_money_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            rows = [
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=0,
                        required_score=500,
                        money=4,
                        deck_size=10,
                        hand=["AS", "KD"],
                    ),
                    "chosen_action": {"type": "play_hand", "card_indices": [0]},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=16,
                        required_score=500,
                        money=9,
                        deck_size=9,
                        hand=["KD"],
                    ),
                    "chosen_action": {"type": "play_hand", "card_indices": [0]},
                },
            ]
            replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = diff_replays((replay_path,))

        self.assertEqual(summary.compared_transitions, 1)
        self.assertEqual(summary.field_mismatches["money"], 1)
        self.assertEqual(summary.action_field_mismatches["play_hand.money"], 1)
        self.assertIn("money: simulated=4 actual=9", summary.to_text())

    def test_diff_replays_validates_cash_out_transition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            rows = [
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="round_eval",
                        current_score=500,
                        required_score=300,
                        hands_remaining=2,
                        discards_remaining=3,
                        money=10,
                        deck_size=52,
                        hand=[],
                    ),
                    "chosen_action": {"type": "cash_out"},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="shop",
                        current_score=0,
                        required_score=300,
                        hands_remaining=4,
                        discards_remaining=4,
                        money=17,
                        deck_size=52,
                        hand=[],
                    ),
                },
            ]
            replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = diff_replays((replay_path,))

        self.assertEqual(summary.compared_transitions, 1)
        self.assertEqual(summary.exact_counts["cash_out"], 1)
        self.assertEqual(summary.field_mismatches, {})

    def test_diff_replays_validates_select_blind_transition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            rows = [
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="blind_select",
                        current_score=120,
                        required_score=300,
                        hands_remaining=4,
                        discards_remaining=3,
                        deck_size=4,
                        hand=[],
                        known_deck=["AS", "KH", "QD", "JC"],
                    ),
                    "chosen_action": {"type": "select_blind"},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="selecting_hand",
                        current_score=0,
                        required_score=300,
                        hands_remaining=4,
                        discards_remaining=3,
                        deck_size=2,
                        hand=["AS", "KH"],
                        known_deck=["QD", "JC"],
                    ),
                },
            ]
            replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = diff_replays((replay_path,))

        self.assertEqual(summary.compared_transitions, 1)
        self.assertEqual(summary.exact_counts["select_blind"], 1)
        self.assertEqual(summary.field_mismatches, {})

    def test_diff_replays_validates_select_blind_created_cards(self) -> None:
        certificate = {"name": "Certificate", "sell_value": 2}
        marble = {"name": "Marble Joker", "sell_value": 2}
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            rows = [
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="blind_select",
                        current_score=120,
                        required_score=300,
                        hands_remaining=4,
                        discards_remaining=3,
                        deck_size=4,
                        hand=[],
                        known_deck=["AS", "KH", "QD", "JC"],
                        jokers=[certificate, marble],
                    ),
                    "chosen_action": {"type": "select_blind"},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="selecting_hand",
                        current_score=0,
                        required_score=300,
                        hands_remaining=4,
                        discards_remaining=3,
                        deck_size=3,
                        hand=["AS", "KH", "7D"],
                        known_deck=["9C", "QD", "JC"],
                        jokers=[certificate, marble],
                    ),
                },
            ]
            replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = diff_replays((replay_path,))

        self.assertEqual(summary.compared_transitions, 1)
        self.assertEqual(summary.exact_counts["select_blind"], 1)
        self.assertEqual(summary.field_mismatches, {})

    def test_diff_replays_validates_shop_transitions(self) -> None:
        joker = {"name": "Joker", "set": "JOKER", "cost": {"buy": 4, "sell": 2}}
        mars = {"name": "Mars", "set": "PLANET", "cost": {"buy": 3, "sell": 1}}
        green_joker = {"name": "Green Joker", "set": "JOKER", "cost": {"sell": 2}}
        buffoon_pack = {"name": "Buffoon Pack", "set": "BOOSTER", "cost": {"buy": 4, "sell": 2}}
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            rows = [
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="shop",
                        money=10,
                        hand=[],
                        shop=[joker],
                        booster_packs=[buffoon_pack],
                    ),
                    "chosen_action": {"type": "buy", "target_id": "card", "amount": 0, "metadata": {"kind": "card", "index": 0}},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="shop",
                        money=6,
                        hand=[],
                        jokers=[{"name": "Joker", "sell_value": 2}],
                        shop=[],
                        booster_packs=[buffoon_pack],
                    ),
                    "chosen_action": {"type": "reroll"},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="shop",
                        money=6,
                        hand=[],
                        jokers=[{"name": "Joker", "sell_value": 2}],
                        shop=[mars],
                        booster_packs=[buffoon_pack],
                    ),
                    "chosen_action": {"type": "open_pack", "target_id": "pack", "amount": 0, "metadata": {"kind": "pack", "index": 0}},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="booster_opened",
                        money=2,
                        hand=[],
                        jokers=[{"name": "Joker", "sell_value": 2}],
                        shop=[mars],
                        pack=[green_joker],
                    ),
                    "chosen_action": {
                        "type": "choose_pack_card",
                        "target_id": "card",
                        "amount": 0,
                        "metadata": {"kind": "card", "index": 0},
                    },
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="shop",
                        money=2,
                        hand=[],
                        jokers=[{"name": "Joker", "sell_value": 2}, {"name": "Green Joker", "sell_value": 2}],
                        shop=[mars],
                    ),
                    "chosen_action": {"type": "sell", "target_id": "joker", "amount": 0, "metadata": {"kind": "joker", "index": 0}},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="shop",
                        money=4,
                        hand=[],
                        jokers=[{"name": "Green Joker", "sell_value": 2}],
                        shop=[mars],
                    ),
                    "chosen_action": {"type": "end_shop"},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="blind_select",
                        money=4,
                        hand=[],
                        jokers=[{"name": "Green Joker", "sell_value": 2}],
                    ),
                },
            ]
            replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = diff_replays((replay_path,))

        self.assertEqual(summary.compared_transitions, 6)
        self.assertEqual(summary.exact_counts["buy"], 1)
        self.assertEqual(summary.exact_counts["reroll"], 1)
        self.assertEqual(summary.exact_counts["open_pack"], 1)
        self.assertEqual(summary.exact_counts["choose_pack_card"], 1)
        self.assertEqual(summary.exact_counts["sell"], 1)
        self.assertEqual(summary.exact_counts["end_shop"], 1)
        self.assertEqual(summary.field_mismatches, {})

    def test_diff_replays_reconstructs_played_hand_history_for_mouth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            rows = [
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="selecting_hand",
                        current_score=0,
                        required_score=500,
                        hands_remaining=4,
                        deck_size=10,
                        hand=["AS", "AH", "KD", "2C"],
                    )
                    | {"blind": "The Mouth"},
                    "chosen_action": {"type": "play_hand", "card_indices": [0, 1]},
                    "extra": {"score_audit": {"hand_type": "Pair"}},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="selecting_hand",
                        current_score=64,
                        required_score=500,
                        hands_remaining=3,
                        deck_size=8,
                        hand=["KD", "2C", "QS", "JC"],
                    )
                    | {"blind": "The Mouth"},
                    "chosen_action": {"type": "play_hand", "card_indices": [0]},
                    "extra": {"score_audit": {"hand_type": "High Card"}},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="selecting_hand",
                        current_score=64,
                        required_score=500,
                        hands_remaining=2,
                        deck_size=6,
                        hand=["2C", "QS", "JC", "9D", "8D"],
                    )
                    | {"blind": "The Mouth"},
                    "chosen_action": {"type": "play_hand", "card_indices": [0]},
                },
            ]
            replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = diff_replays((replay_path,))

        self.assertEqual(summary.compared_transitions, 2)
        self.assertEqual(summary.exact_counts["play_hand"], 2)
        self.assertEqual(summary.field_mismatches, {})

    def test_diff_replays_ignores_hook_visible_hand_randomness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            rows = [
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=0,
                        required_score=500,
                        hands_remaining=4,
                        deck_size=10,
                        hand=["AS", "KD", "QC"],
                    )
                    | {"blind": "The Hook"},
                    "chosen_action": {"type": "play_hand", "card_indices": [0]},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=16,
                        required_score=500,
                        hands_remaining=3,
                        deck_size=7,
                        hand=["9S", "8D"],
                    )
                    | {"blind": "The Hook"},
                    "chosen_action": {"type": "play_hand", "card_indices": [0]},
                },
            ]
            replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = diff_replays((replay_path,))

        self.assertEqual(summary.compared_transitions, 1)
        self.assertEqual(summary.exact_counts["play_hand"], 1)
        self.assertEqual(summary.field_mismatches, {})

    def test_diff_replays_reconstructs_visible_scaling_joker_counters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            rows = [
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=0,
                        required_score=9999,
                        hands_remaining=4,
                        hand=["AS", "KH"],
                    ),
                    "chosen_action": {"type": "play_hand", "card_indices": [0]},
                    "extra": {"score_audit": {"hand_type": "High Card"}},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=16,
                        required_score=9999,
                        hands_remaining=3,
                        deck_size=10,
                        hand=["KH"],
                    ),
                    "chosen_action": {"type": "select_blind"},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=0,
                        required_score=9999,
                        hands_remaining=4,
                        hand=["AS", "KD"],
                        jokers=[
                            {"name": "Ice Cream", "sell_value": 2},
                            {"name": "Popcorn", "sell_value": 2},
                            {"name": "Supernova", "sell_value": 2},
                        ],
                    ),
                    "chosen_action": {"type": "play_hand", "card_indices": [0]},
                    "extra": {"score_audit": {"hand_type": "High Card"}},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=2668,
                        required_score=9999,
                        hands_remaining=3,
                        deck_size=10,
                        hand=["KD"],
                        jokers=[
                            {"name": "Ice Cream", "sell_value": 2},
                            {"name": "Popcorn", "sell_value": 2},
                            {"name": "Supernova", "sell_value": 2},
                        ],
                    ),
                    "chosen_action": {"type": "play_hand", "card_indices": [0]},
                },
            ]
            replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = diff_replays((replay_path,))

        self.assertEqual(summary.compared_transitions, 2)
        self.assertEqual(summary.exact_counts["play_hand"], 2)
        self.assertEqual(summary.field_mismatches, {})

    def test_diff_replays_reconstructs_square_green_and_ramen_counters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            jokers = [
                {"name": "Square Joker", "sell_value": 2},
                {"name": "Green Joker", "sell_value": 2},
                {"name": "Ramen", "sell_value": 2},
            ]
            rows = [
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=0,
                        required_score=9999,
                        hands_remaining=4,
                        deck_size=10,
                        hand=["AS", "KH", "QD", "JC"],
                        jokers=jokers,
                    ),
                    "chosen_action": {"type": "play_hand", "card_indices": [0, 1, 2, 3]},
                    "extra": {"score_audit": {"hand_type": "High Card"}},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=80,
                        required_score=9999,
                        hands_remaining=3,
                        discards_remaining=3,
                        deck_size=6,
                        hand=["9S", "8H", "7D", "6C"],
                        jokers=jokers,
                    ),
                    "chosen_action": {"type": "discard", "card_indices": [0, 1]},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=80,
                        required_score=9999,
                        hands_remaining=3,
                        discards_remaining=2,
                        deck_size=4,
                        hand=["7D", "6C", "5S", "4H"],
                        jokers=jokers,
                    ),
                    "chosen_action": {"type": "play_hand", "card_indices": [0, 1, 2, 3]},
                    "extra": {"score_audit": {"hand_type": "High Card"}},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=159,
                        required_score=9999,
                        hands_remaining=2,
                        discards_remaining=2,
                        deck_size=4,
                        hand=[],
                        jokers=jokers,
                    ),
                    "chosen_action": {"type": "play_hand", "card_indices": []},
                },
            ]
            replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = diff_replays((replay_path,))

        self.assertEqual(summary.compared_transitions, 3)
        self.assertEqual(summary.exact_counts["play_hand"], 2)
        self.assertEqual(summary.exact_counts["discard"], 1)
        self.assertEqual(summary.field_mismatches, {})

    def test_diff_replays_reconstructs_loyalty_card_countdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            jokers = [{"name": "Loyalty Card", "sell_value": 2}]
            rows = []
            current_score = 0
            deck_size = 10
            hands_remaining = 10
            for index in range(6):
                rows.append(
                    {
                        "seed": 1,
                        "state_detail": state_detail(
                            current_score=current_score,
                            required_score=9999,
                            hands_remaining=hands_remaining,
                            deck_size=deck_size,
                            hand=["AS"],
                            jokers=jokers,
                        ),
                        "chosen_action": {"type": "play_hand", "card_indices": [0]},
                        "extra": {"score_audit": {"hand_type": "High Card"}},
                    }
                )
                current_score += 64 if index == 5 else 16
                deck_size -= 1
                hands_remaining -= 1
            rows.append(
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=current_score,
                        required_score=9999,
                        hands_remaining=hands_remaining,
                        deck_size=deck_size,
                        hand=["AS"],
                        jokers=jokers,
                    ),
                    "chosen_action": {"type": "play_hand", "card_indices": []},
                }
            )
            replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = diff_replays((replay_path,))

        self.assertEqual(summary.compared_transitions, 6)
        self.assertEqual(summary.exact_counts["play_hand"], 6)
        self.assertEqual(summary.field_mismatches, {})


if __name__ == "__main__":
    unittest.main()
