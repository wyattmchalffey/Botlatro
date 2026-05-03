from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import context  # noqa: F401
from balatro_ai.sim.replay_validator import validate_local_sim_replays


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
    round_data: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "phase": phase,
        "ante": ante,
        "blind": blind,
        "current_score": current_score,
        "required_score": required_score,
        "hands_remaining": hands_remaining,
        "discards_remaining": discards_remaining,
        "money": money,
        "deck_size": deck_size,
        "hand": [{"rank": card[:-1], "suit": card[-1], "name": card} for card in (hand or [])],
        "known_deck": [{"rank": card[:-1], "suit": card[-1], "name": card} for card in (known_deck or [])],
        "hand_levels": {},
        "hands": {},
        "jokers": jokers or [],
        "consumables": [],
        "owned_vouchers": [],
        "vouchers": [],
        "round": round_data or {},
        "shop": [],
        "voucher_shop": [],
        "booster_packs": [],
        "pack": [],
    }


class LocalSimReplayValidatorTests(unittest.TestCase):
    def test_validate_local_sim_replays_reports_exact_carried_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            rows = [
                {
                    "seed": 1,
                    "state_detail": state_detail(hand=["AS", "AH", "KD", "2C"]),
                    "chosen_action": {"type": "play_hand", "card_indices": [0, 1]},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=64,
                        hands_remaining=3,
                        deck_size=8,
                        hand=["KD", "QS", "QC", "2C"],
                    ),
                    "chosen_action": {"type": "discard", "card_indices": [3]},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=64,
                        hands_remaining=3,
                        discards_remaining=2,
                        deck_size=7,
                        hand=["KD", "QS", "QC", "JD"],
                    ),
                },
            ]
            replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = validate_local_sim_replays((replay_path,))

        self.assertEqual(summary.files_scanned, 1)
        self.assertEqual(summary.transitions_checked, 2)
        self.assertEqual(summary.exact_transition_counts["play_hand"], 1)
        self.assertEqual(summary.exact_transition_counts["discard"], 1)
        self.assertEqual(summary.divergences, ())
        self.assertIn("Post-transition exact: 2/2", summary.to_text())

    def test_validate_local_sim_replays_uses_observed_cash_out_delta_when_round_oracle_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            rows = [
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        hand=["AS", "AH"],
                        deck_size=0,
                        required_score=10,
                        discards_remaining=3,
                    ),
                    "chosen_action": {"type": "play_hand", "card_indices": [0, 1]},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="round_eval",
                        current_score=64,
                        required_score=10,
                        hands_remaining=3,
                        discards_remaining=3,
                        money=4,
                        deck_size=0,
                        round_data={"chips": 64, "dollars": 3, "hands_left": 3, "discards_left": 3},
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
                        money=9,
                        deck_size=2,
                    ),
                },
            ]
            replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = validate_local_sim_replays((replay_path,))

        self.assertEqual(summary.exact_transition_counts["cash_out"], 1)
        self.assertEqual(summary.divergences, ())

    def test_validate_local_sim_replays_reports_post_transition_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            rows = [
                {
                    "seed": 1,
                    "state_detail": state_detail(hand=["AS", "AH", "KD", "2C"]),
                    "chosen_action": {"type": "play_hand", "card_indices": [0, 1]},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=99,
                        hands_remaining=3,
                        deck_size=8,
                        hand=["KD", "QS", "QC", "2C"],
                    ),
                },
            ]
            replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = validate_local_sim_replays((replay_path,))

        self.assertEqual(summary.transitions_checked, 1)
        self.assertEqual(summary.exact_transition_counts["play_hand"], 0)
        self.assertEqual(len(summary.divergences), 1)
        self.assertEqual(summary.divergences[0].stage, "post_transition")
        self.assertEqual(summary.divergences[0].fields, ("current_score",))
        self.assertEqual(summary.stage_field_mismatches["post_transition.current_score"], 1)
        self.assertEqual(summary.action_field_mismatches["play_hand.current_score"], 1)
        self.assertEqual(summary.to_json_dict()["examples"][0]["mismatches"], ["current_score"])

    def test_bridge_joker_smoke_rows_reset_between_independent_scenarios(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "smoke.jsonl"
            rows = [
                {
                    "record_type": "bridge_joker_smoke",
                    "stage": "pre",
                    "seed": 1,
                    "scenario_id": "joker",
                    "state_detail": state_detail(hand=["AS", "AH"], jokers=[{"name": "Joker", "cost": {"sell": 1}}]),
                    "chosen_action": {"type": "play_hand", "card_indices": [0, 1]},
                },
                {
                    "record_type": "bridge_joker_smoke",
                    "stage": "post",
                    "seed": 1,
                    "scenario_id": "joker",
                    "state_detail": state_detail(
                        current_score=192,
                        hands_remaining=3,
                        hand=[],
                        jokers=[{"name": "Joker", "cost": {"sell": 1}}],
                    ),
                },
                {
                    "record_type": "bridge_joker_smoke",
                    "stage": "pre",
                    "seed": 2,
                    "scenario_id": "greedy_joker",
                    "state_detail": state_detail(
                        hand=["AH", "KH"],
                        jokers=[{"name": "Greedy Joker", "cost": {"sell": 2}}],
                    ),
                    "chosen_action": {"type": "play_hand", "card_indices": [0, 1]},
                },
                {
                    "record_type": "bridge_joker_smoke",
                    "stage": "post",
                    "seed": 2,
                    "scenario_id": "greedy_joker",
                    "state_detail": state_detail(
                        current_score=16,
                        hands_remaining=3,
                        hand=[],
                        jokers=[{"name": "Greedy Joker", "cost": {"sell": 2}}],
                    ),
                },
            ]
            replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = validate_local_sim_replays((replay_path,))

        self.assertEqual(summary.transitions_checked, 2)
        self.assertEqual(summary.exact_transition_counts["play_hand"], 2)
        self.assertEqual(summary.divergences, ())

    def test_no_resync_mode_reports_carried_pre_state_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            rows = [
                {
                    "seed": 1,
                    "state_detail": state_detail(hand=["AS", "AH", "KD", "2C"]),
                    "chosen_action": {"type": "play_hand", "card_indices": [0, 1]},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=99,
                        hands_remaining=3,
                        deck_size=8,
                        hand=["KD", "QS", "QC", "2C"],
                    ),
                    "chosen_action": {"type": "discard", "card_indices": [3]},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=99,
                        hands_remaining=3,
                        discards_remaining=2,
                        deck_size=7,
                        hand=["KD", "QS", "QC", "JD"],
                    ),
                },
            ]
            replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = validate_local_sim_replays((replay_path,), resync_on_divergence=False)

        self.assertGreaterEqual(len(summary.divergences), 2)
        self.assertEqual(summary.divergences[0].stage, "post_transition")
        self.assertEqual(summary.divergences[1].stage, "pre_state")
        self.assertIn("pre_state.current_score", summary.stage_field_mismatches)

    def test_resync_preserves_hidden_round_card_zones_for_later_round_eval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            rows = [
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        hand=["AS", "AH", "KD", "QC"],
                        deck_size=4,
                        required_score=100,
                    ),
                    "chosen_action": {"type": "play_hand", "card_indices": [0, 1]},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=99,
                        hands_remaining=3,
                        deck_size=2,
                        hand=["KD", "QC", "QS", "JC"],
                        required_score=100,
                    ),
                    "chosen_action": {"type": "play_hand", "card_indices": [0]},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="round_eval",
                        current_score=114,
                        hands_remaining=2,
                        deck_size=8,
                        hand=[],
                        required_score=100,
                    ),
                },
            ]
            replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = validate_local_sim_replays((replay_path,))

        self.assertEqual(summary.transitions_checked, 2)
        self.assertEqual(summary.exact_transition_counts["play_hand"], 1)
        self.assertEqual(summary.action_field_mismatches["play_hand.current_score"], 1)
        self.assertNotIn("play_hand.deck_size", summary.action_field_mismatches)

    def test_transition_validation_uses_observed_deck_surface_after_hidden_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            rows = [
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="shop",
                        blind="Small Blind",
                        current_score=0,
                        required_score=300,
                        hands_remaining=4,
                        discards_remaining=4,
                        deck_size=30,
                    ),
                    "chosen_action": {"type": "end_shop"},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="blind_select",
                        blind="Big Blind",
                        current_score=0,
                        required_score=450,
                        hands_remaining=4,
                        discards_remaining=4,
                        deck_size=52,
                    ),
                    "chosen_action": {"type": "select_blind"},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="selecting_hand",
                        blind="Big Blind",
                        current_score=0,
                        required_score=450,
                        hands_remaining=4,
                        discards_remaining=4,
                        deck_size=44,
                        hand=["AS", "AH", "AD", "AC", "KS", "KH", "KD", "KC"],
                    ),
                },
            ]
            replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = validate_local_sim_replays((replay_path,))

        self.assertEqual(summary.transitions_checked, 2)
        self.assertEqual(summary.exact_transition_counts["select_blind"], 1)
        self.assertNotIn("select_blind.deck_size", summary.action_field_mismatches)

    def test_impossible_run_summary_win_flag_is_known_data_gap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            rows = [
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=900,
                        required_score=1000,
                        hands_remaining=1,
                        deck_size=0,
                        hand=["AS"],
                    ),
                    "chosen_action": {"type": "play_hand", "card_indices": [0]},
                },
                {
                    "record_type": "run_summary",
                    "seed": 1,
                    "won": True,
                    "final_score": 916,
                    "final_money": 4,
                    "final_state_detail": state_detail(
                        phase="run_over",
                        current_score=916,
                        required_score=1000,
                        hands_remaining=0,
                        deck_size=0,
                        hand=[],
                    ),
                },
            ]
            replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = validate_local_sim_replays((replay_path,))

        self.assertEqual(summary.comparable_divergences, 0)
        self.assertEqual(summary.known_gap_counts["Run summary win flag conflicts with final score below required score"], 1)

    def test_cash_out_keeps_cleared_blind_surface_in_shop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            rows = [
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="round_eval",
                        blind="Small Blind",
                        current_score=500,
                        required_score=300,
                        hands_remaining=2,
                        discards_remaining=3,
                        money=10,
                        deck_size=52,
                    ),
                    "chosen_action": {"type": "cash_out"},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="shop",
                        blind="Small Blind",
                        current_score=0,
                        required_score=300,
                        hands_remaining=4,
                        discards_remaining=4,
                        money=17,
                        deck_size=52,
                    ),
                },
            ]
            replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = validate_local_sim_replays((replay_path,))

        self.assertEqual(summary.transitions_checked, 1)
        self.assertEqual(summary.exact_transition_counts["cash_out"], 1)
        self.assertEqual(summary.divergences, ())

    def test_end_shop_advances_visible_blind_surface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            rows = [
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="shop",
                        blind="Small Blind",
                        current_score=0,
                        required_score=300,
                        hands_remaining=4,
                        discards_remaining=4,
                        money=10,
                        deck_size=52,
                    ),
                    "chosen_action": {"type": "end_shop"},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="blind_select",
                        blind="Big Blind",
                        current_score=0,
                        required_score=450,
                        hands_remaining=4,
                        discards_remaining=4,
                        money=10,
                        deck_size=52,
                    ),
                },
            ]
            replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = validate_local_sim_replays((replay_path,))

        self.assertEqual(summary.transitions_checked, 1)
        self.assertEqual(summary.exact_transition_counts["end_shop"], 1)
        self.assertEqual(summary.divergences, ())

    def test_cash_out_preserves_hidden_cleared_blind_for_next_shop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            rows = [
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="blind_select",
                        blind="Big Blind",
                        current_score=0,
                        required_score=450,
                        hands_remaining=4,
                        discards_remaining=4,
                        money=10,
                        deck_size=52,
                    ),
                    "chosen_action": {"type": "select_blind"},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="round_eval",
                        blind="Big Blind",
                        current_score=500,
                        required_score=450,
                        hands_remaining=2,
                        discards_remaining=3,
                        money=10,
                        deck_size=52,
                    ),
                    "chosen_action": {"type": "cash_out"},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="shop",
                        blind="Small Blind",
                        current_score=0,
                        required_score=300,
                        hands_remaining=4,
                        discards_remaining=4,
                        money=17,
                        deck_size=52,
                    ),
                    "chosen_action": {"type": "end_shop"},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        phase="blind_select",
                        blind="The Club",
                        current_score=0,
                        required_score=600,
                        hands_remaining=4,
                        discards_remaining=4,
                        money=17,
                        deck_size=52,
                    ),
                },
            ]
            replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = validate_local_sim_replays((replay_path,))

        self.assertEqual(summary.transitions_checked, 3)
        self.assertEqual(summary.exact_transition_counts["cash_out"], 1)
        self.assertEqual(summary.exact_transition_counts["end_shop"], 1)


if __name__ == "__main__":
    unittest.main()
