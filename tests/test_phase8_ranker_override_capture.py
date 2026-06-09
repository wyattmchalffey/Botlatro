from __future__ import annotations

import os
from pathlib import Path
import unittest

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.ml.shop_candidate_dataset import action_key
from scripts import phase8_ranker_override_capture as script


def _shop_state() -> GameState:
    return GameState(
        phase=GamePhase.SHOP,
        ante=2,
        money=17,
        modifiers={
            "shop_cards": (
                {"key": "j_joker", "name": "Joker", "set": "JOKER", "cost": {"buy": 4}},
            ),
            "booster_packs": (
                {"key": "p_buffoon_normal_1", "name": "Buffoon Pack", "set": "BOOSTER", "cost": {"buy": 4}},
            ),
        },
    )


class Phase8RankerOverrideCaptureTests(unittest.TestCase):
    def test_seed_strings_match_phase8_offset_convention(self) -> None:
        self.assertEqual(script._seed_strings(540000, 3), ["0540001", "0540002", "0540003"])

    def test_validate_action_types_accepts_names_and_values(self) -> None:
        self.assertEqual(
            script._validate_action_types(("BUY", "open_pack", "end_shop", "buy")),
            ("buy", "open_pack", "end_shop"),
        )

    def test_ranker_env_sets_and_restores_values(self) -> None:
        old_ckpt = os.environ.get("BALATRO_SHOP_RANKER_CKPT")
        os.environ["BALATRO_SHOP_RANKER_CKPT"] = "old.pt"
        try:
            with script._ranker_env(
                ranker_ckpt=Path("new.pt"),
                max_actions=4,
                candidate_action_types=("buy", "open_pack"),
                candidate_action_kinds=("card", "pack"),
                min_margin=0.5,
                min_baseline_margin=0.25,
            ):
                self.assertEqual(os.environ["BALATRO_SHOP_RANKER_CKPT"], "new.pt")
                self.assertEqual(os.environ["BALATRO_SHOP_RANKER_ACTION_KINDS"], "card,pack")
                self.assertEqual(os.environ["BALATRO_SHOP_RANKER_MIN_BASELINE_MARGIN"], "0.25")
            self.assertEqual(os.environ["BALATRO_SHOP_RANKER_CKPT"], "old.pt")
        finally:
            if old_ckpt is None:
                os.environ.pop("BALATRO_SHOP_RANKER_CKPT", None)
            else:
                os.environ["BALATRO_SHOP_RANKER_CKPT"] = old_ckpt

    def test_proposal_record_matches_deepening_probe_contract(self) -> None:
        state = _shop_state()
        baseline = Action(ActionType.END_SHOP)
        ranker = Action(
            ActionType.OPEN_PACK,
            target_id="pack",
            amount=0,
            metadata={
                "kind": "pack",
                "index": 0,
                "shop_ranker": {
                    "score": 1.25,
                    "margin": 0.75,
                    "baseline_score": 0.1,
                    "baseline_margin": 1.15,
                    "baseline_key": action_key(baseline),
                },
            },
        )

        record = script._proposal_record(
            seed="0540001",
            state_index=12,
            source_bot="solver_shop_basic_play_bot",
            state=state,
            baseline_action=baseline,
            ranker_action=ranker,
            ranker_ckpt=Path(".data/ranker.pt"),
            max_actions=4,
            candidate_action_types=("buy", "open_pack", "end_shop"),
            candidate_action_kinds=("card", "pack"),
            min_margin=0.5,
            min_baseline_margin=0.0,
        )

        self.assertEqual(record["proposal_source"], "shop_ranker_override_capture")
        self.assertEqual(record["deepening_candidate_action_key"], action_key(ranker))
        self.assertEqual(record["deepening_candidate_action_type"], "open_pack")
        self.assertEqual(record["deepening_candidate_action_kind"], "pack")
        self.assertEqual(record["deepening_heuristic_action_key"], action_key(baseline))
        self.assertEqual(record["ranker_baseline_key"], action_key(baseline))
        self.assertIn("state_snapshot", record)
        self.assertIn("encoded_state", record)


if __name__ == "__main__":
    unittest.main()
