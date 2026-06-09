from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import context  # noqa: F401
from scripts import phase8_select_shop_state_pool as script


class Phase8SelectShopStatePoolTests(unittest.TestCase):
    def test_parse_action_types_csv_accepts_values_and_names(self) -> None:
        parsed = script._parse_action_types_csv("buy, OPEN_PACK, buy")

        self.assertEqual([action.value for action in parsed], ["buy", "open_pack"])

    def test_select_balanced_spreads_marginal_fields_by_score(self) -> None:
        items = [
            _item("bot_a", 2, 1.0, "a_low"),
            _item("bot_a", 2, 5.0, "a_high"),
            _item("bot_b", 2, 2.0, "b_low"),
            _item("bot_b", 2, 6.0, "b_high"),
            _item("bot_a", 3, 3.0, "a3"),
            _item("bot_b", 3, 4.0, "b3"),
        ]

        selected = script._select_balanced(
            items,
            limit=4,
            seed=0,
            balance_fields=("source_bot", "ante"),
        )

        self.assertEqual(len(selected), 4)
        self.assertEqual(
            {(item.source_bot, item.ante) for item in selected},
            {("bot_a", 2), ("bot_a", 3), ("bot_b", 2), ("bot_b", 3)},
        )
        self.assertIn("a_high", {item.record["id"] for item in selected})
        self.assertIn("b_high", {item.record["id"] for item in selected})

    def test_filter_excluded_uses_source_seed_state_key(self) -> None:
        records = [
            {"source_bot": "bot_a", "seed": "0000001", "state_index": 1},
            {"source_bot": "bot_a", "seed": "0000002", "state_index": 1},
            {"source_bot": "bot_b", "seed": "0000001", "state_index": 1},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "exclude.jsonl"
            path.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")

            excluded = script._exclude_keys([path])
            filtered = script._filter_excluded(records, excluded)

        self.assertEqual(len(excluded), 1)
        self.assertEqual(filtered, records[1:])


def _item(source_bot: str, ante: int, score: float, ident: str) -> script.ScoredShopState:
    return script.ScoredShopState(
        record={"id": ident},
        source_bot=source_bot,
        ante=ante,
        money=10,
        heuristic_action_type="sell",
        heuristic_in_candidates=False,
        candidate_action_types=("buy", "end_shop"),
        score=score,
    )


if __name__ == "__main__":
    unittest.main()
