from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import context  # noqa: F401
from balatro_ai.eval.import_multiplayer_logs import import_multiplayer_logs


class ImportMultiplayerLogsTests(unittest.TestCase):
    def test_import_multiplayer_log_extracts_sanitized_weak_signals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "logs"
            source.mkdir()
            log_path = source / "lovely-2026.05.05-10.00.00.log"
            log_path.write_text(
                "\n".join(
                    (
                        'INFO - [G] 2026-05-05 10:00:00 :: TRACE :: MULTIPLAYER :: Client sent message: {"action":"connect"}',
                        (
                            'INFO - [G] 2026-05-05 10:00:01 :: TRACE :: MULTIPLAYER :: Client sent message: '
                            '{"username":"PlayerOne","modHash":"preview=true;Multiplayer-0.3.3;Steamodded-1.0.0",'
                            '"action":"username"}'
                        ),
                        (
                            "INFO - [G] 2026-05-05 10:00:02 :: TRACE :: MULTIPLAYER :: Client got joinedLobby message:  "
                            "(type: attrition)  (action: joinedLobby)  (code: SECRET)  "
                            "(reconnectToken: abc123) "
                        ),
                        (
                            "INFO - [G] 2026-05-05 10:00:03 :: TRACE :: MULTIPLAYER :: Client got lobbyOptions message:  "
                            "(ruleset: ruleset_mp_standard_ranked)  (back: Ghost Deck)  (stake: 1)  "
                            "(custom_seed: random)  (action: lobbyOptions) "
                        ),
                        (
                            "INFO - [G] 2026-05-05 10:00:04 :: TRACE :: MULTIPLAYER :: Client got startGame message:  "
                            "(deck: c_multiplayer_1)  (action: startGame) "
                        ),
                        'INFO - [G] 2026-05-05 10:00:05 :: TRACE :: MULTIPLAYER :: Client sent message: {"ante":1,"action":"setAnte"}',
                        (
                            'INFO - [G] 2026-05-05 10:00:06 :: TRACE :: MULTIPLAYER :: Client sent message: '
                            '{"location":"loc_shop","action":"setLocation"}'
                        ),
                        (
                            "INFO - [G] 2026-05-05 10:00:07 :: TRACE :: MULTIPLAYER :: Client sent message: "
                            "action:boughtCardFromShop,card:Hologram,cost:5"
                        ),
                        (
                            "INFO - [G] 2026-05-05 10:00:08 :: TRACE :: MULTIPLAYER :: Client sent message: "
                            "action:moneyMoved,amount:-5"
                        ),
                        (
                            "INFO - [G] 2026-05-05 10:00:09 :: TRACE :: MULTIPLAYER :: Client sent message: "
                            "action:usedCard,card:The Hermit"
                        ),
                        (
                            'INFO - [G] 2026-05-05 10:00:10 :: TRACE :: MULTIPLAYER :: Client sent message: '
                            '{"action":"playHand","handsLeft":3,"score":"12345"}'
                        ),
                        (
                            "INFO - [G] 2026-05-05 10:00:11 :: DEBUG :: DefaultLogger :: "
                            "(Idol) Selected card Ace of Hearts with weight 1 of total 52"
                        ),
                        (
                            "INFO - [G] 2026-05-05 10:00:12 :: TRACE :: MULTIPLAYER :: "
                            "Sending end game jokers: ;j_hologram-none-none-none;j_baron-polychrome-none-none"
                        ),
                        (
                            'INFO - [G] 2026-05-05 10:00:13 :: TRACE :: MULTIPLAYER :: Client sent message: '
                            '{"cards":";H-A-m_lucky-none-Gold;S-K-c_base-foil-none","action":"receiveNemesisDeck"}'
                        ),
                        (
                            "INFO - [G] 2026-05-05 10:00:14 :: TRACE :: MULTIPLAYER :: Client got receiveEndGameJokers message:  "
                            "(keys: H4sIABLOB)  (action: receiveEndGameJokers)  (seed: 4MU6C8WJ) "
                        ),
                        "INFO - [G] 2026-05-05 10:00:15 :: TRACE :: MULTIPLAYER :: Client got winGame message:  (action: winGame) ",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            dest = root / "dataset"

            summary = import_multiplayer_logs((source,), dest=dest, player_id="tester", min_support=1)
            run_rows = _load_jsonl(summary.output_files["runs"])
            replay_rows = _load_jsonl(summary.output_files["replay_summaries"])
            shop_rows = _load_jsonl(summary.output_files["shop_actions"])
            build_rows = _load_jsonl(summary.output_files["endgame_builds"])
            events_text = summary.output_files["events"].read_text(encoding="utf-8")
            card_stats = json.loads(summary.output_files["card_stats"].read_text(encoding="utf-8"))

        self.assertEqual(summary.files_scanned, 1)
        self.assertEqual(summary.logs_imported, 1)
        self.assertEqual(summary.observed_wins, 1)
        self.assertEqual(run_rows[0]["won"], True)
        self.assertEqual(run_rows[0]["source_seed"], "4MU6C8WJ")
        self.assertEqual(run_rows[0]["final_score"], 12345)
        self.assertEqual(replay_rows[0]["record_type"], "run_summary")
        self.assertEqual(replay_rows[0]["outcome"], "win")
        self.assertIn("Hologram", {row.get("item") for row in shop_rows})
        self.assertIn("The Hermit", replay_rows[0]["tarot_usage"])
        self.assertEqual(build_rows[0]["jokers"][0]["key"], "j_hologram")
        self.assertEqual(build_rows[0]["deck_card_count"], 2)
        self.assertIn("Hologram", card_stats["by_action"]["boughtCardFromShop"])
        self.assertNotIn("PlayerOne", events_text)
        self.assertNotIn("SECRET", events_text)
        self.assertNotIn("abc123", events_text)
        self.assertNotIn("H4sIABLOB", events_text)

    def test_import_multiplayer_logs_uses_new_directory_when_outputs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "run.log"
            source.write_text(
                "INFO - [G] 2026-05-05 10:00:00 :: TRACE :: MULTIPLAYER :: "
                "Client got winGame message:  (action: winGame) \n",
                encoding="utf-8",
            )
            dest = root / "dataset"
            dest.mkdir()
            (dest / "events.jsonl").write_text("old\n", encoding="utf-8")

            summary = import_multiplayer_logs((source,), dest=dest)

        self.assertEqual(summary.output_dir.name, "dataset_2")


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    unittest.main()
