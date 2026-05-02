from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import context  # noqa: F401
from balatro_ai.eval.import_user_logs import import_user_logs


class ImportUserLogsTests(unittest.TestCase):
    def test_import_user_logs_copies_jsonl_rows_with_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            log_path = source / "run.jsonl"
            log_path.write_text(
                "\n".join(
                    (
                        json.dumps(
                            {
                                "record_type": "user_step",
                                "state": "phase=shop ante=1 blind=Small Blind score=0/300 money=8 hands=4 discards=4",
                                "chosen_action": {"type": "end_shop", "card_indices": []},
                            }
                        ),
                        "{bad-json}",
                        json.dumps(
                            {
                                "record_type": "run_summary",
                                "seed": "ABC",
                                "won": False,
                                "ante": 1,
                            }
                        ),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            dest = root / "dest"

            summary = import_user_logs((source,), dest=dest, player_id="Wyatt Test")

            output = summary.output_files[0]
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(summary.files_imported, 1)
        self.assertEqual(summary.rows_imported, 2)
        self.assertEqual(summary.summary_rows, 1)
        self.assertEqual(summary.malformed_rows, 1)
        self.assertEqual(output.name, "Wyatt_Test_run.jsonl")
        self.assertEqual(rows[0]["dataset_source"]["player_id"], "Wyatt Test")
        self.assertEqual(rows[0]["dataset_source"]["source_line"], 1)

    def test_import_user_logs_avoids_overwriting_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "run.jsonl"
            source.write_text('{"record_type": "run_summary"}\n', encoding="utf-8")
            dest = root / "dest"
            dest.mkdir()
            (dest / "human_run.jsonl").write_text('{"old": true}\n', encoding="utf-8")

            summary = import_user_logs((source,), dest=dest, player_id="human")

        self.assertEqual(summary.output_files[0].name, "human_run_2.jsonl")


if __name__ == "__main__":
    unittest.main()
