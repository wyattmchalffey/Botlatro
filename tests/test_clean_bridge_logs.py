from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import context  # noqa: F401
from balatro_ai.tools.clean_bridge_logs import clean_log_file, clean_logs, summarize_results


class CleanBridgeLogsTests(unittest.TestCase):
    def test_clean_log_file_writes_sidecar_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "bridge.log"
            log_file.write_text(
                "\n".join(
                    [
                        "INFO - BalatroBot started",
                        "INFO - [G] LONG DT @ 123: 0.083166666666667",
                        "ERROR - real problem",
                        ":: DEBUG :: BB.REQUEST :: gamestate",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = clean_log_file(log_file)

            self.assertEqual(result.original_lines, 4)
            self.assertEqual(result.kept_lines, 2)
            self.assertEqual(result.dropped_lines, 2)
            self.assertEqual(result.output_path, Path(temp_dir) / "bridge.clean.log")
            self.assertEqual(result.output_path.read_text(encoding="utf-8"), "INFO - BalatroBot started\nERROR - real problem\n")
            self.assertIn("LONG DT", log_file.read_text(encoding="utf-8"))

    def test_clean_logs_replaces_nested_logs_and_skips_clean_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "worker" / "timestamp"
            nested.mkdir(parents=True)
            log_file = nested / "12346.log"
            clean_sidecar = nested / "12346.clean.log"
            log_file.write_text("keep\nINFO - [G] LONG DT @ 1: 0.08\n", encoding="utf-8")
            clean_sidecar.write_text("already clean\n", encoding="utf-8")

            results = clean_logs([root], replace=True)

            self.assertEqual(len(results), 1)
            self.assertEqual(log_file.read_text(encoding="utf-8"), "keep\n")
            self.assertEqual(clean_sidecar.read_text(encoding="utf-8"), "already clean\n")
            self.assertEqual(summarize_results(results), "Cleaned 1 log file(s): dropped 1 noisy line(s), kept 1 line(s).")


if __name__ == "__main__":
    unittest.main()
