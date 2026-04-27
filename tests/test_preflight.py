from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.tools.preflight import Check, checks_to_text


class PreflightTests(unittest.TestCase):
    def test_checks_to_text_contains_status_and_detail(self) -> None:
        text = checks_to_text(
            (
                Check("one", True, "ready"),
                Check("two", False, "missing"),
            )
        )

        self.assertIn("OK", text)
        self.assertIn("MISSING", text)
        self.assertIn("ready", text)
        self.assertIn("missing", text)


if __name__ == "__main__":
    unittest.main()

