from __future__ import annotations

import unittest

import context  # noqa: F401
from scripts import phase8_rollout_cost_profile as script


class Phase8RolloutCostProfileTests(unittest.TestCase):
    def test_summarize_rollout_timings(self) -> None:
        samples = [
            script.TimedRollout(
                value=3.0,
                wall_s=10.0,
                apply_action_s=0.5,
                bot_create_s=0.25,
                choose_action_s=7.0,
                sim_step_s=2.0,
                terminal_value_s=0.25,
                steps=4,
                phases={"shop": 2, "blind_select": 2},
                action_types={"buy": 1, "end_shop": 1},
                termination="horizon",
            ),
            script.TimedRollout(
                value=2.0,
                wall_s=5.0,
                apply_action_s=0.5,
                bot_create_s=0.25,
                choose_action_s=2.0,
                sim_step_s=2.5,
                terminal_value_s=0.25,
                steps=2,
                phases={"shop": 1, "play": 1},
                action_types={"play": 1},
                termination="run_over",
            ),
        ]

        summary = script._summarize(samples)

        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["mean_wall_s"], 7.5)
        self.assertEqual(summary["choose_action_share"], 0.6)
        self.assertEqual(summary["sim_step_share"], 0.3)
        self.assertEqual(summary["phases"], {"blind_select": 2, "play": 1, "shop": 3})
        self.assertEqual(summary["terminations"], {"horizon": 1, "run_over": 1})


if __name__ == "__main__":
    unittest.main()
