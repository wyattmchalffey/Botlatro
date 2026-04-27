from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GameState
from balatro_ai.env.action_space import ActionCatalog


class ActionSpaceTests(unittest.TestCase):
    def test_legal_action_mask_marks_allowed_actions(self) -> None:
        play = Action(ActionType.PLAY_HAND, card_indices=(0, 1))
        discard = Action(ActionType.DISCARD, card_indices=(2, 3))
        reroll = Action(ActionType.REROLL)

        catalog = ActionCatalog(actions=(play, discard, reroll))
        state = GameState(legal_actions=(discard,))

        self.assertEqual(catalog.legal_action_mask(state), (False, True, False))


if __name__ == "__main__":
    unittest.main()

