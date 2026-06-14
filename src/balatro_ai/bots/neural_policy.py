"""Deployed decision-shaped policy bot (Phase B, component 5).

Turns a trained `DecisionPolicyNet` into a whole-policy bot: at each decision,
encode the state, build the same schema-v2 candidate features used in training,
score them, and return the argmax candidate as a legal `Action`. Any failure
(no checkpoint, illegal/empty output, inference error) falls back to a heuristic
bot — so a half-trained net can never crash a run, only underperform.

Checkpoint via `BALATRO_POLICY_CKPT`; honest-mode blinding is applied at the
observation boundary like every other deployed bot.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GameState
from balatro_ai.bots.no_foresight import blind_known_deck


@dataclass(slots=True)
class NeuralPolicyBot:
    seed: int | None = None
    name: str = "neural_policy_bot"
    ckpt: str | None = None
    _net: object = field(default=None, init=False, repr=False)
    _fallback: object = field(default=None, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot

        self._fallback = BasicStrategyBot(seed=self.seed)

    def _ensure_net(self) -> object | None:
        if self._loaded:
            return self._net
        self._loaded = True
        path = self.ckpt or os.environ.get("BALATRO_POLICY_CKPT", "").strip()
        if path and os.path.exists(path):
            try:
                from balatro_ai.ml.policy_net import load_policy

                self._net = load_policy(path)
            except Exception:  # noqa: BLE001 — never let a bad ckpt crash a run
                self._net = None
        return self._net

    def choose_action(self, state: GameState) -> Action:
        state = blind_known_deck(state)
        net = self._ensure_net()
        if net is None:
            return self._fallback.choose_action(state)
        try:
            from balatro_ai.ml.dataset import candidate_tokens_for_state
            from balatro_ai.ml.encoding import encode_state
            from balatro_ai.ml.policy_net import best_candidate_index

            legals = state.legal_actions
            candidates = candidate_tokens_for_state(state)
            if not candidates or len(candidates) != len(legals):
                return self._fallback.choose_action(state)
            idx = best_candidate_index(net, encode_state(state), candidates)
            if not 0 <= idx < len(legals):
                return self._fallback.choose_action(state)
            action = legals[idx]
            if action is None or action.action_type == ActionType.NO_OP:
                return self._fallback.choose_action(state)
            return action
        except Exception:  # noqa: BLE001 — inference must never crash a run
            return self._fallback.choose_action(state)
