"""Rollout-based shop policy (engine path, first brick).

Instead of scoring shop choices with a hand-crafted or learned value, COMPUTE
their value with the exact forward model: at a shop decision, take each candidate
action, then roll the run forward a few antes with a cheap rollout policy, and
score by how far it gets. Pick the best candidate. The economy-side analog of the
play-side rollout leaf — it leans on the project's strongest asset (the exact sim)
and naturally avoids build-destroying sells (selling jokers tanks the rollout).

This is also the self-play data seed: each decision yields `(state, action,
rolled-out value)` triples — the strong targets a learned value head can later
distill (the neural value's missing ingredient was a strong target, not a better
encoder).

Cost is ~|candidates| x samples x (horizon antes of rollout) per shop decision;
keep the rollout policy cheap (basic_strategy_bot) and horizon small.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState

_SHOP_ACTION_TYPES = frozenset({
    ActionType.BUY, ActionType.SELL, ActionType.REROLL,
    ActionType.OPEN_PACK, ActionType.END_SHOP,
})


def _frac_progress(state: GameState) -> float:
    """Fractional progress toward the current blind's target, in [0, 1]."""
    if state.required_score and state.required_score > 0:
        return min(1.0, max(0.0, state.current_score / state.required_score))
    return 0.0


@dataclass
class RolloutShopPolicy:
    """Greedy 1-ply + rollout shop policy. `choose_action` returns the legal shop
    action whose post-action rollout reaches the furthest.

    horizon: antes to simulate past the current one before scoring.
    samples: rollouts averaged per candidate (variance reduction; common seeds
             across candidates = fair comparison).
    """

    horizon: int = 2
    samples: int = 2
    rollout_bot: str = "basic_strategy_bot"
    stake: str = "white"
    base_seed: int = 0
    max_steps: int = 400
    fallback: object | None = None

    def choose_action(self, state: GameState) -> Action:
        legal = tuple(a for a in state.legal_actions if a.action_type in _SHOP_ACTION_TYPES)
        if not legal:
            return self._fallback_action(state)
        if len(legal) == 1:
            return legal[0]
        best, best_val = legal[0], float("-inf")
        for action in legal:
            val = self._value(state, action)
            if val > best_val:
                best_val, best = val, action
        return best

    def _value(self, state: GameState, action: Action) -> float:
        total = 0.0
        for s in range(self.samples):
            total += self._rollout(state, action, seed=self.base_seed + 9973 * (s + 1))
        return total / self.samples

    def _rollout(self, state: GameState, action: Action, *, seed: int) -> float:
        from balatro_ai.bots.registry import create_bot
        from balatro_ai.sim.local_runner import LocalBalatroSimulator

        sim = LocalBalatroSimulator(seed=seed, stake=self.stake)
        sim.state = state
        start_ante = state.ante
        try:
            sim.step(action)
        except (ValueError, IndexError, KeyError, TypeError, AttributeError):
            return float("-inf")  # illegal/unsupported -> never pick it
        bot = create_bot(self.rollout_bot, seed=seed)
        for _ in range(self.max_steps):
            s = sim.state
            if s.won:
                return float(self.horizon) + 2.0  # winning dominates
            if s.run_over or s.phase == GamePhase.RUN_OVER:
                break
            if s.ante - start_ante >= self.horizon:
                break
            nxt = bot.choose_action(s)
            if nxt is None or nxt.action_type == ActionType.NO_OP:
                break
            try:
                sim.step(nxt)
            except (ValueError, IndexError, KeyError, TypeError, AttributeError):
                break
        f = sim.state
        return (f.ante - start_ante) + _frac_progress(f)

    def _fallback_action(self, state: GameState) -> Action:
        if self.fallback is not None:
            return self.fallback.choose_action(state)
        for a in state.legal_actions:
            if a.action_type == ActionType.END_SHOP:
                return a
        return state.legal_actions[0]
