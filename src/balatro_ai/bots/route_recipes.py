"""Route-recipe bots: forced strategy diversity for self-play data generation.

The S0 result is the standing proof that value iteration on the baseline's own
trajectories cannot discover strategies the baseline never exhibits (suit
concentration had ZERO variance under the baseline, so flush commitment was
unlearnable). These wrappers force diverse whole-run strategies — the same
grammar the P1.1 route oracle measured — so generated data contains variance
along the dimensions a learner needs.

Registry names: ``recipe_<policy>_bot`` (see RECIPE_POLICIES), each wrapping
the deployed ``solver_shop_basic_play_bot``. Recipes activate from
``BALATRO_RECIPE_START_ANTE`` (default 1). Outcome labels stay honest — the
recipe shapes BEHAVIOR, never the reward.
"""

from __future__ import annotations

import os

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState

RECIPE_POLICIES = (
    "buy_best",        # every shop: buy best affordable joker offer (sell weakest if full)
    "hoard2_spend",    # skip purchases for 2 shops, then buy the 2 best affordable jokers
    "level_focus",     # every shop: buy planet/celestial offers; skip jokers
    "reroll_hunt",     # every shop: reroll up to 2x before normal decisions
    "skip_smalls",     # skip every Small Blind (tags over blind money)
    "churn_weakest",   # every shop: sell the weakest joker first, then default
    "buy_best_skip",   # buy_best + skip_smalls combined
)


def _shop_joker_offers(state: GameState):
    cards = state.modifiers.get("shop_cards", []) or []
    return [
        (i, sc) for i, sc in enumerate(cards)
        if isinstance(sc, dict) and str(sc.get("set", "")).upper() == "JOKER" and sc.get("name")
    ]


def _affordable(state: GameState, offer: dict) -> bool:
    try:
        cost = int((offer.get("cost") or {}).get("buy", offer.get("price", 10**9)))
    except (TypeError, ValueError):
        return False
    return cost <= int(state.money)


class RouteRecipeBot:
    """Wraps a base bot; overrides SHOP / BLIND_SELECT decisions per the route
    recipe, delegates everything else (plays stay the base bot's)."""

    def __init__(self, policy: str, base_bot, *, start_ante: int | None = None):
        if policy not in RECIPE_POLICIES and policy != "null":
            raise ValueError(f"unknown route recipe: {policy!r}")
        self.policy = policy
        self.bot = base_bot
        self.start_ante = (
            int(os.environ.get("BALATRO_RECIPE_START_ANTE", "1") or 1)
            if start_ante is None
            else int(start_ante)
        )
        self._shop_ante = None
        self.shop_seen = 0
        self.rerolls_this_shop = 0
        self.bought_this_shop = 0
        self.sold_this_shop = False

    def _new_shop(self, state: GameState) -> None:
        if self._shop_ante != int(state.ante):
            self._shop_ante = int(state.ante)
            self.shop_seen += 1
            self.rerolls_this_shop = 0
            self.bought_this_shop = 0
            self.sold_this_shop = False

    def _buy_best_offer(self, state: GameState) -> Action | None:
        offers = [(i, sc) for i, sc in _shop_joker_offers(state) if _affordable(state, sc)]
        if not offers:
            return None
        slot_limit = int(state.modifiers.get("joker_slot_limit", 5) or 5)
        if len(state.jokers) >= slot_limit:
            if self.sold_this_shop or not state.jokers:
                return None
            weakest = min(
                range(len(state.jokers)),
                key=lambda j: int(state.jokers[j].sell_value or 0),
            )
            self.sold_this_shop = True
            return Action(ActionType.SELL, target_id="joker", amount=weakest,
                          metadata={"kind": "joker", "index": weakest})
        i, _sc = max(offers, key=lambda o: int((o[1].get("cost") or {}).get("buy", 0) or 0))
        self.bought_this_shop += 1
        return Action(ActionType.BUY, target_id="card", amount=i,
                      metadata={"kind": "card", "index": i})

    def choose_action(self, state: GameState) -> Action:
        p = self.policy
        if p == "null" or int(state.ante) < self.start_ante:
            return self.bot.choose_action(state)

        if state.phase == GamePhase.BLIND_SELECT and p in ("skip_smalls", "buy_best_skip"):
            blind = str((state.modifiers.get("current_blind") or {}).get("name", state.blind or ""))
            if "Small" in blind:
                skip = next(
                    (a for a in state.legal_actions if a.action_type == ActionType.SKIP_BLIND),
                    None,
                )
                if skip is not None:
                    return skip

        if state.phase == GamePhase.SHOP:
            self._new_shop(state)
            if p == "reroll_hunt" and self.rerolls_this_shop < 2:
                rr = next(
                    (a for a in state.legal_actions
                     if a.action_type == ActionType.REROLL
                     and str(a.metadata.get("kind", "")) != "boss"),
                    None,
                )
                if rr is not None:
                    self.rerolls_this_shop += 1
                    return rr
            if p in ("buy_best", "buy_best_skip") and self.bought_this_shop < 1:
                action = self._buy_best_offer(state)
                if action is not None:
                    return action
            if p == "hoard2_spend":
                if self.shop_seen <= 2:
                    end = next(
                        (a for a in state.legal_actions if a.action_type == ActionType.END_SHOP),
                        None,
                    )
                    if end is not None:
                        return end
                elif self.bought_this_shop < 2:
                    action = self._buy_best_offer(state)
                    if action is not None:
                        return action
            if p == "level_focus":
                cards = state.modifiers.get("shop_cards", []) or []
                for i, sc in enumerate(cards):
                    if not isinstance(sc, dict) or not _affordable(state, sc):
                        continue
                    if str(sc.get("set", "")).upper() in ("PLANET", "CELESTIAL") and self.bought_this_shop < 2:
                        self.bought_this_shop += 1
                        return Action(ActionType.BUY, target_id="card", amount=i,
                                      metadata={"kind": "card", "index": i})
            if p == "churn_weakest" and not self.sold_this_shop and len(state.jokers) >= 2:
                weakest = min(range(len(state.jokers)),
                              key=lambda j: int(state.jokers[j].sell_value or 0))
                self.sold_this_shop = True
                return Action(ActionType.SELL, target_id="joker", amount=weakest,
                              metadata={"kind": "joker", "index": weakest})

        return self.bot.choose_action(state)
