"""Per-action derived context for BasicStrategyBot decisions."""

from __future__ import annotations

from dataclasses import dataclass, field

from balatro_ai.api.state import GameState
from balatro_ai.bots.basic_strategy.blind_solver import _BlindSolution, _solve_blind
from balatro_ai.bots.basic_strategy.build_profile import _build_profile
from balatro_ai.bots.basic_strategy.hand_models import _BlindContext
from balatro_ai.bots.basic_strategy.hand_preferences import _preferred_hand_type
from balatro_ai.bots.basic_strategy.profile import _BuildProfile, _ShopContext, _ShopPressure
from balatro_ai.bots.basic_strategy.shop_pressure import _shop_pressure
from balatro_ai.rules.hand_evaluator import HandType


_UNSET = object()


@dataclass(slots=True)
class _DecisionContext:
    """Derived state shared by the policy helpers for one action choice.

    The existing decision cache already memoizes many helpers. This context gives
    the bot a single explicit object to pass around as we move toward a common
    evaluator interface, while keeping expensive values lazy so cheap branches
    do not compute shop or build data they never need.
    """

    state: GameState
    shop: _ShopContext
    blind: _BlindContext
    _build_profile_value: _BuildProfile | None = field(default=None, init=False, repr=False)
    _shop_pressure_value: _ShopPressure | None = field(default=None, init=False, repr=False)
    _preferred_hand_value: HandType | None | object = field(default=_UNSET, init=False, repr=False)
    _blind_solution_value: _BlindSolution | None = field(default=None, init=False, repr=False)

    @property
    def build_profile(self) -> _BuildProfile:
        if self._build_profile_value is None:
            self._build_profile_value = _build_profile(self.state)
        return self._build_profile_value

    @property
    def shop_pressure(self) -> _ShopPressure:
        if self._shop_pressure_value is None:
            self._shop_pressure_value = _shop_pressure(self.state)
        return self._shop_pressure_value

    @property
    def blind_solution(self) -> _BlindSolution:
        if self._blind_solution_value is None:
            self._blind_solution_value = _solve_blind(self.state, self.blind)
        return self._blind_solution_value

    @property
    def preferred_hand(self) -> HandType | None:
        if self._preferred_hand_value is _UNSET:
            self._preferred_hand_value = _preferred_hand_type(self.state)
        return self._preferred_hand_value if isinstance(self._preferred_hand_value, HandType) else None
