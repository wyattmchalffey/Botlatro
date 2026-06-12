"""Bot lookup helpers for CLI commands."""

from __future__ import annotations

from dataclasses import dataclass, field
import os

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot
from balatro_ai.bots.base import Bot
from balatro_ai.bots.config import BotConfig, bot_config_scope
from balatro_ai.bots.greedy_bot import GreedyBot
from balatro_ai.bots.neural_shop import RankerGuidedShopBot
from balatro_ai.bots.no_foresight import blind_known_deck
from balatro_ai.bots.random_bot import RandomBot
from balatro_ai.bots.search_bot import SearchBot
from balatro_ai.bots.search_bot_v2 import SearchBotV2


@dataclass(slots=True)
class SolverPolicyBot:
    """Registry adapter for the composed offline solver policy."""

    seed: int | None = None
    name: str = "solver_policy_bot"
    _policy: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from balatro_ai.solver.policy import SolverPolicy

        self._policy = SolverPolicy(seed=self.seed or 0)

    def choose_action(self, state: GameState) -> Action:
        return self._policy.choose_action(blind_known_deck(state))


@dataclass(slots=True)
class _BlindedPolicyBot:
    """Honest-mode adapter for raw policies that bypass the registry
    wrappers: blinds the observation (no_foresight) then delegates."""

    policy: object
    name: str = "blinded_policy_bot"

    def choose_action(self, state: GameState) -> Action:
        return self.policy.choose_action(blind_known_deck(state))


@dataclass(slots=True)
class SolverShopBasicPlayBot:
    """Solver shop policy with BasicStrategy play decisions."""

    seed: int | None = None
    name: str = "solver_shop_basic_play_bot"
    prefer_fallback_info_first_shop: bool = True
    fallback_negative_shop_sells: bool = True
    prefer_opening_buffoon_pack: bool = False
    fallback_opening_buffoon_over_non_joker_buy: bool = False
    fallback_unfunded_open_slot_sells: bool = False
    fallback_negative_end_shop: bool = False
    fallback_ante1_single_joker_planet_buy: bool = False
    fallback_negative_shop_actions: bool = False
    fallback_shop_through_ante: int = 0
    config: BotConfig | None = None
    _fallback: BasicStrategyBot = field(init=False, repr=False)
    _policy: object = field(init=False, repr=False)
    _deep_play: object = field(default=None, init=False, repr=False)
    _deep_play_used: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        from balatro_ai.solver.policy import SolverPolicy

        self._fallback = BasicStrategyBot(seed=self.seed, config=self.config)
        self._policy = SolverPolicy(
            play_policy=self._fallback,
            fallback=self._fallback,
            seed=self.seed or 0,
            prefer_fallback_info_first_shop=self.prefer_fallback_info_first_shop,
            fallback_negative_shop_sells=self.fallback_negative_shop_sells,
            prefer_opening_buffoon_pack=self.prefer_opening_buffoon_pack,
            fallback_opening_buffoon_over_non_joker_buy=self.fallback_opening_buffoon_over_non_joker_buy,
            fallback_unfunded_open_slot_sells=self.fallback_unfunded_open_slot_sells,
            fallback_negative_end_shop=self.fallback_negative_end_shop,
            fallback_ante1_single_joker_planet_buy=self.fallback_ante1_single_joker_planet_buy,
            fallback_negative_shop_actions=self.fallback_negative_shop_actions,
        )

    def choose_action(self, state: GameState) -> Action:
        state = blind_known_deck(state)
        with bot_config_scope(self.config):
            deep_action = self._deep_play_action(state)
            if deep_action is not None:
                return deep_action
            if (
                self.fallback_shop_through_ante > 0
                and state.phase == GamePhase.SHOP
                and state.ante <= self.fallback_shop_through_ante
            ):
                return self._fallback.choose_action(state)
            return self._policy.choose_action(state)

    def _deep_play_action(self, state: GameState) -> Action | None:
        """P1.5 experiment knob: from BALATRO_DEEP_PLAY_ANTE on, ENDANGERED
        blind-phase decisions route to a deep v2 play beam (the honest
        fork-audit's driver: 29.7% of honest losses were clearable by deep
        play at the death blind). Endangered = the basic heuristic's best
        immediate play, repeated for every remaining hand, still misses the
        blind by the margin — i.e. the bot is projected to FAIL. Safe blinds
        keep basic play (co-designed with the build, and ~30x cheaper;
        blanket delegation measured win->loss flips). Default OFF; knobs:
        BALATRO_DEEP_PLAY_DEPTH/WIDTH/MARGIN. Returns None for normal path."""

        deep_ante = int(os.environ.get("BALATRO_DEEP_PLAY_ANTE", "0") or 0)
        if deep_ante <= 0 or state.ante < deep_ante:
            return None
        if state.phase != GamePhase.SELECTING_HAND:
            return None
        # Cost rail: bound worst-case run cost (a degenerate seed could
        # otherwise grind thousands of multi-second deep decisions).
        budget = int(os.environ.get("BALATRO_DEEP_PLAY_BUDGET", "25") or 25)
        if self._deep_play_used >= budget:
            return None
        remaining = state.required_score - state.current_score
        if remaining <= 0:
            return None
        # Endangerment via the bot's OWN redraw-aware pace model: delegate
        # only when _solve_blind projects an outright miss (hands_needed >
        # hands_remaining). Immediate-play extrapolation was measured too
        # imprecise (fired on clearable blinds; the beam then lost them).
        try:
            from balatro_ai.bots.basic_strategy.blind_solver import _solve_blind
            from balatro_ai.bots.basic_strategy.cache import decision_cache_scope
            from balatro_ai.bots.basic_strategy.hand_models import _BlindContext

            with decision_cache_scope():
                solution = _solve_blind(state, _BlindContext())
        except Exception:  # noqa: BLE001
            return None
        if solution is None or solution.can_clear_now:
            return None
        if solution.hands_needed <= max(1, solution.hands_remaining):
            return None  # bot's own pace model projects a clear -> basic play
        if self._deep_play is None:
            from balatro_ai.solver.policy import SolverPolicy

            self._deep_play = SolverPolicy(
                play_backend="v2",
                play_depth=int(os.environ.get("BALATRO_DEEP_PLAY_DEPTH", "4") or 4),
                play_width=int(os.environ.get("BALATRO_DEEP_PLAY_WIDTH", "4") or 4),
                seed=self.seed or 0,
            )
        try:
            action = self._deep_play.choose_action(state)
        except Exception:  # noqa: BLE001 — never let the experiment knob kill a run
            return None
        if action is None or action.action_type == ActionType.NO_OP:
            return None
        self._deep_play_used += 1
        return action


@dataclass(slots=True)
class SolverShopBasicPlayNoSellBot:
    """Solver shop policy with BasicStrategy play and sell fallback guard."""

    seed: int | None = None
    name: str = "solver_shop_basic_play_no_sell_bot"
    _policy: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from balatro_ai.solver.policy import SolverPolicy

        fallback = BasicStrategyBot(seed=self.seed)
        self._policy = SolverPolicy(
            play_policy=fallback,
            fallback=fallback,
            seed=self.seed or 0,
            allow_shop_sells=False,
        )

    def choose_action(self, state: GameState) -> Action:
        return self._policy.choose_action(state)


@dataclass(slots=True)
class ValueGuidedBot:
    """Enable Phase 8 value guidance only while the wrapped bot chooses."""

    wrapped: Bot
    seed: int | None = None
    name: str = "value_guided_bot"

    def choose_action(self, state: GameState) -> Action:
        ckpt = os.environ.get("BALATRO_VALUE_GUIDED_CKPT", "").strip()
        head = os.environ.get("BALATRO_VALUE_GUIDED_HEAD", "ante").strip() or "ante"
        scale = os.environ.get("BALATRO_VALUE_GUIDED_SCALE", "250").strip() or "250"
        if not ckpt:
            return self.wrapped.choose_action(state)

        old_ckpt = os.environ.get("BALATRO_VALUE_MODEL_CKPT")
        old_head = os.environ.get("BALATRO_VALUE_MODEL_HEAD")
        old_scale = os.environ.get("BALATRO_VALUE_SCALE")
        os.environ["BALATRO_VALUE_MODEL_CKPT"] = ckpt
        os.environ["BALATRO_VALUE_MODEL_HEAD"] = head
        os.environ["BALATRO_VALUE_SCALE"] = scale
        try:
            return self.wrapped.choose_action(state)
        finally:
            _restore_env("BALATRO_VALUE_MODEL_CKPT", old_ckpt)
            _restore_env("BALATRO_VALUE_MODEL_HEAD", old_head)
            _restore_env("BALATRO_VALUE_SCALE", old_scale)


@dataclass(slots=True)
class PortfolioBot:
    """Choose a full-run policy by simulating candidate bots from the seed."""

    seed: int | None = None
    name: str = "portfolio_basic_solver_bot"
    candidate_names: tuple[str, ...] = ("basic_strategy_bot", "solver_shop_basic_play_bot")
    _chosen: Bot | None = field(default=None, init=False, repr=False)
    _chosen_name: str | None = field(default=None, init=False, repr=False)
    _action_plan: tuple[str, ...] = field(default=(), init=False, repr=False)
    _action_plan_index: int = field(default=0, init=False, repr=False)

    def choose_action(self, state: GameState) -> Action:
        if self._chosen is None:
            self._chosen_name, result = self._select_candidate(state)
            self._action_plan = tuple(str(key) for key in result.get("action_plan", ()))
            self._action_plan_index = 0
        planned = self._planned_action(state)
        if planned is not None:
            return planned
        if self._chosen is None:
            self._chosen = create_bot(self._chosen_name, seed=self.seed)
        return self._chosen.choose_action(state)

    def _select_candidate(self, state: GameState) -> tuple[str, dict]:
        seed = _portfolio_seed(state)
        if not seed:
            return self.candidate_names[0], {}
        results = [
            (candidate, _portfolio_candidate_result(candidate, seed))
            for candidate in self.candidate_names
        ]
        return max(results, key=lambda item: _portfolio_result_key(item[1]))

    def _planned_action(self, state: GameState) -> Action | None:
        if self._action_plan_index >= len(self._action_plan):
            return None
        stable_key = self._action_plan[self._action_plan_index]
        for action in state.legal_actions:
            if action.stable_key == stable_key:
                self._action_plan_index += 1
                return action
        self._action_plan = ()
        self._action_plan_index = 0
        return None


def _portfolio_seed(state: GameState) -> str | None:
    raw = os.environ.get("BALATRO_RUN_SEED", "").strip() or os.environ.get("BALATRO_PORTFOLIO_SEED", "").strip()
    if raw:
        return raw
    modifier_seed = state.modifiers.get("balatro_seed")
    return str(modifier_seed) if modifier_seed else None


def _portfolio_candidate_result(bot_name: str, seed: str) -> dict:
    from dataclasses import replace as dc_replace

    from balatro_ai.api.actions import ActionType
    from balatro_ai.api.state import GamePhase, with_derived_legal_actions
    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = with_derived_legal_actions(SeedGame(seed, stake="white").initial_state())
    bot = create_bot(bot_name, seed=0)
    termination = "max_steps"
    action_plan: list[str] = []
    with bot_config_scope(dc_replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
        for _steps in range(1, 4001):
            state = sim.state
            if state.run_over or state.phase == GamePhase.RUN_OVER:
                termination = "run_over"
                break
            action = bot.choose_action(state)
            if action is None or action.action_type == ActionType.NO_OP:
                termination = "no_action"
                break
            action_plan.append(action.stable_key)
            try:
                sim.step(action)
            except (ValueError, IndexError, KeyError, TypeError, AttributeError) as exc:
                termination = f"sim_error:{type(exc).__name__}"
                break

    final = sim.state
    return {
        "won": bool(final.won),
        "run_over": bool(final.run_over or final.phase == GamePhase.RUN_OVER),
        "termination": termination,
        "ante": int(final.ante),
        "score": int(final.current_score),
        "required_score": int(final.required_score),
        "loss_frac": (final.current_score / final.required_score)
        if (not final.won and final.required_score > 0)
        else None,
        "action_plan": tuple(action_plan),
    }


def _portfolio_result_key(result: dict) -> tuple[object, ...]:
    loss_frac = result.get("loss_frac")
    return (
        int(bool(result.get("won"))),
        int(result.get("ante", 0)),
        float(loss_frac) if loss_frac is not None else 2.0,
        int(result.get("score", 0)),
    )


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def create_bot(name: str, seed: int | None = None) -> Bot:
    normalized = name.lower().replace("-", "_")
    if normalized == "recipe_flush_commit_bot":
        # Suit-stacking diversity (the S0 antidote dimension): the deployed
        # composition with the flush archetype leaf committed from ante 1 —
        # the exact bot the S0 oracle measured (flush best on 72/200 seeds).
        from balatro_ai.solver.archetypes import FLUSH_ARCHETYPE
        from balatro_ai.solver.policy import SolverPolicy

        fb = BasicStrategyBot(seed=seed)
        return _BlindedPolicyBot(
            SolverPolicy(
                play_policy=fb,
                fallback=fb,
                seed=seed or 0,
                prefer_fallback_info_first_shop=True,
                fallback_negative_shop_sells=True,
                archetype=FLUSH_ARCHETYPE,
            ),
            name="recipe_flush_commit_bot",
        )
    if normalized.startswith("recipe_") and normalized.endswith("_bot"):
        # Exploration-diversity wrappers for self-play data generation
        # (route_recipes.py): recipe_<policy>_bot over the deployed bot.
        from balatro_ai.bots.route_recipes import RouteRecipeBot

        policy = normalized[len("recipe_"):-len("_bot")]
        return RouteRecipeBot(policy, SolverShopBasicPlayBot(seed=seed))
    if normalized == "random_bot":
        return RandomBot(seed=seed)
    if normalized == "greedy_bot":
        return GreedyBot(seed=seed)
    if normalized in {"basic_strategy_bot", "basic_bot", "rule_bot"}:
        return BasicStrategyBot(seed=seed)
    if normalized in {"basic_strategy_value_guided", "basic_strategy_value_guided_bot"}:
        return ValueGuidedBot(
            wrapped=BasicStrategyBot(seed=seed),
            seed=seed,
            name="basic_strategy_value_guided_bot",
        )
    if normalized in {"basic_strategy_shop_ranker", "basic_strategy_shop_ranker_bot"}:
        return RankerGuidedShopBot(
            wrapped=BasicStrategyBot(seed=seed),
            seed=seed,
            name="basic_strategy_shop_ranker_bot",
        )
    if normalized in {"basic_strategy_legacy", "basic_strategy_bot_legacy", "rule_bot_legacy"}:
        return BasicStrategyBot(
            seed=seed,
            name="basic_strategy_legacy",
            config=BotConfig(calibrated_shop_planner_enabled=False),
        )
    if normalized in {"search_bot", "search_bot_v0"}:
        return SearchBot(seed=seed)
    if normalized in {"search_bot_v1", "shop_search_bot"}:
        return SearchBot(seed=seed, enable_shop_search=True, name="search_bot_v1")
    if normalized == "search_bot_v2":
        return SearchBotV2(seed=seed)
    if normalized in {"solver_policy", "solver_policy_bot", "offline_solver"}:
        return SolverPolicyBot(seed=seed)
    if normalized in {"solver_shop_basic_play", "solver_shop_basic_play_bot"}:
        return SolverShopBasicPlayBot(seed=seed)
    if normalized in {"solver_shop_basic_play_value_guided", "solver_shop_basic_play_value_guided_bot"}:
        return ValueGuidedBot(
            wrapped=SolverShopBasicPlayBot(seed=seed),
            seed=seed,
            name="solver_shop_basic_play_value_guided_bot",
        )
    if normalized in {"solver_shop_basic_play_shop_ranker", "solver_shop_basic_play_shop_ranker_bot"}:
        return RankerGuidedShopBot(
            wrapped=SolverShopBasicPlayBot(seed=seed),
            seed=seed,
            name="solver_shop_basic_play_shop_ranker_bot",
        )
    if normalized in {"solver_shop_basic_play_info", "solver_shop_basic_play_info_bot"}:
        return SolverShopBasicPlayBot(
            seed=seed,
            name="solver_shop_basic_play_info_bot",
            prefer_fallback_info_first_shop=True,
            fallback_negative_shop_sells=False,
        )
    if normalized in {"solver_shop_basic_play_neg_sell", "solver_shop_basic_play_neg_sell_bot"}:
        return SolverShopBasicPlayBot(
            seed=seed,
            name="solver_shop_basic_play_neg_sell_bot",
            prefer_fallback_info_first_shop=False,
            fallback_negative_shop_sells=True,
        )
    if normalized in {"solver_shop_basic_play_buffoon", "solver_shop_basic_play_buffoon_bot"}:
        return SolverShopBasicPlayBot(
            seed=seed,
            name="solver_shop_basic_play_buffoon_bot",
            prefer_fallback_info_first_shop=True,
            fallback_negative_shop_sells=True,
            prefer_opening_buffoon_pack=True,
        )
    if normalized in {"solver_shop_basic_play_buffoon_nonjoker", "solver_shop_basic_play_buffoon_nonjoker_bot"}:
        return SolverShopBasicPlayBot(
            seed=seed,
            name="solver_shop_basic_play_buffoon_nonjoker_bot",
            prefer_fallback_info_first_shop=True,
            fallback_negative_shop_sells=True,
            fallback_opening_buffoon_over_non_joker_buy=True,
        )
    if normalized in {"solver_shop_basic_play_sell_guard", "solver_shop_basic_play_sell_guard_bot"}:
        return SolverShopBasicPlayBot(
            seed=seed,
            name="solver_shop_basic_play_sell_guard_bot",
            prefer_fallback_info_first_shop=True,
            fallback_negative_shop_sells=True,
            fallback_unfunded_open_slot_sells=True,
        )
    if normalized in {"solver_shop_basic_play_basic_ante1", "solver_shop_basic_play_basic_ante1_bot"}:
        return SolverShopBasicPlayBot(
            seed=seed,
            name="solver_shop_basic_play_basic_ante1_bot",
            prefer_fallback_info_first_shop=True,
            fallback_negative_shop_sells=True,
            fallback_shop_through_ante=1,
        )
    if normalized in {"solver_shop_basic_play_basic_ante2", "solver_shop_basic_play_basic_ante2_bot"}:
        return SolverShopBasicPlayBot(
            seed=seed,
            name="solver_shop_basic_play_basic_ante2_bot",
            prefer_fallback_info_first_shop=True,
            fallback_negative_shop_sells=True,
            fallback_shop_through_ante=2,
        )
    if normalized in {"solver_shop_basic_play_neg_action", "solver_shop_basic_play_neg_action_bot"}:
        return SolverShopBasicPlayBot(
            seed=seed,
            name="solver_shop_basic_play_neg_action_bot",
            prefer_fallback_info_first_shop=True,
            fallback_negative_shop_sells=True,
            fallback_negative_shop_actions=True,
        )
    if normalized in {"solver_shop_basic_play_neg_end", "solver_shop_basic_play_neg_end_bot"}:
        return SolverShopBasicPlayBot(
            seed=seed,
            name="solver_shop_basic_play_neg_end_bot",
            prefer_fallback_info_first_shop=True,
            fallback_negative_shop_sells=True,
            fallback_negative_end_shop=True,
        )
    if normalized in {"solver_shop_basic_play_ante1_planet", "solver_shop_basic_play_ante1_planet_bot"}:
        return SolverShopBasicPlayBot(
            seed=seed,
            name="solver_shop_basic_play_ante1_planet_bot",
            prefer_fallback_info_first_shop=True,
            fallback_negative_shop_sells=True,
            fallback_ante1_single_joker_planet_buy=True,
        )
    if normalized in {"solver_shop_basic_play_no_sell", "solver_shop_basic_play_no_sell_bot"}:
        return SolverShopBasicPlayNoSellBot(seed=seed)
    if normalized in {"portfolio_basic_solver", "portfolio_basic_solver_bot"}:
        return PortfolioBot(seed=seed)
    raise ValueError(f"Unknown bot: {name}")
