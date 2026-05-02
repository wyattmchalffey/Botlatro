"""Run one seeded game with a bot and a local Balatro client."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

from balatro_ai.api.client import BalatroBridgeError, BalatroClient, JsonRpcBalatroClient
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GameState
from balatro_ai.bots.base import Bot
from balatro_ai.bots.registry import create_bot
from balatro_ai.data.replay_logger import ReplayLogger
from balatro_ai.env.balatro_env import BalatroEnv
from balatro_ai.eval.metrics import RunResult
from balatro_ai.rules.hand_evaluator import debuffed_suits_for_blind, evaluate_played_cards

REPLAY_MODES = ("off", "summary", "light", "score_audit")


@dataclass(frozen=True, slots=True)
class RunSeedOptions:
    seed: int
    stake: str = "white"
    max_steps: int = 500
    run_timeout_seconds: float | None = 1800.0
    print_states: bool = False
    replay_path: Path | None = None
    replay_mode: str = "score_audit"
    profile_path: Path | None = None
    start_retries: int = 1


def run_single_seed(
    *,
    bot: Bot,
    client: BalatroClient,
    options: RunSeedOptions,
) -> RunResult:
    env = BalatroEnv(client=client, stake=options.stake)
    if options.replay_mode not in REPLAY_MODES:
        raise ValueError(f"Unknown replay mode: {options.replay_mode}")
    logger = ReplayLogger(options.replay_path) if options.replay_path and options.replay_mode != "off" else None
    profiler = _RunProfiler(options.profile_path)
    started_at = perf_counter()

    reset_started_at = perf_counter()
    _, info = _reset_with_retries(env, seed=options.seed, max_retries=options.start_retries)
    profiler.add_timing("reset_seconds", perf_counter() - reset_started_at)
    state = _with_standard_win_boundary(info["state"])
    if options.print_states:
        print(state.debug_summary)

    steps = 0
    stale_state_recoveries = 0
    timed_out = False
    while not state.run_over and steps < options.max_steps:
        if _run_timeout_exceeded(started_at, options.run_timeout_seconds):
            timed_out = True
            break
        if not state.legal_actions:
            if stale_state_recoveries >= 5:
                raise ValueError("No legal actions available in nonterminal state")
            stale_state_recoveries += 1
            sleep_seconds = 0.15 * stale_state_recoveries
            sleep_started_at = perf_counter()
            sleep(sleep_seconds)
            profiler.add_timing("stale_sleep_seconds", perf_counter() - sleep_started_at)
            refresh_started_at = perf_counter()
            state = _refreshed_state(env, fallback=state)
            profiler.add_timing("refresh_seconds", perf_counter() - refresh_started_at)
            profiler.stale_state_recoveries += 1
            env.state = state
            continue

        legal_actions = state.legal_actions
        decision_started_at = perf_counter()
        action = bot.choose_action(state)
        decision_seconds = perf_counter() - decision_started_at
        step_started_at = perf_counter()
        try:
            _, reward, _, _, info = env.step(action)
        except (BalatroBridgeError, ConnectionError) as exc:
            if not _is_recoverable_step_error(exc) or stale_state_recoveries >= 5:
                raise
            stale_state_recoveries += 1
            sleep_seconds = 0.15 * stale_state_recoveries
            sleep_started_at = perf_counter()
            sleep(sleep_seconds)
            profiler.add_timing("stale_sleep_seconds", perf_counter() - sleep_started_at)
            refresh_started_at = perf_counter()
            state = _refreshed_state(env, fallback=state)
            profiler.add_timing("refresh_seconds", perf_counter() - refresh_started_at)
            profiler.stale_state_recoveries += 1
            env.state = state
            continue
        env_step_seconds = perf_counter() - step_started_at

        next_state = _with_standard_win_boundary(info["state"])
        score_audit_seconds = 0.0
        replay_log_seconds = 0.0
        if logger is not None and options.replay_mode != "summary":
            score_audit_started_at = perf_counter()
            extra = (
                _step_extra(state=state, next_state=next_state, action=action)
                if options.replay_mode == "score_audit"
                else {}
            )
            score_audit_seconds = perf_counter() - score_audit_started_at
            replay_log_started_at = perf_counter()
            logger.log_step(
                state=state,
                legal_actions=legal_actions,
                chosen_action=action,
                reward=reward,
                extra=extra,
            )
            replay_log_seconds = perf_counter() - replay_log_started_at

        profiler.record_step(
            state=state,
            next_state=next_state,
            action=action,
            decision_seconds=decision_seconds,
            env_step_seconds=env_step_seconds,
            score_audit_seconds=score_audit_seconds,
            replay_log_seconds=replay_log_seconds,
        )

        state = next_state
        steps += 1
        stale_state_recoveries = 0

        if options.print_states:
            print(state.debug_summary)

    runtime = perf_counter() - started_at
    result = RunResult(
        bot_version=bot.name,
        seed=options.seed,
        stake=options.stake,
        won=state.won,
        ante_reached=state.ante,
        final_score=state.current_score,
        final_money=state.money,
        runtime_seconds=runtime,
        death_reason=None if state.won else _death_reason(state=state, timed_out=timed_out, options=options),
    )
    if logger is not None:
        logger.log_summary(
            bot_version=result.bot_version,
            seed=result.seed,
            stake=result.stake,
            won=result.won,
            ante_reached=result.ante_reached,
            final_score=result.final_score,
            final_money=result.final_money,
            runtime_seconds=result.runtime_seconds,
            death_reason=result.death_reason,
            final_state=state,
        )
    profiler.write(
        result=result,
        steps=steps,
        max_steps=options.max_steps,
        timed_out=timed_out,
    )
    return result


class _RunProfiler:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.timings: dict[str, float] = {
            "reset_seconds": 0.0,
            "bot_decision_seconds": 0.0,
            "env_step_seconds": 0.0,
            "score_audit_seconds": 0.0,
            "replay_log_seconds": 0.0,
            "refresh_seconds": 0.0,
            "stale_sleep_seconds": 0.0,
        }
        self.action_counts: dict[str, int] = {}
        self.phase_counts: dict[str, int] = {}
        self.ante_stats: dict[str, dict[str, float | int]] = {}
        self.slowest_steps: list[dict[str, Any]] = []
        self.stale_state_recoveries = 0

    def add_timing(self, name: str, seconds: float) -> None:
        if self.path is None:
            return
        self.timings[name] = self.timings.get(name, 0.0) + max(0.0, seconds)

    def record_step(
        self,
        *,
        state: GameState,
        next_state: GameState,
        action: Action,
        decision_seconds: float,
        env_step_seconds: float,
        score_audit_seconds: float,
        replay_log_seconds: float,
    ) -> None:
        if self.path is None:
            return

        action_name = action.action_type.value
        phase_name = state.phase.value
        self.action_counts[action_name] = self.action_counts.get(action_name, 0) + 1
        self.phase_counts[phase_name] = self.phase_counts.get(phase_name, 0) + 1
        self.add_timing("bot_decision_seconds", decision_seconds)
        self.add_timing("env_step_seconds", env_step_seconds)
        self.add_timing("score_audit_seconds", score_audit_seconds)
        self.add_timing("replay_log_seconds", replay_log_seconds)

        total_seconds = decision_seconds + env_step_seconds + score_audit_seconds + replay_log_seconds
        ante_key = str(state.ante)
        ante = self.ante_stats.setdefault(
            ante_key,
            {
                "steps": 0,
                "total_seconds": 0.0,
                "bot_decision_seconds": 0.0,
                "env_step_seconds": 0.0,
                "score_audit_seconds": 0.0,
                "replay_log_seconds": 0.0,
                "score_delta": 0,
            },
        )
        ante["steps"] = int(ante["steps"]) + 1
        ante["total_seconds"] = float(ante["total_seconds"]) + total_seconds
        ante["bot_decision_seconds"] = float(ante["bot_decision_seconds"]) + decision_seconds
        ante["env_step_seconds"] = float(ante["env_step_seconds"]) + env_step_seconds
        ante["score_audit_seconds"] = float(ante["score_audit_seconds"]) + score_audit_seconds
        ante["replay_log_seconds"] = float(ante["replay_log_seconds"]) + replay_log_seconds
        ante["score_delta"] = int(ante["score_delta"]) + max(0, next_state.current_score - state.current_score)

        self.slowest_steps.append(
            {
                "ante": state.ante,
                "blind": state.blind,
                "phase": phase_name,
                "action": action_name,
                "action_key": action.stable_key,
                "score_before": state.current_score,
                "score_after": next_state.current_score,
                "required_score": state.required_score,
                "money_before": state.money,
                "money_after": next_state.money,
                "total_seconds": total_seconds,
                "bot_decision_seconds": decision_seconds,
                "env_step_seconds": env_step_seconds,
                "score_audit_seconds": score_audit_seconds,
                "replay_log_seconds": replay_log_seconds,
            }
        )
        self.slowest_steps = sorted(
            self.slowest_steps,
            key=lambda row: float(row["total_seconds"]),
            reverse=True,
        )[:20]

    def write(self, *, result: Any, steps: int, max_steps: int, timed_out: bool) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "record_type": "run_profile",
            "bot_version": result.bot_version,
            "seed": result.seed,
            "stake": result.stake,
            "won": result.won,
            "ante": result.ante_reached,
            "final_score": result.final_score,
            "final_money": result.final_money,
            "runtime_seconds": result.runtime_seconds,
            "death_reason": result.death_reason,
            "steps": steps,
            "max_steps": max_steps,
            "timed_out": timed_out,
            "stale_state_recoveries": self.stale_state_recoveries,
            "timings": self.timings,
            "action_counts": dict(sorted(self.action_counts.items())),
            "phase_counts": dict(sorted(self.phase_counts.items())),
            "ante_stats": self.ante_stats,
            "slowest_steps": self.slowest_steps,
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_timeout_exceeded(started_at: float, timeout_seconds: float | None) -> bool:
    return timeout_seconds is not None and timeout_seconds > 0 and perf_counter() - started_at >= timeout_seconds


def _death_reason(*, state: GameState, timed_out: bool, options: RunSeedOptions) -> str | None:
    if timed_out:
        timeout = options.run_timeout_seconds
        return f"error:RunTimeout: exceeded {timeout:.1f}s" if timeout is not None else "error:RunTimeout"
    return state.blind or None


def _reset_with_retries(
    env: BalatroEnv,
    *,
    seed: int,
    max_retries: int,
) -> tuple[tuple[float, ...], dict[str, object]]:
    attempts = 0
    while True:
        try:
            return env.reset(seed=seed)
        except (BalatroBridgeError, ConnectionError):
            if attempts >= max_retries:
                raise
            attempts += 1
            sleep(0.25 * attempts)


def _with_standard_win_boundary(state: GameState) -> GameState:
    if state.ante >= 9:
        return replace(state, ante=8, run_over=True, won=True, legal_actions=())
    if state.run_over or state.won:
        return state
    return state


def _step_extra(*, state: GameState, next_state: GameState, action: Action) -> dict[str, object]:
    if action.action_type != ActionType.PLAY_HAND or not action.card_indices:
        return {}

    selected_cards = tuple(state.hand[index] for index in action.card_indices)
    held_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
    evaluation = evaluate_played_cards(
        selected_cards,
        state.hand_levels,
        debuffed_suits=debuffed_suits_for_blind(state.blind),
        blind_name=state.blind,
        jokers=state.jokers,
        discards_remaining=state.discards_remaining,
        hands_remaining=state.hands_remaining,
        held_cards=held_cards,
        deck_size=state.deck_size,
        money=state.money,
        played_hand_types_this_round=_played_hand_types_this_round(state),
        played_hand_counts=_played_hand_counts(state),
    )
    actual_score_delta = next_state.current_score - state.current_score
    return {
        "score_audit": {
            "cards": [card.short_name for card in selected_cards],
            "card_details": [_card_audit_payload(card) for card in selected_cards],
            "hand_before": [card.short_name for card in state.hand],
            "hand_before_details": [_card_audit_payload(card) for card in state.hand],
            "held_cards": [card.short_name for card in held_cards],
            "held_card_details": [_card_audit_payload(card) for card in held_cards],
            "hand_type": evaluation.hand_type.value,
            "scoring_indices": list(evaluation.scoring_indices),
            "predicted_chips": evaluation.chips,
            "predicted_mult": evaluation.mult,
            "predicted_score": evaluation.score,
            "actual_score_delta": actual_score_delta,
            "score_before": state.current_score,
            "score_after": next_state.current_score,
            "ante": state.ante,
            "blind": state.blind,
            "money": state.money,
            "hands_remaining": state.hands_remaining,
            "discards_remaining": state.discards_remaining,
            "deck_size": state.deck_size,
            "hand_levels": dict(state.hand_levels),
            "hands": dict(state.modifiers.get("hands", {})) if isinstance(state.modifiers.get("hands"), dict) else {},
            "played_hand_counts": _played_hand_counts(state),
            "jokers": [joker.name for joker in state.jokers],
            "joker_details": [_joker_audit_payload(joker) for joker in state.jokers],
        }
    }


def _played_hand_types_this_round(state: GameState) -> tuple[str, ...]:
    hands = state.modifiers.get("hands", {})
    if not isinstance(hands, dict):
        return ()

    played: list[tuple[int, str]] = []
    for name, value in hands.items():
        if not isinstance(value, dict):
            continue
        try:
            played_count = int(value.get("played_this_round", 0) or 0)
        except (TypeError, ValueError):
            played_count = 0
        if played_count <= 0:
            continue
        try:
            order = int(value.get("order", 0) or 0)
        except (TypeError, ValueError):
            order = 0
        played.extend((order, str(name)) for _ in range(played_count))
    return tuple(name for _, name in sorted(played, key=lambda item: item[0]))


def _played_hand_counts(state: GameState) -> dict[str, int]:
    hands = state.modifiers.get("hands", {})
    if not isinstance(hands, dict):
        return {}

    counts: dict[str, int] = {}
    for name, value in hands.items():
        if isinstance(value, dict):
            counts[str(name)] = _int_value(value.get("played"))
        else:
            counts[str(name)] = _int_value(value)
    return counts


def _card_audit_payload(card) -> dict[str, str | None]:
    return {
        "rank": card.rank,
        "suit": card.suit,
        "enhancement": card.enhancement,
        "seal": card.seal,
        "edition": card.edition,
        "debuffed": card.debuffed,
        "metadata": dict(card.metadata),
    }


def _joker_audit_payload(joker) -> dict[str, object]:
    return {
        "name": joker.name,
        "edition": joker.edition,
        "sell_value": joker.sell_value,
        "metadata": dict(joker.metadata),
    }


def _int_value(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _is_recoverable_step_error(exc: Exception) -> bool:
    return isinstance(exc, ConnectionError) or (
        isinstance(exc, BalatroBridgeError) and exc.name == "INVALID_STATE"
    )


def _refreshed_state(env: BalatroEnv, *, fallback: GameState) -> GameState:
    try:
        return env.client.get_state()
    except ConnectionError:
        return fallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Botlatro bot on one seed.")
    parser.add_argument("--bot", default="random_bot", help="Bot name.")
    parser.add_argument("--seed", type=int, required=True, help="Game seed.")
    parser.add_argument("--stake", default="white", help="Stake name.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:12346", help="JSON-RPC endpoint.")
    parser.add_argument("--timeout-seconds", type=float, default=10.0, help="JSON-RPC request timeout.")
    parser.add_argument("--max-steps", type=int, default=500, help="Safety cap for one run.")
    parser.add_argument(
        "--run-timeout-seconds",
        type=float,
        default=1800.0,
        help="Wall-clock cap for one full seed; use 0 to disable.",
    )
    parser.add_argument("--print-states", action="store_true", help="Print state summaries while running.")
    parser.add_argument("--replay-path", type=Path, help="Optional JSONL replay output path.")
    parser.add_argument("--profile-path", type=Path, help="Optional JSON profile timing output path.")
    parser.add_argument(
        "--replay-mode",
        choices=REPLAY_MODES,
        default="score_audit",
        help="Replay detail: off, summary-only JSONL, light JSONL, or score-audit JSONL.",
    )
    parser.add_argument("--start-retries", type=int, default=1, help="Retries for bridge start/reset failures.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bot = create_bot(args.bot, seed=args.seed)
    client = JsonRpcBalatroClient(endpoint=args.endpoint, timeout_seconds=args.timeout_seconds)
    options = RunSeedOptions(
        seed=args.seed,
        stake=args.stake,
        max_steps=args.max_steps,
        run_timeout_seconds=_optional_positive_float(args.run_timeout_seconds),
        print_states=args.print_states,
        replay_path=args.replay_path,
        replay_mode=args.replay_mode,
        profile_path=args.profile_path,
        start_retries=args.start_retries,
    )
    result = run_single_seed(bot=bot, client=client, options=options)
    print(
        f"Bot={result.bot_version} seed={result.seed} stake={result.stake} "
        f"won={result.won} ante={result.ante_reached} score={result.final_score} "
        f"money={result.final_money} runtime={result.runtime_seconds:.2f}s"
    )
    return 0


def _optional_positive_float(value: float) -> float | None:
    return value if value > 0 else None


if __name__ == "__main__":
    raise SystemExit(main())
