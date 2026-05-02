"""Run one seeded game with a bot and a local Balatro client."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter, sleep

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
    print_states: bool = False
    replay_path: Path | None = None
    replay_mode: str = "score_audit"
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
    started_at = perf_counter()

    _, info = _reset_with_retries(env, seed=options.seed, max_retries=options.start_retries)
    state = _with_standard_win_boundary(info["state"])
    if options.print_states:
        print(state.debug_summary)

    steps = 0
    stale_state_recoveries = 0
    while not state.run_over and steps < options.max_steps:
        if not state.legal_actions:
            if stale_state_recoveries >= 5:
                raise ValueError("No legal actions available in nonterminal state")
            stale_state_recoveries += 1
            sleep(0.15 * stale_state_recoveries)
            state = _refreshed_state(env, fallback=state)
            env.state = state
            continue

        legal_actions = state.legal_actions
        action = bot.choose_action(state)
        try:
            _, reward, _, _, info = env.step(action)
        except (BalatroBridgeError, ConnectionError) as exc:
            if not _is_recoverable_step_error(exc) or stale_state_recoveries >= 5:
                raise
            stale_state_recoveries += 1
            sleep(0.15 * stale_state_recoveries)
            state = _refreshed_state(env, fallback=state)
            env.state = state
            continue

        next_state = _with_standard_win_boundary(info["state"])
        if logger is not None and options.replay_mode != "summary":
            extra = (
                _step_extra(state=state, next_state=next_state, action=action)
                if options.replay_mode == "score_audit"
                else {}
            )
            logger.log_step(
                state=state,
                legal_actions=legal_actions,
                chosen_action=action,
                reward=reward,
                extra=extra,
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
        death_reason=None if state.won else state.blind or None,
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
    return result


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
    parser.add_argument("--print-states", action="store_true", help="Print state summaries while running.")
    parser.add_argument("--replay-path", type=Path, help="Optional JSONL replay output path.")
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
        print_states=args.print_states,
        replay_path=args.replay_path,
        replay_mode=args.replay_mode,
        start_retries=args.start_retries,
    )
    result = run_single_seed(bot=bot, client=client, options=options)
    print(
        f"Bot={result.bot_version} seed={result.seed} stake={result.stake} "
        f"won={result.won} ante={result.ante_reached} score={result.final_score} "
        f"money={result.final_money} runtime={result.runtime_seconds:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
