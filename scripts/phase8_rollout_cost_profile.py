"""Profile the cost of one Phase 8 shop rollout continuation."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.ml.shop_candidate_dataset import action_key, candidate_shop_actions


@dataclass(frozen=True, slots=True)
class TimedRollout:
    value: float | None
    wall_s: float
    apply_action_s: float
    bot_create_s: float
    choose_action_s: float
    sim_step_s: float
    terminal_value_s: float
    steps: int
    phases: dict[str, int]
    action_types: dict[str, int]
    termination: str


def _configure_rust_bestplay(enabled: bool) -> None:
    os.environ["BALATRO_RUST_BESTPLAY"] = "1" if enabled else "0"
    module = sys.modules.get("balatro_ai.rules.hand_evaluator")
    if module is not None:
        setattr(module, "_RUST_BESTPLAY_ENABLED", enabled)


def _load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


def _parse_action_types_csv(value: str | None) -> tuple[ActionType, ...] | None:
    if value is None or not value.strip():
        return None
    by_value = {action_type.value: action_type for action_type in ActionType}
    by_name = {action_type.name.lower(): action_type for action_type in ActionType}
    out: list[ActionType] = []
    seen: set[ActionType] = set()
    for raw in value.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        action_type = by_value.get(key) or by_name.get(key)
        if action_type is None:
            allowed = ", ".join(action.value for action in ActionType)
            raise ValueError(f"unknown action type {raw!r}; expected one of: {allowed}")
        if action_type not in seen:
            seen.add(action_type)
            out.append(action_type)
    return tuple(out)


def _select_action(
    state: GameState,
    *,
    action_key_value: str,
    action_index: int,
    max_actions: int,
    action_types: tuple[ActionType, ...] | None,
    priority: str,
) -> Action:
    actions = candidate_shop_actions(
        state,
        max_actions=max_actions,
        action_types=action_types,
        priority=priority,
    )
    if not actions:
        raise ValueError("selected state has no candidate shop actions")
    if action_key_value:
        for action in actions:
            if action_key(action) == action_key_value:
                return action
        raise ValueError(f"candidate action key not found: {action_key_value}")
    if not 0 <= action_index < len(actions):
        raise ValueError(f"action index {action_index} outside {len(actions)} candidates")
    return actions[action_index]


def profile_rollout_after_action(
    state: GameState,
    action: Action,
    *,
    seed: int,
    rollout_bot: str,
    max_antes: int,
    max_steps: int,
) -> TimedRollout:
    from dataclasses import replace

    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.ml.shop_candidate_dataset import _rollout_terminal_value
    from balatro_ai.sim.local_runner import LocalBalatroSimulator

    started = time.perf_counter()
    sim = LocalBalatroSimulator(seed=seed, stake="white")
    sim.state = state
    apply_started = time.perf_counter()
    try:
        sim.step(action)
    except (ValueError, IndexError, KeyError, TypeError, AttributeError):
        return TimedRollout(
            value=None,
            wall_s=time.perf_counter() - started,
            apply_action_s=time.perf_counter() - apply_started,
            bot_create_s=0.0,
            choose_action_s=0.0,
            sim_step_s=0.0,
            terminal_value_s=0.0,
            steps=0,
            phases={},
            action_types={},
            termination="candidate_error",
        )
    apply_action_s = time.perf_counter() - apply_started
    bot_started = time.perf_counter()
    bot = create_bot(rollout_bot, seed=seed)
    bot_create_s = time.perf_counter() - bot_started
    start_ante = state.ante
    choose_action_s = 0.0
    sim_step_s = 0.0
    terminal_value_s = 0.0
    phases: Counter[str] = Counter()
    action_types: Counter[str] = Counter()
    termination = "max_steps"
    value: float | None = None
    steps = 0
    with bot_config_scope(replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
        for steps in range(max_steps):
            current = sim.state
            phases[current.phase.value] += 1
            if current.won:
                terminal_started = time.perf_counter()
                value = _rollout_terminal_value(current, root_state=state)
                terminal_value_s += time.perf_counter() - terminal_started
                termination = "won"
                break
            if current.run_over or current.phase == GamePhase.RUN_OVER:
                termination = "run_over"
                break
            if current.ante - start_ante >= max_antes:
                terminal_started = time.perf_counter()
                value = _rollout_terminal_value(current, root_state=state)
                terminal_value_s += time.perf_counter() - terminal_started
                termination = "horizon"
                break
            choose_started = time.perf_counter()
            next_action = bot.choose_action(current)
            choose_action_s += time.perf_counter() - choose_started
            if next_action is None or next_action.action_type == ActionType.NO_OP:
                termination = "no_action"
                break
            action_types[next_action.action_type.value] += 1
            step_started = time.perf_counter()
            try:
                sim.step(next_action)
            except (ValueError, IndexError, KeyError, TypeError, AttributeError):
                sim_step_s += time.perf_counter() - step_started
                termination = "sim_error"
                break
            sim_step_s += time.perf_counter() - step_started
        else:
            steps = max_steps
    if value is None:
        terminal_started = time.perf_counter()
        value = _rollout_terminal_value(sim.state, root_state=state)
        terminal_value_s += time.perf_counter() - terminal_started
    return TimedRollout(
        value=value,
        wall_s=time.perf_counter() - started,
        apply_action_s=apply_action_s,
        bot_create_s=bot_create_s,
        choose_action_s=choose_action_s,
        sim_step_s=sim_step_s,
        terminal_value_s=terminal_value_s,
        steps=steps + 1 if termination != "max_steps" else steps,
        phases=dict(sorted(phases.items())),
        action_types=dict(sorted(action_types.items())),
        termination=termination,
    )


def _summarize(samples: list[TimedRollout]) -> dict[str, Any]:
    if not samples:
        return {"samples": 0}
    walls = [sample.wall_s for sample in samples]
    choose = [sample.choose_action_s for sample in samples]
    sim_step = [sample.sim_step_s for sample in samples]
    steps = [sample.steps for sample in samples]
    phases: Counter[str] = Counter()
    action_types: Counter[str] = Counter()
    terminations: Counter[str] = Counter()
    for sample in samples:
        phases.update(sample.phases)
        action_types.update(sample.action_types)
        terminations[sample.termination] += 1
    return {
        "samples": len(samples),
        "mean_wall_s": statistics.mean(walls),
        "max_wall_s": max(walls),
        "mean_choose_action_s": statistics.mean(choose),
        "mean_sim_step_s": statistics.mean(sim_step),
        "mean_steps": statistics.mean(steps),
        "choose_action_share": sum(choose) / sum(walls) if sum(walls) > 0 else None,
        "sim_step_share": sum(sim_step) / sum(walls) if sum(walls) > 0 else None,
        "phases": dict(sorted(phases.items())),
        "action_types": dict(sorted(action_types.items())),
        "terminations": dict(sorted(terminations.items())),
    }


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile one or more rollout continuations from shop snapshots.")
    parser.add_argument("--input-records", type=Path, action="append", required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--action-index", type=int, default=0)
    parser.add_argument("--action-key", default="")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--rollout-bot", action="append", default=[])
    parser.add_argument("--max-antes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-actions", type=int, default=4)
    parser.add_argument("--candidate-action-types", default="buy,open_pack,end_shop")
    parser.add_argument("--candidate-priority", default="deep_advantage", choices=("legal", "deep_advantage"))
    parser.add_argument("--no-rust-bestplay", dest="rust_bestplay", action="store_false", default=True)
    args = parser.parse_args()
    _configure_rust_bestplay(bool(args.rust_bestplay))
    try:
        action_types = _parse_action_types_csv(args.candidate_action_types)
    except ValueError as exc:
        parser.error(str(exc))
    records = _load_records(args.input_records)
    if not 0 <= args.state_index < len(records):
        parser.error(f"--state-index outside {len(records)} records")
    snapshot = records[args.state_index].get("state_snapshot")
    if not isinstance(snapshot, dict):
        parser.error("selected record has no state_snapshot")
    state = GameState.from_mapping(snapshot)
    action = _select_action(
        state,
        action_key_value=args.action_key,
        action_index=args.action_index,
        max_actions=args.max_actions,
        action_types=action_types,
        priority=args.candidate_priority,
    )
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    bots = args.rollout_bot or ["solver_shop_basic_play_bot"]
    started = time.perf_counter()
    by_bot: dict[str, dict[str, Any]] = {}
    for bot_name in bots:
        samples = [
            profile_rollout_after_action(
                state,
                action,
                seed=seed,
                rollout_bot=bot_name,
                max_antes=args.max_antes,
                max_steps=args.max_steps,
            )
            for seed in seeds
        ]
        by_bot[bot_name] = {
            "summary": _summarize(samples),
            "samples": [asdict(sample) for sample in samples],
        }
    metrics = {
        "input_records": [str(path) for path in args.input_records],
        "state_index": args.state_index,
        "seed": records[args.state_index].get("seed"),
        "source_bot": records[args.state_index].get("source_bot"),
        "ante": int(state.ante),
        "money": int(state.money),
        "action_key": action_key(action),
        "action": action.to_json(),
        "rollout_bots": bots,
        "seeds": seeds,
        "max_antes": args.max_antes,
        "max_steps": args.max_steps,
        "rust_bestplay": bool(args.rust_bestplay),
        "wall_s": round(time.perf_counter() - started, 3),
        "by_bot": by_bot,
    }
    _atomic_write_json(args.metrics, metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
