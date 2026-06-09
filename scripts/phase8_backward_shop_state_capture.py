"""Capture late shop snapshots from winning or near-winning trajectories.

This is the first step of a backward-reanalysis label lane:

1. Run a baseline/solver bot on fresh seeds.
2. Keep every real shop state in memory.
3. If the trajectory wins or reaches a requested late ante, write only the last
   N shop snapshots.
4. Feed the output to ``phase8_shop_candidate_dataset.py --input-records`` to
   branch each saved shop across legal options and roll forward from there.

Example:

    python scripts/phase8_backward_shop_state_capture.py \
        --bot solver_shop_basic_play_bot --seed-offset 600000 --seed-count 64 \
        --shops-per-run 2 --min-final-ante 8 --jobs 8 \
        --out .data/phase8_backward_shops_600000_64.jsonl \
        --metrics .data/phase8_backward_shops_600000_64.metrics.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import statistics
import time
from typing import Any


def _seed_strings(offset: int, count: int) -> list[str]:
    return [f"{offset + index:07d}" for index in range(1, count + 1)]


def _chunks(items: list[str], n_chunks: int) -> list[list[str]]:
    n_chunks = max(1, min(n_chunks, len(items) or 1))
    size = max(1, (len(items) + n_chunks - 1) // n_chunks)
    return [items[start:start + size] for start in range(0, len(items), size)]


def _parse_action_types_csv(value: str | None):
    if value is None or not value.strip():
        return None
    from balatro_ai.api.actions import ActionType

    parsed: list[ActionType] = []
    seen: set[ActionType] = set()
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        try:
            action_type = ActionType(token)
        except ValueError as exc:
            try:
                action_type = ActionType[token.upper()]
            except KeyError:
                allowed = ", ".join(action.value for action in ActionType)
                raise ValueError(f"unknown action type {token!r}; expected one of: {allowed}") from exc
        if action_type not in seen:
            parsed.append(action_type)
            seen.add(action_type)
    return tuple(parsed) if parsed else None


def _run_seed(args: tuple[Any, ...]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    (
        seed,
        bot_name,
        stake,
        max_steps,
        shops_per_run,
        min_capture_ante,
        max_capture_ante,
        min_final_ante,
        require_win,
        exclude_wins,
        max_actions,
        candidate_action_types,
        candidate_priority,
    ) = args

    from dataclasses import replace

    from balatro_ai.api.actions import ActionType
    from balatro_ai.api.state import GamePhase
    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.ml.shop_candidate_dataset import candidate_shop_actions, state_snapshot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake=stake)
    sim.state = SeedGame(seed, stake=stake).initial_state()
    bot = create_bot(bot_name, seed=0)
    shops: list[dict[str, Any]] = []
    steps = 0
    error: str | None = None
    with bot_config_scope(replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
        while steps < max_steps:
            state = sim.state
            if state is None or state.run_over or state.phase == GamePhase.RUN_OVER:
                break
            if state.phase == GamePhase.SHOP:
                ante = int(state.ante)
                in_range = ante >= min_capture_ante and (max_capture_ante is None or ante <= max_capture_ante)
                candidates = candidate_shop_actions(
                    state,
                    max_actions=max_actions,
                    action_types=candidate_action_types,
                    priority=candidate_priority,
                )
                has_build_action = any(
                    action.action_type in {ActionType.BUY, ActionType.OPEN_PACK}
                    for action in candidates
                )
                if in_range and len(candidates) >= 2 and has_build_action:
                    shops.append(
                        {
                            "record_type": "phase8_backward_shop_state",
                            "seed": seed,
                            "state_index": steps,
                            "source_bot": bot_name,
                            "phase": state.phase.value,
                            "ante": ante,
                            "money": int(state.money),
                            "candidate_action_count": len(candidates),
                            "state_snapshot": state_snapshot(state),
                        }
                    )
            try:
                action = bot.choose_action(state)
                if action is None or action.action_type == ActionType.NO_OP:
                    error = "no_action"
                    break
                sim.step(action)
            except Exception as exc:  # noqa: BLE001 - capture should record bad local transitions.
                error = f"{type(exc).__name__}:{exc}"
                break
            steps += 1

    final = sim.state
    final_ante = int(getattr(final, "ante", 0) or 0)
    won = bool(getattr(final, "won", False)) if final is not None else False
    qualifies = _trajectory_qualifies(
        has_shops=bool(shops),
        final_ante=final_ante,
        won=won,
        min_final_ante=min_final_ante,
        require_win=require_win,
        exclude_wins=exclude_wins,
    )
    selected: list[dict[str, Any]] = []
    if qualifies:
        tail = shops[-max(0, shops_per_run):] if shops_per_run > 0 else []
        total_shops = len(shops)
        for reverse_index, record in enumerate(reversed(tail)):
            enriched = dict(record)
            enriched["trajectory_shop_count"] = total_shops
            enriched["shops_from_terminal"] = reverse_index
            enriched["terminal_won"] = won
            enriched["terminal_ante"] = final_ante
            enriched["terminal_money"] = int(getattr(final, "money", 0) or 0) if final is not None else 0
            enriched["terminal_score"] = int(getattr(final, "current_score", 0) or 0) if final is not None else 0
            enriched["terminal_required_score"] = (
                int(getattr(final, "required_score", 0) or 0) if final is not None else 0
            )
            enriched["selection_reason"] = "win" if won else f"reached_ante_{final_ante}"
            selected.append(enriched)
    summary = {
        "seed": seed,
        "steps": steps,
        "shops_seen": len(shops),
        "records": len(selected),
        "qualifies": qualifies,
        "won": won,
        "terminal_ante": final_ante,
        "terminal_money": int(getattr(final, "money", 0) or 0) if final is not None else 0,
        "error": error,
    }
    return selected, summary


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    tmp.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _summarize(
    records: list[dict[str, Any]],
    seed_summaries: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    wall_s: float,
) -> dict[str, Any]:
    terminal_antes = [int(item["terminal_ante"]) for item in seed_summaries if item.get("terminal_ante") is not None]
    qualifying = [item for item in seed_summaries if item["qualifies"]]
    by_ante = Counter(str(record["ante"]) for record in records)
    by_distance = Counter(str(record["shops_from_terminal"]) for record in records)
    return {
        "records": len(records),
        "wall_s": round(wall_s, 3),
        "jobs": args.jobs,
        "bot": args.bot,
        "stake": args.stake,
        "seed_offset": args.seed_offset,
        "seed_count": args.seed_count,
        "shops_per_run": args.shops_per_run,
        "min_capture_ante": args.min_capture_ante,
        "max_capture_ante": args.max_capture_ante,
        "min_final_ante": args.min_final_ante,
        "require_win": bool(args.require_win),
        "exclude_wins": bool(args.exclude_wins),
        "max_actions": args.max_actions,
        "candidate_action_types": args.candidate_action_types,
        "candidate_priority": args.candidate_priority,
        "qualifying_runs": len(qualifying),
        "winning_runs": sum(1 for item in seed_summaries if item["won"]),
        "mean_terminal_ante": round(statistics.mean(terminal_antes), 4) if terminal_antes else None,
        "records_by_ante": dict(sorted(by_ante.items())),
        "records_by_shops_from_terminal": dict(sorted(by_distance.items())),
        "seed_summaries": seed_summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture backward late-shop snapshots from strong trajectories.")
    parser.add_argument("--bot", default="solver_shop_basic_play_bot")
    parser.add_argument("--stake", default="white")
    parser.add_argument("--seed-offset", type=int, default=600000)
    parser.add_argument("--seed-count", type=int, default=16)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--shops-per-run", type=int, default=2)
    parser.add_argument("--min-capture-ante", type=int, default=5)
    parser.add_argument("--max-capture-ante", type=int, default=None)
    parser.add_argument("--min-final-ante", type=int, default=8)
    parser.add_argument("--require-win", action="store_true")
    parser.add_argument(
        "--exclude-wins",
        action="store_true",
        help="Keep only non-winning trajectories that still reach --min-final-ante.",
    )
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap after all workers finish; 0 keeps all.")
    parser.add_argument("--max-actions", type=int, default=12)
    parser.add_argument(
        "--candidate-action-types",
        default="buy,open_pack,end_shop",
        help="Comma-separated shop action types required for captured-state candidate counting.",
    )
    parser.add_argument("--candidate-priority", default="legal", choices=("legal", "deep_advantage"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()
    if args.require_win and args.exclude_wins:
        parser.error("--require-win and --exclude-wins are mutually exclusive")

    try:
        action_types = _parse_action_types_csv(args.candidate_action_types)
    except ValueError as exc:
        parser.error(str(exc))

    seeds = _seed_strings(args.seed_offset, args.seed_count)
    worker_args = [
        (
            seed,
            args.bot,
            args.stake,
            args.max_steps,
            args.shops_per_run,
            args.min_capture_ante,
            args.max_capture_ante,
            args.min_final_ante,
            bool(args.require_win),
            bool(args.exclude_wins),
            args.max_actions,
            action_types,
            args.candidate_priority,
        )
        for seed in seeds
    ]
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []
    if args.jobs <= 1:
        for item in worker_args:
            selected, summary = _run_seed(item)
            records.extend(selected)
            seed_summaries.append(summary)
    else:
        chunks = _chunks(worker_args, args.jobs)
        with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = [pool.submit(_run_seed_chunk, chunk) for chunk in chunks]
            for future in as_completed(futures):
                chunk_records, chunk_summaries = future.result()
                records.extend(chunk_records)
                seed_summaries.extend(chunk_summaries)
    records.sort(key=_record_priority_key)
    if args.max_records > 0:
        records = records[:args.max_records]
    seed_summaries.sort(key=lambda item: str(item["seed"]))
    wall_s = time.perf_counter() - started
    metrics = _summarize(records, seed_summaries, args=args, wall_s=wall_s)
    _write_jsonl(args.out, records)
    _write_json(args.metrics, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return 0


def _run_seed_chunk(items: list[tuple[Any, ...]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for item in items:
        selected, summary = _run_seed(item)
        records.extend(selected)
        summaries.append(summary)
    return records, summaries


def _trajectory_qualifies(
    *,
    has_shops: bool,
    final_ante: int,
    won: bool,
    min_final_ante: int,
    require_win: bool,
    exclude_wins: bool,
) -> bool:
    if not has_shops or final_ante < min_final_ante:
        return False
    if require_win and not won:
        return False
    if exclude_wins and won:
        return False
    return True


def _record_priority_key(record: dict[str, Any]) -> tuple[int, int, int, str]:
    """Prefer capped records from wins, then later terminal antes, then last shops."""

    terminal_won = 1 if record.get("terminal_won") else 0
    try:
        terminal_ante = int(record.get("terminal_ante", 0))
    except (TypeError, ValueError):
        terminal_ante = 0
    try:
        shops_from_terminal = int(record.get("shops_from_terminal", 9999))
    except (TypeError, ValueError):
        shops_from_terminal = 9999
    return (-terminal_won, -terminal_ante, shops_from_terminal, str(record.get("seed", "")))


if __name__ == "__main__":
    raise SystemExit(main())
