"""Audit first decision divergences between paired benchmark traces."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from balatro_ai.eval.compare import load_run_results
from balatro_ai.eval.metrics import RunResult


TraceRow = dict[str, object]


@dataclass(frozen=True, slots=True)
class DecisionDivergence:
    seed: int
    flip_type: str
    bot_a_won: bool
    bot_b_won: bool
    bot_a_ante: int
    bot_b_ante: int
    index: int | None
    bot_a_row: TraceRow | None
    bot_b_row: TraceRow | None

    @property
    def category(self) -> str:
        return f"{_action_type(self.bot_a_row)}->{_action_type(self.bot_b_row)}"


@dataclass(frozen=True, slots=True)
class DecisionFlipAudit:
    bot_a: str
    bot_b: str
    divergences: tuple[DecisionDivergence, ...]

    @property
    def category_counts(self) -> Counter[str]:
        return Counter(divergence.category for divergence in self.divergences)

    @property
    def flip_counts(self) -> Counter[str]:
        return Counter(divergence.flip_type for divergence in self.divergences)

    def to_text(self, *, limit: int = 20) -> str:
        lines = [
            "Decision flip audit",
            f"Bot A: {self.bot_a}",
            f"Bot B: {self.bot_b}",
            f"Seeds audited: {len(self.divergences)}",
            f"Flip counts: {_format_counter(self.flip_counts)}",
            f"First-divergence categories: {_format_counter(self.category_counts)}",
        ]
        for divergence in self.divergences[:limit]:
            lines.extend(("", _divergence_text(divergence)))
        return "\n".join(lines)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "bot_a": self.bot_a,
            "bot_b": self.bot_b,
            "flip_counts": dict(self.flip_counts),
            "category_counts": dict(self.category_counts),
            "divergences": [_divergence_json(divergence) for divergence in self.divergences],
        }


def load_decision_traces(paths: Iterable[Path]) -> dict[int, tuple[TraceRow, ...]]:
    grouped: dict[int, list[TraceRow]] = {}
    for path in _expand_paths(paths):
        with path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("record_type") != "local_decision_trace":
                    continue
                seed = int(row["seed"])
                grouped.setdefault(seed, []).append(row)
    return {seed: tuple(sorted(rows, key=lambda row: int(row.get("step", 0)))) for seed, rows in grouped.items()}


def audit_decision_flips(
    bot_a_results: Iterable[RunResult],
    bot_b_results: Iterable[RunResult],
    bot_a_traces: dict[int, tuple[TraceRow, ...]],
    bot_b_traces: dict[int, tuple[TraceRow, ...]],
    *,
    focus: str = "flips",
) -> DecisionFlipAudit:
    by_seed_a = {result.seed: result for result in bot_a_results}
    by_seed_b = {result.seed: result for result in bot_b_results}
    seeds = sorted(set(by_seed_a) & set(by_seed_b) & set(bot_a_traces) & set(bot_b_traces))
    divergences: list[DecisionDivergence] = []
    for seed in seeds:
        result_a = by_seed_a[seed]
        result_b = by_seed_b[seed]
        flip_type = _flip_type(result_a, result_b)
        if not _include_flip(flip_type, focus):
            continue
        index, row_a, row_b = first_action_divergence(bot_a_traces[seed], bot_b_traces[seed])
        divergences.append(
            DecisionDivergence(
                seed=seed,
                flip_type=flip_type,
                bot_a_won=result_a.won,
                bot_b_won=result_b.won,
                bot_a_ante=result_a.ante_reached,
                bot_b_ante=result_b.ante_reached,
                index=index,
                bot_a_row=row_a,
                bot_b_row=row_b,
            )
        )
    bot_a = next(iter(by_seed_a.values())).bot_version if by_seed_a else "bot_a"
    bot_b = next(iter(by_seed_b.values())).bot_version if by_seed_b else "bot_b"
    return DecisionFlipAudit(bot_a=bot_a, bot_b=bot_b, divergences=tuple(divergences))


def first_action_divergence(
    bot_a_rows: tuple[TraceRow, ...],
    bot_b_rows: tuple[TraceRow, ...],
) -> tuple[int | None, TraceRow | None, TraceRow | None]:
    for index, (row_a, row_b) in enumerate(zip(bot_a_rows, bot_b_rows, strict=False)):
        if row_a.get("action_stable_key") != row_b.get("action_stable_key"):
            return index, row_a, row_b
    if len(bot_a_rows) != len(bot_b_rows):
        index = min(len(bot_a_rows), len(bot_b_rows))
        row_a = bot_a_rows[index] if index < len(bot_a_rows) else None
        row_b = bot_b_rows[index] if index < len(bot_b_rows) else None
        return index, row_a, row_b
    return None, None, None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit first decision divergences in paired local-sim traces.")
    parser.add_argument("--bot-a", required=True, help="Baseline bot name.")
    parser.add_argument("--bot-b", required=True, help="Candidate bot name.")
    parser.add_argument("--bot-a-results", nargs="+", type=Path, required=True)
    parser.add_argument("--bot-b-results", nargs="+", type=Path, required=True)
    parser.add_argument("--bot-a-traces", nargs="+", type=Path, required=True)
    parser.add_argument("--bot-b-traces", nargs="+", type=Path, required=True)
    parser.add_argument("--stake", default="white")
    parser.add_argument(
        "--focus",
        choices=("flips", "candidate-wins", "candidate-losses", "all"),
        default="flips",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bot_a_results = load_run_results(args.bot_a_results, default_bot=args.bot_a, default_stake=args.stake)
    bot_b_results = load_run_results(args.bot_b_results, default_bot=args.bot_b, default_stake=args.stake)
    audit = audit_decision_flips(
        bot_a_results,
        bot_b_results,
        load_decision_traces(args.bot_a_traces),
        load_decision_traces(args.bot_b_traces),
        focus=args.focus,
    )
    if args.json:
        print(json.dumps(audit.to_json_dict(), indent=2, sort_keys=True))
    else:
        print(audit.to_text(limit=args.limit))
    return 0


def _expand_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*.jsonl")))
        else:
            expanded.append(path)
    return tuple(expanded)


def _flip_type(result_a: RunResult, result_b: RunResult) -> str:
    if not result_a.won and result_b.won:
        return "candidate_win"
    if result_a.won and not result_b.won:
        return "candidate_loss"
    return "same_win" if result_a.won else "same_loss"


def _include_flip(flip_type: str, focus: str) -> bool:
    if focus == "all":
        return True
    if focus == "flips":
        return flip_type in {"candidate_win", "candidate_loss"}
    if focus == "candidate-wins":
        return flip_type == "candidate_win"
    if focus == "candidate-losses":
        return flip_type == "candidate_loss"
    raise ValueError(f"Unknown focus: {focus}")


def _divergence_text(divergence: DecisionDivergence) -> str:
    lines = [
        (
            f"seed={divergence.seed} {divergence.flip_type} "
            f"ante={divergence.bot_a_ante}->{divergence.bot_b_ante} "
            f"first_diff={divergence.index if divergence.index is not None else '-'} "
            f"category={divergence.category}"
        )
    ]
    lines.append("  A " + _row_summary(divergence.bot_a_row))
    lines.append("  B " + _row_summary(divergence.bot_b_row))
    for label, row in (("A", divergence.bot_a_row), ("B", divergence.bot_b_row)):
        audit = _shop_audit(row)
        if not audit:
            continue
        lines.append(f"  {label} shop decision={audit.get('decision')} threshold={audit.get('threshold')}")
        for option in list(audit.get("options", ()))[:4]:
            lines.append(f"    {label} opt {_option_summary(option)}")
    return "\n".join(lines)


def _divergence_json(divergence: DecisionDivergence) -> dict[str, object]:
    return {
        "seed": divergence.seed,
        "flip_type": divergence.flip_type,
        "bot_a_won": divergence.bot_a_won,
        "bot_b_won": divergence.bot_b_won,
        "bot_a_ante": divergence.bot_a_ante,
        "bot_b_ante": divergence.bot_b_ante,
        "first_diff_index": divergence.index,
        "category": divergence.category,
        "bot_a": _row_json(divergence.bot_a_row),
        "bot_b": _row_json(divergence.bot_b_row),
    }


def _row_summary(row: TraceRow | None) -> str:
    if row is None:
        return "<no row>"
    item = _item_name(row.get("chosen_item"))
    return (
        f"step={row.get('step')} ante={row.get('ante')} blind={row.get('blind')} "
        f"phase={row.get('phase')} money={row.get('money')} "
        f"action={row.get('action_stable_key')} item={item or '-'} "
        f"reason={str(row.get('action_reason', ''))[:120]}"
    )


def _row_json(row: TraceRow | None) -> dict[str, object] | None:
    if row is None:
        return None
    audit = _shop_audit(row)
    return {
        "step": row.get("step"),
        "ante": row.get("ante"),
        "blind": row.get("blind"),
        "phase": row.get("phase"),
        "money": row.get("money"),
        "action_stable_key": row.get("action_stable_key"),
        "chosen_item": _item_name(row.get("chosen_item")),
        "action_reason": row.get("action_reason"),
        "shop_decision": audit.get("decision") if audit else None,
        "shop_options": [_option_json(option) for option in list(audit.get("options", ()))[:4]] if audit else [],
    }


def _shop_audit(row: TraceRow | None) -> dict[str, object]:
    if row is None:
        return {}
    action = row.get("action")
    if not isinstance(action, dict):
        return {}
    metadata = action.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    audit = metadata.get("shop_audit")
    return audit if isinstance(audit, dict) else {}


def _option_summary(option: object) -> str:
    if not isinstance(option, dict):
        return str(option)
    terms = option.get("planner_terms") if isinstance(option.get("planner_terms"), dict) else {}
    term_text = ""
    if terms:
        nonzero_terms = {
            key: value
            for key, value in terms.items()
            if key != "enabled" and isinstance(value, int | float) and abs(float(value)) > 0.001
        }
        term_text = f" terms={nonzero_terms}" if nonzero_terms else ""
    item = option.get("item")
    return f"{option.get('stable_key')} {_item_name(item) or '-'} value={option.get('value')}{term_text}"


def _option_json(option: object) -> dict[str, object]:
    if not isinstance(option, dict):
        return {"raw": str(option)}
    return {
        "stable_key": option.get("stable_key"),
        "item": _item_name(option.get("item")),
        "value": option.get("value"),
        "planner_terms": option.get("planner_terms", {}),
    }


def _action_type(row: TraceRow | None) -> str:
    if row is None:
        return "missing"
    action = row.get("action")
    if isinstance(action, dict) and action.get("type") is not None:
        return str(action["type"])
    stable_key = str(row.get("action_stable_key", ""))
    return stable_key.split("|", 1)[0] if stable_key else "unknown"


def _item_name(item: object) -> str:
    if isinstance(item, dict):
        return str(item.get("name") or item.get("label") or "")
    if item is None:
        return ""
    return str(item)


def _format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in counter.most_common())


if __name__ == "__main__":
    raise SystemExit(main())
