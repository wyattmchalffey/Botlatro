"""Audit late losses for recoverable win-rate signals."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from balatro_ai.eval.compare import load_run_results
from balatro_ai.eval.decision_flip_audit import TraceRow, load_decision_traces
from balatro_ai.eval.metrics import RunResult


BOSS_REROLL_VOUCHERS = frozenset({"Director's Cut", "Retcon"})


@dataclass(frozen=True, slots=True)
class VoucherEvent:
    step: int
    ante: int
    blind: str
    name: str
    money: int


@dataclass(frozen=True, slots=True)
class LateShopChoice:
    step: int
    ante: int
    blind: str
    money: int
    action: str
    item: str
    value: float | None
    pressure_ratio: float | None
    build_capacity: float | None
    top_options: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LossTraceSummary:
    seed: int
    death_reason: str
    ante: int
    final_score: int
    final_required_score: int | None
    final_money: int
    final_blind: str
    boss_rerolls: int
    vouchers_seen: tuple[VoucherEvent, ...]
    vouchers_bought: tuple[VoucherEvent, ...]
    late_shop_choices: tuple[LateShopChoice, ...]

    @property
    def score_gap(self) -> int | None:
        if self.final_required_score is None:
            return None
        return max(0, self.final_required_score - self.final_score)

    @property
    def signals(self) -> tuple[str, ...]:
        signals: list[str] = []
        if self.final_required_score and self.final_score >= self.final_required_score * 0.75:
            signals.append("close_score_gap")
        if self.final_money >= 25:
            signals.append("late_bank_unspent")
        seen_reroll = {event.name for event in self.vouchers_seen} & BOSS_REROLL_VOUCHERS
        bought_reroll = {event.name for event in self.vouchers_bought} & BOSS_REROLL_VOUCHERS
        if seen_reroll and not bought_reroll and self.boss_rerolls == 0:
            signals.append("boss_reroll_voucher_seen_unbought")
        if bought_reroll and self.boss_rerolls == 0:
            signals.append("boss_reroll_voucher_bought_unused")
        if self.death_reason == "Violet Vessel":
            signals.append("violet_vessel_loss")
        return tuple(signals)


@dataclass(frozen=True, slots=True)
class LossAudit:
    bot: str
    losses: tuple[LossTraceSummary, ...]

    @property
    def signal_counts(self) -> Counter[str]:
        return Counter(signal for loss in self.losses for signal in loss.signals)

    @property
    def death_counts(self) -> Counter[str]:
        return Counter(loss.death_reason for loss in self.losses)

    def to_text(self, *, limit: int = 20, shop_limit: int = 4) -> str:
        lines = [
            "Late loss audit",
            f"Bot: {self.bot}",
            f"Losses audited: {len(self.losses)}",
            f"Death reasons: {_format_counter(self.death_counts)}",
            f"Signals: {_format_counter(self.signal_counts)}",
        ]
        for loss in self.losses[:limit]:
            lines.extend(("", _loss_text(loss, shop_limit=shop_limit)))
        return "\n".join(lines)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "bot": self.bot,
            "death_counts": dict(self.death_counts),
            "signal_counts": dict(self.signal_counts),
            "losses": [_loss_json(loss) for loss in self.losses],
        }


def audit_late_losses(
    results: Iterable[RunResult],
    traces: dict[int, tuple[TraceRow, ...]],
    *,
    min_ante: int = 7,
    death_reason: str | None = None,
) -> LossAudit:
    losses: list[LossTraceSummary] = []
    result_list = tuple(results)
    for result in result_list:
        if result.won or result.ante_reached < min_ante:
            continue
        if death_reason is not None and result.death_reason != death_reason:
            continue
        rows = traces.get(result.seed, ())
        if not rows:
            continue
        losses.append(_summarize_loss(result, rows))
    bot = result_list[0].bot_version if result_list else "bot"
    return LossAudit(bot=bot, losses=tuple(sorted(losses, key=_loss_sort_key)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit late local-sim losses for recoverable signals.")
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--traces", nargs="+", type=Path, required=True)
    parser.add_argument("--bot", default="basic_strategy_bot")
    parser.add_argument("--stake", default="white")
    parser.add_argument("--min-ante", type=int, default=7)
    parser.add_argument("--death-reason")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--shop-limit", type=int, default=4)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit = audit_late_losses(
        load_run_results(args.results, default_bot=args.bot, default_stake=args.stake),
        load_decision_traces(args.traces),
        min_ante=args.min_ante,
        death_reason=args.death_reason,
    )
    if args.json:
        print(json.dumps(audit.to_json_dict(), indent=2, sort_keys=True))
    else:
        print(audit.to_text(limit=args.limit, shop_limit=args.shop_limit))
    return 0


def _summarize_loss(result: RunResult, rows: tuple[TraceRow, ...]) -> LossTraceSummary:
    final_row = _final_decision_row(rows)
    final_required = _int_or_none(final_row.get("post_required_score") or final_row.get("required_score"))
    return LossTraceSummary(
        seed=result.seed,
        death_reason=result.death_reason or "",
        ante=result.ante_reached,
        final_score=result.final_score,
        final_required_score=final_required,
        final_money=result.final_money,
        final_blind=str(final_row.get("post_blind") or final_row.get("blind") or ""),
        boss_rerolls=sum(1 for row in rows if _is_boss_reroll(row)),
        vouchers_seen=_voucher_seen_events(rows),
        vouchers_bought=_voucher_bought_events(rows),
        late_shop_choices=_late_shop_choices(rows),
    )


def _final_decision_row(rows: tuple[TraceRow, ...]) -> TraceRow:
    decision_rows = [row for row in rows if row.get("record_type") == "local_decision_trace"]
    if not decision_rows:
        return {}
    return max(decision_rows, key=lambda row: int(row.get("step", 0)))


def _voucher_seen_events(rows: tuple[TraceRow, ...]) -> tuple[VoucherEvent, ...]:
    events: list[VoucherEvent] = []
    seen: set[str] = set()
    for row in rows:
        for voucher in _object_list(row.get("voucher_cards")):
            name = _item_name(voucher)
            if not name or name in seen:
                continue
            seen.add(name)
            events.append(_voucher_event(row, name))
    return tuple(events)


def _voucher_bought_events(rows: tuple[TraceRow, ...]) -> tuple[VoucherEvent, ...]:
    events: list[VoucherEvent] = []
    for row in rows:
        action = row.get("action")
        metadata = action.get("metadata", {}) if isinstance(action, dict) else {}
        if row.get("action_stable_key", "").startswith("buy||voucher") or metadata.get("kind") == "voucher":
            name = _item_name(row.get("chosen_item"))
            if name:
                events.append(_voucher_event(row, name))
    return tuple(events)


def _late_shop_choices(rows: tuple[TraceRow, ...]) -> tuple[LateShopChoice, ...]:
    choices: list[LateShopChoice] = []
    for row in rows:
        if row.get("phase") != "shop":
            continue
        if int(row.get("ante", 0)) < 7:
            continue
        audit = _shop_audit(row)
        if not audit:
            continue
        pressure = audit.get("pressure") if isinstance(audit.get("pressure"), dict) else {}
        choices.append(
            LateShopChoice(
                step=int(row.get("step", 0)),
                ante=int(row.get("ante", 0)),
                blind=str(row.get("blind", "")),
                money=int(row.get("money", 0)),
                action=str(row.get("action_stable_key", "")),
                item=_item_name(row.get("chosen_item")),
                value=_float_or_none(audit.get("chosen_value")),
                pressure_ratio=_float_or_none(pressure.get("ratio")),
                build_capacity=_float_or_none(pressure.get("build_capacity")),
                top_options=tuple(_option_summary(option) for option in _object_list(audit.get("options"))[:4]),
            )
        )
    return tuple(choices[-8:])


def _voucher_event(row: TraceRow, name: str) -> VoucherEvent:
    return VoucherEvent(
        step=int(row.get("step", 0)),
        ante=int(row.get("ante", 0)),
        blind=str(row.get("blind", "")),
        name=name,
        money=int(row.get("money", 0)),
    )


def _is_boss_reroll(row: TraceRow) -> bool:
    action = row.get("action")
    metadata = action.get("metadata", {}) if isinstance(action, dict) else {}
    return row.get("action_stable_key", "").startswith("reroll|||boss") or (
        action.get("type") == "reroll" if isinstance(action, dict) else False
    ) and metadata.get("kind") == "boss"


def _loss_sort_key(loss: LossTraceSummary) -> tuple[int, int, int]:
    gap = loss.score_gap if loss.score_gap is not None else 10**12
    return (-loss.ante, gap, -loss.final_score)


def _loss_text(loss: LossTraceSummary, *, shop_limit: int) -> str:
    required = loss.final_required_score if loss.final_required_score is not None else "?"
    gap = loss.score_gap if loss.score_gap is not None else "?"
    lines = [
        (
            f"seed={loss.seed} death={loss.death_reason or '-'} ante={loss.ante} "
            f"final={loss.final_score}/{required} gap={gap} money={loss.final_money} "
            f"boss_rerolls={loss.boss_rerolls} signals={','.join(loss.signals) or '-'}"
        )
    ]
    reroll_seen = [event for event in loss.vouchers_seen if event.name in BOSS_REROLL_VOUCHERS]
    if reroll_seen:
        lines.append("  boss-reroll vouchers seen: " + "; ".join(_voucher_event_text(event) for event in reroll_seen))
    if loss.vouchers_bought:
        lines.append("  vouchers bought: " + "; ".join(_voucher_event_text(event) for event in loss.vouchers_bought))
    for choice in loss.late_shop_choices[-shop_limit:]:
        lines.append("  shop " + _shop_choice_text(choice))
    return "\n".join(lines)


def _loss_json(loss: LossTraceSummary) -> dict[str, object]:
    return {
        "seed": loss.seed,
        "death_reason": loss.death_reason,
        "ante": loss.ante,
        "final_score": loss.final_score,
        "final_required_score": loss.final_required_score,
        "score_gap": loss.score_gap,
        "final_money": loss.final_money,
        "final_blind": loss.final_blind,
        "boss_rerolls": loss.boss_rerolls,
        "signals": list(loss.signals),
        "vouchers_seen": [_voucher_event_json(event) for event in loss.vouchers_seen],
        "vouchers_bought": [_voucher_event_json(event) for event in loss.vouchers_bought],
        "late_shop_choices": [_shop_choice_json(choice) for choice in loss.late_shop_choices],
    }


def _voucher_event_text(event: VoucherEvent) -> str:
    return f"{event.name}@step{event.step}/a{event.ante}/${event.money}"


def _voucher_event_json(event: VoucherEvent) -> dict[str, object]:
    return {
        "step": event.step,
        "ante": event.ante,
        "blind": event.blind,
        "name": event.name,
        "money": event.money,
    }


def _shop_choice_text(choice: LateShopChoice) -> str:
    ratio = "-" if choice.pressure_ratio is None else f"{choice.pressure_ratio:.2f}"
    capacity = "-" if choice.build_capacity is None else f"{choice.build_capacity:.0f}"
    options = " | ".join(choice.top_options[:3])
    return (
        f"step={choice.step} a{choice.ante} ${choice.money} action={choice.action} item={choice.item or '-'} "
        f"value={choice.value} ratio={ratio} capacity={capacity} options={options or '-'}"
    )


def _shop_choice_json(choice: LateShopChoice) -> dict[str, object]:
    return {
        "step": choice.step,
        "ante": choice.ante,
        "blind": choice.blind,
        "money": choice.money,
        "action": choice.action,
        "item": choice.item,
        "value": choice.value,
        "pressure_ratio": choice.pressure_ratio,
        "build_capacity": choice.build_capacity,
        "top_options": list(choice.top_options),
    }


def _shop_audit(row: TraceRow) -> dict[str, object]:
    audit = row.get("shop_audit")
    if isinstance(audit, dict):
        return audit
    action = row.get("action")
    metadata = action.get("metadata") if isinstance(action, dict) else None
    audit = metadata.get("shop_audit") if isinstance(metadata, dict) else None
    return audit if isinstance(audit, dict) else {}


def _option_summary(option: object) -> str:
    if not isinstance(option, dict):
        return str(option)
    item = _item_name(option.get("item"))
    return f"{option.get('stable_key')}:{item or '-'}:{option.get('value')}"


def _object_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list | tuple):
        return []
    return [item for item in value if isinstance(item, dict)]


def _item_name(item: object) -> str:
    if isinstance(item, dict):
        return str(item.get("name") or item.get("label") or "")
    if item is None:
        return ""
    return str(item)


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in counter.most_common())


if __name__ == "__main__":
    raise SystemExit(main())
