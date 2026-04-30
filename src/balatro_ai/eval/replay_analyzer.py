"""Summarize replay logs for bot behavior and decision quality."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from statistics import mean
from typing import Iterable

from balatro_ai.api.state import Card, Joker
from balatro_ai.bots.basic_strategy_bot import _joker_roles
from balatro_ai.rules.hand_evaluator import evaluate_played_cards


ANTE_PATTERN = re.compile(r"ante=(\d+)")
BLIND_PATTERN = re.compile(r"ante=(\d+) blind=(.*?) score=")
HAND_PATTERN = re.compile(r"hand=\[(?P<hand>.*?)\]")
PRESSURE_PATTERN = re.compile(r"pressure=([0-9]+(?:\.[0-9]+)?)")
STATE_PATTERN = re.compile(
    r"phase=(?P<phase>\S+) ante=(?P<ante>\d+) blind=(?P<blind>.*?) "
    r"score=(?P<score>\d+)/(?P<required>\d+) money=(?P<money>-?\d+) "
    r"hands=(?P<hands>\d+) discards=(?P<discards>\d+) .* jokers=\[(?P<jokers>.*?)\]"
)
BOSS_BLINDS = frozenset(
    {
        "The Hook",
        "The Ox",
        "The House",
        "The Wall",
        "The Wheel",
        "The Arm",
        "The Club",
        "The Fish",
        "The Psychic",
        "The Goad",
        "The Water",
        "The Window",
        "The Manacle",
        "The Eye",
        "The Mouth",
        "The Plant",
        "The Serpent",
        "The Pillar",
        "The Needle",
        "The Head",
        "The Tooth",
        "The Flint",
        "The Mark",
        "Amber Acorn",
        "Verdant Leaf",
        "Violet Vessel",
        "Crimson Heart",
        "Cerulean Bell",
    }
)
WEAK_HAND_TYPES = frozenset({"High Card", "Pair", "Two Pair"})
CRITICAL_ROLES = frozenset({"chips", "mult", "xmult", "scaling"})


@dataclass(frozen=True, slots=True)
class RunReplaySummary:
    path: Path
    seed: str
    max_ante: int
    row_count: int
    action_counts: Counter[str]
    shop_reason_counts: Counter[str]
    pressure_values: tuple[float, ...]
    hands_per_blind: tuple[int, ...]
    sell_actions: int
    observed_win: bool = False
    outcome: str | None = None
    final_phase: str = ""
    final_blind: str = ""
    final_score: int = 0
    final_required_score: int = 0
    final_money: int = 0
    final_hands: int = 0
    final_jokers: tuple[str, ...] = ()
    shop_action_count: int = 0
    shop_audit_count: int = 0
    shop_skip_count: int = 0
    played_hand_types: Counter[str] = field(default_factory=Counter)
    preferred_hand_counts: Counter[str] = field(default_factory=Counter)
    missing_role_counts: Counter[str] = field(default_factory=Counter)
    final_preferred_hand: str | None = None
    final_missing_roles: tuple[str, ...] = ()
    final_blind_played_hand_types: Counter[str] = field(default_factory=Counter)
    final_blind_predicted_score: int = 0
    final_blind_actual_score: int = 0
    final_blind_zero_score_hands: int = 0
    postmortem_labels: tuple[str, ...] = ()
    chosen_shop_items: Counter[str] = field(default_factory=Counter)
    shop_choices: tuple[str, ...] = ()

    @property
    def dominant_played_hand(self) -> str | None:
        if not self.played_hand_types:
            return None
        return self.played_hand_types.most_common(1)[0][0]


@dataclass(frozen=True, slots=True)
class ReplayAnalysis:
    runs: tuple[RunReplaySummary, ...]
    malformed_rows: int = 0
    action_counts: Counter[str] = field(default_factory=Counter)
    shop_reason_counts: Counter[str] = field(default_factory=Counter)
    pressure_values: tuple[float, ...] = ()
    hands_per_blind: tuple[int, ...] = ()
    played_hand_types: Counter[str] = field(default_factory=Counter)
    dominant_played_hands: Counter[str] = field(default_factory=Counter)
    preferred_hand_counts: Counter[str] = field(default_factory=Counter)
    final_preferred_hands: Counter[str] = field(default_factory=Counter)
    missing_role_counts: Counter[str] = field(default_factory=Counter)
    chosen_shop_items: Counter[str] = field(default_factory=Counter)
    postmortem_label_counts: Counter[str] = field(default_factory=Counter)
    shop_audit_count: int = 0
    shop_skip_count: int = 0

    @property
    def files_scanned(self) -> int:
        return len(self.runs)

    @property
    def average_ante(self) -> float:
        return mean(run.max_ante for run in self.runs) if self.runs else 0.0

    @property
    def ante_distribution(self) -> Counter[int]:
        return Counter(run.max_ante for run in self.runs)

    @property
    def observed_wins(self) -> int:
        return sum(1 for run in self.runs if run.observed_win)

    @property
    def outcome_distribution(self) -> Counter[str]:
        return Counter(run.outcome for run in self.runs if run.outcome)

    @property
    def average_hands_per_blind(self) -> float:
        return mean(self.hands_per_blind) if self.hands_per_blind else 0.0

    @property
    def hands_per_blind_distribution(self) -> Counter[int]:
        return Counter(self.hands_per_blind)

    @property
    def average_pressure(self) -> float:
        return mean(self.pressure_values) if self.pressure_values else 0.0

    @property
    def early_failures(self) -> tuple[RunReplaySummary, ...]:
        return tuple(run for run in self.runs if not run.observed_win and run.max_ante <= 2)

    @property
    def deep_losses(self) -> tuple[RunReplaySummary, ...]:
        return tuple(run for run in self.runs if not run.observed_win and 6 <= run.max_ante < 8)

    def to_text(self) -> str:
        lines = [
            "Replay analysis",
            f"Files scanned: {self.files_scanned}",
            f"Malformed rows: {self.malformed_rows}",
            f"Observed wins in replay rows: {self.observed_wins}",
            f"Outcome distribution: {_format_counter(self.outcome_distribution)}",
            f"Average max ante: {self.average_ante:.2f}",
            f"Ante distribution: {_format_counter(self.ante_distribution)}",
        ]
        if self.runs:
            for threshold in (2, 3, 4, 5, 6, 8):
                reached = sum(1 for run in self.runs if run.max_ante >= threshold)
                lines.append(f"Reached ante >= {threshold}: {reached}/{self.files_scanned}")

        lines.extend(
            (
                "",
                f"Action counts: {_format_counter(self.action_counts)}",
                f"Shop reason counts: {_format_counter(self.shop_reason_counts)}",
                f"Sell actions: {self.action_counts.get('sell', 0)}",
                f"Shop audit rows: {self.shop_audit_count}",
                f"Shop skips: {self.shop_skip_count}",
                f"Chosen shop items: {_format_counter(_most_common_counter(self.chosen_shop_items, 15))}",
            )
        )
        if self.pressure_values:
            lines.append(
                "Pressure: "
                f"avg={self.average_pressure:.2f} "
                f"min={min(self.pressure_values):.2f} "
                f"max={max(self.pressure_values):.2f} "
                f">=1.0={sum(1 for value in self.pressure_values if value >= 1.0)}"
            )
        if self.hands_per_blind:
            lines.extend(
                (
                    "",
                    f"Average played hands per blind: {self.average_hands_per_blind:.2f}",
                    f"Played hands per blind: {_format_counter(self.hands_per_blind_distribution)}",
                    f"One-hand clears: {self.hands_per_blind_distribution.get(1, 0)}/{len(self.hands_per_blind)}",
                f"Four-hand clears/deaths: {self.hands_per_blind_distribution.get(4, 0)}/{len(self.hands_per_blind)}",
                )
            )

        if self.played_hand_types or self.preferred_hand_counts:
            lines.extend(
                (
                    "",
                    "Archetypes:",
                    f"Played hand types: {_format_counter(_most_common_counter(self.played_hand_types, 12))}",
                    f"Dominant played hand by run: {_format_counter(self.dominant_played_hands)}",
                    f"Shop preferred hand signals: {_format_counter(_most_common_counter(self.preferred_hand_counts, 12))}",
                    f"Final preferred hand signals: {_format_counter(self.final_preferred_hands)}",
                    f"Missing role signals: {_format_counter(self.missing_role_counts)}",
                )
            )

        if self.postmortem_label_counts:
            lines.extend(
                (
                    "",
                    "Postmortem labels:",
                    f"Cause labels: {_format_counter(_most_common_counter(self.postmortem_label_counts, 14))}",
                )
            )

        if self.early_failures:
            early_jokers = Counter(joker for run in self.early_failures for joker in run.final_jokers)
            early_items = Counter()
            early_play_counts = Counter(
                hand_count
                for run in self.early_failures
                for hand_count in run.hands_per_blind
            )
            early_played_hands = Counter(
                hand_type
                for run in self.early_failures
                for hand_type, count in run.played_hand_types.items()
                for _ in range(count)
            )
            for run in self.early_failures:
                early_items.update(run.chosen_shop_items)
            lines.extend(
                (
                    "",
                    "Early failures:",
                    f"Ante <= 2 losses: {len(self.early_failures)}/{self.files_scanned}",
                    f"Average shop actions: {mean(run.shop_action_count for run in self.early_failures):.2f}",
                    f"Final joker counts: {_format_counter(early_jokers)}",
                    f"Chosen shop items: {_format_counter(_most_common_counter(early_items, 12))}",
                    f"Played hands per blind: {_format_counter(early_play_counts)}",
                    f"Played hand types: {_format_counter(_most_common_counter(early_played_hands, 8))}",
                )
            )
            for run in sorted(self.early_failures, key=lambda item: (item.max_ante, item.seed))[:5]:
                lines.append(f"- {_failure_line(run)}")
                for choice in run.shop_choices[:4]:
                    lines.append(f"  shop: {choice}")

        if self.deep_losses:
            deep_jokers = Counter(joker for run in self.deep_losses for joker in run.final_jokers)
            deep_items = Counter()
            deep_play_counts = Counter(
                hand_count
                for run in self.deep_losses
                for hand_count in run.hands_per_blind
            )
            deep_played_hands = Counter(
                hand_type
                for run in self.deep_losses
                for hand_type, count in run.played_hand_types.items()
                for _ in range(count)
            )
            for run in self.deep_losses:
                deep_items.update(run.chosen_shop_items)
            lines.extend(
                (
                    "",
                    "Deep losses:",
                    f"Ante 6-7 losses: {len(self.deep_losses)}/{self.files_scanned}",
                    f"Final joker counts: {_format_counter(deep_jokers)}",
                    f"Chosen shop items: {_format_counter(_most_common_counter(deep_items, 12))}",
                    f"Played hands per blind: {_format_counter(deep_play_counts)}",
                    f"Played hand types: {_format_counter(_most_common_counter(deep_played_hands, 8))}",
                )
            )
            for run in sorted(self.deep_losses, key=lambda item: (-item.max_ante, item.seed))[:5]:
                lines.append(f"- {_failure_line(run)}")
                for choice in run.shop_choices[-4:]:
                    lines.append(f"  shop: {choice}")

        if self.runs:
            lines.extend(("", "Deepest runs:"))
            for run in sorted(self.runs, key=lambda item: item.max_ante, reverse=True)[:5]:
                lines.append(f"- seed={run.seed} ante={run.max_ante} file={run.path.name}")
        return "\n".join(lines)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "files_scanned": self.files_scanned,
            "malformed_rows": self.malformed_rows,
            "observed_wins": self.observed_wins,
            "outcome_distribution": _counter_to_json_dict(self.outcome_distribution),
            "average_max_ante": self.average_ante,
            "ante_distribution": _counter_to_json_dict(self.ante_distribution),
            "reach_counts": {
                str(threshold): sum(1 for run in self.runs if run.max_ante >= threshold)
                for threshold in (2, 3, 4, 5, 6, 8)
            },
            "action_counts": _counter_to_json_dict(self.action_counts),
            "shop_reason_counts": _counter_to_json_dict(self.shop_reason_counts),
            "shop_audit_count": self.shop_audit_count,
            "shop_skip_count": self.shop_skip_count,
            "chosen_shop_items": _counter_to_json_dict(self.chosen_shop_items),
            "pressure": {
                "count": len(self.pressure_values),
                "average": self.average_pressure,
                "minimum": min(self.pressure_values) if self.pressure_values else 0.0,
                "maximum": max(self.pressure_values) if self.pressure_values else 0.0,
                "at_least_1": sum(1 for value in self.pressure_values if value >= 1.0),
            },
            "hands_per_blind": {
                "average": self.average_hands_per_blind,
                "distribution": _counter_to_json_dict(self.hands_per_blind_distribution),
                "one_hand": self.hands_per_blind_distribution.get(1, 0),
                "four_hand": self.hands_per_blind_distribution.get(4, 0),
                "total": len(self.hands_per_blind),
            },
            "archetypes": {
                "played_hand_types": _counter_to_json_dict(self.played_hand_types),
                "dominant_played_hands": _counter_to_json_dict(self.dominant_played_hands),
                "preferred_hand_counts": _counter_to_json_dict(self.preferred_hand_counts),
                "final_preferred_hands": _counter_to_json_dict(self.final_preferred_hands),
                "missing_role_counts": _counter_to_json_dict(self.missing_role_counts),
            },
            "postmortem_labels": _counter_to_json_dict(self.postmortem_label_counts),
            "early_failures": {
                "count": len(self.early_failures),
                "sample": [_run_json_summary(run) for run in sorted(self.early_failures, key=lambda item: (item.max_ante, item.seed))[:5]],
            },
            "deep_losses": {
                "count": len(self.deep_losses),
                "sample": [_run_json_summary(run) for run in sorted(self.deep_losses, key=lambda item: (-item.max_ante, item.seed))[:5]],
            },
            "deepest_runs": [
                {"seed": run.seed, "max_ante": run.max_ante, "file": run.path.name}
                for run in sorted(self.runs, key=lambda item: item.max_ante, reverse=True)[:5]
            ],
        }


def analyze_replays(paths: Iterable[Path]) -> ReplayAnalysis:
    run_summaries: list[RunReplaySummary] = []
    malformed_rows = 0
    for replay_path in _expand_paths(paths):
        summary, bad_rows = _analyze_file(replay_path)
        malformed_rows += bad_rows
        if summary is not None:
            run_summaries.append(summary)

    action_counts: Counter[str] = Counter()
    shop_reason_counts: Counter[str] = Counter()
    chosen_shop_items: Counter[str] = Counter()
    pressure_values: list[float] = []
    hands_per_blind: list[int] = []
    played_hand_types: Counter[str] = Counter()
    dominant_played_hands: Counter[str] = Counter()
    preferred_hand_counts: Counter[str] = Counter()
    final_preferred_hands: Counter[str] = Counter()
    missing_role_counts: Counter[str] = Counter()
    shop_audit_count = 0
    shop_skip_count = 0
    postmortem_label_counts: Counter[str] = Counter()
    for run in run_summaries:
        action_counts.update(run.action_counts)
        shop_reason_counts.update(run.shop_reason_counts)
        chosen_shop_items.update(run.chosen_shop_items)
        pressure_values.extend(run.pressure_values)
        hands_per_blind.extend(run.hands_per_blind)
        played_hand_types.update(run.played_hand_types)
        if run.dominant_played_hand:
            dominant_played_hands[run.dominant_played_hand] += 1
        preferred_hand_counts.update(run.preferred_hand_counts)
        if run.final_preferred_hand:
            final_preferred_hands[run.final_preferred_hand] += 1
        missing_role_counts.update(run.missing_role_counts)
        postmortem_label_counts.update(run.postmortem_labels)
        shop_audit_count += run.shop_audit_count
        shop_skip_count += run.shop_skip_count

    return ReplayAnalysis(
        runs=tuple(run_summaries),
        malformed_rows=malformed_rows,
        action_counts=action_counts,
        shop_reason_counts=shop_reason_counts,
        pressure_values=tuple(pressure_values),
        hands_per_blind=tuple(hands_per_blind),
        played_hand_types=played_hand_types,
        dominant_played_hands=dominant_played_hands,
        preferred_hand_counts=preferred_hand_counts,
        final_preferred_hands=final_preferred_hands,
        missing_role_counts=missing_role_counts,
        chosen_shop_items=chosen_shop_items,
        postmortem_label_counts=postmortem_label_counts,
        shop_audit_count=shop_audit_count,
        shop_skip_count=shop_skip_count,
    )


def _analyze_file(path: Path) -> tuple[RunReplaySummary | None, int]:
    row_count = 0
    malformed_rows = 0
    max_ante = 0
    seed = path.stem
    action_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    chosen_shop_items: Counter[str] = Counter()
    pressure_values: list[float] = []
    hands_per_blind: list[int] = []
    played_hand_types: Counter[str] = Counter()
    preferred_hand_counts: Counter[str] = Counter()
    missing_role_counts: Counter[str] = Counter()
    final_preferred_hand: str | None = None
    final_missing_roles: tuple[str, ...] = ()
    shop_choices: list[str] = []
    current_blind: str | None = None
    current_play_count = 0
    current_blind_played_hand_types: Counter[str] = Counter()
    current_blind_predicted_score = 0
    current_blind_actual_score = 0
    current_blind_zero_score_hands = 0
    final_blind_played_hand_types: Counter[str] = Counter()
    final_blind_predicted_score = 0
    final_blind_actual_score = 0
    final_blind_zero_score_hands = 0
    observed_win = False
    outcome: str | None = None
    last_state: dict[str, object] = {}
    shop_action_count = 0
    shop_audit_count = 0
    shop_skip_count = 0

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed_rows += 1
                continue

            row_count += 1
            seed = str(row.get("seed", seed))
            if row.get("record_type") == "run_summary":
                max_ante = max(max_ante, _int_value(row.get("ante")))
                summary_state = _state_from_summary_row(row)
                if summary_state:
                    last_state = summary_state
                row_outcome = row.get("outcome")
                if row_outcome is not None:
                    outcome = str(row_outcome)
                if row.get("outcome") == "win" or row.get("won") is True:
                    observed_win = True
                continue

            state_text = str(row.get("state", ""))
            max_ante = max(max_ante, _ante_from_state(state_text))
            parsed_state = _parse_state_text(state_text)
            if parsed_state:
                last_state = parsed_state
            if row.get("outcome") == "win" or row.get("won") is True:
                observed_win = True

            action = row.get("chosen_action", {})
            action_type = str(action.get("type", "unknown")) if isinstance(action, dict) else "unknown"
            action_counts[action_type] += 1
            hand_type = None
            if action_type == "play_hand" and isinstance(action, dict):
                hand_type = _played_hand_type(row, action)
                if hand_type:
                    played_hand_types[hand_type] += 1
            if action_type in {"buy", "sell", "reroll", "open_pack", "choose_pack_card"}:
                shop_action_count += 1

            metadata = action.get("metadata", {}) if isinstance(action, dict) else {}
            reason = metadata.get("reason") if isinstance(metadata, dict) else None
            if reason:
                reason_text = str(reason)
                reason_counts[reason_text.split()[0]] += 1
                pressure_values.extend(_pressure_values(reason_text))
            audit = metadata.get("shop_audit") if isinstance(metadata, dict) else None
            if isinstance(audit, dict):
                shop_audit_count += 1
                profile = audit.get("build_profile")
                if isinstance(profile, dict):
                    preferred_hand = profile.get("preferred_hand")
                    if preferred_hand:
                        final_preferred_hand = str(preferred_hand)
                        preferred_hand_counts[final_preferred_hand] += 1
                    missing_roles = profile.get("missing_roles")
                    if isinstance(missing_roles, list):
                        final_missing_roles = tuple(str(role) for role in missing_roles)
                        missing_role_counts.update(final_missing_roles)
                if action_type == "end_shop":
                    shop_skip_count += 1
                item_label = _chosen_shop_item_label(row, action_type, audit)
                if item_label:
                    chosen_shop_items[item_label] += 1
                shop_choices.append(_shop_choice_line(row, action_type, audit))

            blind_key = _blind_key_from_state(state_text)
            if action_type == "play_hand":
                if current_blind is None:
                    current_blind = blind_key
                    current_play_count = 0
                    current_blind_played_hand_types = Counter()
                    current_blind_predicted_score = 0
                    current_blind_actual_score = 0
                    current_blind_zero_score_hands = 0
                if blind_key != current_blind:
                    if current_play_count:
                        hands_per_blind.append(current_play_count)
                        final_blind_played_hand_types = current_blind_played_hand_types
                        final_blind_predicted_score = current_blind_predicted_score
                        final_blind_actual_score = current_blind_actual_score
                        final_blind_zero_score_hands = current_blind_zero_score_hands
                    current_blind = blind_key
                    current_play_count = 0
                    current_blind_played_hand_types = Counter()
                    current_blind_predicted_score = 0
                    current_blind_actual_score = 0
                    current_blind_zero_score_hands = 0
                current_play_count += 1
                if hand_type:
                    current_blind_played_hand_types[hand_type] += 1
                predicted, actual = _score_audit_totals(row)
                current_blind_predicted_score += predicted
                current_blind_actual_score += actual
                if predicted > 0 and actual == 0:
                    current_blind_zero_score_hands += 1
            elif action_type in {"cash_out", "select_blind", "skip_blind"}:
                if current_play_count:
                    hands_per_blind.append(current_play_count)
                    final_blind_played_hand_types = current_blind_played_hand_types
                    final_blind_predicted_score = current_blind_predicted_score
                    final_blind_actual_score = current_blind_actual_score
                    final_blind_zero_score_hands = current_blind_zero_score_hands
                    current_play_count = 0
                    current_blind = None
                    current_blind_played_hand_types = Counter()
                    current_blind_predicted_score = 0
                    current_blind_actual_score = 0
                    current_blind_zero_score_hands = 0

    if current_play_count:
        hands_per_blind.append(current_play_count)
        final_blind_played_hand_types = current_blind_played_hand_types
        final_blind_predicted_score = current_blind_predicted_score
        final_blind_actual_score = current_blind_actual_score
        final_blind_zero_score_hands = current_blind_zero_score_hands
    if row_count == 0:
        return None, malformed_rows

    postmortem_labels = _postmortem_labels(
        observed_win=observed_win,
        max_ante=max_ante,
        final_blind=str(last_state.get("blind", "")),
        final_score=_int_value(last_state.get("score")),
        final_required_score=_int_value(last_state.get("required")),
        final_money=_int_value(last_state.get("money")),
        final_jokers=tuple(last_state.get("jokers", ())),
        final_missing_roles=final_missing_roles,
        played_hand_types=played_hand_types,
        final_blind_played_hand_types=final_blind_played_hand_types,
        final_blind_predicted_score=final_blind_predicted_score,
        final_blind_actual_score=final_blind_actual_score,
        final_blind_zero_score_hands=final_blind_zero_score_hands,
    )

    return (
        RunReplaySummary(
            path=path,
            seed=seed,
            max_ante=max_ante,
            row_count=row_count,
            action_counts=action_counts,
            shop_reason_counts=reason_counts,
            pressure_values=tuple(pressure_values),
            hands_per_blind=tuple(hands_per_blind),
            sell_actions=action_counts.get("sell", 0),
            observed_win=observed_win,
            outcome=outcome,
            final_phase=str(last_state.get("phase", "")),
            final_blind=str(last_state.get("blind", "")),
            final_score=_int_value(last_state.get("score")),
            final_required_score=_int_value(last_state.get("required")),
            final_money=_int_value(last_state.get("money")),
            final_hands=_int_value(last_state.get("hands")),
            final_jokers=tuple(last_state.get("jokers", ())),
            shop_action_count=shop_action_count,
            shop_audit_count=shop_audit_count,
            shop_skip_count=shop_skip_count,
            played_hand_types=played_hand_types,
            preferred_hand_counts=preferred_hand_counts,
            missing_role_counts=missing_role_counts,
            final_preferred_hand=final_preferred_hand,
            final_missing_roles=final_missing_roles,
            final_blind_played_hand_types=final_blind_played_hand_types,
            final_blind_predicted_score=final_blind_predicted_score,
            final_blind_actual_score=final_blind_actual_score,
            final_blind_zero_score_hands=final_blind_zero_score_hands,
            postmortem_labels=postmortem_labels,
            chosen_shop_items=chosen_shop_items,
            shop_choices=tuple(shop_choices),
        ),
        malformed_rows,
    )


def _expand_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*.jsonl")))
        elif path.exists():
            expanded.append(path)
    return tuple(expanded)


def _ante_from_state(state_text: str) -> int:
    match = ANTE_PATTERN.search(state_text)
    return int(match.group(1)) if match else 0


def _parse_state_text(state_text: str) -> dict[str, object]:
    match = STATE_PATTERN.search(state_text)
    if not match:
        return {}
    values = match.groupdict()
    joker_text = values["jokers"]
    jokers = () if joker_text == "-" else tuple(item.strip() for item in joker_text.split(",") if item.strip())
    return {
        "phase": values["phase"],
        "ante": int(values["ante"]),
        "blind": values["blind"],
        "score": int(values["score"]),
        "required": int(values["required"]),
        "money": int(values["money"]),
        "hands": int(values["hands"]),
        "discards": int(values["discards"]),
        "jokers": jokers,
    }


def _played_hand_type(row: dict[str, object], action: dict[str, object]) -> str | None:
    extra = row.get("extra")
    if isinstance(extra, dict):
        score_audit = extra.get("score_audit")
        if isinstance(score_audit, dict) and score_audit.get("hand_type"):
            return str(score_audit["hand_type"])

    hand = _parse_detail_hand_cards(row.get("state_detail"))
    if not hand:
        state_text = str(row.get("state", ""))
        hand = _parse_hand_cards(state_text)
    if not hand:
        return None
    indices = action.get("card_indices")
    if not isinstance(indices, list):
        return None
    try:
        cards = tuple(hand[int(index)] for index in indices if 0 <= int(index) < len(hand))
    except (TypeError, ValueError):
        return None
    if not cards:
        return None
    try:
        return evaluate_played_cards(cards).hand_type.value
    except ValueError:
        return None


def _score_audit_totals(row: dict[str, object]) -> tuple[int, int]:
    extra = row.get("extra")
    if not isinstance(extra, dict):
        return (0, 0)
    score_audit = extra.get("score_audit")
    if not isinstance(score_audit, dict):
        return (0, 0)

    predicted = _int_value(score_audit.get("predicted_score"))
    actual = score_audit.get("actual_score_delta")
    if actual is None:
        before = score_audit.get("score_before")
        after = score_audit.get("score_after")
        if before is not None and after is not None:
            actual = _int_value(after) - _int_value(before)
    return (predicted, _int_value(actual))


def _postmortem_labels(
    *,
    observed_win: bool,
    max_ante: int,
    final_blind: str,
    final_score: int,
    final_required_score: int,
    final_money: int,
    final_jokers: tuple[str, ...],
    final_missing_roles: tuple[str, ...],
    played_hand_types: Counter[str],
    final_blind_played_hand_types: Counter[str],
    final_blind_predicted_score: int,
    final_blind_actual_score: int,
    final_blind_zero_score_hands: int,
) -> tuple[str, ...]:
    if observed_win:
        return ()

    labels: list[str] = []
    ratio = final_score / final_required_score if final_required_score > 0 else 0.0
    dominant_hand = played_hand_types.most_common(1)[0][0] if played_hand_types else None
    weak_final_plays = sum(final_blind_played_hand_types.get(hand, 0) for hand in WEAK_HAND_TYPES)
    final_plays = sum(final_blind_played_hand_types.values())
    missing = set(final_missing_roles)
    roles = _joker_role_set(final_jokers)

    if max_ante <= 2:
        labels.append("early_death")
    if final_blind in BOSS_BLINDS:
        labels.append("boss_death")
    if ratio >= 0.95:
        labels.append("near_miss_95")
    elif ratio >= 0.85:
        labels.append("close_loss")
    elif ratio < 0.50:
        labels.append("blowout_loss")
    if final_plays >= 3 and weak_final_plays >= 2:
        labels.append("weak_final_hand_mix")
    if dominant_hand in {"Pair", "Two Pair"}:
        labels.append("low_ceiling_pair_archetype")
    if final_money >= 50 and missing & CRITICAL_ROLES:
        labels.append("money_held_while_missing_power")
    if final_money >= 75:
        labels.append("very_high_money_death")
    if len(final_jokers) >= 5 and "xmult" not in roles:
        labels.append("full_slots_no_xmult")
    if len(final_jokers) >= 5 and "scaling" not in roles:
        labels.append("full_slots_no_scaling")
    if (
        final_blind_predicted_score
        and final_blind_predicted_score > final_blind_actual_score * 1.5
        and final_blind_predicted_score - final_blind_actual_score >= 5000
    ):
        labels.append("score_model_overconfident")
    if final_blind in {"The Eye", "The Mouth"} and final_blind_zero_score_hands > 0:
        labels.append("boss_restriction_zero_score")
    return tuple(labels)


def _joker_role_set(joker_names: tuple[str, ...]) -> set[str]:
    roles: set[str] = set()
    for name in joker_names:
        try:
            roles.update(_joker_roles(Joker(name=name)))
        except Exception:
            continue
    return roles


def _parse_hand_cards(state_text: str) -> tuple[Card, ...]:
    match = HAND_PATTERN.search(state_text)
    if not match:
        return ()
    hand_text = match.group("hand").strip()
    if not hand_text or hand_text == "-":
        return ()
    return tuple(_card_from_short_name(token) for token in hand_text.split())


def _parse_detail_hand_cards(detail: object) -> tuple[Card, ...]:
    if not isinstance(detail, dict):
        return ()
    cards = detail.get("hand")
    if not isinstance(cards, list):
        return ()
    parsed: list[Card] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        rank = card.get("rank")
        suit = card.get("suit")
        if rank is None or suit is None:
            name = card.get("name")
            if isinstance(name, str) and len(name) >= 2:
                parsed.append(_card_from_short_name(name))
            continue
        parsed.append(Card(rank=str(rank), suit=str(suit)))
    return tuple(parsed)


def _card_from_short_name(short_name: str) -> Card:
    return Card(rank=short_name[:-1], suit=short_name[-1])


def _state_from_summary_row(row: dict[str, object]) -> dict[str, object]:
    detail = row.get("final_state_detail")
    if isinstance(detail, dict):
        return {
            "phase": str(detail.get("phase", "")),
            "ante": _int_value(detail.get("ante")),
            "blind": str(detail.get("blind", "")),
            "score": _int_value(detail.get("current_score")),
            "required": _int_value(detail.get("required_score")),
            "money": _int_value(detail.get("money", row.get("final_money"))),
            "hands": _int_value(detail.get("hands_remaining")),
            "discards": _int_value(detail.get("discards_remaining")),
            "jokers": tuple(
                str(joker.get("name", ""))
                for joker in detail.get("jokers", ())
                if isinstance(joker, dict) and joker.get("name")
            ),
        }
    return _parse_state_text(str(row.get("final_state", "")))


def _int_value(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _blind_key_from_state(state_text: str) -> str:
    match = BLIND_PATTERN.search(state_text)
    return match.group(0) if match else ""


def _pressure_values(reason: str) -> tuple[float, ...]:
    return tuple(float(match.group(1)) for match in PRESSURE_PATTERN.finditer(reason))


def _chosen_shop_item_label(row: dict[str, object], action_type: str, audit: dict[str, object]) -> str:
    if action_type == "reroll":
        return "reroll"
    if action_type == "end_shop":
        return "skip"
    name = _item_name(row.get("chosen_item"))
    if not name:
        name = _item_name(audit.get("chosen_item"))
    if name:
        return f"{action_type}:{name}"
    return action_type


def _shop_choice_line(row: dict[str, object], action_type: str, audit: dict[str, object]) -> str:
    chosen = _chosen_shop_item_label(row, action_type, audit)
    pressure = audit.get("pressure") if isinstance(audit.get("pressure"), dict) else {}
    ratio = _float_value(pressure.get("ratio") if isinstance(pressure, dict) else None)
    threshold = _float_value(audit.get("threshold"))
    chosen_value = _float_value(audit.get("chosen_value", audit.get("chosen_upgrade")))
    top = _top_audit_option(audit)
    return (
        f"{_row_state_brief(row)} {chosen} "
        f"value={chosen_value:.1f} threshold={threshold:.1f} pressure={ratio:.2f} top={top}"
    )


def _row_state_brief(row: dict[str, object]) -> str:
    detail = row.get("state_detail")
    if isinstance(detail, dict):
        return f"a{_int_value(detail.get('ante'))} ${_int_value(detail.get('money'))}"
    state = _parse_state_text(str(row.get("state", "")))
    if state:
        return f"a{_int_value(state.get('ante'))} ${_int_value(state.get('money'))}"
    return "a? $?"


def _top_audit_option(audit: dict[str, object]) -> str:
    options = audit.get("options")
    if isinstance(options, list) and options:
        top = options[0]
        if isinstance(top, dict):
            name = _item_name(top.get("item")) or str(top.get("type", "option"))
            return f"{name}:{_float_value(top.get('value')):.1f}"
    replacement_options = audit.get("replacement_options")
    if isinstance(replacement_options, list) and replacement_options:
        top = replacement_options[0]
        if isinstance(top, dict):
            return f"{top.get('name', 'replacement')}:{_float_value(top.get('upgrade')):.1f}"
    return "-"


def _item_name(item: object) -> str:
    if isinstance(item, dict):
        name = item.get("name")
        return str(name) if name else ""
    return ""


def _float_value(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _most_common_counter(counter: Counter, limit: int) -> Counter:
    return Counter(dict(counter.most_common(limit)))


def _format_counter(counter: Counter) -> str:
    if not counter:
        return "{}"
    return "{" + ", ".join(f"{key}: {counter[key]}" for key in sorted(counter)) + "}"


def _counter_to_json_dict(counter: Counter) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter)}


def _failure_line(run: RunReplaySummary) -> str:
    jokers = ", ".join(run.final_jokers) if run.final_jokers else "-"
    return (
        f"seed={run.seed} ante={run.max_ante} blind={run.final_blind or '-'} "
        f"score={run.final_score}/{run.final_required_score} hands={run.final_hands} "
        f"money={run.final_money} jokers=[{jokers}] file={run.path.name}"
    )


def _run_json_summary(run: RunReplaySummary) -> dict[str, object]:
    return {
        "seed": run.seed,
        "file": run.path.name,
        "max_ante": run.max_ante,
        "final_blind": run.final_blind,
        "final_score": run.final_score,
        "final_required_score": run.final_required_score,
        "final_hands": run.final_hands,
        "final_money": run.final_money,
        "final_jokers": list(run.final_jokers),
        "dominant_played_hand": run.dominant_played_hand,
        "played_hand_types": _counter_to_json_dict(run.played_hand_types),
        "final_preferred_hand": run.final_preferred_hand,
        "final_missing_roles": list(run.final_missing_roles),
        "postmortem_labels": list(run.postmortem_labels),
        "final_blind_played_hand_types": _counter_to_json_dict(run.final_blind_played_hand_types),
        "final_blind_predicted_score": run.final_blind_predicted_score,
        "final_blind_actual_score": run.final_blind_actual_score,
        "final_blind_zero_score_hands": run.final_blind_zero_score_hands,
        "preferred_hand_counts": _counter_to_json_dict(run.preferred_hand_counts),
        "missing_role_counts": _counter_to_json_dict(run.missing_role_counts),
        "shop_action_count": run.shop_action_count,
        "shop_audit_count": run.shop_audit_count,
        "shop_skip_count": run.shop_skip_count,
        "chosen_shop_items": _counter_to_json_dict(run.chosen_shop_items),
        "shop_choices": list(run.shop_choices[:8]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Botlatro replay behavior.")
    parser.add_argument("paths", nargs="*", type=Path, help="Replay JSONL file(s) or directories.")
    parser.add_argument("--replay-dir", type=Path, help="Replay directory to scan recursively.")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable JSON summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = list(args.paths)
    if args.replay_dir is not None:
        paths.append(args.replay_dir)
    if not paths:
        raise SystemExit("Provide at least one replay path or --replay-dir.")
    analysis = analyze_replays(paths)
    if args.json:
        print(json.dumps(analysis.to_json_dict(), indent=2, sort_keys=True))
    else:
        print(analysis.to_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
