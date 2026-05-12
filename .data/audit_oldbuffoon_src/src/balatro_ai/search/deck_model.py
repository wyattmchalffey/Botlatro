"""Deck inference and draw sampling for Phase 7 search."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Iterable

from balatro_ai.api.actions import ActionType
from balatro_ai.api.state import Card, GamePhase, GameState
from balatro_ai.search import replay_diff

CardKey = tuple[str, str, str | None, str | None, str | None]

STANDARD_RANKS = ("A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2")
STANDARD_SUITS = ("S", "H", "D", "C")


@dataclass(frozen=True, slots=True)
class ImpossibleDraw:
    path: Path
    row_number: int
    seed: str
    card: Card
    exact_model: bool

    def to_text(self) -> str:
        model = "exact" if self.exact_model else "candidate"
        return f"{self.path.name}:{self.row_number} seed={self.seed} impossible_draw={self.card.short_name} model={model}"


@dataclass(frozen=True, slots=True)
class DeckValidationSummary:
    files_scanned: int
    transitions_checked: int
    draws_checked: int
    impossible_draws: tuple[ImpossibleDraw, ...]

    @property
    def passed(self) -> bool:
        return not any(item.exact_model for item in self.impossible_draws)

    @property
    def exact_impossible_draws(self) -> tuple[ImpossibleDraw, ...]:
        return tuple(item for item in self.impossible_draws if item.exact_model)

    @property
    def inexact_candidate_misses(self) -> tuple[ImpossibleDraw, ...]:
        return tuple(item for item in self.impossible_draws if not item.exact_model)

    def to_text(self, *, example_limit: int = 10) -> str:
        lines = [
            "Deck draw validation",
            f"Files scanned: {self.files_scanned}",
            f"Transitions checked: {self.transitions_checked}",
            f"Draws checked: {self.draws_checked}",
            f"Exact-model impossible draws: {len(self.exact_impossible_draws)}",
            f"Inexact candidate misses: {len(self.inexact_candidate_misses)}",
        ]
        if self.impossible_draws:
            lines.extend(("", "Examples:"))
            for item in self.impossible_draws[:example_limit]:
                lines.append(f"- {item.to_text()}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class DeckModel:
    """A multiset of cards that may still be drawn."""

    counts: Counter[CardKey]
    representatives: dict[CardKey, Card]
    exact: bool = True

    @classmethod
    def from_cards(cls, cards: Iterable[Card], *, exact: bool = True) -> "DeckModel":
        counts: Counter[CardKey] = Counter()
        representatives: dict[CardKey, Card] = {}
        for card in cards:
            key = card_key(card)
            counts[key] += 1
            representatives.setdefault(key, card)
        return cls(counts=counts, representatives=representatives, exact=exact)

    @classmethod
    def from_state(cls, state: GameState) -> "DeckModel":
        if state.known_deck:
            return cls.from_cards(state.known_deck, exact=len(state.known_deck) == state.deck_size)

        standard = cls.from_cards(_standard_deck(), exact=False)
        visible_out_of_deck = (
            state.hand
            + _zone_cards(state.modifiers, "played_pile")
            + _zone_cards(state.modifiers, "discard_pile")
            + _zone_cards(state.modifiers, "destroyed_cards")
        )
        return standard.remove_seen(visible_out_of_deck, strict=False)

    @property
    def total_cards(self) -> int:
        return sum(self.counts.values())

    def card_count(self, card: Card) -> int:
        return self.counts.get(card_key(card), 0)

    def contains(self, card: Card) -> bool:
        return self.card_count(card) > 0

    def remove_seen(self, cards: Iterable[Card], *, strict: bool = True) -> "DeckModel":
        counts = Counter(self.counts)
        representatives = dict(self.representatives)
        for card in cards:
            key = card_key(card)
            if counts[key] <= 0:
                if strict:
                    raise ValueError(f"Cannot remove unseen card from deck model: {card.short_name}")
                continue
            counts[key] -= 1
            if counts[key] <= 0:
                del counts[key]
                representatives.pop(key, None)
        return DeckModel(counts=counts, representatives=representatives, exact=self.exact)

    def sample_draws(self, n: int, rng: Random) -> tuple[Card, ...]:
        if n < 0:
            raise ValueError("Draw count must be non-negative")
        pool = self._expanded_pool()
        if n > len(pool):
            raise ValueError(f"Cannot draw {n} cards from deck model with {len(pool)} cards")
        return tuple(pool[index] for index in rng.sample(range(len(pool)), n))

    def all_possible_draws(self, n: int) -> tuple[tuple[Card, ...], ...]:
        if n < 0:
            raise ValueError("Draw count must be non-negative")
        if n > 3:
            raise ValueError("all_possible_draws is intentionally limited to n <= 3")
        if n > self.total_cards:
            return ()
        keys = tuple(sorted(self.counts, key=_key_sort))
        outcomes: list[tuple[Card, ...]] = []

        def visit(start: int, remaining: int, chosen: list[Card]) -> None:
            if remaining == 0:
                outcomes.append(tuple(chosen))
                return
            for index in range(start, len(keys)):
                key = keys[index]
                available = self.counts[key]
                max_take = min(available, remaining)
                for take in range(1, max_take + 1):
                    chosen.extend(self.representatives[key] for _ in range(take))
                    visit(index + 1, remaining - take, chosen)
                    del chosen[-take:]

        visit(0, n, [])
        return tuple(outcomes)

    def _expanded_pool(self) -> tuple[Card, ...]:
        cards: list[Card] = []
        for key in sorted(self.counts, key=_key_sort):
            cards.extend(self.representatives[key] for _ in range(self.counts[key]))
        return tuple(cards)


def validate_replay_draws(paths: Iterable[Path]) -> DeckValidationSummary:
    replay_paths = replay_diff._expand_paths(paths)
    impossible: list[ImpossibleDraw] = []
    transitions_checked = 0
    draws_checked = 0
    for path in replay_paths:
        rows, _ = replay_diff._load_rows(path)
        for index, row in enumerate(rows):
            action = replay_diff._action_from_row(row)
            if action is None or action.action_type not in {ActionType.PLAY_HAND, ActionType.DISCARD}:
                continue
            pre_state = replay_diff._state_from_detail(row.get("state_detail"), audit=replay_diff._score_audit_from_row(row))
            post_state = replay_diff._post_state_after(rows, index)
            if pre_state is None or post_state is None or post_state.phase == GamePhase.RUN_OVER:
                continue
            held_cards = tuple(card for hand_index, card in enumerate(pre_state.hand) if hand_index not in set(action.card_indices))
            drawn_cards = replay_diff._drawn_cards_from_post_state(held_cards, post_state.hand)
            if not drawn_cards:
                continue
            transitions_checked += 1
            model = DeckModel.from_state(pre_state)
            for card in drawn_cards:
                draws_checked += 1
                if not model.contains(card):
                    impossible.append(
                        ImpossibleDraw(
                            path=path,
                            row_number=index + 1,
                            seed=str(row.get("seed", path.stem)),
                            card=card,
                            exact_model=model.exact,
                        )
                    )
                else:
                    model = model.remove_seen((card,), strict=False)
    return DeckValidationSummary(
        files_scanned=len(replay_paths),
        transitions_checked=transitions_checked,
        draws_checked=draws_checked,
        impossible_draws=tuple(impossible),
    )


def card_key(card: Card) -> CardKey:
    return (_normalize_rank(card.rank), card.suit, card.enhancement, card.seal, card.edition)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate replay draws against DeckModel candidates.")
    parser.add_argument("paths", nargs="+", type=Path, help="Replay JSONL files or directories.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = validate_replay_draws(args.paths)
    print(summary.to_text())
    return 0 if summary.passed else 1


def _standard_deck() -> tuple[Card, ...]:
    return tuple(Card(rank, suit) for suit in STANDARD_SUITS for rank in STANDARD_RANKS)


def _zone_cards(modifiers: dict[str, object], key: str) -> tuple[Card, ...]:
    raw = modifiers.get(key, ())
    if isinstance(raw, Card):
        return (raw,)
    if isinstance(raw, dict):
        raw = raw.get("cards", ())
    if not isinstance(raw, list | tuple):
        return ()

    cards: list[Card] = []
    for item in raw:
        if isinstance(item, Card):
            cards.append(item)
        elif isinstance(item, dict | str):
            cards.append(Card.from_mapping(item))
    return tuple(cards)


def _normalize_rank(rank: str) -> str:
    return "10" if rank == "T" else rank


def _key_sort(key: CardKey) -> tuple[str, ...]:
    return tuple(str(part) for part in key)


if __name__ == "__main__":
    raise SystemExit(main())
