"""Small hand-planning data containers for the basic strategy bot."""

from __future__ import annotations

from dataclasses import dataclass

from balatro_ai.api.actions import Action
from balatro_ai.api.state import Card
from balatro_ai.rules.hand_evaluator import HandType


@dataclass(frozen=True, slots=True)
class _PlayCandidate:
    action: Action
    score: int
    hand_type: HandType
    scoring_card_indices: tuple[int, ...]
    cycle_value: float
    cycle_count: int


@dataclass(frozen=True, slots=True)
class _BlindContext:
    played_hand_types: tuple[HandType, ...] = ()
    discards_taken: int = 0


@dataclass(frozen=True, slots=True)
class _StraightDrawEvaluation:
    present_count: int
    missing_count: int
    missing_values: tuple[int, ...]
    out_count: int
    top_draw_out_count: int
    completion_probability: float
    completion_score: int
    quality: float
    window_high: int
    open_ended: bool
    gutshot: bool
    completes_from_known_draw: bool


@dataclass(frozen=True, slots=True)
class _TargetDrawEvaluation:
    hand_type: HandType
    label: str
    present_count: int
    missing_count: int
    out_count: int
    completion_probability: float
    completion_score: int
    quality: float


@dataclass(frozen=True, slots=True)
class _SampleHand:
    cards: tuple[Card, ...]
    held_cards: tuple[Card, ...] = ()
    weight: float = 1.0
