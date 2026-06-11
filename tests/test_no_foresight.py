"""Tests for the BALATRO_NO_FORESIGHT information-set ablation."""

from __future__ import annotations

from collections import Counter

import pytest

from balatro_ai.api.state import Card, GamePhase, GameState
from balatro_ai.bots.no_foresight import blind_known_deck, no_foresight_mode


def _state_with_deck(n: int = 30) -> GameState:
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    suits = ["Hearts", "Spades", "Diamonds", "Clubs"]
    deck = tuple(
        Card.from_mapping({"rank": ranks[i % 13], "suit": suits[i % 4]}) for i in range(n)
    )
    hand = tuple(
        Card.from_mapping({"rank": ranks[(i + 5) % 13], "suit": suits[(i + 1) % 4]})
        for i in range(8)
    )
    return GameState(
        phase=GamePhase.SELECTING_HAND,
        ante=3,
        blind="Big Blind",
        required_score=2000,
        current_score=500,
        money=14,
        hands_remaining=3,
        discards_remaining=2,
        deck_size=n,
        hand=hand,
        known_deck=deck,
    )


def test_off_is_identity(monkeypatch):
    monkeypatch.delenv("BALATRO_NO_FORESIGHT", raising=False)
    state = _state_with_deck()
    assert no_foresight_mode() == ""
    assert blind_known_deck(state) is state
    monkeypatch.setenv("BALATRO_NO_FORESIGHT", "0")
    assert blind_known_deck(state) is state


def test_invalid_mode_raises(monkeypatch):
    monkeypatch.setenv("BALATRO_NO_FORESIGHT", "bogus")
    with pytest.raises(ValueError):
        no_foresight_mode()


def test_hide_empties_known_deck(monkeypatch):
    monkeypatch.setenv("BALATRO_NO_FORESIGHT", "hide")
    state = _state_with_deck()
    blinded = blind_known_deck(state)
    assert blinded.known_deck == ()
    assert blinded.hand == state.hand
    assert blinded.deck_size == state.deck_size


def test_shuffle_preserves_multiset_and_hides_order(monkeypatch):
    monkeypatch.setenv("BALATRO_NO_FORESIGHT", "shuffle")
    state = _state_with_deck(40)
    blinded = blind_known_deck(state)
    assert Counter(repr(c) for c in blinded.known_deck) == Counter(
        repr(c) for c in state.known_deck
    )
    # A fixed permutation of 40 cards matching the true order is ~1/40!.
    assert tuple(repr(c) for c in blinded.known_deck) != tuple(
        repr(c) for c in state.known_deck
    )


def test_shuffle_is_deterministic_per_decision(monkeypatch):
    monkeypatch.setenv("BALATRO_NO_FORESIGHT", "shuffle")
    state = _state_with_deck()
    a = blind_known_deck(state)
    b = blind_known_deck(state)
    assert tuple(repr(c) for c in a.known_deck) == tuple(repr(c) for c in b.known_deck)


def test_shuffle_varies_across_decisions(monkeypatch):
    monkeypatch.setenv("BALATRO_NO_FORESIGHT", "shuffle")
    from dataclasses import replace

    state = _state_with_deck()
    after_play = replace(state, current_score=900, hands_remaining=2)
    a = blind_known_deck(state)
    b = blind_known_deck(after_play)
    assert tuple(repr(c) for c in a.known_deck) != tuple(repr(c) for c in b.known_deck)


def test_shuffle_does_not_depend_on_true_order(monkeypatch):
    """The belief permutation must be a function of the multiset, not the true order."""
    monkeypatch.setenv("BALATRO_NO_FORESIGHT", "shuffle")
    from dataclasses import replace

    state = _state_with_deck()
    reversed_truth = replace(state, known_deck=tuple(reversed(state.known_deck)))
    a = blind_known_deck(state)
    b = blind_known_deck(reversed_truth)
    # Same multiset => identical belief order regardless of the true order.
    assert tuple(repr(c) for c in a.known_deck) == tuple(repr(c) for c in b.known_deck)
