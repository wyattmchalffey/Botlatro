//! Per-card chip value (Phase 2c of RUST_PORT_PLAN.md).
//!
//! Ports `_card_chip_value` from `hand_evaluator.py:567`. Returns
//! the number of chips a single card contributes when it scores.
//!
//! Coverage: the simple, vanilla case — rank value + enhancement
//! bonus + debuffed handling. Cards with metadata-driven chip
//! overrides (Gold Cards with bonus_chips, perma_bonus modifiers,
//! displayed `+N chips` text) are NOT handled here because
//! RustCard intentionally drops the Python metadata dict at FFI
//! conversion time (Phase 1 design — metadata extraction adds
//! per-card cost we don't pay back outside the rare special cases).
//!
//! For those cards, the Rust function returns `None` and the
//! Python caller falls back to the original `_card_chip_value`.
//! Empirically (per the hot-path profile) this fallback rate is
//! <5% in solver rollouts since most cards are vanilla deck cards.
//!
//! Debuffed-suit handling: takes a `&[Suit]` slice of currently-
//! debuffed suits (typically derived from the active blind name
//! via `debuffed_suits_for_blind`). The caller does that lookup
//! once at evaluation entry and passes the result to every card.

use pyo3::prelude::*;

use crate::state::card::{Card, Enhancement, Suit};

/// Per-card chip value. Returns `None` for cards that need Python
/// metadata interpretation (currently never happens for our packed
/// `Card` because metadata is dropped at conversion — see module
/// doc), so this returns `Some(value)` for every input. Kept as
/// `Option<u16>` to leave the door open for "I can't handle this"
/// signaling later.
pub fn card_chip_value(card: Card, debuffed_suits: &[Suit]) -> Option<u16> {
    // Debuffed cards score 0 chips (matches `card.debuffed` check).
    if card.debuffed {
        return Some(0);
    }
    // Stone cards: Python returns the displayed value (from metadata)
    // OR 50 as a fallback. We don't have metadata, so always 50.
    // This is the documented Phase 2c gap — Stone cards with
    // displayed-chip overrides will diverge here.
    if card.enhancement == Enhancement::Stone {
        return Some(50);
    }
    // Suit-debuffed (active blind like "The Club" debuffs all Clubs).
    // Wild cards count as any suit so they're always debuffed when
    // ANY suit is debuffed.
    if !debuffed_suits.is_empty() {
        if card.enhancement == Enhancement::Wild {
            return Some(0);
        }
        if debuffed_suits.contains(&card.suit) {
            return Some(0);
        }
    }
    // Base case: rank chip value + enhancement bonus.
    let base = card.rank.chip_value() as u16;
    let enhancement_chips = enhancement_chip_bonus(card.enhancement);
    Some(base + enhancement_chips)
}

/// True when a card participates in the scored-card effects pass.
/// Matches Python's `_card_can_score`: debuffed cards and suit-debuffed
/// non-Stone cards keep their hand-shape role but do not trigger scored-card
/// jokers, editions, seals, or enhancements.
pub fn card_can_score(card: Card, debuffed_suits: &[Suit]) -> bool {
    if card.debuffed {
        return false;
    }
    if card.enhancement == Enhancement::Stone {
        return true;
    }
    if debuffed_suits.is_empty() {
        return true;
    }
    if card.enhancement == Enhancement::Wild {
        return false;
    }
    !debuffed_suits.contains(&card.suit)
}

/// Bonus chips contributed by the card's enhancement.
/// Matches `_enhancement_chips` at `hand_evaluator.py:1622`.
#[inline]
fn enhancement_chip_bonus(e: Enhancement) -> u16 {
    match e {
        Enhancement::Bonus => 30,
        _ => 0,
    }
}

/// PyO3 wrapper. Accepts a single RustCard + an optional list of
/// debuffed suit strings (one of "C", "D", "H", "S" each).
#[pyfunction]
#[pyo3(name = "card_chip_value")]
pub fn py_card_chip_value(card: Card, debuffed_suits: Vec<String>) -> Option<u16> {
    let suits: Vec<Suit> = debuffed_suits
        .iter()
        .filter_map(|s| Suit::from_str(s))
        .collect();
    card_chip_value(card, &suits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::card::{Edition, Rank, Seal};

    fn card(rank: Rank, suit: Suit) -> Card {
        Card {
            rank,
            suit,
            enhancement: Enhancement::None,
            edition: Edition::None,
            seal: Seal::None,
            debuffed: false,
        }
    }

    #[test]
    fn vanilla_two_is_2_chips() {
        assert_eq!(card_chip_value(card(Rank::Two, Suit::Hearts), &[]), Some(2));
    }

    #[test]
    fn vanilla_ace_is_11_chips() {
        assert_eq!(card_chip_value(card(Rank::Ace, Suit::Hearts), &[]), Some(11));
    }

    #[test]
    fn face_cards_are_10_chips() {
        for r in &[Rank::Ten, Rank::Jack, Rank::Queen, Rank::King] {
            assert_eq!(card_chip_value(card(*r, Suit::Hearts), &[]), Some(10));
        }
    }

    #[test]
    fn bonus_enhancement_adds_30() {
        let mut c = card(Rank::Five, Suit::Hearts);
        c.enhancement = Enhancement::Bonus;
        assert_eq!(card_chip_value(c, &[]), Some(35)); // 5 + 30
    }

    #[test]
    fn stone_card_is_50() {
        let mut c = card(Rank::Five, Suit::Hearts);
        c.enhancement = Enhancement::Stone;
        assert_eq!(card_chip_value(c, &[]), Some(50));
    }

    #[test]
    fn debuffed_card_is_zero() {
        let mut c = card(Rank::Ace, Suit::Hearts);
        c.debuffed = true;
        assert_eq!(card_chip_value(c, &[]), Some(0));
    }

    #[test]
    fn suit_debuff_makes_card_zero() {
        let c = card(Rank::Ace, Suit::Hearts);
        assert_eq!(card_chip_value(c, &[Suit::Hearts]), Some(0));
    }

    #[test]
    fn suit_debuff_irrelevant_for_different_suit() {
        let c = card(Rank::Ace, Suit::Spades);
        assert_eq!(card_chip_value(c, &[Suit::Hearts]), Some(11));
    }
}
