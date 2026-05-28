//! Deck-draw simulation (Phase 3a of RUST_PORT_PLAN.md).
//!
//! Ports `_draw_from_deck` from `src/balatro_ai/search/forward_sim.py:927`.
//!
//! Given the current `known_deck` + `deck_size` + a list of cards
//! drawn this turn, return the post-draw `known_deck` + `deck_size`.
//! Two flavors:
//! - **Exact-known deck**: `len(known_deck) == deck_size`. Drawn cards
//!   MUST appear in `known_deck`; we remove them and the new
//!   `deck_size` is the new length.
//! - **Partial-known deck**: `len(known_deck) < deck_size`. Drawn
//!   cards MAY appear in `known_deck`; we remove matches but the new
//!   `deck_size` decrements regardless.
//!
//! Card matching is done by `(rank, suit, enhancement, edition, seal)`
//! — the matching predicate is `_find_matching_card_index` in Python.
//! We mirror that order: first match wins.

use pyo3::prelude::*;

use crate::state::card::Card;

/// Result of a draw operation.
#[derive(Clone, Debug, PartialEq)]
pub struct DrawResult {
    pub deck_size: u32,
    pub known_deck: Vec<Card>,
}

/// Native draw. Returns `Err` only when `exact_known_deck` AND a
/// drawn card is not found — Python raises `ValueError` in that
/// case, and we propagate the same condition so the PyO3 wrapper
/// can mirror it.
pub fn draw_from_deck(
    known_deck: &[Card],
    deck_size: u32,
    drawn: &[Card],
) -> Result<DrawResult, &'static str> {
    if drawn.is_empty() {
        return Ok(DrawResult {
            deck_size,
            known_deck: known_deck.to_vec(),
        });
    }
    if known_deck.is_empty() {
        // Python: when known_deck is empty, deck_size shrinks but
        // known_deck stays empty (unknown contents).
        let new_size = deck_size.saturating_sub(drawn.len() as u32);
        return Ok(DrawResult { deck_size: new_size, known_deck: Vec::new() });
    }

    let exact = known_deck.len() as u32 == deck_size;
    let mut remaining: Vec<Card> = known_deck.to_vec();
    for d in drawn {
        match find_matching_card_index(&remaining, *d) {
            Some(i) => { remaining.remove(i); }
            None => {
                if exact {
                    return Err("drawn card not present in exact known_deck");
                }
                // partial known deck: silently skip the missing draw
            }
        }
    }
    let new_known_len = remaining.len() as u32;
    let new_size = if exact {
        new_known_len
    } else {
        new_known_len.max(deck_size.saturating_sub(drawn.len() as u32))
    };
    Ok(DrawResult { deck_size: new_size, known_deck: remaining })
}

/// First card in `deck` matching `target` on the canonical 5-field
/// equality (rank, suit, enhancement, edition, seal). Python uses
/// the same predicate via `_find_matching_card_index`.
#[inline]
fn find_matching_card_index(deck: &[Card], target: Card) -> Option<usize> {
    deck.iter().position(|c| {
        c.rank == target.rank
            && c.suit == target.suit
            && c.enhancement == target.enhancement
            && c.edition == target.edition
            && c.seal == target.seal
    })
}

/// PyO3 wrapper. Accepts the known_deck + deck_size + drawn lists
/// as Vec<Card> / u32. Returns `(deck_size, known_deck)` on success.
/// Raises ValueError when an exact known_deck is missing a drawn
/// card (mirrors Python's `_draw_from_deck` behavior).
#[pyfunction]
#[pyo3(name = "draw_from_deck")]
pub fn py_draw_from_deck(
    known_deck: Vec<Card>,
    deck_size: u32,
    drawn: Vec<Card>,
) -> PyResult<(u32, Vec<Card>)> {
    match draw_from_deck(&known_deck, deck_size, &drawn) {
        Ok(r) => Ok((r.deck_size, r.known_deck)),
        Err(msg) => Err(pyo3::exceptions::PyValueError::new_err(msg)),
    }
}

/// Index-returning variant of `draw_from_deck`. Returns the indices
/// in `known_deck` that should be REMOVED to satisfy `drawn`, plus
/// the new deck_size. This lets the Python caller slice its OWN
/// known_deck tuple (preserving Card metadata that RustCard drops).
///
/// Returns `(deck_size, indices_to_remove_sorted)`. Indices are in
/// the original `known_deck` (pre-removal). The caller drops them
/// from highest to lowest to maintain index validity.
pub fn draw_indices_to_remove(
    known_deck: &[Card],
    deck_size: u32,
    drawn: &[Card],
) -> Result<(u32, Vec<usize>), &'static str> {
    if drawn.is_empty() {
        return Ok((deck_size, Vec::new()));
    }
    if known_deck.is_empty() {
        return Ok((deck_size.saturating_sub(drawn.len() as u32), Vec::new()));
    }
    let exact = known_deck.len() as u32 == deck_size;
    // Track which indices are already used (to handle duplicates).
    let mut used = vec![false; known_deck.len()];
    let mut removed: Vec<usize> = Vec::new();
    for d in drawn {
        let idx = known_deck.iter().enumerate().find_map(|(i, c)| {
            if used[i] { return None; }
            if c.rank == d.rank && c.suit == d.suit
                && c.enhancement == d.enhancement && c.edition == d.edition
                && c.seal == d.seal {
                Some(i)
            } else { None }
        });
        match idx {
            Some(i) => {
                used[i] = true;
                removed.push(i);
            }
            None => {
                if exact {
                    return Err("drawn card not present in exact known_deck");
                }
            }
        }
    }
    removed.sort_unstable();
    let new_known_len = (known_deck.len() - removed.len()) as u32;
    let new_size = if exact {
        new_known_len
    } else {
        new_known_len.max(deck_size.saturating_sub(drawn.len() as u32))
    };
    Ok((new_size, removed))
}

/// PyO3 wrapper for the index-returning variant.
#[pyfunction]
#[pyo3(name = "draw_indices_to_remove")]
pub fn py_draw_indices_to_remove(
    known_deck: Vec<Card>,
    deck_size: u32,
    drawn: Vec<Card>,
) -> PyResult<(u32, Vec<usize>)> {
    match draw_indices_to_remove(&known_deck, deck_size, &drawn) {
        Ok(r) => Ok(r),
        Err(msg) => Err(pyo3::exceptions::PyValueError::new_err(msg)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::card::{Edition, Enhancement, Rank, Seal, Suit};

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
    fn empty_drawn_returns_unchanged_deck() {
        let deck = vec![card(Rank::Ace, Suit::Hearts), card(Rank::Two, Suit::Spades)];
        let r = draw_from_deck(&deck, 2, &[]).unwrap();
        assert_eq!(r.deck_size, 2);
        assert_eq!(r.known_deck, deck);
    }

    #[test]
    fn empty_known_deck_just_decrements_size() {
        let r = draw_from_deck(&[], 40, &[card(Rank::Ace, Suit::Hearts)]).unwrap();
        assert_eq!(r.deck_size, 39);
        assert!(r.known_deck.is_empty());
    }

    #[test]
    fn exact_deck_removes_matched_cards() {
        let deck = vec![
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Two, Suit::Spades),
            card(Rank::Three, Suit::Clubs),
        ];
        let r = draw_from_deck(&deck, 3, &[card(Rank::Two, Suit::Spades)]).unwrap();
        assert_eq!(r.deck_size, 2);
        assert_eq!(r.known_deck.len(), 2);
        assert_eq!(r.known_deck[0], card(Rank::Ace, Suit::Hearts));
        assert_eq!(r.known_deck[1], card(Rank::Three, Suit::Clubs));
    }

    #[test]
    fn exact_deck_missing_card_returns_err() {
        let deck = vec![card(Rank::Ace, Suit::Hearts)];
        let r = draw_from_deck(&deck, 1, &[card(Rank::Two, Suit::Spades)]);
        assert!(r.is_err());
    }

    #[test]
    fn partial_deck_silently_skips_missing() {
        let deck = vec![card(Rank::Ace, Suit::Hearts)]; // known partial of 40-size deck
        let r = draw_from_deck(&deck, 40, &[card(Rank::Two, Suit::Spades)]).unwrap();
        // Drawn card not in known partial → known unchanged, size -1.
        assert_eq!(r.known_deck.len(), 1);
        assert_eq!(r.deck_size, 39);
    }

    #[test]
    fn multiple_draws_remove_in_order() {
        let deck = vec![
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Ace, Suit::Hearts), // duplicate
            card(Rank::Two, Suit::Spades),
        ];
        let r = draw_from_deck(
            &deck, 3,
            &[card(Rank::Ace, Suit::Hearts), card(Rank::Ace, Suit::Hearts)],
        ).unwrap();
        assert_eq!(r.deck_size, 1);
        assert_eq!(r.known_deck, vec![card(Rank::Two, Suit::Spades)]);
    }
}
