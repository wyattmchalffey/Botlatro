//! Scoring-card index selection (Phase 2b of RUST_PORT_PLAN.md).
//!
//! Ports `_scoring_indices` from `hand_evaluator.py:450`. Given a
//! list of played cards and the hand type, returns the indices of
//! cards that COUNT toward scoring (the others "go along for the
//! ride" but don't add chips/mult).
//!
//! Hand-type-specific rules:
//! - Full House, Five of a Kind, Flush House, Flush Five → all cards
//! - Pair, Two Pair, Three of a Kind, Four of a Kind → the n-of-a-kind
//!   cards only
//! - Straight → all 5 of the straight (or 4 if Four Fingers — bailing
//!   to Python for now)
//! - Flush → all 5 of the flush
//! - Straight Flush → union of straight + flush indices
//! - High Card → just the highest-ranked card
//!
//! Phase 2b coverage: no-joker fast path. Returns `None` for:
//! - Splash joker (would force all-cards) — caller's responsibility
//! - Stone cards (handled via add-in but harder; bail to Python)
//! - Wild cards (suit handling — bail)
//! - Four Fingers / Smeared (bail)
//! - Hands where Rust identify_hand_type returned None (already
//!   bailed at the previous step)

use pyo3::prelude::*;

use crate::hand_eval::hand_type::HandType;
use crate::state::card::{Card, Enhancement, Rank};

/// Return the scoring-card indices for the simple case. `None` means
/// "fall back to Python".
///
/// `has_splash` short-circuits to "all cards score" when the player
/// holds the Splash joker — matches Python's first-line check.
pub fn scoring_indices_simple(
    cards: &[Card],
    hand_type: HandType,
) -> Option<Vec<usize>> {
    scoring_indices_simple_with_splash(cards, hand_type, false)
}

/// Scoring indices with joker-aware identification. Picks the
/// right 4/5-card subset for Flush/Straight/StraightFlush when
/// Four Fingers / Smeared / Shortcut are active. Adds Stone-card
/// indices back. Bails on Wild cards.
pub fn scoring_indices_with_jokers(
    cards: &[Card],
    hand_type: HandType,
    has_splash: bool,
    j: crate::hand_eval::hand_type::IdJokers,
) -> Option<Vec<usize>> {
    if has_splash {
        return Some((0..cards.len()).collect());
    }

    // Stone indices always score.
    let stone_indices: Vec<usize> = cards.iter()
        .enumerate()
        .filter(|(_, c)| c.enhancement == Enhancement::Stone)
        .map(|(i, _)| i)
        .collect();
    let ranked: Vec<(usize, Card)> = cards.iter()
        .copied()
        .enumerate()
        .filter(|(_, c)| c.enhancement != Enhancement::Stone)
        .collect();

    let merge = |mut idx: Vec<usize>| -> Vec<usize> {
        idx.extend(stone_indices.iter().copied());
        idx.sort_unstable();
        idx.dedup();
        idx
    };

    // Pick the right card indices for Flush / Straight / StraightFlush
    // when joker modifiers shrink the requirement.
    let n_ranked = ranked.len();
    let scoring = match hand_type {
        HandType::FullHouse | HandType::FiveOfAKind | HandType::FlushHouse | HandType::FlushFive => {
            return Some((0..cards.len()).collect());
        }
        HandType::Flush => {
            // Try 5-card first; fall back to 4-card only if Four
            // Fingers AND no 5-card match (matches Python's
            // _five_or_four_fingers_indices).
            best_flush_indices(&ranked, 5, j.smeared)
                .or_else(|| if j.four_fingers { best_flush_indices(&ranked, 4, j.smeared) } else { None })?
        }
        HandType::Straight => {
            best_straight_indices(&ranked, 5, j.shortcut)
                .or_else(|| if j.four_fingers { best_straight_indices(&ranked, 4, j.shortcut) } else { None })?
        }
        HandType::StraightFlush => {
            best_straight_flush_indices(&ranked, 5, j.smeared, j.shortcut)
                .or_else(|| if j.four_fingers { best_straight_flush_indices(&ranked, 4, j.smeared, j.shortcut) } else { None })?
        }
        HandType::FourOfAKind => indices_with_count(&ranked, 4),
        HandType::ThreeOfAKind => indices_with_count(&ranked, 3),
        HandType::TwoPair => indices_with_count(&ranked, 2),
        HandType::Pair => indices_with_count(&ranked, 2),
        HandType::HighCard => {
            match highest_rank_index(&ranked) {
                Some(i) => vec![i],
                None => vec![],
            }
        }
    };
    let _ = n_ranked;
    Some(merge(scoring))
}

/// Find the largest suit-key group of size ≥ `needed`. Wild cards
/// match every suit key. Returns the original indices (sorted) of
/// the chosen group.
fn best_flush_indices(
    ranked: &[(usize, Card)],
    needed: usize,
    smeared: bool,
) -> Option<Vec<usize>> {
    use crate::hand_eval::hand_type::flush_suit_key;
    use crate::state::card::Enhancement;
    let n_keys: usize = if smeared { 2 } else { 4 };
    let mut groups: [Vec<usize>; 4] = [vec![], vec![], vec![], vec![]];
    for &(orig_i, c) in ranked {
        if c.enhancement == Enhancement::Wild {
            for i in 0..n_keys { groups[i].push(orig_i); }
        } else {
            let key = flush_suit_key(c.suit, smeared) as usize;
            groups[key].push(orig_i);
        }
    }
    // Pick the largest group meeting the size threshold; tie-break by
    // rank-sum (Python's `max(candidates, key=_rank_sum)`).
    let mut best: Option<Vec<usize>> = None;
    let mut best_sum: u32 = 0;
    for g in groups.iter().take(n_keys) {
        if g.len() >= needed {
            let sum: u32 = g.iter().map(|&i| {
                ranked.iter()
                    .find(|(oi, _)| *oi == i)
                    .map(|(_, c)| c.rank as u32)
                    .unwrap_or(0)
            }).sum();
            if best.is_none() || sum > best_sum {
                let mut sorted = g.clone();
                sorted.sort_unstable();
                best = Some(sorted);
                best_sum = sum;
            }
        }
    }
    best
}

/// Find the best straight subset of size `needed`. Returns the
/// original indices.
fn best_straight_indices(
    ranked: &[(usize, Card)],
    needed: usize,
    shortcut: bool,
) -> Option<Vec<usize>> {
    enumerate_subsets_pick_best(ranked, needed, |subset_cards| {
        is_straight_with_jokers_local(subset_cards, shortcut)
    })
}

/// Find the best straight-flush subset of size `needed` — must be
/// both a straight (with shortcut) AND share a suit-key (with
/// smeared). Wild cards match any suit-key.
fn best_straight_flush_indices(
    ranked: &[(usize, Card)],
    needed: usize,
    smeared: bool,
    shortcut: bool,
) -> Option<Vec<usize>> {
    use crate::hand_eval::hand_type::flush_suit_key;
    use crate::state::card::Enhancement;
    enumerate_subsets_pick_best(ranked, needed, |subset_cards| {
        if !is_straight_with_jokers_local(subset_cards, shortcut) {
            return false;
        }
        if subset_cards.is_empty() { return true; }
        // The subset is a "flush" if some suit-key is shared by ALL
        // cards (Wild cards count as that suit too).
        let n_keys: u8 = if smeared { 2 } else { 4 };
        (0..n_keys).any(|key| {
            subset_cards.iter().all(|c| {
                c.enhancement == Enhancement::Wild
                    || flush_suit_key(c.suit, smeared) == key
            })
        })
    })
}

/// Enumerate `needed`-sized subsets of `ranked`, find the one with
/// the highest rank-sum that passes `predicate(subset_cards)`.
fn enumerate_subsets_pick_best<F: Fn(&[Card]) -> bool>(
    ranked: &[(usize, Card)],
    needed: usize,
    predicate: F,
) -> Option<Vec<usize>> {
    let n = ranked.len();
    if needed > n { return None; }
    let mut idx = vec![0usize; needed];
    for i in 0..needed { idx[i] = i; }
    let mut best_indices: Option<Vec<usize>> = None;
    let mut best_sum: u32 = 0;
    loop {
        let subset_cards: Vec<Card> = idx.iter().map(|&i| ranked[i].1).collect();
        if predicate(&subset_cards) {
            let sum: u32 = subset_cards.iter().map(|c| c.rank as u32).sum();
            if best_indices.is_none() || sum > best_sum {
                best_indices = Some(idx.iter().map(|&i| ranked[i].0).collect());
                best_sum = sum;
            }
        }
        // Advance combination.
        let mut k: Option<usize> = None;
        for i in (0..needed).rev() {
            if idx[i] < n - needed + i {
                k = Some(i);
                break;
            }
        }
        let Some(k) = k else { return best_indices };
        idx[k] += 1;
        for i in k+1..needed {
            idx[i] = idx[i-1] + 1;
        }
    }
}

/// Local copy of `is_straight_with_jokers` to avoid a circular
/// dep import (hand_type::is_straight_with_jokers is private).
fn is_straight_with_jokers_local(cards: &[Card], shortcut: bool) -> bool {
    let mut values: Vec<u8> = cards.iter().map(|c| c.rank as u8).collect();
    values.sort_unstable();
    values.dedup();
    if values.len() != cards.len() { return false; }
    if values.len() == 5 && values == [2, 3, 4, 5, 14] { return true; }
    if values.len() == 4 && values == [2, 3, 4, 14] { return true; }
    if shortcut {
        return values.windows(2).all(|w| w[1] >= w[0] + 1 && w[1] <= w[0] + 2);
    }
    let lo = values[0];
    values.iter().enumerate().all(|(i, &v)| v == lo + i as u8)
}

pub fn scoring_indices_simple_with_splash(
    cards: &[Card],
    hand_type: HandType,
    has_splash: bool,
) -> Option<Vec<usize>> {
    // Splash forces ALL cards to score (Python: first line of
    // _scoring_indices). When active, return the full index range.
    // Bail on Wild cards (their suit semantics differ); Stone cards
    // are fine because card_chip_value handles them (returns 50).
    if has_splash {
        for c in cards {
            if matches!(c.enhancement, Enhancement::Wild) {
                return None;
            }
        }
        return Some((0..cards.len()).collect());
    }

    // Fast-path filter: Wild cards bail. Stone cards are now
    // supported (filtered out of identification, added back here).
    for c in cards {
        if matches!(c.enhancement, Enhancement::Wild) {
            return None;
        }
    }

    // Stone indices always score (Python `_with_stone_indices`).
    let stone_indices: Vec<usize> = cards.iter()
        .enumerate()
        .filter(|(_, c)| c.enhancement == Enhancement::Stone)
        .map(|(i, _)| i)
        .collect();
    // The ranked subset (non-stone) drives normal scoring decisions.
    let ranked: Vec<(usize, Card)> = cards.iter()
        .copied()
        .enumerate()
        .filter(|(_, c)| c.enhancement != Enhancement::Stone)
        .collect();

    // Merge ranked scoring indices with stone indices, deduplicated
    // and sorted (Python `_with_stone_indices`).
    let merge = |mut ranked_idxs: Vec<usize>| -> Vec<usize> {
        ranked_idxs.extend(stone_indices.iter().copied());
        ranked_idxs.sort_unstable();
        ranked_idxs.dedup();
        ranked_idxs
    };

    let n = ranked.len();
    let scoring = match hand_type {
        HandType::FullHouse | HandType::FiveOfAKind | HandType::FlushHouse | HandType::FlushFive => {
            // All cards (ranked + stone) score for these hand types.
            return Some((0..cards.len()).collect());
        }
        HandType::Flush | HandType::Straight | HandType::StraightFlush => {
            if n == 5 {
                // The five ranked cards form the run/flush; all of
                // them score, plus any stone cards.
                ranked.iter().map(|(i, _)| *i).collect()
            } else {
                return None;
            }
        }
        HandType::FourOfAKind => indices_with_count(&ranked, 4),
        HandType::ThreeOfAKind => indices_with_count(&ranked, 3),
        HandType::TwoPair => indices_with_count(&ranked, 2),
        HandType::Pair => indices_with_count(&ranked, 2),
        HandType::HighCard => {
            // Just the highest-ranked card.
            match highest_rank_index(&ranked) {
                Some(i) => vec![i],
                None => vec![],
            }
        }
    };
    Some(merge(scoring))
}

/// Indices of cards whose rank appears exactly `target` times among
/// the ranked cards. Operates on `(original_index, Card)` tuples
/// so callers can pass a Stone-filtered subset while preserving
/// the original indices in the returned vector.
fn indices_with_count(ranked: &[(usize, Card)], target: u8) -> Vec<usize> {
    let mut rank_counts = [0u8; 16];
    for (_, c) in ranked {
        rank_counts[c.rank as usize] += 1;
    }
    ranked
        .iter()
        .filter(|(_, c)| rank_counts[c.rank as usize] == target)
        .map(|(i, _)| *i)
        .collect()
}

/// Index of the single highest-ranked card by CHIP value (matches
/// Python's `max(ranked_indices, key=lambda i: RANK_VALUES[...])`).
///
/// Two important nuances:
/// 1. Uses `Rank::chip_value()` (face cards all = 10, ten = 10),
///    NOT straight value (which differentiates them). This is what
///    Python `RANK_VALUES` does — see card.rs `chip_value()` for
///    the source-of-truth comment.
/// 2. Python's `max(..., key=...)` returns the FIRST maximum on tie.
///    Rust's `max_by_key` returns the LAST. We invert by iterating
///    in reverse + using `min_by_key` with negated index — or
///    equivalently, scan forward and only update on strict-greater.
fn highest_rank_index(ranked: &[(usize, Card)]) -> Option<usize> {
    if ranked.is_empty() {
        return None;
    }
    let mut best_idx = ranked[0].0;
    let mut best_value = ranked[0].1.rank.chip_value();
    for &(orig_idx, c) in &ranked[1..] {
        let v = c.rank.chip_value();
        if v > best_value {
            best_value = v;
            best_idx = orig_idx;
        }
    }
    Some(best_idx)
}

/// PyO3 wrapper. Accepts a list of `RustCard` + a hand type string
/// (matching the Python `HandType` enum value), returns a list of
/// indices or None.
#[pyfunction]
#[pyo3(name = "scoring_indices")]
pub fn py_scoring_indices(cards: Vec<Card>, hand_type_str: &str) -> Option<Vec<usize>> {
    let hand_type = match hand_type_str {
        "High Card" => HandType::HighCard,
        "Pair" => HandType::Pair,
        "Two Pair" => HandType::TwoPair,
        "Three of a Kind" => HandType::ThreeOfAKind,
        "Straight" => HandType::Straight,
        "Flush" => HandType::Flush,
        "Full House" => HandType::FullHouse,
        "Four of a Kind" => HandType::FourOfAKind,
        "Straight Flush" => HandType::StraightFlush,
        "Five of a Kind" => HandType::FiveOfAKind,
        "Flush House" => HandType::FlushHouse,
        "Flush Five" => HandType::FlushFive,
        _ => return None, // unknown hand type — bail
    };
    scoring_indices_simple(&cards, hand_type)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::card::{Edition, Seal, Suit};

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
    fn pair_returns_only_pair_indices() {
        let cards = [
            card(Rank::Ace, Suit::Spades),
            card(Rank::Ace, Suit::Hearts),
            card(Rank::King, Suit::Diamonds),
            card(Rank::Nine, Suit::Clubs),
        ];
        assert_eq!(scoring_indices_simple(&cards, HandType::Pair), Some(vec![0, 1]));
    }

    #[test]
    fn three_of_a_kind_returns_three_indices() {
        let cards = [
            card(Rank::King, Suit::Diamonds),
            card(Rank::Ace, Suit::Spades),
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Ace, Suit::Diamonds),
            card(Rank::Nine, Suit::Clubs),
        ];
        assert_eq!(
            scoring_indices_simple(&cards, HandType::ThreeOfAKind),
            Some(vec![1, 2, 3]),
        );
    }

    #[test]
    fn full_house_returns_all_cards() {
        let cards = [
            card(Rank::King, Suit::Diamonds),
            card(Rank::King, Suit::Hearts),
            card(Rank::Ace, Suit::Spades),
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Ace, Suit::Diamonds),
        ];
        assert_eq!(scoring_indices_simple(&cards, HandType::FullHouse), Some(vec![0, 1, 2, 3, 4]));
    }

    #[test]
    fn high_card_returns_highest_only() {
        let cards = [
            card(Rank::Two, Suit::Spades),
            card(Rank::Five, Suit::Hearts),
            card(Rank::King, Suit::Diamonds),
            card(Rank::Seven, Suit::Clubs),
        ];
        assert_eq!(scoring_indices_simple(&cards, HandType::HighCard), Some(vec![2]));
    }

    #[test]
    fn flush_with_5_cards_returns_all() {
        let cards = [
            card(Rank::Two, Suit::Hearts),
            card(Rank::Five, Suit::Hearts),
            card(Rank::Seven, Suit::Hearts),
            card(Rank::Nine, Suit::Hearts),
            card(Rank::King, Suit::Hearts),
        ];
        assert_eq!(scoring_indices_simple(&cards, HandType::Flush), Some(vec![0, 1, 2, 3, 4]));
    }

    #[test]
    fn stone_card_always_scores() {
        // Stone at index 0 + a Two at index 1 → HighCard hand picks
        // the Two as the highest ranked card, but the Stone is added
        // back as a scoring index. Result: [0, 1] sorted.
        let mut stone = card(Rank::Ace, Suit::Spades);
        stone.enhancement = Enhancement::Stone;
        let cards = [stone, card(Rank::Two, Suit::Hearts)];
        assert_eq!(scoring_indices_simple(&cards, HandType::HighCard), Some(vec![0, 1]));
    }

    #[test]
    fn stone_with_pair_includes_stone_and_pair() {
        // [Stone, Two, Two] as a Pair → pair indices [1, 2] + stone [0]
        // = [0, 1, 2].
        let mut stone = card(Rank::Ace, Suit::Spades);
        stone.enhancement = Enhancement::Stone;
        let cards = [
            stone,
            card(Rank::Two, Suit::Hearts),
            card(Rank::Two, Suit::Spades),
        ];
        assert_eq!(scoring_indices_simple(&cards, HandType::Pair), Some(vec![0, 1, 2]));
    }
}
