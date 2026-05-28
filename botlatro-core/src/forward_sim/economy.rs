//! End-of-round economy (Phase 3e of RUST_PORT_PLAN.md).
//!
//! Ports the held-card-money portion of `simulate_play`:
//!   `_held_end_of_round_money_delta(held, jokers)` →
//!   3 × (1 + Mime count) × (held Gold cards).
//!
//! Gift card / blue seal / cash-out interest are larger pieces
//! that may be added here as their own helpers later.

use pyo3::prelude::*;

use crate::state::card::{Card, Enhancement};

/// Held Gold cards each yield $3 at end of round; Mime jokers
/// retrigger that. Matches `_held_end_of_round_money_delta`.
pub fn held_end_of_round_money_delta(
    held_cards: &[Card],
    mime_count: u32,
) -> i32 {
    let trigger_count = 1 + mime_count as i32;
    let gold_count = held_cards.iter()
        .filter(|c| !c.debuffed && c.enhancement == Enhancement::Gold)
        .count() as i32;
    3 * trigger_count * gold_count
}

/// PyO3 wrapper. Caller passes the precomputed Mime count
/// (Python's `_joker_is_disabled` filter is too detailed to
/// duplicate in Rust for this small helper).
#[pyfunction]
#[pyo3(name = "held_end_of_round_money_delta")]
pub fn py_held_end_of_round_money_delta(
    held_cards: Vec<Card>,
    mime_count: u32,
) -> i32 {
    held_end_of_round_money_delta(&held_cards, mime_count)
}

/// Money change from a discard. Ports `_discard_money_delta` at
/// `forward_sim.py:3377`.
///
/// Components:
/// - **Trading Card**: +N (default $3) if first discard AND
///   exactly one card discarded.
/// - **Faceless Joker**: +$5 if 3+ face cards discarded.
/// - **Mail-In Rebate**: +$5 per non-debuffed discarded card whose
///   rank matches the joker's target_rank.
///
/// The caller (Python wire-in) supplies the relevant flags +
/// counts so this function stays free of metadata-reading logic.
pub fn discard_money_delta(
    discarded_cards: &[Card],
    pareidolia_active: bool,
    has_trading_card_first_discard: bool,
    trading_card_amount: i32,
    has_faceless_joker: bool,
    mail_in_rebate_target_rank: Option<crate::state::card::Rank>,
) -> i32 {
    use crate::state::card::Rank;
    let mut delta: i32 = 0;
    if has_trading_card_first_discard && discarded_cards.len() == 1 {
        delta += trading_card_amount;
    }
    if has_faceless_joker {
        let face_count = discarded_cards.iter()
            .filter(|c| {
                if pareidolia_active { true }
                else { matches!(c.rank, Rank::Jack | Rank::Queen | Rank::King) }
            })
            .count();
        if face_count >= 3 {
            delta += 5;
        }
    }
    if let Some(target) = mail_in_rebate_target_rank {
        let match_count = discarded_cards.iter()
            .filter(|c| !c.debuffed && c.rank == target)
            .count() as i32;
        delta += 5 * match_count;
    }
    delta
}

/// PyO3 wrapper. target_rank as Option<&str>.
#[pyfunction]
#[pyo3(name = "discard_money_delta")]
#[allow(clippy::too_many_arguments)]
pub fn py_discard_money_delta(
    discarded_cards: Vec<Card>,
    pareidolia_active: bool,
    has_trading_card_first_discard: bool,
    trading_card_amount: i32,
    has_faceless_joker: bool,
    mail_in_rebate_target_rank: Option<String>,
) -> i32 {
    let target = mail_in_rebate_target_rank
        .as_deref()
        .and_then(crate::state::card::Rank::from_str);
    discard_money_delta(
        &discarded_cards,
        pareidolia_active,
        has_trading_card_first_discard,
        trading_card_amount,
        has_faceless_joker,
        target,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::card::{Edition, Rank, Seal, Suit};

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

    fn gold(rank: Rank, suit: Suit) -> Card {
        Card { enhancement: Enhancement::Gold, ..card(rank, suit) }
    }

    #[test]
    fn no_gold_no_money() {
        let held = [card(Rank::Ace, Suit::Hearts)];
        assert_eq!(held_end_of_round_money_delta(&held, 0), 0);
    }

    #[test]
    fn three_per_gold_no_mime() {
        let held = [gold(Rank::Ace, Suit::Hearts), gold(Rank::Two, Suit::Spades)];
        assert_eq!(held_end_of_round_money_delta(&held, 0), 6);
    }

    #[test]
    fn mime_doubles_payout() {
        let held = [gold(Rank::Ace, Suit::Hearts)];
        // 3 * (1 + 1) * 1 = 6
        assert_eq!(held_end_of_round_money_delta(&held, 1), 6);
    }

    #[test]
    fn debuffed_gold_skipped() {
        let mut g = gold(Rank::Ace, Suit::Hearts);
        g.debuffed = true;
        assert_eq!(held_end_of_round_money_delta(&[g], 0), 0);
    }
}
