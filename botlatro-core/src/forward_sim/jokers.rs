//! Joker scaling-counter updates after a play (Phase 3b of
//! RUST_PORT_PLAN.md).
//!
//! Ports `_jokers_after_play` from
//! `src/balatro_ai/search/forward_sim.py:3054`.
//!
//! Given the current joker metadata + the just-played hand's shape,
//! returns the NEW metadata values for each joker. Python uses these
//! to rebuild Joker objects with updated `.metadata` dicts.
//!
//! Coverage (mirrors Python's `_jokers_after_play`):
//! - **Pure delta**: Green Joker (+1), Square Joker (+4 if 4-card),
//!   Runner (+15 if straight), Spare Trousers (+2 if two-pair-shape),
//!   Wee Joker (+8 per scored-2-trigger), Vampire (+precomputed gain),
//!   Lucky Cat (+0.25 per lucky trigger; default 0).
//! - **Reset-or-delta**: Ride the Bus (0 if any face scored else +1),
//!   Loyalty Card (5 if remaining<=0 else -1), Obelisk (xmult+gain if
//!   should_scale else reset to 1.0).
//! - **Decay-with-removal**: Ice Cream (chips - 5; remove when <= 0),
//!   Seltzer (remaining - 1; remove when <= 0).
//!
//! Bail policy: jokers not in the supported set are returned with
//! `JokerUpdate::default()` (no change). The Python caller then
//! falls back to its own logic for any joker it doesn't see updated.

use pyo3::prelude::*;

use crate::hand_eval::hand_type::HandType;
use crate::state::card::{Card, Rank};

/// Per-joker post-play update. `None` fields mean "unchanged".
/// `remove = true` instructs the caller to drop this joker.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct JokerUpdate {
    pub new_plus_chips: Option<i32>,
    pub new_plus_mult: Option<i32>,
    pub new_xmult: Option<f64>,
    pub new_remaining: Option<i32>,
    pub remove: bool,
}

/// Per-play context the updaters consume.
#[derive(Clone, Debug)]
pub struct AfterPlayContext<'a> {
    pub hand_type: HandType,
    pub played_cards: &'a [Card],
    pub scoring_indices: &'a [usize],
    pub n_played: usize,
    pub hands_remaining: u32,
    pub pareidolia_active: bool,
}

/// Compute per-joker updates. `current_*` arrays are parallel to
/// `joker_names`; `vampire_gain` / `obelisk_should_scale` / etc.
/// are precomputed by the Python wire-in (Vampire's gain depends on
/// pre-play card enhancements which are easier to read in Python).
#[allow(clippy::too_many_arguments)]
pub fn jokers_after_play(
    joker_names: &[String],
    current_plus_chips: &[i32],
    current_plus_mult: &[i32],
    current_xmult: &[f64],
    current_remaining: &[i32],
    vampire_gain: &[f64],
    obelisk_should_scale: &[bool],
    obelisk_gain: &[f64],
    lucky_triggers: u32,
    ctx: AfterPlayContext<'_>,
) -> Vec<JokerUpdate> {
    let n = joker_names.len();
    let mut out = vec![JokerUpdate::default(); n];

    // Pre-compute hand-shape flags used by several jokers.
    let is_straight_shape = matches!(
        ctx.hand_type,
        HandType::Straight | HandType::StraightFlush
    );
    // "Two-pair shape" for Spare Trousers: TwoPair, FullHouse,
    // FlushHouse all contain at least one pair-of-pairs structure.
    let is_two_pair_shape = matches!(
        ctx.hand_type,
        HandType::TwoPair | HandType::FullHouse | HandType::FlushHouse
    );
    let any_face_scored = ctx.scoring_indices.iter().any(|&i| {
        let r = ctx.played_cards[i].rank;
        if ctx.pareidolia_active { true }
        else { matches!(r, Rank::Jack | Rank::Queen | Rank::King) }
    });

    // Wee Joker: sum scored-2 trigger counts. The trigger count
    // formula matches `scored_card_trigger_count` in effects.rs
    // (Hack +1 per 2-5, Seltzer +1, Hanging Chad +2 on first scored).
    // We re-derive it from `joker_names` to avoid duplicating state.
    let scored_2s_trigger_total: u32 = if joker_names.iter().any(|n| n == "Wee Joker") {
        let first_scored = ctx.scoring_indices.first().copied();
        ctx.scoring_indices.iter().filter_map(|&i| {
            if ctx.played_cards[i].rank == Rank::Two {
                let is_first = first_scored == Some(i);
                Some(crate::hand_eval::effects::scored_card_trigger_count(
                    ctx.played_cards[i], is_first, joker_names,
                    ctx.hands_remaining, ctx.pareidolia_active,
                ))
            } else { None }
        }).sum()
    } else { 0 };

    for (i, name) in joker_names.iter().enumerate() {
        let cur_chips = *current_plus_chips.get(i).unwrap_or(&0);
        let cur_mult = *current_plus_mult.get(i).unwrap_or(&0);
        let cur_xmult = *current_xmult.get(i).unwrap_or(&1.0);
        let cur_remaining = *current_remaining.get(i).unwrap_or(&0);
        let upd = &mut out[i];
        match name.as_str() {
            "Green Joker" => {
                upd.new_plus_mult = Some(cur_mult + 1);
            }
            "Ride the Bus" => {
                upd.new_plus_mult = Some(if any_face_scored { 0 } else { cur_mult + 1 });
            }
            "Square Joker" => {
                let gain = if ctx.n_played == 4 { 4 } else { 0 };
                upd.new_plus_chips = Some(cur_chips + gain);
            }
            "Runner" => {
                let gain = if is_straight_shape { 15 } else { 0 };
                upd.new_plus_chips = Some(cur_chips + gain);
            }
            "Spare Trousers" => {
                let gain = if is_two_pair_shape { 2 } else { 0 };
                upd.new_plus_mult = Some(cur_mult + gain);
            }
            "Wee Joker" => {
                upd.new_plus_chips = Some(cur_chips + 8 * (scored_2s_trigger_total as i32));
            }
            "Loyalty Card" => {
                let next = if cur_remaining <= 0 { 5 } else { cur_remaining - 1 };
                upd.new_remaining = Some(next);
            }
            "Vampire" => {
                let gain = *vampire_gain.get(i).unwrap_or(&0.0);
                upd.new_xmult = Some(cur_xmult + gain);
            }
            "Obelisk" => {
                if *obelisk_should_scale.get(i).unwrap_or(&false) {
                    let gain = *obelisk_gain.get(i).unwrap_or(&0.2);
                    upd.new_xmult = Some(cur_xmult + gain);
                } else {
                    upd.new_xmult = Some(1.0);
                }
            }
            "Lucky Cat" => {
                if lucky_triggers > 0 {
                    upd.new_xmult = Some(cur_xmult + 0.25 * (lucky_triggers as f64));
                }
            }
            "Ice Cream" => {
                // Python default is 100 when metadata absent. The
                // wire-in passes that as `current_plus_chips`.
                let next = cur_chips - 5;
                if next > 0 {
                    upd.new_plus_chips = Some(next);
                } else {
                    upd.remove = true;
                }
            }
            "Seltzer" => {
                // Decay 1 per play; remove when at 0.
                let next = cur_remaining - 1;
                if next > 0 {
                    upd.new_remaining = Some(next);
                } else {
                    upd.remove = true;
                }
            }
            _ => {} // unchanged — Python's else-branch
        }
    }
    out
}

/// PyO3 wrapper. Returns five parallel optional lists + a remove
/// list. Python applies non-None values to update joker metadata.
#[pyfunction]
#[pyo3(name = "jokers_after_play")]
#[allow(clippy::too_many_arguments)]
pub fn py_jokers_after_play(
    joker_names: Vec<String>,
    current_plus_chips: Vec<i32>,
    current_plus_mult: Vec<i32>,
    current_xmult: Vec<f64>,
    current_remaining: Vec<i32>,
    vampire_gain: Vec<f64>,
    obelisk_should_scale: Vec<bool>,
    obelisk_gain: Vec<f64>,
    lucky_triggers: u32,
    hand_type_str: &str,
    played_cards: Vec<Card>,
    scoring_indices: Vec<usize>,
    hands_remaining: u32,
    pareidolia_active: bool,
) -> Option<(
    Vec<Option<i32>>,  // new_plus_chips
    Vec<Option<i32>>,  // new_plus_mult
    Vec<Option<f64>>,  // new_xmult
    Vec<Option<i32>>,  // new_remaining
    Vec<bool>,         // remove
)> {
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
        _ => return None, // unknown hand type — bail to Python
    };
    let ctx = AfterPlayContext {
        hand_type,
        played_cards: &played_cards,
        scoring_indices: &scoring_indices,
        n_played: played_cards.len(),
        hands_remaining,
        pareidolia_active,
    };
    let updates = jokers_after_play(
        &joker_names,
        &current_plus_chips,
        &current_plus_mult,
        &current_xmult,
        &current_remaining,
        &vampire_gain,
        &obelisk_should_scale,
        &obelisk_gain,
        lucky_triggers,
        ctx,
    );
    let n = updates.len();
    let mut new_chips = Vec::with_capacity(n);
    let mut new_mult = Vec::with_capacity(n);
    let mut new_xmult = Vec::with_capacity(n);
    let mut new_remaining = Vec::with_capacity(n);
    let mut remove = Vec::with_capacity(n);
    for u in updates {
        new_chips.push(u.new_plus_chips);
        new_mult.push(u.new_plus_mult);
        new_xmult.push(u.new_xmult);
        new_remaining.push(u.new_remaining);
        remove.push(u.remove);
    }
    Some((new_chips, new_mult, new_xmult, new_remaining, remove))
}

/// After-discard joker counter updates. Ports
/// `_jokers_after_discard` from `forward_sim.py:3208`.
///
/// Covered:
/// - Ramen: xmult -= 0.01 per discard; if drops to <=1, REMOVE.
/// - Green Joker: max(0, mult - 1).
/// - Hit the Road: xmult += 0.5 per Jack discarded.
/// - Castle: chips += 3 per discarded card matching target_suit.
/// - Yorick: per discard, if remaining<=1 → remaining=23 + xmult+1,
///   else remaining-=1.
#[allow(clippy::too_many_arguments)]
pub fn jokers_after_discard(
    joker_names: &[String],
    current_plus_chips: &[i32],
    current_plus_mult: &[i32],
    current_xmult: &[f64],
    current_remaining: &[i32],
    target_suit: &[Option<u8>],
    discarded_cards: &[Card],
) -> Vec<JokerUpdate> {
    use crate::state::card::Rank;
    let n = joker_names.len();
    let mut out = vec![JokerUpdate::default(); n];

    let jack_count: i32 = discarded_cards.iter()
        .filter(|c| !c.debuffed && c.rank == Rank::Jack)
        .count() as i32;

    for (i, name) in joker_names.iter().enumerate() {
        let cur_chips = *current_plus_chips.get(i).unwrap_or(&0);
        let cur_mult = *current_plus_mult.get(i).unwrap_or(&0);
        let cur_xmult = *current_xmult.get(i).unwrap_or(&1.0);
        let cur_remaining = *current_remaining.get(i).unwrap_or(&0);
        let upd = &mut out[i];
        match name.as_str() {
            "Green Joker" => {
                upd.new_plus_mult = Some((cur_mult - 1).max(0));
            }
            "Hit the Road" => {
                upd.new_xmult = Some(cur_xmult + 0.5 * (jack_count as f64));
            }
            "Castle" => {
                if let Some(Some(target)) = target_suit.get(i) {
                    let gain = 3 * discarded_cards.iter()
                        .filter(|c| !c.debuffed && c.suit as u8 == *target)
                        .count() as i32;
                    upd.new_plus_chips = Some(cur_chips + gain);
                }
                // No target → no change (Python's `continue`).
            }
            "Ramen" => {
                // Python: for _ in discarded_cards: if current - 0.01 <= 1: eaten; else current -= 0.01
                let mut current = if cur_xmult == 0.0 { 2.0 } else { cur_xmult };
                let mut eaten = false;
                for _ in discarded_cards {
                    if current - 0.01 <= 1.0 {
                        eaten = true;
                        break;
                    }
                    current -= 0.01;
                }
                if eaten {
                    upd.remove = true;
                } else {
                    upd.new_xmult = Some(current);
                }
            }
            "Yorick" => {
                let mut remaining = cur_remaining;
                let mut xmult = cur_xmult;
                for _ in discarded_cards {
                    if remaining <= 1 {
                        remaining = 23;
                        xmult += 1.0;
                    } else {
                        remaining -= 1;
                    }
                }
                upd.new_remaining = Some(remaining);
                upd.new_xmult = Some(xmult);
            }
            _ => {}
        }
    }
    out
}

/// PyO3 wrapper for `jokers_after_discard`.
#[pyfunction]
#[pyo3(name = "jokers_after_discard")]
#[allow(clippy::too_many_arguments)]
pub fn py_jokers_after_discard(
    joker_names: Vec<String>,
    current_plus_chips: Vec<i32>,
    current_plus_mult: Vec<i32>,
    current_xmult: Vec<f64>,
    current_remaining: Vec<i32>,
    target_suit: Vec<Option<String>>,
    discarded_cards: Vec<Card>,
) -> (
    Vec<Option<i32>>,
    Vec<Option<i32>>,
    Vec<Option<f64>>,
    Vec<Option<i32>>,
    Vec<bool>,
) {
    use crate::state::card::Suit;
    let target_suit_u8: Vec<Option<u8>> = target_suit.iter()
        .map(|s| s.as_deref().and_then(Suit::from_str).map(|s| s as u8))
        .collect();
    let updates = jokers_after_discard(
        &joker_names,
        &current_plus_chips,
        &current_plus_mult,
        &current_xmult,
        &current_remaining,
        &target_suit_u8,
        &discarded_cards,
    );
    let n = updates.len();
    let mut new_chips = Vec::with_capacity(n);
    let mut new_mult = Vec::with_capacity(n);
    let mut new_xmult = Vec::with_capacity(n);
    let mut new_remaining = Vec::with_capacity(n);
    let mut remove = Vec::with_capacity(n);
    for u in updates {
        new_chips.push(u.new_plus_chips);
        new_mult.push(u.new_plus_mult);
        new_xmult.push(u.new_xmult);
        new_remaining.push(u.new_remaining);
        remove.push(u.remove);
    }
    (new_chips, new_mult, new_xmult, new_remaining, remove)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::card::{Edition, Enhancement, Seal, Suit};

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

    fn ctx<'a>(ht: HandType, played: &'a [Card], scoring: &'a [usize]) -> AfterPlayContext<'a> {
        AfterPlayContext {
            hand_type: ht,
            played_cards: played,
            scoring_indices: scoring,
            n_played: played.len(),
            hands_remaining: 4,
            pareidolia_active: false,
        }
    }

    #[test]
    fn green_joker_increments_mult() {
        let names = vec!["Green Joker".to_string()];
        let played = [card(Rank::Ace, Suit::Hearts)];
        let scoring = [0];
        let upd = jokers_after_play(
            &names, &[0], &[3], &[1.0], &[0],
            &[0.0], &[false], &[0.0], 0,
            ctx(HandType::HighCard, &played, &scoring),
        );
        assert_eq!(upd[0].new_plus_mult, Some(4));
    }

    #[test]
    fn ride_the_bus_resets_on_face() {
        let names = vec!["Ride the Bus".to_string()];
        let played = [card(Rank::King, Suit::Hearts)];
        let scoring = [0];
        let upd = jokers_after_play(
            &names, &[0], &[7], &[1.0], &[0],
            &[0.0], &[false], &[0.0], 0,
            ctx(HandType::HighCard, &played, &scoring),
        );
        assert_eq!(upd[0].new_plus_mult, Some(0));
    }

    #[test]
    fn ride_the_bus_increments_without_face() {
        let names = vec!["Ride the Bus".to_string()];
        let played = [card(Rank::Two, Suit::Hearts)];
        let scoring = [0];
        let upd = jokers_after_play(
            &names, &[0], &[7], &[1.0], &[0],
            &[0.0], &[false], &[0.0], 0,
            ctx(HandType::HighCard, &played, &scoring),
        );
        assert_eq!(upd[0].new_plus_mult, Some(8));
    }

    #[test]
    fn ice_cream_decays_until_removed() {
        let names = vec!["Ice Cream".to_string()];
        let played = [card(Rank::Ace, Suit::Hearts)];
        let scoring = [0];
        let upd = jokers_after_play(
            &names, &[5], &[0], &[1.0], &[0],
            &[0.0], &[false], &[0.0], 0,
            ctx(HandType::HighCard, &played, &scoring),
        );
        assert!(upd[0].remove); // 5 - 5 = 0, not > 0, so remove.

        let upd2 = jokers_after_play(
            &names, &[10], &[0], &[1.0], &[0],
            &[0.0], &[false], &[0.0], 0,
            ctx(HandType::HighCard, &played, &scoring),
        );
        assert_eq!(upd2[0].new_plus_chips, Some(5));
        assert!(!upd2[0].remove);
    }

    #[test]
    fn loyalty_card_cycle() {
        let names = vec!["Loyalty Card".to_string()];
        let played = [card(Rank::Ace, Suit::Hearts)];
        let scoring = [0];
        // remaining = 0 → reset to 5
        let upd = jokers_after_play(
            &names, &[0], &[0], &[1.0], &[0],
            &[0.0], &[false], &[0.0], 0,
            ctx(HandType::HighCard, &played, &scoring),
        );
        assert_eq!(upd[0].new_remaining, Some(5));
        // remaining = 5 → 4
        let upd2 = jokers_after_play(
            &names, &[0], &[0], &[1.0], &[5],
            &[0.0], &[false], &[0.0], 0,
            ctx(HandType::HighCard, &played, &scoring),
        );
        assert_eq!(upd2[0].new_remaining, Some(4));
    }

    #[test]
    fn obelisk_scales_or_resets() {
        let names = vec!["Obelisk".to_string()];
        let played = [card(Rank::Ace, Suit::Hearts)];
        let scoring = [0];
        // should scale → +gain
        let upd = jokers_after_play(
            &names, &[0], &[0], &[1.4], &[0],
            &[0.0], &[true], &[0.2], 0,
            ctx(HandType::HighCard, &played, &scoring),
        );
        // 1.4 + 0.2 = 1.6 ± float error
        let v = upd[0].new_xmult.unwrap();
        assert!((v - 1.6).abs() < 1e-9, "got {v}");
        // should NOT scale → reset to 1.0
        let upd2 = jokers_after_play(
            &names, &[0], &[0], &[2.0], &[0],
            &[0.0], &[false], &[0.2], 0,
            ctx(HandType::HighCard, &played, &scoring),
        );
        assert_eq!(upd2[0].new_xmult, Some(1.0));
    }

    #[test]
    fn unknown_joker_returns_default() {
        let names = vec!["Future Joker".to_string()];
        let played = [card(Rank::Ace, Suit::Hearts)];
        let scoring = [0];
        let upd = jokers_after_play(
            &names, &[0], &[0], &[1.0], &[0],
            &[0.0], &[false], &[0.0], 0,
            ctx(HandType::HighCard, &played, &scoring),
        );
        assert_eq!(upd[0], JokerUpdate::default());
    }
}
