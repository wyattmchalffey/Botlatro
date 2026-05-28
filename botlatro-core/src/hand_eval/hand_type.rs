//! Poker hand type identification (Phase 2a of RUST_PORT_PLAN.md).
//!
//! Ports `_identify_hand_type` from `hand_evaluator.py:394`.
//!
//! Coverage: the "fast path" — no Stone cards, no Wild cards, no
//! Smeared/Four Fingers/Shortcut jokers. For hands that contain
//! any of those, the Rust function returns `None` and the Python
//! wrapper falls back to the original Python implementation.
//!
//! Rationale: those special cases account for <5% of evaluated
//! hands in typical solver runs (most opening hands are vanilla
//! 8-card draws with no joker yet); porting them adds complexity
//! and parity risk that's not worth the marginal speedup.
//!
//! When/if profiling shows the Python fallback becoming a
//! bottleneck (e.g. mid-run with several enhancement jokers), we
//! extend the Rust function to handle those cases.

use pyo3::prelude::*;

use crate::state::card::{Card, Enhancement, Rank};

/// The 12 hand types Balatro recognizes. Variants and their
/// string representations match `HandType` in
/// `hand_evaluator.py:25`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum HandType {
    HighCard = 0,
    Pair = 1,
    TwoPair = 2,
    ThreeOfAKind = 3,
    Straight = 4,
    Flush = 5,
    FullHouse = 6,
    FourOfAKind = 7,
    StraightFlush = 8,
    FiveOfAKind = 9,
    FlushHouse = 10,
    FlushFive = 11,
}

impl HandType {
    /// String form matching the Python enum's `.value`. Used by
    /// the PyO3 wrapper so callers can compare against Python's
    /// `HandType.PAIR.value` etc.
    pub fn to_str(self) -> &'static str {
        match self {
            Self::HighCard => "High Card",
            Self::Pair => "Pair",
            Self::TwoPair => "Two Pair",
            Self::ThreeOfAKind => "Three of a Kind",
            Self::Straight => "Straight",
            Self::Flush => "Flush",
            Self::FullHouse => "Full House",
            Self::FourOfAKind => "Four of a Kind",
            Self::StraightFlush => "Straight Flush",
            Self::FiveOfAKind => "Five of a Kind",
            Self::FlushHouse => "Flush House",
            Self::FlushFive => "Flush Five",
        }
    }
}

/// Identify the hand type for the simple case: no Stone cards, no
/// Wild cards, no Four Fingers / Smeared / Shortcut jokers.
///
/// Returns `None` if any of the above conditions are present —
/// caller should fall back to the Python `_identify_hand_type`.
///
/// The classification logic mirrors `hand_evaluator.py:394-432`:
/// 1. Count card ranks (Counter equivalent)
/// 2. Determine is_flush (all same suit, ≥5 cards) and is_straight
///    (consecutive ranks, ≥5 cards, with A-2-3-4-5 wheel exception)
/// 3. Apply the classification ladder (flush_five > flush_house >
///    five_of_a_kind > straight_flush > four_of_a_kind > full_house
///    > flush > straight > three_of_a_kind > two_pair > pair > high_card)
pub fn identify_hand_type_simple(cards: &[Card]) -> Option<HandType> {
    // Wild cards still bail — they can stand in for any suit which
    // requires a much more involved flush/identification path.
    for c in cards {
        if matches!(c.enhancement, Enhancement::Wild) {
            return None;
        }
    }

    if cards.is_empty() {
        return Some(HandType::HighCard);
    }

    // Stone cards are filtered out for identification — they don't
    // have a rank/suit for poker-shape detection. They still score
    // (via 50 chips) and are added back by scoring_indices_simple.
    let ranked: Vec<Card> = cards.iter()
        .copied()
        .filter(|c| c.enhancement != Enhancement::Stone)
        .collect();

    // All-stone hand → HighCard (mirrors Python's "no ranked cards"
    // return-HIGH_CARD branch).
    if ranked.is_empty() {
        return Some(HandType::HighCard);
    }

    // Rank counts: bucket by enum discriminant for cheap counting.
    let mut rank_counts = [0u8; 16]; // indexed by Rank as u8
    for c in &ranked {
        rank_counts[c.rank as usize] += 1;
    }

    let mut counts_sorted: Vec<u8> = rank_counts.iter().copied().filter(|&n| n > 0).collect();
    counts_sorted.sort_unstable();
    let max_count = *counts_sorted.last().unwrap_or(&0);
    let pair_count = counts_sorted.iter().filter(|&&n| n == 2).count();

    let is_flush = ranked.len() >= 5 && all_same_suit(&ranked);
    let is_straight = ranked.len() >= 5 && is_straight_simple(&ranked);

    // Classification ladder (top-down — first matching wins).
    // Note: comparisons against "len" use ranked.len(), matching
    // Python's `len(ranked_cards)` after stone-filtering.
    if is_flush && max_count == ranked.len() as u8 && ranked.len() >= 5 {
        return Some(HandType::FlushFive);
    }
    if is_flush && ranked.len() == 5 && counts_sorted == [2, 3] {
        return Some(HandType::FlushHouse);
    }
    if max_count == ranked.len() as u8 && ranked.len() >= 5 {
        return Some(HandType::FiveOfAKind);
    }
    if is_flush && is_straight {
        return Some(HandType::StraightFlush);
    }
    if max_count == 4 {
        return Some(HandType::FourOfAKind);
    }
    if ranked.len() == 5 && counts_sorted == [2, 3] {
        return Some(HandType::FullHouse);
    }
    if is_flush {
        return Some(HandType::Flush);
    }
    if is_straight {
        return Some(HandType::Straight);
    }
    if max_count == 3 {
        return Some(HandType::ThreeOfAKind);
    }
    if pair_count == 2 {
        return Some(HandType::TwoPair);
    }
    if pair_count == 1 {
        return Some(HandType::Pair);
    }
    Some(HandType::HighCard)
}

/// All cards share one suit. Simple case — no Wild handling.
#[inline]
fn all_same_suit(cards: &[Card]) -> bool {
    let first = cards[0].suit;
    cards.iter().all(|c| c.suit == first)
}

/// Identification modifiers from jokers. Default values mean the
/// vanilla path. Mirrors Python's joker-aware identification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct IdJokers {
    /// Smeared Joker: Hearts/Diamonds count as same suit; Clubs/
    /// Spades count as same suit.
    pub smeared: bool,
    /// Four Fingers: flushes and straights need only 4 cards.
    pub four_fingers: bool,
    /// Shortcut: straights may skip exactly one rank between
    /// consecutive values.
    pub shortcut: bool,
}

/// "Suit key" for flush detection — collapses to a color when
/// Smeared is active.
#[inline]
pub fn flush_suit_key(suit: crate::state::card::Suit, smeared: bool) -> u8 {
    use crate::state::card::Suit;
    if smeared {
        match suit {
            Suit::Hearts | Suit::Diamonds => 0, // Red
            Suit::Clubs | Suit::Spades => 1,    // Black
        }
    } else {
        suit as u8
    }
}

/// True if the given (ranked) cards share a suit key. With Smeared
/// active, "share a suit" becomes "share a color".
fn all_same_suit_with_jokers(cards: &[Card], smeared: bool) -> bool {
    if cards.is_empty() {
        return true;
    }
    let first = flush_suit_key(cards[0].suit, smeared);
    cards.iter().all(|c| flush_suit_key(c.suit, smeared) == first)
}

/// Straight detection that respects Shortcut. Mirrors Python's
/// `_is_straight(cards, shortcut=...)` at hand_evaluator.py:435.
/// Stone-filtering is the caller's responsibility.
fn is_straight_with_jokers(cards: &[Card], shortcut: bool) -> bool {
    let mut values: Vec<u8> = cards.iter().map(|c| c.rank as u8).collect();
    values.sort_unstable();
    values.dedup();
    if values.len() != cards.len() {
        return false; // duplicates → not a straight
    }
    // 5-card wheel: A-2-3-4-5
    if values.len() == 5 && values == [2, 3, 4, 5, 14] {
        return true;
    }
    // 4-card wheel: A-2-3-4 (Four Fingers).
    if values.len() == 4 && values == [2, 3, 4, 14] {
        return true;
    }
    if shortcut {
        // Consecutive gaps must each be 1 or 2.
        return values
            .windows(2)
            .all(|w| w[1] >= w[0] + 1 && w[1] <= w[0] + 2);
    }
    let lo = values[0];
    values.iter().enumerate().all(|(i, &v)| v == lo + i as u8)
}

/// Identification with joker-aware modifiers. When all flags are
/// false (and no Wild cards), behaves identically to
/// `identify_hand_type_simple`. Otherwise handles Smeared / Four
/// Fingers / Shortcut / Wild.
pub fn identify_hand_type_with_jokers(
    cards: &[Card],
    j: IdJokers,
) -> Option<HandType> {
    if cards.is_empty() {
        return Some(HandType::HighCard);
    }

    // Filter out stone cards for identification.
    let ranked: Vec<Card> = cards.iter()
        .copied()
        .filter(|c| c.enhancement != Enhancement::Stone)
        .collect();
    if ranked.is_empty() {
        return Some(HandType::HighCard);
    }

    let needed_flush = if j.four_fingers { 4 } else { 5 };
    let needed_straight = if j.four_fingers { 4 } else { 5 };

    // Rank counts (over ranked cards).
    let mut rank_counts = [0u8; 16];
    for c in &ranked {
        rank_counts[c.rank as usize] += 1;
    }
    let mut counts_sorted: Vec<u8> = rank_counts.iter().copied().filter(|&n| n > 0).collect();
    counts_sorted.sort_unstable();
    let max_count = *counts_sorted.last().unwrap_or(&0);
    let pair_count = counts_sorted.iter().filter(|&&n| n == 2).count();

    // Flush: pick the largest suit-key group (≥ needed_flush).
    // Wild cards count for EVERY suit key.
    let is_flush = ranked.len() >= needed_flush
        && largest_suit_group_size_wild(&ranked, j.smeared) >= needed_flush;

    // Straight: try the best subset of `needed_straight` cards.
    let is_straight = ranked.len() >= needed_straight
        && best_straight_subset_size(&ranked, needed_straight, j.shortcut).is_some();

    // Classification ladder mirrors `identify_hand_type_simple` but
    // uses the joker-aware flags. Note: FlushFive / FlushHouse /
    // FiveOfAKind still require ALL 5 cards to match — Four Fingers
    // doesn't help here (Python: `is_flush and max_count == len(...) and len() >= 5`).
    if is_flush && max_count == ranked.len() as u8 && ranked.len() >= 5 {
        return Some(HandType::FlushFive);
    }
    if is_flush && ranked.len() == 5 && counts_sorted == [2, 3] {
        return Some(HandType::FlushHouse);
    }
    if max_count == ranked.len() as u8 && ranked.len() >= 5 {
        return Some(HandType::FiveOfAKind);
    }
    if is_flush && is_straight {
        return Some(HandType::StraightFlush);
    }
    if max_count == 4 {
        return Some(HandType::FourOfAKind);
    }
    if ranked.len() == 5 && counts_sorted == [2, 3] {
        return Some(HandType::FullHouse);
    }
    if is_flush {
        return Some(HandType::Flush);
    }
    if is_straight {
        return Some(HandType::Straight);
    }
    if max_count == 3 {
        return Some(HandType::ThreeOfAKind);
    }
    if pair_count == 2 {
        return Some(HandType::TwoPair);
    }
    if pair_count == 1 {
        return Some(HandType::Pair);
    }
    Some(HandType::HighCard)
}

/// Size of the largest suit-key group among `cards`.
fn largest_suit_group_size(cards: &[Card], smeared: bool) -> usize {
    let mut counts = [0usize; 4];
    for c in cards {
        counts[flush_suit_key(c.suit, smeared) as usize] += 1;
    }
    *counts.iter().max().unwrap_or(&0)
}

/// Same as `largest_suit_group_size` but Wild cards count for EVERY
/// suit key — they could stand in for any suit. With Smeared active,
/// "every suit" collapses to the 2 color keys, so Wild adds 1 to
/// both color counts.
pub fn largest_suit_group_size_wild(cards: &[Card], smeared: bool) -> usize {
    let n_keys: usize = if smeared { 2 } else { 4 };
    let mut counts = [0usize; 4];
    for c in cards {
        if c.enhancement == Enhancement::Wild {
            for i in 0..n_keys { counts[i] += 1; }
        } else {
            counts[flush_suit_key(c.suit, smeared) as usize] += 1;
        }
    }
    *counts[..n_keys].iter().max().unwrap_or(&0)
}

/// True if some `size`-sized subset of `cards` forms a straight
/// (respecting shortcut). Returns the size of the matching subset
/// when found (helpful for downstream scoring-index selection).
fn best_straight_subset_size(cards: &[Card], size: usize, shortcut: bool) -> Option<usize> {
    let n = cards.len();
    if size > n {
        return None;
    }
    if size == n {
        return if is_straight_with_jokers(cards, shortcut) { Some(size) } else { None };
    }
    // Try all C(n, size) subsets. For our hot path (n ≤ 5, size ≤ 5),
    // this is at most C(5,4) = 5 subsets.
    let mut idx = vec![0usize; size];
    for i in 0..size { idx[i] = i; }
    loop {
        let subset: Vec<Card> = idx.iter().map(|&i| cards[i]).collect();
        if is_straight_with_jokers(&subset, shortcut) {
            return Some(size);
        }
        // Advance the lexicographically-next combination.
        let mut k: Option<usize> = None;
        for i in (0..size).rev() {
            if idx[i] < n - size + i {
                k = Some(i);
                break;
            }
        }
        let Some(k) = k else { return None };
        idx[k] += 1;
        for i in k+1..size {
            idx[i] = idx[i-1] + 1;
        }
    }
}

/// Straight detection for the simple case. Ports the no-shortcut
/// path of `_is_straight` (hand_evaluator.py:435).
///
/// Special cases:
/// - 5 cards with values {2,3,4,5,14} (Ace-low straight, the
///   "wheel") → straight
/// - Otherwise consecutive: values == range(min, min+len)
fn is_straight_simple(cards: &[Card]) -> bool {
    // Get unique rank "straight values" (Ace counted as 14).
    let mut values: Vec<u8> = cards.iter().map(|c| c.rank as u8).collect();
    values.sort_unstable();
    values.dedup();
    if values.len() != cards.len() {
        return false; // duplicates → not a straight
    }
    // Wheel: A-2-3-4-5
    if values.len() == 5 && values == [2, 3, 4, 5, 14] {
        return true;
    }
    // Standard: consecutive
    let lo = values[0];
    values.iter().enumerate().all(|(i, &v)| v == lo + i as u8)
}

/// PyO3 wrapper: accept a Python list of `RustCard` objects, return
/// the HandType as a string (matching Python's `HandType` enum
/// `.value`), or None if the simple fast path doesn't apply.
///
/// The Python caller pattern is:
///     hand_type_str = balatro_core.identify_hand_type(rust_cards)
///     if hand_type_str is None:
///         # Fall back to Python _identify_hand_type for special cases
///         ...
#[pyfunction]
#[pyo3(name = "identify_hand_type")]
pub fn py_identify_hand_type(cards: Vec<Card>) -> Option<&'static str> {
    identify_hand_type_simple(&cards).map(HandType::to_str)
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
    fn empty_hand_is_high_card() {
        assert_eq!(identify_hand_type_simple(&[]), Some(HandType::HighCard));
    }

    #[test]
    fn single_card_is_high_card() {
        let cards = [card(Rank::Ace, Suit::Spades)];
        assert_eq!(identify_hand_type_simple(&cards), Some(HandType::HighCard));
    }

    #[test]
    fn pair_of_aces() {
        let cards = [
            card(Rank::Ace, Suit::Spades),
            card(Rank::Ace, Suit::Hearts),
        ];
        assert_eq!(identify_hand_type_simple(&cards), Some(HandType::Pair));
    }

    #[test]
    fn two_pair() {
        let cards = [
            card(Rank::Ace, Suit::Spades),
            card(Rank::Ace, Suit::Hearts),
            card(Rank::King, Suit::Diamonds),
            card(Rank::King, Suit::Clubs),
        ];
        assert_eq!(identify_hand_type_simple(&cards), Some(HandType::TwoPair));
    }

    #[test]
    fn five_card_straight() {
        let cards = [
            card(Rank::Five, Suit::Hearts),
            card(Rank::Six, Suit::Diamonds),
            card(Rank::Seven, Suit::Clubs),
            card(Rank::Eight, Suit::Spades),
            card(Rank::Nine, Suit::Hearts),
        ];
        assert_eq!(identify_hand_type_simple(&cards), Some(HandType::Straight));
    }

    #[test]
    fn ace_low_wheel() {
        // A-2-3-4-5 is a straight (the "wheel")
        let cards = [
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Two, Suit::Diamonds),
            card(Rank::Three, Suit::Clubs),
            card(Rank::Four, Suit::Spades),
            card(Rank::Five, Suit::Hearts),
        ];
        assert_eq!(identify_hand_type_simple(&cards), Some(HandType::Straight));
    }

    #[test]
    fn five_card_flush() {
        let cards = [
            card(Rank::Two, Suit::Hearts),
            card(Rank::Five, Suit::Hearts),
            card(Rank::Seven, Suit::Hearts),
            card(Rank::Nine, Suit::Hearts),
            card(Rank::King, Suit::Hearts),
        ];
        assert_eq!(identify_hand_type_simple(&cards), Some(HandType::Flush));
    }

    #[test]
    fn straight_flush_detected_over_separate_straight_or_flush() {
        let cards = [
            card(Rank::Five, Suit::Hearts),
            card(Rank::Six, Suit::Hearts),
            card(Rank::Seven, Suit::Hearts),
            card(Rank::Eight, Suit::Hearts),
            card(Rank::Nine, Suit::Hearts),
        ];
        assert_eq!(identify_hand_type_simple(&cards), Some(HandType::StraightFlush));
    }

    #[test]
    fn full_house() {
        let cards = [
            card(Rank::Ace, Suit::Spades),
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Ace, Suit::Diamonds),
            card(Rank::King, Suit::Clubs),
            card(Rank::King, Suit::Hearts),
        ];
        assert_eq!(identify_hand_type_simple(&cards), Some(HandType::FullHouse));
    }

    #[test]
    fn four_of_a_kind() {
        let cards = [
            card(Rank::Ace, Suit::Spades),
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Ace, Suit::Diamonds),
            card(Rank::Ace, Suit::Clubs),
            card(Rank::King, Suit::Hearts),
        ];
        assert_eq!(identify_hand_type_simple(&cards), Some(HandType::FourOfAKind));
    }

    #[test]
    fn flush_five_outranks_five_of_a_kind() {
        let cards = [
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Ace, Suit::Hearts),
        ];
        assert_eq!(identify_hand_type_simple(&cards), Some(HandType::FlushFive));
    }

    #[test]
    fn flush_house() {
        let cards = [
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Ace, Suit::Hearts),
            card(Rank::Ace, Suit::Hearts),
            card(Rank::King, Suit::Hearts),
            card(Rank::King, Suit::Hearts),
        ];
        assert_eq!(identify_hand_type_simple(&cards), Some(HandType::FlushHouse));
    }

    #[test]
    fn stone_card_excluded_from_identification() {
        // Stone cards are filtered out of identification — they have
        // no rank/suit. A Stone + a Two should identify as HighCard
        // (driven by the Two alone), NOT as a pair.
        let mut stone = card(Rank::Ace, Suit::Spades);
        stone.enhancement = Enhancement::Stone;
        let cards = [stone, card(Rank::Two, Suit::Hearts)];
        assert_eq!(identify_hand_type_simple(&cards), Some(HandType::HighCard));
    }

    #[test]
    fn stone_pair_identifies_via_ranked_only() {
        // Pair of Twos + 1 stone → still Pair (stone doesn't count).
        let mut stone = card(Rank::Ace, Suit::Spades);
        stone.enhancement = Enhancement::Stone;
        let cards = [
            stone,
            card(Rank::Two, Suit::Hearts),
            card(Rank::Two, Suit::Spades),
        ];
        assert_eq!(identify_hand_type_simple(&cards), Some(HandType::Pair));
    }

    #[test]
    fn all_stone_hand_is_high_card() {
        // 5 stones, no ranked cards → HighCard (Python behavior).
        let mut s = card(Rank::Two, Suit::Hearts);
        s.enhancement = Enhancement::Stone;
        let cards = [s; 5];
        assert_eq!(identify_hand_type_simple(&cards), Some(HandType::HighCard));
    }

    #[test]
    fn wild_card_falls_back_to_none() {
        let mut wild = card(Rank::Ace, Suit::Spades);
        wild.enhancement = Enhancement::Wild;
        let cards = [wild, card(Rank::Two, Suit::Hearts)];
        assert_eq!(identify_hand_type_simple(&cards), None);
    }
}
