//! Direct best-play finder (Phase: all-native rewrite, the 10-30x lever).
//!
//! Instead of brute-forcing all C(8,1..5)=218 card subsets and scoring each
//! (what `best_play_action` / `top_k_plays` do today, and what the rollouts
//! call thousands of times per game), generate only the CANONICAL best subset
//! per achievable hand type (~10-20 candidates), score those, return the max.
//!
//! Correctness strategy: the generator must produce a SUPERSET that always
//! contains the brute-force best play. `evaluate_simple` (joker-aware) then
//! scores the candidates, so the generator does NOT need to classify hands —
//! it only needs to include every subset that could be optimal. Validated by
//! `candidate_plays` COVERAGE vs brute force on real states.
//!
//! Bail (return empty / None) on the hard joker cases — Four Fingers (4-card
//! straights/flushes), Shortcut (gapped straights), Smeared (suit merge), Wild
//! cards — where the canonical subsets differ; the caller falls back to brute
//! force for those (rare).

use pyo3::prelude::*;

use crate::state::card::{Card, Rank};

#[inline]
fn rank_value(r: Rank) -> u8 {
    match r {
        Rank::Two => 2, Rank::Three => 3, Rank::Four => 4, Rank::Five => 5,
        Rank::Six => 6, Rank::Seven => 7, Rank::Eight => 8, Rank::Nine => 9,
        Rank::Ten => 10, Rank::Jack => 11, Rank::Queen => 12, Rank::King => 13,
        Rank::Ace => 14,
    }
}

/// Highest 5-card straight (indices), optionally restricted to one suit (for
/// straight flush). Handles Ace-high and Ace-low (A-2-3-4-5). None if no
/// 5-consecutive run. No Shortcut/Four-Fingers (caller bails on those).
fn best_straight(hand: &[Card], suit_filter: Option<u8>) -> Option<Vec<usize>> {
    // Representative index per rank value (first seen = arbitrary; suit-filtered
    // for straight flush). Index 1..=14 (1 unused; Ace handled as 14 and low).
    let mut rep: [Option<usize>; 15] = [None; 15];
    for (i, c) in hand.iter().enumerate() {
        if let Some(s) = suit_filter {
            if c.suit as u8 != s { continue; }
        }
        let rv = rank_value(c.rank) as usize;
        if rep[rv].is_none() { rep[rv] = Some(i); }
    }
    let present = |rv: usize| rep[rv].is_some();
    // High straights: top card 14 down to 6 (run top-4..=top).
    for top in (6..=14usize).rev() {
        if (top - 4..=top).all(present) {
            return Some((top - 4..=top).rev().map(|rv| rep[rv].unwrap()).collect());
        }
    }
    // Ace-low: A(low)-2-3-4-5.
    if present(14) && (2..=5).all(present) {
        let mut v: Vec<usize> = (2..=5).rev().map(|rv| rep[rv].unwrap()).collect();
        v.push(rep[14].unwrap());
        return Some(v);
    }
    None
}

/// All k-subsets of `items` (indices), each preserving input order. Bounded by
/// design (rank groups are tiny); k<=5. Used so per-card jokers/debuffs that
/// favor a SPECIFIC same-rank/same-suit card (not the highest) are covered.
fn combos(items: &[usize], k: usize) -> Vec<Vec<usize>> {
    let n = items.len();
    if k == 0 || k > n {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        out.push(idx.iter().map(|&i| items[i]).collect());
        let mut p: Option<usize> = None;
        for i in (0..k).rev() {
            if idx[i] != i + n - k { p = Some(i); break; }
        }
        let Some(p) = p else { break };
        idx[p] += 1;
        for j in p + 1..k { idx[j] = idx[j - 1] + 1; }
    }
    out
}

/// Generate the candidate plays (index subsets) for a hand. Each is a sorted
/// Vec of card indices. Designed to CONTAIN the brute-force best play even when
/// per-card jokers (suit/odd/even) or suit-debuffs make a non-highest-rank card
/// optimal — by generating per-suit/per-combo variants. ~40 candidates vs 218.
/// Returns empty if the hand is empty.
pub(crate) fn candidate_plays(hand: &[Card]) -> Vec<Vec<usize>> {
    let n = hand.len();
    if n == 0 {
        return Vec::new();
    }
    // Indices sorted by rank value descending (highest first).
    let mut by_rank: Vec<usize> = (0..n).collect();
    by_rank.sort_by(|&a, &b| rank_value(hand[b].rank).cmp(&rank_value(hand[a].rank)));

    // Rank groups: rank_value -> indices (rank-desc order preserved from by_rank).
    let mut rank_to_idx: Vec<Vec<usize>> = vec![Vec::new(); 15];
    for &i in &by_rank {
        rank_to_idx[rank_value(hand[i].rank) as usize].push(i);
    }
    let groups_desc: Vec<usize> = (2..=14usize).rev().filter(|&rv| !rank_to_idx[rv].is_empty()).collect();

    // Suit groups: suit -> indices (rank-desc within).
    let mut suit_to_idx: [Vec<usize>; 4] = Default::default();
    for &i in &by_rank {
        suit_to_idx[hand[i].suit as usize].push(i);
    }

    let mut cands: Vec<Vec<usize>> = Vec::new();

    // High card: EVERY single card. Per-card jokers stack across dimensions
    // (suit x odd/even x specific rank: e.g. Odd Todd + Wrathful both favor 5S
    // over a higher KS), so the best single can be any card. 8 singles is cheap.
    for i in 0..n {
        cands.push(vec![i]);
    }

    // Per-rank-group: all 2/3/4/5-card sub-selections (which SAME-rank cards to
    // include matters under suit/debuff jokers).
    for &rv in &groups_desc {
        let g = &rank_to_idx[rv];
        for k in 2..=g.len().min(5) {
            cands.extend(combos(g, k));
        }
    }

    // Two pair: every pair of distinct pair-ranks, each rank's 2-card variants.
    let pair_ranks: Vec<usize> = groups_desc.iter().copied().filter(|&rv| rank_to_idx[rv].len() >= 2).collect();
    for a in 0..pair_ranks.len() {
        for b in a + 1..pair_ranks.len() {
            for pa in combos(&rank_to_idx[pair_ranks[a]], 2) {
                for pb in combos(&rank_to_idx[pair_ranks[b]], 2) {
                    let mut tp = pa.clone();
                    tp.extend_from_slice(&pb);
                    cands.push(tp);
                }
            }
        }
    }

    // Full house: every (trip-rank, different pair-rank) combo, with variants.
    let trip_ranks: Vec<usize> = groups_desc.iter().copied().filter(|&rv| rank_to_idx[rv].len() >= 3).collect();
    for &trv in &trip_ranks {
        for &prv in &pair_ranks {
            if prv == trv { continue; }
            for t in combos(&rank_to_idx[trv], 3) {
                for p in combos(&rank_to_idx[prv], 2) {
                    let mut fh = t.clone();
                    fh.extend_from_slice(&p);
                    cands.push(fh);
                }
            }
        }
    }

    // Flush: 5-card subsets of any suit with >=5. count==5 -> the only subset;
    // count>=6 -> all C(count,5) (drop a low/debuffed card). Capped: suit groups
    // rarely exceed 6, and combos() is bounded.
    for s in 0..4 {
        let g = &suit_to_idx[s];
        if g.len() >= 5 {
            cands.extend(combos(g, 5));
        }
    }

    // Straight (any suits) + straight flush (within each >=5 suit).
    if let Some(st) = best_straight(hand, None) {
        cands.push(st);
    }
    for s in 0..4 {
        if suit_to_idx[s].len() >= 5 {
            if let Some(sf) = best_straight(hand, Some(s as u8)) {
                cands.push(sf);
            }
        }
    }

    // 5-card fills: for each candidate < 5 cards, add the highest remaining
    // cards up to 5. Covers Splash (all cards score) and held-card synergies
    // where extra cards add value.
    let base_len = cands.len();
    for ci in 0..base_len {
        let c = &cands[ci];
        if c.len() < 5 {
            let mut filled = c.clone();
            let in_set: u16 = c.iter().fold(0u16, |acc, &i| acc | (1 << i));
            for &i in &by_rank {
                if filled.len() >= 5 { break; }
                if in_set & (1 << i) == 0 { filled.push(i); }
            }
            if filled.len() > c.len() {
                cands.push(filled);
            }
        }
    }

    // Normalize (sort each) + dedup.
    for c in cands.iter_mut() { c.sort_unstable(); }
    cands.sort();
    cands.dedup();
    cands
}

/// PyO3: expose the candidate index subsets for coverage validation against
/// the brute-force best play. Each inner Vec is sorted ascending.
#[pyfunction]
#[pyo3(name = "candidate_plays")]
pub fn py_candidate_plays(hand: Vec<Card>) -> Vec<Vec<usize>> {
    candidate_plays(&hand)
}
