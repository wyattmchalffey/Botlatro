//! Next-phase decision (Phase 3d of RUST_PORT_PLAN.md).
//!
//! Ports the phase-transition portion of `simulate_play` at
//! `forward_sim.py:283-295`:
//!
//! ```text
//! if state.required_score > 0 and next_score >= state.required_score:
//!     next_phase = ROUND_EVAL
//! elif next_hands <= 0:
//!     if _mr_bones_saves(state, next_score):
//!         next_phase = ROUND_EVAL
//!         next_jokers = _jokers_after_mr_bones_save(next_jokers)
//!     else:
//!         next_phase = RUN_OVER
//! ```
//!
//! Mr. Bones save fires when score >= 25% of required AND a Mr. Bones
//! joker is present. The save consumes Mr. Bones (so we report whether
//! it fired so Python can remove it).

use pyo3::prelude::*;

/// Phase transition decision. Mirrors Python's enum string values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NextPhase {
    /// No transition — stay in SELECTING_HAND.
    KeepPlaying,
    /// Round defeated (score met) — go to ROUND_EVAL.
    RoundEval,
    /// Out of hands AND Mr. Bones saved → ROUND_EVAL + remove Bones.
    MrBonesSave,
    /// Out of hands AND no save → RUN_OVER.
    RunOver,
}

impl NextPhase {
    pub fn to_str(self) -> &'static str {
        match self {
            Self::KeepPlaying => "SELECTING_HAND",
            Self::RoundEval => "ROUND_EVAL",
            Self::MrBonesSave => "ROUND_EVAL",  // same phase, separate signal
            Self::RunOver => "RUN_OVER",
        }
    }
}

/// Decide the next phase + whether Mr. Bones fired.
///
/// `has_mr_bones` is whether ANY non-disabled Mr. Bones joker is
/// held. The caller (Python) is responsible for the disabled check
/// and for removing the actual joker on save.
pub fn next_phase(
    required_score: i64,
    next_score: i64,
    next_hands_remaining: u32,
    has_mr_bones: bool,
) -> NextPhase {
    if required_score > 0 && next_score >= required_score {
        return NextPhase::RoundEval;
    }
    if next_hands_remaining == 0 {
        // Mr. Bones gate: score >= 25% of required AND a Bones joker.
        if required_score > 0
            && (next_score as f64) >= (required_score as f64) * 0.25
            && has_mr_bones
        {
            return NextPhase::MrBonesSave;
        }
        return NextPhase::RunOver;
    }
    NextPhase::KeepPlaying
}

/// PyO3 wrapper. Returns the next-phase string + a bool indicating
/// whether Mr. Bones fired (caller removes the joker if true).
#[pyfunction]
#[pyo3(name = "next_phase")]
pub fn py_next_phase(
    required_score: i64,
    next_score: i64,
    next_hands_remaining: u32,
    has_mr_bones: bool,
) -> (&'static str, bool) {
    let phase = next_phase(required_score, next_score, next_hands_remaining, has_mr_bones);
    let mr_bones_fired = matches!(phase, NextPhase::MrBonesSave);
    (phase.to_str(), mr_bones_fired)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_met_goes_to_round_eval() {
        assert_eq!(next_phase(300, 350, 3, false), NextPhase::RoundEval);
    }

    #[test]
    fn under_score_with_hands_keeps_playing() {
        assert_eq!(next_phase(300, 100, 3, false), NextPhase::KeepPlaying);
    }

    #[test]
    fn out_of_hands_no_bones_run_over() {
        assert_eq!(next_phase(300, 100, 0, false), NextPhase::RunOver);
    }

    #[test]
    fn out_of_hands_with_bones_but_too_low_run_over() {
        // 50 / 300 = 16.7%, below 25% threshold → no save.
        assert_eq!(next_phase(300, 50, 0, true), NextPhase::RunOver);
    }

    #[test]
    fn out_of_hands_with_bones_above_threshold_saves() {
        // 100 / 300 = 33.3%, above 25% → save fires.
        assert_eq!(next_phase(300, 100, 0, true), NextPhase::MrBonesSave);
    }

    #[test]
    fn no_required_score_keeps_playing() {
        // required_score=0 (uninitialized) shouldn't trigger anything.
        assert_eq!(next_phase(0, 1000, 3, false), NextPhase::KeepPlaying);
    }
}
