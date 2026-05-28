//! Hand-level updates (Phase 3f of RUST_PORT_PLAN.md).
//!
//! Ports `_hand_levels_after_play` at `forward_sim.py:2312`.
//!
//! After a play, the played hand_type's stored level may change:
//! - **Space Joker**: random chance to level up — passed in as
//!   `space_joker_triggers` (0 in the solver's deterministic
//!   search context).
//! - **The Arm** boss blind: -1 to the played hand type's stored
//!   level (minimum 1).
//!
//! If both deltas are zero, return None (signal: caller can reuse
//! the existing hand_levels dict without copying).

use pyo3::prelude::*;

/// Compute the new level for the played hand type. Returns Some(new)
/// when a change is needed, None otherwise.
pub fn hand_level_after_play(
    current_level: u32,
    space_joker_triggers: i32,
    is_the_arm_blind: bool,
) -> Option<u32> {
    let arm_decrement: i32 = if is_the_arm_blind { 1 } else { 0 };
    if space_joker_triggers <= 0 && arm_decrement <= 0 {
        return None;
    }
    let current = current_level.max(1) as i32;
    let next = (current + space_joker_triggers - arm_decrement).max(1) as u32;
    if next == current_level { None } else { Some(next) }
}

/// PyO3 wrapper. Caller looks up the current level from their
/// own hand_levels dict and passes it as `current_level`.
#[pyfunction]
#[pyo3(name = "hand_level_after_play")]
pub fn py_hand_level_after_play(
    current_level: u32,
    space_joker_triggers: i32,
    is_the_arm_blind: bool,
) -> Option<u32> {
    hand_level_after_play(current_level, space_joker_triggers, is_the_arm_blind)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_change_when_no_triggers_or_arm() {
        assert_eq!(hand_level_after_play(3, 0, false), None);
    }

    #[test]
    fn space_joker_increments() {
        assert_eq!(hand_level_after_play(3, 1, false), Some(4));
    }

    #[test]
    fn the_arm_decrements_minimum_1() {
        assert_eq!(hand_level_after_play(3, 0, true), Some(2));
        assert_eq!(hand_level_after_play(1, 0, true), None);
        assert_eq!(hand_level_after_play(2, 0, true), Some(1));
    }

    #[test]
    fn space_and_arm_cancel() {
        // +1 from space, -1 from arm = no net change.
        assert_eq!(hand_level_after_play(3, 1, true), None);
    }

    #[test]
    fn current_zero_treated_as_one() {
        // A fresh hand_type with no entry → current=0 → treat as 1.
        assert_eq!(hand_level_after_play(0, 1, false), Some(2));
    }
}
