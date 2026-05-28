// Native Rust core for the Balatro AI solver.
//
// PyO3 entry point. As the porting plan in RUST_PORT_PLAN.md is
// executed, this module grows function registrations grouped by
// subsystem.
//
// FFI strategy (per RUST_PORT_PLAN.md §3): port at COARSE
// granularity, not fine helpers. The 2026-05-26 PoC showed that
// crossing the Python↔Rust boundary for a small function like
// `is_stone_card` costs ~3.6× MORE than the Python implementation
// itself. So the public surface here exposes large operations
// (RustCard construction, eventually evaluate_played_cards,
// simulate_play, solver_beam_play_action) rather than helpers.
// The fine-grained helpers stay as pure-Rust `#[inline]` functions
// inside the modules that compose them.

use pyo3::prelude::*;

mod card;        // is_stone_card (PoC; kept for reference)
mod state;       // Phase 1: Card, Joker, GameState
mod hand_eval;   // Phase 2: identify_hand_type, ...
mod forward_sim; // Phase 3: simulate_play / simulate_discard helpers
mod search;      // Phase 4: native solver search

/// Return the crate version. Smoke-test entry point.
#[pyfunction]
fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Smoke-test: returns True so Python callers can verify the
/// extension loaded and basic FFI works.
#[pyfunction]
fn is_native_available() -> bool {
    true
}

/// PyO3 module definition. Phase 1 surface: RustCard + the
/// is_stone_card PoC (still exported so existing parity tests pass).
#[pymodule]
fn balatro_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(is_native_available, m)?)?;
    m.add_function(wrap_pyfunction!(card::is_stone_card, m)?)?;
    m.add_class::<state::Card>()?;
    m.add_class::<state::Joker>()?;
    m.add_class::<state::GameStateNative>()?;
    m.add_function(wrap_pyfunction!(hand_eval::hand_type::py_identify_hand_type, m)?)?;
    m.add_function(wrap_pyfunction!(hand_eval::scoring::py_scoring_indices, m)?)?;
    m.add_function(wrap_pyfunction!(hand_eval::chips::py_card_chip_value, m)?)?;
    m.add_function(wrap_pyfunction!(hand_eval::evaluate::py_evaluate_simple, m)?)?;
    m.add_function(wrap_pyfunction!(hand_eval::evaluate::py_evaluate_simple_with_levels, m)?)?;
    m.add_function(wrap_pyfunction!(forward_sim::deck::py_draw_from_deck, m)?)?;
    m.add_function(wrap_pyfunction!(forward_sim::deck::py_draw_indices_to_remove, m)?)?;
    m.add_function(wrap_pyfunction!(forward_sim::jokers::py_jokers_after_play, m)?)?;
    m.add_function(wrap_pyfunction!(forward_sim::jokers::py_jokers_after_discard, m)?)?;
    m.add_function(wrap_pyfunction!(forward_sim::phase::py_next_phase, m)?)?;
    m.add_function(wrap_pyfunction!(forward_sim::economy::py_held_end_of_round_money_delta, m)?)?;
    m.add_function(wrap_pyfunction!(forward_sim::economy::py_discard_money_delta, m)?)?;
    m.add_function(wrap_pyfunction!(forward_sim::levels::py_hand_level_after_play, m)?)?;
    m.add_function(wrap_pyfunction!(forward_sim::simulate::py_simulate_play_simple, m)?)?;
    m.add_function(wrap_pyfunction!(search::scorer::py_score_play_actions_batch, m)?)?;
    m.add_function(wrap_pyfunction!(search::scorer::py_best_play_action, m)?)?;
    m.add_function(wrap_pyfunction!(search::scorer::py_discard_candidate_score, m)?)?;
    m.add_function(wrap_pyfunction!(search::scorer::py_discard_candidate_scores_batch, m)?)?;
    m.add_function(wrap_pyfunction!(search::rollout::py_clear_probability_native, m)?)?;
    m.add_function(wrap_pyfunction!(search::beam::py_beam_play_value_native, m)?)?;
    Ok(())
}
