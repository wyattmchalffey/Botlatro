//! Hand evaluation (Phase 2 of RUST_PORT_PLAN.md).
//!
//! Ports the hot-path scoring from `balatro_ai.rules.hand_evaluator`.
//! Functions here operate on `&[Card]` slices and return typed
//! results — no Python data structures touched in the inner loop.
//!
//! Public surface (registered in `lib.rs`):
//! - `identify_hand_type` — pure HandType classification
//!
//! Incremental porting strategy: each function ports a SUBSET of
//! the Python equivalent's cases (the common-case fast path) and
//! returns `None` for cases it can't handle. The Python wrapper
//! falls back to the original Python implementation when the Rust
//! version returns None. This lets us land speedup on the 95% case
//! without having to port every joker effect on day one.

pub mod chips;
pub mod effects;
pub mod evaluate;
pub mod hand_type;
pub mod scoring;

pub use chips::card_chip_value;
pub use effects::{
    ability_joker_effect, enhancement_effect, is_supported_joker,
    per_card_joker_effect, HandContext, JokerEffect,
};
pub use evaluate::evaluate_simple;
pub use hand_type::{HandType, identify_hand_type_simple};
pub use scoring::scoring_indices_simple;
