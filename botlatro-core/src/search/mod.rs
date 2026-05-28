//! Native solver search (Phase 4 of RUST_PORT_PLAN.md).
//!
//! Ports parts of `src/balatro_ai/solver/search_v2/play.py` and its
//! candidate-scoring helpers into Rust. The win here is FFI
//! AMORTIZATION: batched/recursive operations that today do N
//! cross-boundary calls become ONE call.
//!
//! Phased build-out:
//! - `scorer` (Phase 4a) — batched per-action scorer that takes
//!   shared inputs once + a list of action-index sets and returns
//!   scores for each.
//! - `enumerate` (Phase 4b — pending) — native legal-play enumeration.
//! - `rollout` (Phase 4c — pending) — single-subtree beam rollout.
//! - `beam` (Phase 4d — pending) — full `solver_beam_play_action_native`.

pub mod beam;
pub mod rollout;
pub mod scorer;
