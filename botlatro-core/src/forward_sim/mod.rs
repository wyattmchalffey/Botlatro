//! Forward simulation (Phase 3 of RUST_PORT_PLAN.md).
//!
//! Ports `simulate_play` / `simulate_discard` and their helpers
//! from `src/balatro_ai/search/forward_sim.py`. The Python file is
//! ~4500 lines so this is staged into focused submodules:
//!
//! - `deck` — `_draw_from_deck` and deck-pile bookkeeping (Phase 3a)
//! - `jokers` — `_jokers_after_play` / `_jokers_after_discard`
//!   scaling-counter updates (Phase 3b/3c — pending)
//! - `phase` — next-phase + Mr Bones logic (Phase 3d — pending)
//! - `economy` — held-card money + round-end joker housekeeping
//!   (Phase 3e — pending)
//!
//! Strategy: each helper has its own PyO3 entry point so the Python
//! `simulate_play` can wire individual pieces over to Rust without
//! a single big port. Once enough helpers are native, we can add a
//! top-level `simulate_play_native` that does the whole transition
//! in one FFI call.
//!
//! Bail policy: each function returns `Option<...>` (or `None` for
//! Python callers) when the input falls outside the fast path. The
//! Python caller falls back to the original implementation. This is
//! the same conservative strategy we used in Phase 2d.

pub mod deck;
pub mod economy;
pub mod jokers;
pub mod levels;
pub mod phase;
pub mod simulate;
