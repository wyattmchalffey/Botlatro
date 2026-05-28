//! Game state representation (Phase 1 of RUST_PORT_PLAN.md).
//!
//! Owns the typed structs for Card, Joker, GameState that live on
//! the Rust side. Python interop happens at the module boundary —
//! a Python `Card` becomes a Rust `Card` once at search entry, then
//! stays in Rust for the duration.
//!
//! Layout choice: regular Rust enums + structs, not hand-packed
//! bitfields. The compiler packs aligned enum fields tightly
//! enough; premature bit-packing obscures code without measurable
//! gain at this scope. We can revisit bit-packing in Phase 5
//! optimization if profiling identifies it as a hot path.

pub mod card;
pub mod game_state;
pub mod joker;

pub use card::Card;
pub use game_state::GameStateNative;
pub use joker::Joker;
