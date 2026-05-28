//! GameState representation (Phase 1 of RUST_PORT_PLAN.md).
//!
//! Owns the typed Rust mirror of `balatro_ai.api.state.GameState`.
//! Fields chosen to cover what the solver hot path actually reads;
//! a few low-value fields (stake, seed, shop, pack, legal_actions)
//! are kept as `Py<PyAny>` opaques for round-trip preservation but
//! the Rust side doesn't inspect them.
//!
//! Layout: Vec<Card> for the hand/deck, Vec<Joker> for the jokers,
//! Vec<String> for consumables/vouchers. Could be tightened to
//! SmallVec or fixed-size arrays in Phase 5 optimization once we
//! know the typical sizes — for Phase 1 we lean on Vec's
//! ergonomics.
//!
//! Dict fields (`hand_levels`, `modifiers`): hand_levels becomes
//! a typed `[u8; 12]` (one entry per HandType). modifiers stays as
//! a `Py<PyDict>` since its shape varies and the solver only
//! reads a handful of specific keys; we'll surface those as typed
//! fields as Phase 2 ports them.

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};

use crate::state::card::Card;
use crate::state::joker::Joker;

/// Solver-relevant fields of `balatro_ai.api.state.GameState`.
///
/// Constructed once via `from_python` at search entry. All
/// subsequent native operations read/copy this struct directly,
/// never crossing the FFI boundary.
#[pyclass(name = "RustGameState")]
#[derive(Debug)]
// NOT `#[derive(Clone)]`: `Py<PyDict>` requires the GIL to clone
// (via `clone_ref(py)`). If/when we need GameStateNative::clone,
// add a manual impl that takes `py: Python<'_>` and bumps refcounts
// on modifiers + hand_levels.
pub struct GameStateNative {
    // Scalars
    pub phase: String,         // GamePhase enum value as string
    pub ante: i32,
    pub blind: String,
    pub required_score: i64,
    pub current_score: i64,
    pub hands_remaining: i32,
    pub discards_remaining: i32,
    pub money: i32,
    pub deck_size: i32,
    pub run_over: bool,
    pub won: bool,

    // Collections
    pub hand: Vec<Card>,
    pub known_deck: Vec<Card>,
    pub jokers: Vec<Joker>,
    pub consumables: Vec<String>,
    pub vouchers: Vec<String>,

    // Dict-ish state (kept as Py<PyAny> for round-trip; the hot
    // path will get typed accessors as fields are ported).
    pub modifiers: Py<PyDict>,
    pub hand_levels: Py<PyDict>,
}

#[pymethods]
impl GameStateNative {
    /// Build a GameStateNative from a Python `GameState`.
    ///
    /// One FFI crossing per call — collects all field values up
    /// front and copies into the Rust struct. After this, the
    /// solver can run entirely native without further attribute
    /// lookups against the Python object.
    #[staticmethod]
    pub fn from_python(py_state: &Bound<'_, PyAny>) -> PyResult<Self> {
        // (py token: not needed here yet — kept this comment in case
        // a future change adds an operation requiring `Python<'_>`.)

        // Phase is GamePhase enum; we store its .value (a string).
        let phase: String = py_state.getattr("phase")?.getattr("value")?.extract()?;
        let ante: i32 = py_state.getattr("ante")?.extract()?;
        let blind: String = py_state.getattr("blind")?.extract()?;
        let required_score: i64 = py_state.getattr("required_score")?.extract()?;
        let current_score: i64 = py_state.getattr("current_score")?.extract()?;
        let hands_remaining: i32 = py_state.getattr("hands_remaining")?.extract()?;
        let discards_remaining: i32 = py_state.getattr("discards_remaining")?.extract()?;
        let money: i32 = py_state.getattr("money")?.extract()?;
        let deck_size: i32 = py_state.getattr("deck_size")?.extract()?;
        let run_over: bool = py_state.getattr("run_over")?.extract()?;
        let won: bool = py_state.getattr("won")?.extract()?;

        // Collections: convert each element via the matching from_python.
        let hand = collect_cards(&py_state.getattr("hand")?)?;
        let known_deck = collect_cards(&py_state.getattr("known_deck")?)?;
        let jokers = collect_jokers(&py_state.getattr("jokers")?)?;
        let consumables = collect_strings(&py_state.getattr("consumables")?)?;
        let vouchers = collect_strings(&py_state.getattr("vouchers")?)?;

        // Dict fields: clone the PyDict into our owned Py reference.
        let modifiers_any = py_state.getattr("modifiers")?;
        let modifiers: Py<PyDict> = modifiers_any.downcast::<PyDict>()?.clone().unbind();
        let hand_levels_any = py_state.getattr("hand_levels")?;
        let hand_levels: Py<PyDict> = hand_levels_any.downcast::<PyDict>()?.clone().unbind();

        Ok(Self {
            phase,
            ante,
            blind,
            required_score,
            current_score,
            hands_remaining,
            discards_remaining,
            money,
            deck_size,
            run_over,
            won,
            hand,
            known_deck,
            jokers,
            consumables,
            vouchers,
            modifiers,
            hand_levels,
        })
    }

    /// Round-trip back to a Python GameState. Used by parity tests
    /// + as the "after one beam call" return path. The phase string
    /// is converted back to a GamePhase enum; the dict fields are
    /// returned as-is (same Python dict object).
    pub fn to_python<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let state_module = py.import_bound("balatro_ai.api.state")?;
        let gs_cls = state_module.getattr("GameState")?;
        let phase_enum_cls = state_module.getattr("GamePhase")?;
        let phase_enum = phase_enum_cls.call1((&self.phase,))?;

        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("phase", phase_enum)?;
        kwargs.set_item("ante", self.ante)?;
        kwargs.set_item("blind", &self.blind)?;
        kwargs.set_item("required_score", self.required_score)?;
        kwargs.set_item("current_score", self.current_score)?;
        kwargs.set_item("hands_remaining", self.hands_remaining)?;
        kwargs.set_item("discards_remaining", self.discards_remaining)?;
        kwargs.set_item("money", self.money)?;
        kwargs.set_item("deck_size", self.deck_size)?;
        kwargs.set_item("run_over", self.run_over)?;
        kwargs.set_item("won", self.won)?;

        // Collections as tuples (GameState uses frozen tuples).
        let hand_py: Vec<Bound<'py, PyAny>> = self.hand.iter()
            .map(|c| c.to_python(py))
            .collect::<PyResult<_>>()?;
        kwargs.set_item("hand", pyo3::types::PyTuple::new_bound(py, hand_py))?;

        let known_deck_py: Vec<Bound<'py, PyAny>> = self.known_deck.iter()
            .map(|c| c.to_python(py))
            .collect::<PyResult<_>>()?;
        kwargs.set_item("known_deck", pyo3::types::PyTuple::new_bound(py, known_deck_py))?;

        let jokers_py: Vec<Bound<'py, PyAny>> = self.jokers.iter()
            .map(|j| j.to_python(py))
            .collect::<PyResult<_>>()?;
        kwargs.set_item("jokers", pyo3::types::PyTuple::new_bound(py, jokers_py))?;

        kwargs.set_item(
            "consumables",
            pyo3::types::PyTuple::new_bound(py, self.consumables.iter()),
        )?;
        kwargs.set_item(
            "vouchers",
            pyo3::types::PyTuple::new_bound(py, self.vouchers.iter()),
        )?;

        kwargs.set_item("modifiers", self.modifiers.bind(py))?;
        kwargs.set_item("hand_levels", self.hand_levels.bind(py))?;

        gs_cls.call((), Some(&kwargs))
    }

    // Scalar accessors for quick Python inspection. The hot path
    // doesn't go through these — Rust code reads the fields
    // directly via the pub members.
    #[getter] fn get_ante(&self) -> i32 { self.ante }
    #[getter] fn get_blind(&self) -> &str { &self.blind }
    #[getter] fn get_required_score(&self) -> i64 { self.required_score }
    #[getter] fn get_current_score(&self) -> i64 { self.current_score }
    #[getter] fn get_hands_remaining(&self) -> i32 { self.hands_remaining }
    #[getter] fn get_discards_remaining(&self) -> i32 { self.discards_remaining }
    #[getter] fn get_money(&self) -> i32 { self.money }
    #[getter] fn get_deck_size(&self) -> i32 { self.deck_size }
    #[getter] fn get_phase(&self) -> &str { &self.phase }
    #[getter] fn get_won(&self) -> bool { self.won }
    #[getter] fn get_run_over(&self) -> bool { self.run_over }
    #[getter] fn get_hand_size(&self) -> usize { self.hand.len() }
    #[getter] fn get_n_jokers(&self) -> usize { self.jokers.len() }

    fn __repr__(&self) -> String {
        format!(
            "RustGameState(phase={}, ante={}, blind={:?}, score={}/{}, hands={}, discards={}, money={}, hand_size={}, n_jokers={})",
            self.phase, self.ante, self.blind, self.current_score, self.required_score,
            self.hands_remaining, self.discards_remaining, self.money,
            self.hand.len(), self.jokers.len(),
        )
    }
}

fn collect_cards(seq: &Bound<'_, PyAny>) -> PyResult<Vec<Card>> {
    let mut out = Vec::with_capacity(seq.len().unwrap_or(0));
    for item in seq.iter()? {
        out.push(Card::from_python(&item?)?);
    }
    Ok(out)
}

fn collect_jokers(seq: &Bound<'_, PyAny>) -> PyResult<Vec<Joker>> {
    let mut out = Vec::with_capacity(seq.len().unwrap_or(0));
    for item in seq.iter()? {
        out.push(Joker::from_python(&item?)?);
    }
    Ok(out)
}

fn collect_strings(seq: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let mut out = Vec::with_capacity(seq.len().unwrap_or(0));
    for item in seq.iter()? {
        out.push(item?.extract::<String>()?);
    }
    Ok(out)
}

// Suppress the unused-import warning that PyList triggers when
// nothing references it yet (it's here for future use when we add
// tuple/list distinction handling).
#[allow(dead_code)]
fn _list_marker(_: &PyList) {}
