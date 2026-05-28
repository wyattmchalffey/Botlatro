//! Joker representation (Phase 1 of RUST_PORT_PLAN.md).
//!
//! For Phase 1 we store the joker's display name as a `String`,
//! not a typed `JokerId` enum. The reason: a closed enum of ~150
//! variants is meaningful work and only pays off once the joker
//! effect table is being dispatched. Until then, name-as-String
//! preserves all info we need for round-trip and is trivially
//! constructible from Python's `Joker.name`.
//!
//! Phase 2 (hand evaluation) will introduce `JokerId` for the
//! ported subset of jokers and dispatch through it. Unported
//! jokers keep their name as a fallback signal — the Rust eval
//! returns "skip me, fall back to Python" for them.
//!
//! Metadata + effect: the Python `Joker` has a `metadata` dict
//! and a derived `effect: JokerEffect`. We drop both on the Rust
//! side; round-trip back to Python rebuilds with empty defaults.
//! The fields that matter for evaluation (counters, target ranks,
//! disabled flag) come from `metadata` and we'll surface those as
//! typed fields as the effects are ported.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::state::card::Edition;

/// Joker held by the player. Phase 1 representation — counter
/// and per-joker dynamic state come in Phase 2.
#[pyclass(name = "RustJoker")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Joker {
    /// Display name as the Python `Joker.name` field stores it
    /// (e.g. "Jolly Joker", "Smeared Joker"). String for Phase 1;
    /// will become a JokerId enum once Phase 2 lands.
    pub name: String,
    pub edition: Edition,
    /// `None` when the bridge doesn't surface a sell value.
    pub sell_value: Option<i32>,
}

#[pymethods]
impl Joker {
    /// Build a RustJoker from a Python `balatro_ai.api.state.Joker`.
    #[staticmethod]
    pub fn from_python(py_joker: &Bound<'_, PyAny>) -> PyResult<Self> {
        let name: String = py_joker.getattr("name")?.extract()?;
        let edition_str: Option<String> = py_joker.getattr("edition")?.extract()?;
        let sell_value: Option<i32> = py_joker.getattr("sell_value")?.extract()?;

        Ok(Self {
            name,
            edition: Edition::from_option_str(edition_str.as_deref()),
            sell_value,
        })
    }

    /// Round-trip back to a Python `Joker`. Metadata is empty;
    /// callers that need metadata should not round-trip.
    pub fn to_python<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let state_module = py.import_bound("balatro_ai.api.state")?;
        let joker_cls = state_module.getattr("Joker")?;
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("name", &self.name)?;
        if let Some(s) = self.edition.to_python_str() {
            kwargs.set_item("edition", s)?;
        }
        if let Some(sv) = self.sell_value {
            kwargs.set_item("sell_value", sv)?;
        }
        joker_cls.call((), Some(&kwargs))
    }

    #[getter]
    fn get_name(&self) -> &str { &self.name }

    #[getter]
    fn get_edition(&self) -> Option<&'static str> { self.edition.to_python_str() }

    #[getter]
    fn get_sell_value(&self) -> Option<i32> { self.sell_value }

    fn __repr__(&self) -> String {
        format!(
            "RustJoker(name={:?}, edition={:?}, sell_value={:?})",
            self.name,
            self.edition.to_python_str(),
            self.sell_value,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joker_struct_is_reasonable_size() {
        // String is 24 bytes on 64-bit (ptr + len + cap), Edition is
        // 1 byte, sell_value is 8 bytes (Option<i32> = 8). Plus
        // alignment padding. Roughly 40 bytes — much more than
        // Card's 8, but Joker references heap-allocated name.
        // Phase 2 will replace name with JokerId (1-2 bytes) for
        // the hot path, dropping this to ~16 bytes.
        let size = std::mem::size_of::<Joker>();
        assert!(size <= 64, "Joker grew to {size} bytes — review layout");
    }
}
