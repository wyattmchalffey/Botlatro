//! Card representation (Phase 1 of RUST_PORT_PLAN.md).
//!
//! `Card` is a packed Rust struct with typed enums for rank/suit/
//! enhancement/edition/seal. Python interop via `Card::from_python`
//! and `Card::to_python` round-trip methods.
//!
//! Metadata handling: the Python `Card.metadata` dict is NOT
//! preserved on the Rust side. It's almost never read in the hot
//! path (the only readers are `_displayed_card_chip_value` and
//! `_permanent_card_chips` which handle rare "Gold Card-style"
//! modifiers). When those code paths are eventually ported, we'll
//! either add a metadata field or compute the relevant integers at
//! conversion time. For now, round-trip back to Python builds a
//! Card with an empty metadata dict, which is a known divergence
//! flagged in the parity test.

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Card rank. Ordering matches Balatro's ace-high convention; the
/// numeric value happens to be the chip value for most ranks (only
/// face cards and Ace differ — see RANK_VALUES in the Python port
/// `hand_evaluator.py:40`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Rank {
    Two = 2,
    Three = 3,
    Four = 4,
    Five = 5,
    Six = 6,
    Seven = 7,
    Eight = 8,
    Nine = 9,
    Ten = 10,
    Jack = 11,
    Queen = 12,
    King = 13,
    Ace = 14,
}

impl Rank {
    /// Per-card chip value, matching Python's `RANK_VALUES` dict at
    /// `hand_evaluator.py:40`. Note that all face cards AND ten are
    /// 10, and Ace is 11 — this is NOT the same as the straight
    /// value (where face cards differentiate as J=11, Q=12, K=13,
    /// A=14). Use `straight_value()` for sequencing logic and
    /// `chip_value()` for scoring + "highest card" comparisons.
    #[inline]
    pub fn chip_value(self) -> u8 {
        match self {
            Self::Two => 2,
            Self::Three => 3,
            Self::Four => 4,
            Self::Five => 5,
            Self::Six => 6,
            Self::Seven => 7,
            Self::Eight => 8,
            Self::Nine => 9,
            Self::Ten | Self::Jack | Self::Queen | Self::King => 10,
            Self::Ace => 11,
        }
    }

    /// Straight-ordering value (Python `STRAIGHT_VALUES`). Used for
    /// sequence detection.
    #[inline]
    pub fn straight_value(self) -> u8 {
        self as u8 // discriminant equals straight value
    }

    /// Parse from the Python string form ("2"-"10", "T", "J", "Q",
    /// "K", "A"). "T" is an alias for "10" used by some bridge JSON
    /// payloads.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "2" => Some(Self::Two),
            "3" => Some(Self::Three),
            "4" => Some(Self::Four),
            "5" => Some(Self::Five),
            "6" => Some(Self::Six),
            "7" => Some(Self::Seven),
            "8" => Some(Self::Eight),
            "9" => Some(Self::Nine),
            "10" | "T" => Some(Self::Ten),
            "J" => Some(Self::Jack),
            "Q" => Some(Self::Queen),
            "K" => Some(Self::King),
            "A" => Some(Self::Ace),
            _ => None,
        }
    }

    /// Render back to Python's canonical short form. We pick "T"
    /// for 10 because that's the form used in `Card.short_name` for
    /// brevity, but the Python Card stores `rank="10"` literally.
    /// For round-trip compatibility we return "10" (matches the
    /// Python field value, not the display form).
    pub fn to_str(self) -> &'static str {
        match self {
            Self::Two => "2",
            Self::Three => "3",
            Self::Four => "4",
            Self::Five => "5",
            Self::Six => "6",
            Self::Seven => "7",
            Self::Eight => "8",
            Self::Nine => "9",
            Self::Ten => "10",
            Self::Jack => "J",
            Self::Queen => "Q",
            Self::King => "K",
            Self::Ace => "A",
        }
    }
}

/// Card suit. Single-char codes match Python's canonical form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Suit {
    Clubs = 0,
    Diamonds = 1,
    Hearts = 2,
    Spades = 3,
}

impl Suit {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "C" | "Club" | "Clubs" => Some(Self::Clubs),
            "D" | "Diamond" | "Diamonds" => Some(Self::Diamonds),
            "H" | "Heart" | "Hearts" => Some(Self::Hearts),
            "S" | "Spade" | "Spades" => Some(Self::Spades),
            _ => None,
        }
    }

    pub fn to_str(self) -> &'static str {
        match self {
            Self::Clubs => "C",
            Self::Diamonds => "D",
            Self::Hearts => "H",
            Self::Spades => "S",
        }
    }
}

/// Card enhancement. Closed enum of the values Balatro defines.
/// "None" is the no-enhancement state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Enhancement {
    None = 0,
    Bonus = 1,
    Mult = 2,
    Wild = 3,
    Glass = 4,
    Steel = 5,
    Stone = 6,
    Gold = 7,
    Lucky = 8,
}

impl Enhancement {
    /// Parse from the Python `card.enhancement` value. The Python
    /// version handles `None`, lowercase, mixed-case, "m_" prefix
    /// (`m_stone`), and "* card" suffix variations all via
    /// `_normalize_effect_name`. We mirror that here.
    pub fn from_option_str(name: Option<&str>) -> Self {
        let normalized = match name {
            None => return Self::None,
            Some(s) => normalize_effect_name(s),
        };
        match normalized.as_str() {
            "" => Self::None,
            "bonus" | "bonus card" => Self::Bonus,
            "mult" | "mult card" => Self::Mult,
            "wild" | "wild card" => Self::Wild,
            "glass" | "glass card" => Self::Glass,
            "steel" | "steel card" => Self::Steel,
            "stone" | "stone card" => Self::Stone,
            "gold" | "gold card" => Self::Gold,
            "lucky" | "lucky card" => Self::Lucky,
            _ => Self::None,  // unknown -> treat as no enhancement
        }
    }

    /// Render to the Python canonical form. Returns the lowercase
    /// name (matching what bridge JSON typically sends), or None
    /// for the no-enhancement case.
    pub fn to_python_str(self) -> Option<&'static str> {
        match self {
            Self::None => None,
            Self::Bonus => Some("bonus"),
            Self::Mult => Some("mult"),
            Self::Wild => Some("wild"),
            Self::Glass => Some("glass"),
            Self::Steel => Some("steel"),
            Self::Stone => Some("stone"),
            Self::Gold => Some("gold"),
            Self::Lucky => Some("lucky"),
        }
    }
}

/// Card edition. Five-value enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Edition {
    None = 0,
    Foil = 1,
    Holographic = 2,
    Polychrome = 3,
    Negative = 4,
}

impl Edition {
    /// Edition chip contribution per scored card / per ability joker.
    /// Matches `_edition_chips` in `hand_evaluator.py`.
    #[inline]
    pub fn chips(self) -> i64 {
        match self {
            Self::Foil => 50,
            _ => 0,
        }
    }

    /// Edition additive mult contribution.
    #[inline]
    pub fn mult(self) -> i64 {
        match self {
            Self::Holographic => 10,
            _ => 0,
        }
    }

    /// Edition multiplicative mult (x1.5 for Polychrome, x1 for the rest).
    #[inline]
    pub fn xmult(self) -> f64 {
        match self {
            Self::Polychrome => 1.5,
            _ => 1.0,
        }
    }

    pub fn from_option_str(name: Option<&str>) -> Self {
        let normalized = match name {
            None => return Self::None,
            Some(s) => normalize_effect_name(s),
        };
        match normalized.as_str() {
            "" => Self::None,
            "foil" => Self::Foil,
            "holographic" | "holo" => Self::Holographic,
            "polychrome" => Self::Polychrome,
            "negative" => Self::Negative,
            _ => Self::None,
        }
    }

    pub fn to_python_str(self) -> Option<&'static str> {
        match self {
            Self::None => None,
            Self::Foil => Some("foil"),
            Self::Holographic => Some("holographic"),
            Self::Polychrome => Some("polychrome"),
            Self::Negative => Some("negative"),
        }
    }
}

/// Card seal. Five-value enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Seal {
    None = 0,
    Red = 1,
    Blue = 2,
    Gold = 3,
    Purple = 4,
}

impl Seal {
    pub fn from_option_str(name: Option<&str>) -> Self {
        let normalized = match name {
            None => return Self::None,
            Some(s) => normalize_effect_name(s),
        };
        match normalized.as_str() {
            "" => Self::None,
            "red" | "red seal" => Self::Red,
            "blue" | "blue seal" => Self::Blue,
            "gold" | "gold seal" => Self::Gold,
            "purple" | "purple seal" => Self::Purple,
            _ => Self::None,
        }
    }

    pub fn to_python_str(self) -> Option<&'static str> {
        match self {
            Self::None => None,
            Self::Red => Some("red"),
            Self::Blue => Some("blue"),
            Self::Gold => Some("gold"),
            Self::Purple => Some("purple"),
        }
    }
}

/// Packed card representation. With #[repr(C)] alignment, this is
/// 6 bytes; the compiler typically rounds up to 8 for alignment.
///
/// Compare to Python's Card dataclass which is ~200 bytes (frozen
/// dataclass + tuple of attr names + dict for metadata + string
/// objects for each field). 25-30× smaller, plus zero-allocation
/// since enums are stack-allocated u8s.
#[pyclass(name = "RustCard")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Card {
    pub rank: Rank,
    pub suit: Suit,
    pub enhancement: Enhancement,
    pub edition: Edition,
    pub seal: Seal,
    pub debuffed: bool,
}

#[pymethods]
impl Card {
    /// Build a RustCard from a Python `balatro_ai.api.state.Card`.
    /// Reads attributes via PyO3 getattr — slower than a fully
    /// native construction, but acceptable at the FFI boundary
    /// since cards are constructed once per state import.
    #[staticmethod]
    pub fn from_python(py_card: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rank_str: String = py_card.getattr("rank")?.extract()?;
        let suit_str: String = py_card.getattr("suit")?.extract()?;
        let enhancement: Option<String> = py_card.getattr("enhancement")?.extract()?;
        let edition: Option<String> = py_card.getattr("edition")?.extract()?;
        let seal: Option<String> = py_card.getattr("seal")?.extract()?;
        let debuffed: bool = py_card.getattr("debuffed")?.extract()?;

        let rank = Rank::from_str(&rank_str).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("unknown rank: {rank_str:?}"))
        })?;
        let suit = Suit::from_str(&suit_str).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("unknown suit: {suit_str:?}"))
        })?;

        Ok(Self {
            rank,
            suit,
            enhancement: Enhancement::from_option_str(enhancement.as_deref()),
            edition: Edition::from_option_str(edition.as_deref()),
            seal: Seal::from_option_str(seal.as_deref()),
            debuffed,
        })
    }

    /// Round-trip back to a Python `Card`. Used by tests; not part
    /// of the hot path. Metadata is reconstructed as an empty dict
    /// (the Python field, not the dict value).
    pub fn to_python<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let state_module = py.import_bound("balatro_ai.api.state")?;
        let card_cls = state_module.getattr("Card")?;
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("rank", self.rank.to_str())?;
        kwargs.set_item("suit", self.suit.to_str())?;
        if let Some(s) = self.enhancement.to_python_str() {
            kwargs.set_item("enhancement", s)?;
        }
        if let Some(s) = self.edition.to_python_str() {
            kwargs.set_item("edition", s)?;
        }
        if let Some(s) = self.seal.to_python_str() {
            kwargs.set_item("seal", s)?;
        }
        kwargs.set_item("debuffed", self.debuffed)?;
        card_cls.call((), Some(&kwargs))
    }

    /// Accessor: rank as Python string. Cheap to call from Python.
    #[getter]
    fn get_rank(&self) -> &'static str { self.rank.to_str() }

    #[getter]
    fn get_suit(&self) -> &'static str { self.suit.to_str() }

    #[getter]
    fn get_enhancement(&self) -> Option<&'static str> { self.enhancement.to_python_str() }

    #[getter]
    fn get_edition(&self) -> Option<&'static str> { self.edition.to_python_str() }

    #[getter]
    fn get_seal(&self) -> Option<&'static str> { self.seal.to_python_str() }

    #[getter]
    fn get_debuffed(&self) -> bool { self.debuffed }

    /// `repr()` for debugging.
    fn __repr__(&self) -> String {
        format!(
            "RustCard(rank={}, suit={}, enhancement={:?}, edition={:?}, seal={:?}, debuffed={})",
            self.rank.to_str(),
            self.suit.to_str(),
            self.enhancement.to_python_str(),
            self.edition.to_python_str(),
            self.seal.to_python_str(),
            self.debuffed,
        )
    }
}

/// Pure-Rust port of `_normalize_effect_name` from
/// `hand_evaluator.py:1738`. Strip "m_" prefix, replace
/// underscores with spaces, lowercase. Used by all the enum
/// `from_option_str` parsers.
#[inline]
pub(crate) fn normalize_effect_name(name: &str) -> String {
    let stripped = name.strip_prefix("m_").unwrap_or(name);
    stripped.replace('_', " ").to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_round_trip() {
        for s in &["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"] {
            let r = Rank::from_str(s).expect(s);
            assert_eq!(r.to_str(), *s);
        }
    }

    #[test]
    fn rank_t_alias_normalizes_to_10() {
        let r = Rank::from_str("T").unwrap();
        assert_eq!(r, Rank::Ten);
        assert_eq!(r.to_str(), "10");
    }

    #[test]
    fn suit_round_trip() {
        for s in &["C", "D", "H", "S"] {
            let suit = Suit::from_str(s).expect(s);
            assert_eq!(suit.to_str(), *s);
        }
    }

    #[test]
    fn enhancement_handles_m_prefix() {
        assert_eq!(Enhancement::from_option_str(Some("m_stone")), Enhancement::Stone);
        assert_eq!(Enhancement::from_option_str(Some("m_glass")), Enhancement::Glass);
        assert_eq!(Enhancement::from_option_str(Some("Stone Card")), Enhancement::Stone);
        assert_eq!(Enhancement::from_option_str(None), Enhancement::None);
    }

    #[test]
    fn edition_handles_holo_alias() {
        assert_eq!(Edition::from_option_str(Some("holo")), Edition::Holographic);
        assert_eq!(Edition::from_option_str(Some("Holographic")), Edition::Holographic);
        assert_eq!(Edition::from_option_str(Some("Negative")), Edition::Negative);
    }

    #[test]
    fn card_struct_size_is_compact() {
        // Sanity check: the packed card fits in a single cache line
        // word. If this assertion fires after a field addition,
        // reconsider whether the new field is worth the size hit.
        assert!(std::mem::size_of::<Card>() <= 8,
            "Card grew to {} bytes — review the layout",
            std::mem::size_of::<Card>());
    }
}
