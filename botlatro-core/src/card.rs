// Card-related native functions.
//
// First port (proof-of-concept): `is_stone_card`. This is the most
// frequently called card predicate in the hot path — every call to
// `evaluate_played_cards` calls it once per played card (5 calls per
// hand) and every greedy rollout calls it ~218 plays × 5 cards =
// 1090 times per rollout step. With 4 samples per leaf × 100+ leaves
// per decision × 7 steps per rollout = on the order of 3M calls per
// decision. Even modest per-call speedup compounds.
//
// FFI pattern: accept the Python `Card` object via `&Bound<'_, PyAny>`,
// extract the `enhancement` attribute as `Option<String>`, then run
// the same normalization the Python `_is_stone_enhancement` does.
//
// We don't try to cache (Python does, via lru_cache) — the Python
// cache is process-local and dispatching into Rust for cache lookups
// would add FFI overhead with no gain. Rust's match is fast enough
// that recomputing on every call beats a cache miss + dispatch.

use pyo3::prelude::*;

/// Return True if the card has a Stone enhancement.
///
/// Parity contract: must return the exact same boolean as
/// `balatro_ai.rules.hand_evaluator._is_stone_card` for every Card
/// input. The parity test in
/// `tests/test_rust_card_parity.py` enforces this on the audit
/// corpus.
#[pyfunction]
pub fn is_stone_card(card: &Bound<'_, PyAny>) -> PyResult<bool> {
    // The Python version reads `card.enhancement` (Optional[str]),
    // runs `_normalize_effect_name` (strip "m_" prefix, replace "_"
    // with " ", lowercase), then checks membership in
    // STONE_ENHANCEMENTS = {"stone", "stone card"}.
    let enhancement: Option<String> = card.getattr("enhancement")?.extract()?;
    Ok(is_stone_enhancement(enhancement.as_deref()))
}

/// Helper: pure-Rust version of `_is_stone_enhancement`.
/// Exposed at the module level so it can be reused by future ports
/// (e.g. `_card_chip_value` needs the same check).
#[inline]
fn is_stone_enhancement(name: Option<&str>) -> bool {
    let normalized = normalize_effect_name(name);
    matches!(normalized.as_str(), "stone" | "stone card")
}

/// Pure-Rust port of `_normalize_effect_name`.
/// Strips the "m_" prefix, replaces underscores with spaces, lowercases.
#[inline]
fn normalize_effect_name(name: Option<&str>) -> String {
    match name {
        None => String::new(),
        Some(s) => {
            let stripped = s.strip_prefix("m_").unwrap_or(s);
            stripped.replace('_', " ").to_lowercase()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_strips_m_prefix() {
        assert_eq!(normalize_effect_name(Some("m_stone")), "stone");
        assert_eq!(normalize_effect_name(Some("m_glass_card")), "glass card");
    }

    #[test]
    fn normalize_handles_none() {
        assert_eq!(normalize_effect_name(None), "");
    }

    #[test]
    fn is_stone_matches_aliases() {
        assert!(is_stone_enhancement(Some("stone")));
        assert!(is_stone_enhancement(Some("Stone Card")));
        assert!(is_stone_enhancement(Some("m_stone")));
        assert!(!is_stone_enhancement(Some("bonus")));
        assert!(!is_stone_enhancement(None));
    }
}
