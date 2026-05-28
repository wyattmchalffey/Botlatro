from __future__ import annotations

import json
from pathlib import Path
import unittest

import context  # noqa: F401
from balatro_ai.rng.capture_surfaces import pack_fixture_path
from balatro_ai.rng.surfaces import PredictedCard, predict_pack_contents
from balatro_ai.rng.validate_surfaces import (
    CardSignature,
    actual_card_signature,
    check_pack_fixture,
    load_pack_fixture,
)


FIXTURE_DIR = Path(__file__).resolve().parent.parent / ".data" / "rng-validation"

ENHANCEMENT_LABELS = {
    "m_bonus": "Bonus Card",
    "m_mult": "Mult Card",
    "m_wild": "Wild Card",
    "m_glass": "Glass Card",
    "m_steel": "Steel Card",
    "m_stone": "Stone Card",
    "m_gold": "Gold Card",
    "m_lucky": "Lucky Card",
}


def _bridge_card(card: PredictedCard) -> dict[str, object]:
    if card.front_key is not None:
        payload: dict[str, object] = {
            "key": card.front_key,
            "set": "DEFAULT",
            "label": ENHANCEMENT_LABELS.get(card.key, "Base Card"),
        }
        modifier: dict[str, object] = {}
        if card.key.startswith("m_"):
            modifier["enhancement"] = card.key.removeprefix("m_").upper()
        if card.edition:
            modifier["edition"] = card.edition.upper()
        if card.seal:
            modifier["seal"] = card.seal.upper()
        if modifier:
            payload["modifier"] = modifier
        return payload

    payload = {"key": card.key, "set": card.set.upper(), "label": card.key}
    if card.edition:
        payload["edition"] = card.edition.upper()
    if card.seal:
        payload["seal"] = card.seal.upper()
    return payload


class PackSurfaceValidationHelperTests(unittest.TestCase):
    def test_pack_fixture_path_distinguishes_forced_and_natural_captures(self) -> None:
        natural = pack_fixture_path("AAAAAAA", pack_index=1, pack_key="p_standard_normal_1")
        forced = pack_fixture_path("AAAAAAA", pack_key="p_standard_normal_1", forced=True)
        voucher = pack_fixture_path(
            "AAAAAAA",
            pack_key="p_standard_normal_1",
            forced=True,
            vouchers=("v_omen_globe",),
        )
        self.assertIn("pack1_p_standard_normal_1", natural.name)
        self.assertIn("forced_p_standard_normal_1", forced.name)
        self.assertIn("forced_p_standard_normal_1_with_v_omen_globe", voucher.name)
        self.assertNotEqual(natural, forced)
        self.assertNotEqual(forced, voucher)

    def test_actual_card_signature_normalizes_playing_modifiers(self) -> None:
        card = {
            "key": "S_A",
            "set": "DEFAULT",
            "label": "Bonus Card",
            "modifier": {"edition": "FOIL", "seal": "RED"},
        }
        self.assertEqual(
            actual_card_signature(card),
            CardSignature("Enhanced", "m_bonus", "S_A", edition="foil", seal="red"),
        )

    def test_synthetic_pack_fixture_matches_prediction(self) -> None:
        seed = "AAAAAAA"
        pack_key = "p_arcana_normal_1"
        predicted = predict_pack_contents(seed, ante=1, pack_key=pack_key)
        fixture = {
            "seed": seed,
            "ante": 1,
            "pack_key": pack_key,
            "opened_state": {"state": "SMODS_BOOSTER_OPENED", "pack": {"cards": [_bridge_card(card) for card in predicted]}},
        }
        result = check_pack_fixture(fixture)
        self.assertEqual(result.status, "ok", result.detail)

    def test_synthetic_omen_globe_fixture_matches_prediction(self) -> None:
        seed = "BBBBBBB"
        pack_key = "p_arcana_normal_1"
        predicted = predict_pack_contents(seed, ante=1, pack_key=pack_key, vouchers=("v_omen_globe",))
        self.assertTrue(any(card.set == "Spectral" for card in predicted))
        fixture = {
            "seed": seed,
            "ante": 1,
            "pack_key": pack_key,
            "vouchers": ["v_omen_globe"],
            "opened_state": {
                "state": "SMODS_BOOSTER_OPENED",
                "used_vouchers": {"v_omen_globe": ""},
                "pack": {"cards": [_bridge_card(card) for card in predicted]},
            },
        }
        result = check_pack_fixture(fixture)
        self.assertEqual(result.status, "ok", result.detail)

    def test_synthetic_glow_up_fixture_matches_edition_rate_prediction(self) -> None:
        seed = "CCCCCCC"
        pack_key = "p_standard_normal_1"
        predicted = predict_pack_contents(seed, ante=1, pack_key=pack_key, edition_rate=4.0)
        self.assertTrue(any(card.edition for card in predicted))
        fixture = {
            "seed": seed,
            "ante": 1,
            "pack_key": pack_key,
            "vouchers": ["v_glow_up"],
            "opened_state": {
                "state": "SMODS_BOOSTER_OPENED",
                "used_vouchers": {"v_glow_up": ""},
                "pack": {"cards": [_bridge_card(card) for card in predicted]},
            },
        }
        result = check_pack_fixture(fixture)
        self.assertEqual(result.status, "ok", result.detail)

    def test_synthetic_telescope_fixture_matches_forced_planet_prediction(self) -> None:
        seed = "AAAAAAA"
        pack_key = "p_celestial_normal_1"
        predicted = predict_pack_contents(
            seed,
            ante=1,
            pack_key=pack_key,
            vouchers=("v_telescope",),
            played_hand_types=frozenset({"High Card"}),
            telescope_planet_key="c_pluto",
        )
        self.assertEqual(predicted[0].key, "c_pluto")
        fixture = {
            "seed": seed,
            "ante": 1,
            "pack_key": pack_key,
            "vouchers": ["v_telescope"],
            "played_hands": {"High Card": 3},
            "opened_state": {
                "state": "SMODS_BOOSTER_OPENED",
                "used_vouchers": {"v_telescope": ""},
                "pack": {"cards": [_bridge_card(card) for card in predicted]},
            },
        }
        result = check_pack_fixture(fixture)
        self.assertEqual(result.status, "ok", result.detail)


class CapturedPackSurfaceFixtureTests(unittest.TestCase):
    def test_captured_pack_fixtures_match_predictions(self) -> None:
        paths = sorted(FIXTURE_DIR.glob("pack_seed_*.json"))
        if not paths:
            self.skipTest("No opened-pack fixtures; run `python -m balatro_ai.rng.capture_surfaces --all --all-pack-kinds`")
        for path in paths:
            with self.subTest(path=path.name):
                fixture = load_pack_fixture(path)
                result = check_pack_fixture(fixture)
                self.assertEqual(result.status, "ok", f"{path.name}: {result.detail}")


if __name__ == "__main__":
    unittest.main()
