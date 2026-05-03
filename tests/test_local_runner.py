from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState, Joker, with_derived_legal_actions
from balatro_ai.search.shop_sampler import ShopSampler
from balatro_ai.sim.local_runner import LocalBalatroSimulator, LocalSimOptions, run_local_seed


def tiny_sim_data() -> dict[str, object]:
    return {
        "shop_slots": {"base_joker_max": 2, "booster_slots": 1, "voucher_slots": 1},
        "slot_rates": {"Joker": 1},
        "joker_rarity_rates": {"common": 1.0, "uncommon": 0.0, "rare": 0.0, "legendary": 0.0},
        "edition_rates": {"negative": 0.0, "polychrome": 0.0, "holographic": 0.0, "foil": 0.0},
        "reroll": {"base_cost": 5, "increase_per_reroll": 1},
        "jokers": [
            {"key": "j_joker", "name": "Joker", "set": "Joker", "cost": 2, "rarity_name": "common"},
            {"key": "j_greedy_joker", "name": "Greedy Joker", "set": "Joker", "cost": 5, "rarity_name": "common"},
        ],
        "tarots": [{"key": "c_empress", "name": "The Empress", "set": "Tarot", "cost": 3}],
        "planets": [{"key": "c_jupiter", "name": "Jupiter", "set": "Planet", "cost": 3}],
        "spectrals": [{"key": "c_hex", "name": "Hex", "set": "Spectral", "cost": 4}],
        "vouchers": [{"key": "v_blank", "name": "Blank", "set": "Voucher", "cost": 10, "available_default": True}],
        "boosters": [
            {
                "key": "p_buffoon_normal_1",
                "name": "Buffoon Pack",
                "set": "Booster",
                "kind": "Buffoon",
                "config": {"extra": 2, "choose": 1},
                "cost": 4,
                "weight": 1,
            }
        ],
    }


class ScriptedClearBot:
    name = "scripted_clear_bot"

    def choose_action(self, state: GameState) -> Action:
        if state.phase == GamePhase.BLIND_SELECT:
            return _first_action(state, ActionType.SELECT_BLIND)
        if state.phase == GamePhase.SELECTING_HAND:
            return Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4))
        if state.phase == GamePhase.ROUND_EVAL:
            return _first_action(state, ActionType.CASH_OUT)
        if state.phase == GamePhase.SHOP:
            return _first_action(state, ActionType.END_SHOP)
        return state.legal_actions[0]


def _first_action(state: GameState, action_type: ActionType) -> Action:
    for action in state.legal_actions:
        if action.action_type == action_type:
            return action
    raise AssertionError(f"No {action_type.value} action in {state.legal_actions}")


class FixedRandom:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


class LocalRunnerTests(unittest.TestCase):
    def test_reset_starts_at_ante_one_small_blind_with_exact_deck(self) -> None:
        sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Club",))

        state = sim.reset()

        self.assertEqual(state.phase, GamePhase.BLIND_SELECT)
        self.assertEqual(state.ante, 1)
        self.assertEqual(state.blind, "Small Blind")
        self.assertEqual(state.required_score, 300)
        self.assertEqual(state.money, 4)
        self.assertEqual(state.deck_size, 52)
        self.assertEqual(len(state.known_deck), 52)
        self.assertEqual(state.modifiers["current_blind"]["type"], "SMALL")
        self.assertTrue(state.modifiers["current_blind"]["tag"].endswith(" Tag"))
        self.assertTrue(state.modifiers["blinds"]["small"]["skip_tag"]["key"].startswith("tag_"))
        self.assertNotIn("tag", state.modifiers["blinds"]["boss"])
        self.assertIn(ActionType.SELECT_BLIND, {action.action_type for action in state.legal_actions})

    def test_needle_uses_source_truth_one_x_score_multiplier(self) -> None:
        sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Needle",))

        state = sim.reset()

        self.assertEqual(state.modifiers["blinds"]["boss"]["name"], "The Needle")
        self.assertEqual(state.modifiers["blinds"]["boss"]["score"], 300)

    def test_default_boss_pool_uses_source_min_antes_and_showdown_pool(self) -> None:
        sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()))

        ante_one = set(sim._eligible_boss_pool(1))
        ante_two = set(sim._eligible_boss_pool(2))
        ante_eight = set(sim._eligible_boss_pool(8))

        self.assertIn("The Hook", ante_one)
        self.assertIn("The Manacle", ante_one)
        self.assertNotIn("The Mouth", ante_one)
        self.assertIn("The Mouth", ante_two)
        self.assertNotIn("The Ox", ante_two)
        self.assertEqual(
            ante_eight,
            {"Cerulean Bell", "Verdant Leaf", "Violet Vessel", "Amber Acorn", "Crimson Heart"},
        )

    def test_final_boss_payload_uses_showdown_reward_and_multiplier(self) -> None:
        sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("Violet Vessel",))

        state = sim._with_tagged_blind_selection_surface(GameState(), ante=8, blind_kind="BOSS", boss_name="Violet Vessel")

        self.assertEqual(state.required_score, 300000)
        self.assertEqual(state.modifiers["blind_reward"], 8)
        self.assertEqual(state.modifiers["current_blind"]["key"], "bl_final_vessel")

    def test_manacle_reduces_opening_hand_size_for_the_boss_blind(self) -> None:
        sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Manacle",))
        sim.state = GameState(
            phase=GamePhase.BLIND_SELECT,
            blind="The Manacle",
            deck_size=10,
            known_deck=tuple(Card("A", "S") for _ in range(10)),
            modifiers={"current_blind": {"name": "The Manacle", "type": "BOSS"}, "hand_size": 8},
        )

        next_state = sim.step(Action(ActionType.SELECT_BLIND))

        self.assertEqual(len(next_state.hand), 7)
        self.assertEqual(next_state.modifiers["hand_size"], 7)

    def test_house_mark_and_fish_apply_source_face_down_draw_rules(self) -> None:
        house_sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The House",))
        house_sim.state = GameState(
            phase=GamePhase.BLIND_SELECT,
            blind="The House",
            deck_size=3,
            known_deck=(Card("2", "S"), Card("3", "H"), Card("4", "D")),
            hands_remaining=4,
            discards_remaining=4,
            modifiers={"current_blind": {"name": "The House", "type": "BOSS"}, "hand_size": 3},
        )

        house_state = house_sim.step(Action(ActionType.SELECT_BLIND))

        self.assertTrue(all(card.metadata.get("face_down") for card in house_state.hand))

        mark_sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Mark",))
        mark_sim.state = GameState(
            phase=GamePhase.BLIND_SELECT,
            blind="The Mark",
            deck_size=3,
            known_deck=(Card("K", "S"), Card("3", "H"), Card("Q", "D")),
            hands_remaining=4,
            discards_remaining=4,
            modifiers={"current_blind": {"name": "The Mark", "type": "BOSS"}, "hand_size": 3},
        )

        mark_state = mark_sim.step(Action(ActionType.SELECT_BLIND))

        self.assertEqual(sum(1 for card in mark_state.hand if card.metadata.get("face_down")), 2)

        fish_sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Fish",))
        fish_sim.state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="The Fish",
            required_score=10_000,
            hands_remaining=4,
            deck_size=1,
            hand=(Card("A", "S"), Card("2", "C")),
            known_deck=(Card("3", "D"),),
            modifiers={"current_blind": {"name": "The Fish", "type": "BOSS"}, "hand_size": 2},
        )

        fish_state = fish_sim.step(Action(ActionType.PLAY_HAND, card_indices=(0,)))

        self.assertEqual(sum(1 for card in fish_state.hand if card.metadata.get("face_down")), 1)

    def test_serpent_draws_three_after_play_even_past_normal_hand_space(self) -> None:
        sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Serpent",))
        sim.state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="The Serpent",
            required_score=10_000,
            hands_remaining=4,
            deck_size=3,
            hand=(Card("A", "S"), Card("K", "H"), Card("Q", "D"), Card("J", "C"), Card("10", "S")),
            known_deck=(Card("9", "S"), Card("8", "H"), Card("7", "D")),
            modifiers={"current_blind": {"name": "The Serpent", "type": "BOSS"}, "hand_size": 5},
        )

        next_state = sim.step(Action(ActionType.PLAY_HAND, card_indices=(0,)))

        self.assertEqual(len(next_state.hand), 7)
        self.assertEqual(next_state.deck_size, 0)

    def test_pillar_plant_and_verdant_apply_card_debuffs(self) -> None:
        played = Card("A", "S", metadata={"played_this_ante": True})
        pillar_sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Pillar",))
        pillar_sim.state = GameState(
            phase=GamePhase.BLIND_SELECT,
            blind="The Pillar",
            deck_size=1,
            known_deck=(played,),
            modifiers={"current_blind": {"name": "The Pillar", "type": "BOSS"}, "hand_size": 1},
        )

        pillar_state = pillar_sim.step(Action(ActionType.SELECT_BLIND))

        self.assertTrue(pillar_state.hand[0].debuffed)
        self.assertTrue(pillar_state.hand[0].metadata["debuffed_by_blind"])

        plant_sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Plant",))
        plant_sim.state = GameState(
            phase=GamePhase.BLIND_SELECT,
            blind="The Plant",
            deck_size=2,
            known_deck=(Card("K", "S"), Card("2", "H")),
            modifiers={"current_blind": {"name": "The Plant", "type": "BOSS"}, "hand_size": 2},
        )

        plant_state = plant_sim.step(Action(ActionType.SELECT_BLIND))

        self.assertEqual(sum(1 for card in plant_state.hand if card.debuffed), 1)

        verdant_state = with_derived_legal_actions(
            GameState(
                phase=GamePhase.SELECTING_HAND,
                blind="Verdant Leaf",
                hand=(Card("A", "S", debuffed=True, metadata={"debuffed_by_blind": True, "state": {"debuff": True}}),),
                jokers=(Joker("Joker", sell_value=1),),
                modifiers={"current_blind": {"name": "Verdant Leaf", "type": "BOSS"}},
            )
        )
        verdant_sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("Verdant Leaf",))
        verdant_sim.state = verdant_state

        sold_state = verdant_sim.step(Action(ActionType.SELL, target_id="joker", amount=0, metadata={"kind": "joker", "index": 0}))

        self.assertTrue(sold_state.modifiers["boss_disabled"])
        self.assertFalse(sold_state.hand[0].debuffed)

    def test_cerulean_bell_forces_one_selected_card_in_legal_actions(self) -> None:
        sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("Cerulean Bell",))
        sim.state = GameState(
            phase=GamePhase.BLIND_SELECT,
            blind="Cerulean Bell",
            deck_size=4,
            known_deck=(Card("A", "S"), Card("K", "H"), Card("Q", "D"), Card("J", "C")),
            modifiers={"current_blind": {"name": "Cerulean Bell", "type": "BOSS"}, "hand_size": 4},
        )

        next_state = sim.step(Action(ActionType.SELECT_BLIND))

        forced = {index for index, card in enumerate(next_state.hand) if card.metadata.get("forced_selection")}
        self.assertEqual(len(forced), 1)
        self.assertTrue(
            all(
                forced.issubset(action.card_indices)
                for action in next_state.legal_actions
                if action.action_type in {ActionType.PLAY_HAND, ActionType.DISCARD}
            )
        )

    def test_amber_acorn_flips_jokers_face_down_on_boss_start(self) -> None:
        sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("Amber Acorn",))
        sim.state = GameState(
            phase=GamePhase.BLIND_SELECT,
            blind="Amber Acorn",
            deck_size=1,
            known_deck=(Card("A", "S"),),
            jokers=(Joker("Joker"), Joker("Stuntman"), Joker("Blueprint")),
            modifiers={"current_blind": {"name": "Amber Acorn", "type": "BOSS"}, "hand_size": 1},
        )

        next_state = sim.step(Action(ActionType.SELECT_BLIND))

        self.assertEqual({joker.name for joker in next_state.jokers}, {"Joker", "Stuntman", "Blueprint"})
        self.assertTrue(all(joker.metadata.get("face_down") for joker in next_state.jokers))

    def test_select_blind_draws_opening_hand_and_reduces_deck(self) -> None:
        sim = LocalBalatroSimulator(seed=2, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Club",))
        state = sim.reset()

        state = sim.step(_first_action(state, ActionType.SELECT_BLIND))

        self.assertEqual(state.phase, GamePhase.SELECTING_HAND)
        self.assertEqual(len(state.hand), 8)
        self.assertEqual(state.deck_size, 44)
        self.assertEqual(len(state.known_deck), 44)
        self.assertGreater(len([action for action in state.legal_actions if action.action_type == ActionType.PLAY_HAND]), 0)

    def test_clear_cash_out_creates_shop_then_end_shop_advances_to_big_blind(self) -> None:
        deck = tuple(Card("A", "S") for _ in range(52))
        sim = LocalBalatroSimulator(
            seed=3,
            sampler=ShopSampler(tiny_sim_data()),
            initial_deck=deck,
            boss_pool=("The Club",),
        )
        state = sim.reset()
        state = sim.step(_first_action(state, ActionType.SELECT_BLIND))

        state = sim.step(Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4)))
        self.assertEqual(state.phase, GamePhase.ROUND_EVAL)
        self.assertGreaterEqual(state.current_score, state.required_score)

        state = sim.step(_first_action(state, ActionType.CASH_OUT))
        self.assertEqual(state.phase, GamePhase.SHOP)
        self.assertEqual(state.blind, "Small Blind")
        self.assertEqual(state.required_score, 300)
        self.assertGreater(state.money, 4)
        self.assertTrue(state.modifiers["shop_cards"])
        self.assertTrue(state.modifiers["booster_packs"])

        state = sim.step(_first_action(state, ActionType.END_SHOP))
        self.assertEqual(state.phase, GamePhase.BLIND_SELECT)
        self.assertEqual(state.blind, "Big Blind")
        self.assertEqual(state.required_score, 450)

    def test_skip_blind_adds_tag_and_double_tag_duplicates_next_non_double_tag(self) -> None:
        sim = LocalBalatroSimulator(seed=7, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Club",))
        sim.state = GameState(
            phase=GamePhase.BLIND_SELECT,
            ante=1,
            blind="Small Blind",
            money=10,
            jokers=(Joker("Throwback", metadata={"current_xmult": 1.0}),),
            modifiers={
                "current_blind": {"name": "Small Blind", "type": "SMALL", "score": 300, "dollars": 3, "tag": "Economy Tag"},
                "tags": ("Double Tag",),
            },
        )

        next_state = sim.step(Action(ActionType.SKIP_BLIND))

        self.assertEqual(next_state.phase, GamePhase.BLIND_SELECT)
        self.assertEqual(next_state.blind, "Big Blind")
        self.assertEqual(next_state.money, 40)
        self.assertEqual(next_state.modifiers["tags"], ())
        self.assertEqual(next_state.modifiers["skips"], 1)
        self.assertEqual(next_state.jokers[0].metadata["current_xmult"], 1.25)

    def test_shop_tags_apply_at_cash_out(self) -> None:
        sim = LocalBalatroSimulator(seed=8, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Club",))
        sim.state = GameState(
            phase=GamePhase.ROUND_EVAL,
            ante=1,
            blind="Small Blind",
            required_score=300,
            current_score=300,
            money=20,
            modifiers={
                "current_blind": {"name": "Small Blind", "type": "SMALL", "score": 300, "dollars": 0},
                "tags": ("D6 Tag", "Voucher Tag", "Coupon Tag", "Uncommon Tag", "Foil Tag"),
            },
        )

        next_state = sim.step(Action(ActionType.CASH_OUT))

        self.assertEqual(next_state.phase, GamePhase.SHOP)
        self.assertEqual(next_state.modifiers["reroll_cost"], 0)
        self.assertEqual(next_state.modifiers["current_reroll_cost"], 0)
        self.assertGreaterEqual(len(next_state.modifiers["voucher_cards"]), 2)
        self.assertTrue(all(item["cost"]["buy"] == 0 for item in next_state.modifiers["shop_cards"]))
        self.assertTrue(all(item["cost"]["buy"] == 0 for item in next_state.modifiers["booster_packs"]))
        self.assertIn("FOIL", {item.get("edition") for item in next_state.modifiers["shop_cards"]})
        self.assertEqual(next_state.modifiers["tags"], ())

    def test_free_pack_tag_opens_pack_then_returns_to_blind_select(self) -> None:
        sim = LocalBalatroSimulator(seed=9, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Club",))
        sim.state = GameState(
            phase=GamePhase.BLIND_SELECT,
            ante=2,
            blind="Small Blind",
            modifiers={
                "current_blind": {"name": "Small Blind", "type": "SMALL", "score": 800, "dollars": 3, "tag": "Buffoon Tag"},
            },
        )

        opened = sim.step(Action(ActionType.SKIP_BLIND))

        self.assertEqual(opened.phase, GamePhase.BOOSTER_OPENED)
        self.assertEqual(opened.pack, ("Buffoon Pack",))
        self.assertEqual({item["set"] for item in opened.modifiers["pack_cards"]}, {"JOKER"})

        returned = sim.step(Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip"}))

        self.assertEqual(returned.phase, GamePhase.BLIND_SELECT)
        self.assertEqual(returned.blind, "Big Blind")

    def test_boss_tag_rerolls_upcoming_boss(self) -> None:
        sim = LocalBalatroSimulator(seed=9, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Club", "The Eye"))
        sim.state = GameState(
            phase=GamePhase.BLIND_SELECT,
            ante=1,
            blind="Big Blind",
            modifiers={
                "current_blind": {"name": "Big Blind", "type": "BIG", "score": 450, "dollars": 4, "tag": "Boss Tag"},
                "blinds": {
                    "small": {"name": "Small Blind", "type": "SMALL", "score": 300, "dollars": 3},
                    "big": {"name": "Big Blind", "type": "BIG", "score": 450, "dollars": 4},
                    "boss": {"name": "The Club", "type": "BOSS", "score": 600, "dollars": 5},
                },
                "upcoming_boss": {"name": "The Club", "type": "BOSS", "score": 600, "dollars": 5},
            },
        )

        next_state = sim.step(Action(ActionType.SKIP_BLIND))

        self.assertEqual(next_state.phase, GamePhase.BLIND_SELECT)
        self.assertEqual(next_state.blind, "The Eye")
        self.assertEqual(next_state.modifiers["blinds"]["boss"]["name"], "The Eye")
        self.assertEqual(next_state.modifiers["tags"], ())

    def test_directors_cut_adds_one_paid_boss_reroll_action(self) -> None:
        sim = LocalBalatroSimulator(seed=9, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Club", "The Eye"))
        state = with_derived_legal_actions(
            GameState(
                phase=GamePhase.BLIND_SELECT,
                ante=1,
                blind="Small Blind",
                money=10,
                vouchers=("Director's Cut",),
                modifiers={
                    "current_blind": {"name": "Small Blind", "type": "SMALL", "score": 300, "dollars": 3},
                    "blinds": {
                        "small": {"name": "Small Blind", "type": "SMALL", "score": 300, "dollars": 3},
                        "big": {"name": "Big Blind", "type": "BIG", "score": 450, "dollars": 4},
                        "boss": {"name": "The Club", "type": "BOSS", "score": 600, "dollars": 5},
                    },
                    "upcoming_boss": {"name": "The Club", "type": "BOSS", "score": 600, "dollars": 5},
                },
            )
        )
        sim.state = state
        action = next(action for action in state.legal_actions if action.action_type == ActionType.REROLL)

        next_state = sim.step(action)

        self.assertEqual(action.metadata["kind"], "boss")
        self.assertEqual(next_state.money, 0)
        self.assertEqual(next_state.modifiers["blinds"]["boss"]["name"], "The Eye")
        self.assertTrue(next_state.modifiers["boss_rerolled"])
        self.assertNotIn(ActionType.REROLL, {action.action_type for action in next_state.legal_actions})

    def test_retcon_allows_repeated_paid_boss_rerolls(self) -> None:
        state = with_derived_legal_actions(
            GameState(
                phase=GamePhase.BLIND_SELECT,
                money=20,
                vouchers=("Retcon",),
                modifiers={"current_blind": {"name": "Small Blind", "type": "SMALL"}},
            )
        )

        rerolls = [action for action in state.legal_actions if action.action_type == ActionType.REROLL]

        self.assertEqual(len(rerolls), 1)
        self.assertEqual(rerolls[0].metadata["kind"], "boss")

    def test_investment_tag_pays_when_boss_is_cleared(self) -> None:
        sim = LocalBalatroSimulator(seed=10, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Club",))
        sim.state = GameState(
            phase=GamePhase.ROUND_EVAL,
            ante=1,
            blind="The Club",
            required_score=600,
            current_score=600,
            money=5,
            modifiers={
                "current_blind": {"name": "The Club", "type": "BOSS", "score": 600, "dollars": 0},
                "tags": ("Investment Tag",),
            },
        )

        next_state = sim.step(Action(ActionType.CASH_OUT))

        self.assertEqual(next_state.phase, GamePhase.SHOP)
        self.assertGreaterEqual(next_state.money, 30)
        self.assertEqual(next_state.modifiers["tags"], ())

    def test_juggle_tag_adds_temporary_hand_size_on_blind_select(self) -> None:
        sim = LocalBalatroSimulator(
            seed=11,
            sampler=ShopSampler(tiny_sim_data()),
            initial_deck=tuple(Card("A", "S") for _ in range(10)),
            boss_pool=("The Club",),
        )
        sim.state = GameState(
            phase=GamePhase.BLIND_SELECT,
            ante=1,
            blind="Small Blind",
            deck_size=10,
            known_deck=tuple(Card("A", "S") for _ in range(10)),
            modifiers={
                "current_blind": {"name": "Small Blind", "type": "SMALL", "score": 300, "dollars": 3},
                "hand_size": 2,
                "tags": ("Juggle Tag",),
            },
        )

        next_state = sim.step(Action(ActionType.SELECT_BLIND))

        self.assertEqual(next_state.phase, GamePhase.SELECTING_HAND)
        self.assertEqual(len(next_state.hand), 5)
        self.assertEqual(next_state.modifiers["tags"], ())

    def test_buffoon_pack_contents_are_sampled_as_jokers(self) -> None:
        sampler = ShopSampler(tiny_sim_data())
        pack = {
            "key": "p_buffoon_normal_1",
            "name": "Buffoon Pack",
            "set": "Booster",
            "kind": "Buffoon",
            "config": {"extra": 2, "choose": 1},
        }

        contents = sampler.sample_pack_contents(GameState(), pack)

        self.assertEqual(len(contents), 2)
        self.assertEqual({item["set"] for item in contents}, {"JOKER"})

    def test_derived_shop_actions_honor_visible_reroll_cost(self) -> None:
        state = with_derived_legal_actions(GameState(phase=GamePhase.SHOP, money=5, modifiers={"reroll_cost": 6}))

        self.assertNotIn(ActionType.REROLL, {action.action_type for action in state.legal_actions})

        free_state = with_derived_legal_actions(
            GameState(phase=GamePhase.SHOP, money=0, modifiers={"reroll_cost": 6, "free_rerolls": 1})
        )
        self.assertIn(ActionType.REROLL, {action.action_type for action in free_state.legal_actions})

    def test_buying_overstock_refills_prior_empty_shop_slots(self) -> None:
        data = tiny_sim_data()
        data["vouchers"] = [{"key": "v_overstock_norm", "name": "Overstock", "set": "Voucher", "cost": 10, "available_default": True}]
        data["jokers"] = [
            {"key": "j_visible", "name": "Visible Joker", "set": "Joker", "cost": 2, "rarity_name": "common"},
            {"key": "j_fill_a", "name": "Fill A", "set": "Joker", "cost": 2, "rarity_name": "common"},
            {"key": "j_fill_b", "name": "Fill B", "set": "Joker", "cost": 2, "rarity_name": "common"},
        ]
        sim = LocalBalatroSimulator(seed=5, sampler=ShopSampler(data), boss_pool=("The Club",))
        visible = {"key": "j_visible", "name": "Visible Joker", "set": "Joker", "cost": {"buy": 2}}
        sim.state = GameState(
            phase=GamePhase.SHOP,
            money=20,
            modifiers={
                "shop_cards": (visible,),
                "voucher_cards": ({"key": "v_overstock_norm", "name": "Overstock", "set": "Voucher", "cost": {"buy": 10}},),
            },
        )

        next_state = sim.step(Action(ActionType.BUY, target_id="voucher", amount=0, metadata={"kind": "voucher", "index": 0}))

        shop_cards = next_state.modifiers["shop_cards"]
        self.assertEqual(len(shop_cards), 3)
        self.assertIs(shop_cards[0], visible)
        self.assertEqual(next_state.shop[0], "Visible Joker")

    def test_hook_play_samples_held_discards_and_refills_hand(self) -> None:
        sim = LocalBalatroSimulator(seed=6, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Hook",))
        sim.state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="The Hook",
            required_score=10_000,
            hands_remaining=4,
            deck_size=3,
            known_deck=(Card("9", "S"), Card("8", "D"), Card("7", "C")),
            hand=(Card("A", "S"), Card("K", "H"), Card("Q", "D"), Card("J", "C")),
            modifiers={"hand_size": 4},
        )

        next_state = sim.step(Action(ActionType.PLAY_HAND, card_indices=(0,)))

        self.assertEqual(next_state.phase, GamePhase.SELECTING_HAND)
        self.assertEqual(next_state.deck_size, 0)
        self.assertEqual(len(next_state.hand), 4)
        self.assertEqual(tuple(card.short_name for card in next_state.known_deck), ())
        self.assertEqual(len(next_state.modifiers["discard_pile"]), 2)

    def test_misprint_play_samples_random_mult(self) -> None:
        sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Club",))
        sim.state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=10_000,
            hands_remaining=4,
            deck_size=0,
            hand=(Card("A", "S"),),
            jokers=(Joker("Misprint"),),
            modifiers={"hand_size": 1},
        )

        next_state = sim.step(Action(ActionType.PLAY_HAND, card_indices=(0,)))

        self.assertEqual(next_state.phase, GamePhase.SELECTING_HAND)
        self.assertEqual(next_state.current_score, 80)

    def test_local_runner_counts_blueprint_copied_rng_jokers(self) -> None:
        sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Club",))
        sim.state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            required_score=10_000,
            hands_remaining=4,
            deck_size=0,
            hand=(Card("A", "S"),),
            jokers=(Joker("Blueprint"), Joker("Space Joker")),
            modifiers={"hand_size": 1, "probability_multiplier": 100},
        )

        next_state = sim.step(Action(ActionType.PLAY_HAND, card_indices=(0,)))

        self.assertEqual(next_state.current_score, 108)

    def test_oops_probability_multiplier_makes_wheel_boss_flips_guaranteed(self) -> None:
        sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Wheel",))
        sim.state = GameState(
            phase=GamePhase.BLIND_SELECT,
            blind="The Wheel",
            deck_size=3,
            known_deck=(Card("A", "S"), Card("K", "H"), Card("Q", "D")),
            hands_remaining=4,
            discards_remaining=4,
            jokers=(Joker("Oops! All 6s"),),
            modifiers={
                "current_blind": {"type": "BOSS", "name": "The Wheel"},
                "hand_size": 3,
                "probability_multiplier": 7,
            },
        )

        next_state = sim.step(Action(ActionType.SELECT_BLIND))

        self.assertEqual(len(next_state.hand), 3)
        self.assertTrue(all(card.metadata.get("face_down") for card in next_state.hand))
        self.assertTrue(all(card.metadata.get("state", {}).get("facing") == "back" for card in next_state.hand))

    def test_oops_probability_multiplier_makes_wheel_of_fortune_guaranteed(self) -> None:
        sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Club",))
        sim.state = GameState(
            phase=GamePhase.BOOSTER_OPENED,
            pack=("The Wheel of Fortune",),
            jokers=(Joker("Joker", sell_value=1),),
            modifiers={
                "pack_cards": ({"label": "The Wheel of Fortune", "set": "TAROT"},),
                "probability_multiplier": 4,
            },
        )

        next_state = sim.step(
            Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=0, metadata={"kind": "card", "index": 0})
        )

        self.assertIn(next_state.jokers[0].edition, {"FOIL", "HOLOGRAPHIC", "POLYCHROME"})
        self.assertEqual(next_state.modifiers["tarot_cards_used"]["The Wheel of Fortune"], 1)

    def test_derived_actions_allow_stored_tarot_use_and_consumable_sell_in_blind_and_pack(self) -> None:
        blind_state = with_derived_legal_actions(
            GameState(
                phase=GamePhase.SELECTING_HAND,
                hand=(Card("A", "S"), Card("K", "H")),
                consumables=("The Empress",),
            )
        )
        blind_use_actions = [action for action in blind_state.legal_actions if action.action_type == ActionType.USE_CONSUMABLE]
        blind_sell_actions = [
            action
            for action in blind_state.legal_actions
            if action.action_type == ActionType.SELL and action.metadata.get("kind") == "consumable"
        ]

        self.assertIn((0, 1), {action.card_indices for action in blind_use_actions})
        self.assertEqual(len(blind_sell_actions), 1)

        pack_state = with_derived_legal_actions(
            GameState(
                phase=GamePhase.BOOSTER_OPENED,
                pack=("8S",),
                hand=(Card("A", "S"), Card("K", "H")),
                consumables=("Death",),
                modifiers={"pack_cards": ({"rank": "8", "suit": "S", "set": "Base"},)},
            )
        )
        pack_use_actions = [action for action in pack_state.legal_actions if action.action_type == ActionType.USE_CONSUMABLE]

        self.assertIn((0, 1), {action.card_indices for action in pack_use_actions})
        self.assertTrue(any(action.action_type == ActionType.CHOOSE_PACK_CARD for action in pack_state.legal_actions))

    def test_wheel_of_fortune_edition_poll_uses_source_truth_guaranteed_odds(self) -> None:
        sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("The Club",))

        for poll, expected in (
            (0.49, "FOIL"),
            (0.50, "FOIL"),
            (0.51, "HOLOGRAPHIC"),
            (0.85, "HOLOGRAPHIC"),
            (0.86, "POLYCHROME"),
            (0.99, "POLYCHROME"),
        ):
            sim._rng = FixedRandom(poll)  # type: ignore[assignment]
            self.assertEqual(sim._wheel_of_fortune_edition(), expected)

    def test_crimson_heart_rotates_disabled_joker_after_play(self) -> None:
        sim = LocalBalatroSimulator(seed=1, sampler=ShopSampler(tiny_sim_data()), boss_pool=("Crimson Heart",))
        sim.state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Crimson Heart",
            required_score=10_000,
            hands_remaining=4,
            deck_size=0,
            hand=(Card("A", "S"),),
            jokers=(
                Joker("Joker", metadata={"state": {"debuff": True}, "value": {"effect": "All abilities are disabled"}}),
                Joker("Stuntman", metadata={"value": {"effect": "+250 Chips, -2 hand size"}}),
            ),
            modifiers={"hand_size": 1},
        )

        next_state = sim.step(Action(ActionType.PLAY_HAND, card_indices=(0,)))

        self.assertEqual(next_state.current_score, 266)
        self.assertFalse(next_state.jokers[0].metadata.get("state"))
        self.assertEqual(next_state.jokers[1].metadata["state"]["debuff"], True)

    def test_run_local_seed_records_local_result(self) -> None:
        deck = tuple(Card("A", "S") for _ in range(52))

        result = run_local_seed(
            bot=ScriptedClearBot(),
            options=LocalSimOptions(seed=4, max_steps=4, initial_deck=deck, boss_pool=("The Club",)),
        )

        self.assertFalse(result.won)
        self.assertEqual(result.ante_reached, 1)
        self.assertEqual(result.death_reason, "max_steps")
        self.assertLess(result.runtime_seconds, 1.0)


if __name__ == "__main__":
    unittest.main()
