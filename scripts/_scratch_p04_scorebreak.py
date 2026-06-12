"""Scratch: breakdown of the sim's scoring at a given step for a seed."""
from __future__ import annotations

import os
import sys

os.environ["BALATRO_SEED_FAITHFUL"] = "1"

from balatro_ai.api.actions import ActionType  # noqa: E402
from balatro_ai.api.state import with_derived_legal_actions  # noqa: E402
from balatro_ai.bots.registry import create_bot  # noqa: E402
from balatro_ai.rules.hand_evaluator import debuffed_suits_for_blind, evaluate_played_cards  # noqa: E402
from balatro_ai.sim.local_runner import LocalBalatroSimulator  # noqa: E402
from balatro_ai.solver.seed_game import SeedGame  # noqa: E402
from balatro_ai.solver.trajectory import _stable_seed_int  # noqa: E402
from balatro_ai.search import forward_sim as fs  # noqa: E402


def main():
    seed = sys.argv[1]
    target_step = int(sys.argv[2])
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white", balatro_seed=seed)
    sim.state = with_derived_legal_actions(SeedGame(seed, stake="white").initial_state())
    bot = create_bot("basic_strategy_bot", seed=0)
    for i in range(target_step + 1):
        s = sim.state
        action = bot.choose_action(s)
        if i == target_step:
            assert action.action_type == ActionType.PLAY_HAND, action.action_type
            selected_cards, held_cards = fs._split_action_cards(s, action)
            blind_name = fs._effective_blind_name(s)
            state_jokers = fs._jokers_with_dynamic_state_values(s)
            print(f"step={i} blind={blind_name!r} money={s.money} hands_rem={s.hands_remaining} disc_rem={s.discards_remaining} deck_size={s.deck_size}")
            print(f"hand_levels={dict(s.hand_levels)}")
            for j in state_jokers:
                print(f"JOKER {j.name}: metadata={j.metadata}")
            print(f"played={[f'{c.suit}_{c.rank}' for c in selected_cards]}")
            print(f"held ={[f'{c.suit}_{c.rank}' for c in held_cards]}")
            ev = evaluate_played_cards(
                selected_cards,
                s.hand_levels,
                debuffed_suits=debuffed_suits_for_blind(blind_name),
                blind_name=blind_name,
                jokers=state_jokers,
                discards_remaining=s.discards_remaining,
                hands_remaining=s.hands_remaining,
                held_cards=held_cards,
                deck_size=s.deck_size,
                money=s.money,
                played_hand_types_this_round=fs._played_hand_types_this_round(s),
                played_hand_counts=fs._played_hand_counts(s),
            )
            print(f"hand_type={ev.hand_type.value} level={ev.level}")
            print(f"scoring_indices={ev.scoring_indices} -> {[f'{ev.cards[k].suit}_{ev.cards[k].rank}' for k in ev.scoring_indices]}")
            print(f"base_chips={ev.base_chips} base_mult={ev.base_mult} level_chips={ev.level_chips} level_mult={ev.level_mult}")
            print(f"card_chips={ev.card_chips} effect_chips={ev.effect_chips} effect_mult={ev.effect_mult} effect_xmult={ev.effect_xmult}")
            print(f"chips={ev.chips} mult={ev.mult} SCORE={ev.score} ordered={ev.ordered_score} override={ev.score_override}")
            break
        sim.step(action)


if __name__ == "__main__":
    main()
