"""Training-data pipeline (Step 0.2).

Turns runs into `(encoded_state, action, value_target)` training examples
**without bloating storage with full per-step states**. The mechanism, per
`PHASE8_NEURAL_PLAN.md` Stage 0.2:

1. `capture_run` drives a policy once and records a *thin but replay-complete*
   log: the full `Action.to_json()` for each step plus the run outcome. (The
   existing `solver.trajectory.StepRecord` is lossy — it drops
   `target_id`/`amount`/`metadata`, so shop/pack actions can't be replayed from
   it. The full action JSON can.)
2. `replay_states` re-simulates that action log on a fresh, deterministic
   `LocalBalatroSimulator` (same setup as `generate_trajectory`), yielding the
   full `GameState` before each action. This is where states are reconstructed —
   cheaply, with no policy/search re-run.
3. `examples_from_capture` encodes each reconstructed state and pairs it with the
   action (policy target) and the run outcome (value target).

`verify_capture_roundtrip` is the Stage 0.2 gate: re-simulating the stored log
must reproduce the captured per-step (score, ante, money) exactly. If it does,
the thin log is provably sufficient to reconstruct the run — so the dataset can
store action logs, not states.

Value targets are Monte-Carlo outcome labels (won / final ante / final score)
shared by every step of a run, with `steps_to_end` retained for later
discounting. Dependency-free (stdlib) — tensorization lives in the model layer.
"""

from __future__ import annotations

import json
import os
import pickle
from collections.abc import Iterator
from dataclasses import dataclass, field

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.ml.encoding import ENCODING_VERSION, EncodedState, encode_state
from balatro_ai.sim.local_runner import LocalBalatroSimulator
from balatro_ai.solver.seed_game import SeedGame
from balatro_ai.solver.trajectory import Policy, _stable_seed_int

# Schema version for the expanded-examples cache; bump when CandidateToken or
# the candidate-building logic changes (separate from ENCODING_VERSION).
EXAMPLES_CACHE_VERSION = 1

# Reasons `capture_run` can stop — mirrors `generate_trajectory`.
TERMINAL_REASONS = frozenset(
    {"RUN_OVER", "STEP_LIMIT", "STUCK", "POLICY_NOOP"}
)


def _make_sim(seed: str, stake: str) -> LocalBalatroSimulator:
    """Construct a sim seeded *identically* to `generate_trajectory`.

    Matching the setup byte-for-byte is what lets a captured action log replay
    to the same trajectory (and lets `capture_run` agree with
    `generate_trajectory` on the same seed+policy).
    """
    game = SeedGame(seed, stake=stake)
    initial = game.initial_state()
    int_seed = _stable_seed_int(seed)
    balatro_seed = seed if os.environ.get("BALATRO_SEED_FAITHFUL") == "1" else None
    sim = LocalBalatroSimulator(seed=int_seed, stake=stake, balatro_seed=balatro_seed)
    sim.state = initial
    return sim


def _signature(state: GameState) -> tuple:
    # Same stuck-detection signature as generate_trajectory, so capture stops
    # on exactly the same step the canonical generator would.
    return (
        state.phase,
        state.ante,
        state.current_score,
        state.money,
        state.hands_remaining,
        state.discards_remaining,
        state.deck_size,
        len(state.hand),
        len(state.jokers),
        len(state.consumables),
    )


@dataclass(frozen=True, slots=True)
class ValueTarget:
    """Monte-Carlo outcome label shared by every step of a run."""

    won: bool
    final_ante: int
    final_score: int


# --------------------------------------------------------------------------- #
# Schema v2: per-decision candidate sets (the decision-shaped policy's input).
# --------------------------------------------------------------------------- #

# Action types in a fixed order -> embedding index for candidate tokens.
_ACTION_TYPE_ORDER: tuple[ActionType, ...] = tuple(ActionType)
_ACTION_TYPE_INDEX = {t: i for i, t in enumerate(_ACTION_TYPE_ORDER)}


@dataclass(frozen=True, slots=True)
class CandidateToken:
    """One legal action as a scorable candidate. Structural features + the
    heuristic's own evaluation fused in where cheap (the architecture's core
    idea: the net learns WHEN the heuristic is wrong, not chip math). The
    `has_*` flags are missing-feature indicators — absence is informative
    (the Rust play score legitimately doesn't exist on hard boss states), so
    it is flagged, never zero-pretended."""

    action_type_index: int
    n_cards: float           # len(card_indices) / 8
    amount: float            # (amount or 0) / 20
    has_target: float        # 1.0 if target_id set
    play_score: float        # normalized Rust immediate score (play actions)
    has_play_score: float    # 1.0 when play_score is real
    heuristic_choice: float  # 1.0 if the reference heuristic would take this action


def _candidate_tokens(
    state: GameState, taken: Action
) -> tuple[tuple[CandidateToken, ...], int]:
    """Build the candidate set for `state.legal_actions` and the index of the
    one the policy took (`-1` if the taken action is not among the legals —
    e.g. a metadata-only variant; caller can drop those examples)."""

    tokens = candidate_tokens_for_state(state)
    if not tokens:
        return (), -1
    taken_key = taken.stable_key
    legals = state.legal_actions
    chosen = next((i for i, a in enumerate(legals) if a.stable_key == taken_key), -1)
    return tokens, chosen


def candidate_tokens_for_state(state: GameState) -> tuple[CandidateToken, ...]:
    """The candidate feature set for `state.legal_actions`, parallel to
    `state.legal_actions` by index. Shared by training (schema v2) and
    inference (the deployed policy bot) — same features both sides."""
    legals = state.legal_actions
    if not legals:
        return ()
    play_scores = _play_scores_by_position(state, legals)
    heuristic_pos = _heuristic_choice_position(state)
    tokens: list[CandidateToken] = []
    for pos, act in enumerate(legals):
        score = play_scores.get(pos)
        tokens.append(
            CandidateToken(
                action_type_index=_ACTION_TYPE_INDEX.get(act.action_type, 0),
                n_cards=min(len(act.card_indices), 8) / 8.0,
                amount=min(abs(act.amount or 0), 20) / 20.0,
                has_target=1.0 if act.target_id else 0.0,
                play_score=score if score is not None else 0.0,
                has_play_score=1.0 if score is not None else 0.0,
                heuristic_choice=1.0 if pos == heuristic_pos else 0.0,
            )
        )
    return tuple(tokens)


_REF_HEURISTIC = None


def _heuristic_choice_position(state: GameState) -> int:
    """Index of the candidate the reference heuristic (BasicStrategy — the play
    policy every mixture bot shares) would take, or -1. THE load-bearing fusion
    feature: B0 showed the net was near-chance on plays because the immediate
    score doesn't capture what the heuristic optimizes (min-sufficient + pace);
    'would the heuristic pick this' does. Pure function of the state; used
    identically at training (expansion) and inference (the deployed bot)."""
    global _REF_HEURISTIC
    try:
        if _REF_HEURISTIC is None:
            from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot

            _REF_HEURISTIC = BasicStrategyBot(seed=0)
        action = _REF_HEURISTIC.choose_action(state)
        key = action.stable_key
        return next((i for i, a in enumerate(state.legal_actions) if a.stable_key == key), -1)
    except Exception:  # noqa: BLE001 — feature extraction must never break capture
        return -1


def _play_scores_by_position(state: GameState, legals: tuple[Action, ...]) -> dict[int, float]:
    """Batched Rust immediate score for the PLAY_HAND candidates, normalized to
    a stable log scale. Returns {} (all has_play_score=0) if the Rust path
    bails — the missing-feature flag then carries that honestly."""
    play_positions = [
        i for i, a in enumerate(legals)
        if a.action_type == ActionType.PLAY_HAND and a.card_indices
    ]
    if not play_positions:
        return {}
    try:
        from math import log10

        from balatro_ai.search.rust_bridge import rust_play_action_evals

        play_actions = [legals[i] for i in play_positions]
        evals = rust_play_action_evals(state, play_actions)
        if evals is None:
            return {}
        out: dict[int, float] = {}
        for pos, entry in zip(play_positions, evals):
            if entry is not None:
                out[pos] = min(log10(max(1, int(entry[0]))) / 6.0, 3.0)
        return out
    except Exception:  # noqa: BLE001 — feature extraction must never break capture
        return {}


@dataclass(frozen=True, slots=True)
class TrainingExample:
    step: int
    phase: str
    encoded_state: EncodedState
    action: dict            # Action.to_json() — replay-complete policy target
    value: ValueTarget
    steps_to_end: int       # actions remaining from this step (incl. this one)
    candidates: tuple[CandidateToken, ...] = ()  # schema v2: scorable legal actions
    chosen_index: int = -1                        # which candidate the policy took


@dataclass(frozen=True, slots=True)
class RunCapture:
    """A thin, replay-complete record of one run.

    `actions` is the only thing needed to reconstruct the full run (via
    `replay_states`). `step_summaries` holds the captured post-action
    (score, ante, money) used by the round-trip gate.
    """

    seed: str
    stake: str
    actions: tuple[dict, ...]
    won: bool
    final_ante: int
    final_score: int
    final_money: int
    terminated_reason: str
    step_summaries: tuple[tuple[int, int, int], ...] = field(default_factory=tuple)

    @property
    def n_steps(self) -> int:
        return len(self.actions)

    def to_json_dict(self) -> dict:
        return {
            "seed": self.seed,
            "stake": self.stake,
            "actions": [dict(a) for a in self.actions],
            "won": self.won,
            "final_ante": self.final_ante,
            "final_score": self.final_score,
            "final_money": self.final_money,
            "terminated_reason": self.terminated_reason,
            "step_summaries": [list(s) for s in self.step_summaries],
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> "RunCapture":
        return cls(
            seed=str(data["seed"]),
            stake=data.get("stake", "white"),
            actions=tuple(dict(a) for a in data.get("actions", ())),
            won=bool(data.get("won", False)),
            final_ante=int(data.get("final_ante", 0)),
            final_score=int(data.get("final_score", 0)),
            final_money=int(data.get("final_money", 0)),
            terminated_reason=str(data.get("terminated_reason", "")),
            step_summaries=tuple(tuple(int(x) for x in s) for s in data.get("step_summaries", ())),
        )


@dataclass(frozen=True, slots=True)
class RoundTripResult:
    ok: bool
    n_steps: int
    # (step_index, expected (score,ante,money), got (score,ante,money))
    mismatches: tuple[tuple[int, tuple[int, int, int], tuple[int, int, int]], ...]


def capture_run(
    seed: str,
    policy: Policy,
    *,
    stake: str = "white",
    max_steps: int = 5000,
) -> RunCapture:
    """Drive `policy` once, recording a replay-complete action log + outcome.

    Termination logic mirrors `solver.trajectory.generate_trajectory` exactly,
    so the captured run is the same run the canonical generator produces.
    """
    sim = _make_sim(seed, stake)
    actions: list[dict] = []
    summaries: list[tuple[int, int, int]] = []
    reason = "STEP_LIMIT"
    last_sig: tuple | None = None
    stuck = 0

    for _ in range(max_steps):
        state = sim.state
        if state.phase == GamePhase.RUN_OVER or state.run_over:
            reason = "RUN_OVER"
            break

        sig = _signature(state)
        if sig == last_sig:
            stuck += 1
            if stuck >= 10:
                reason = "STUCK"
                break
        else:
            stuck = 0
        last_sig = sig

        try:
            action = policy(state)
        except Exception as exc:  # noqa: BLE001 — recorded, not raised
            reason = f"POLICY_ERROR: {type(exc).__name__}: {exc}"
            break

        if action.action_type == ActionType.NO_OP:
            reason = "POLICY_NOOP"
            break

        next_state = sim.step(action)
        actions.append(action.to_json())
        summaries.append((next_state.current_score, next_state.ante, next_state.money))

    final = sim.state
    return RunCapture(
        seed=seed,
        stake=stake,
        actions=tuple(actions),
        won=bool(final.won),
        final_ante=final.ante,
        final_score=final.current_score,
        final_money=final.money,
        terminated_reason=reason,
        step_summaries=tuple(summaries),
    )


def replay_states(
    seed: str,
    actions: tuple[dict, ...] | list[dict],
    *,
    stake: str = "white",
) -> Iterator[tuple[GameState, Action]]:
    """Re-simulate an action log, yielding `(state_before, action)` per step.

    This is the offline expansion primitive: no policy, no search — just the
    deterministic sim replaying the stored actions. The yielded state is the
    exact state the policy faced before that action.
    """
    sim = _make_sim(seed, stake)
    for action_json in actions:
        state_before = sim.state
        action = Action.from_mapping(action_json)
        yield state_before, action
        sim.step(action)


def verify_capture_roundtrip(capture: RunCapture) -> RoundTripResult:
    """Stage 0.2 gate: re-simulating the log reproduces the captured run exactly.

    Replays `capture.actions` on a fresh sim and checks each post-action
    (score, ante, money) against `capture.step_summaries`. Exact match proves
    the thin log is sufficient to reconstruct the run.
    """
    mismatches: list[tuple[int, tuple[int, int, int], tuple[int, int, int]]] = []
    sim = _make_sim(capture.seed, capture.stake)
    for idx, action_json in enumerate(capture.actions):
        nxt = sim.step(Action.from_mapping(action_json))
        got = (nxt.current_score, nxt.ante, nxt.money)
        expected = capture.step_summaries[idx] if idx < len(capture.step_summaries) else None
        if expected is not None and got != expected:
            mismatches.append((idx, expected, got))
    return RoundTripResult(
        ok=not mismatches,
        n_steps=len(capture.actions),
        mismatches=tuple(mismatches),
    )


def examples_from_capture(capture: RunCapture) -> list[TrainingExample]:
    """Expand a capture into encoded `(state, action, value)` training examples."""
    total = len(capture.actions)
    value = ValueTarget(
        won=capture.won,
        final_ante=capture.final_ante,
        final_score=capture.final_score,
    )
    examples: list[TrainingExample] = []
    for i, (state, action) in enumerate(
        replay_states(capture.seed, capture.actions, stake=capture.stake)
    ):
        candidates, chosen = _candidate_tokens(state, action)
        examples.append(
            TrainingExample(
                step=i,
                phase=state.phase.value,
                encoded_state=encode_state(state),
                action=action.to_json(),
                value=value,
                steps_to_end=total - i,
                candidates=candidates,
                chosen_index=chosen,
            )
        )
    return examples


def load_or_expand_examples(
    dataset_path: str,
    *,
    cache_path: str | None = None,
    progress_every: int = 100,
) -> list[TrainingExample]:
    """Expand a capture JSONL into examples, CACHED to disk.

    Expansion is expensive (the heuristic_choice fusion feature runs
    basic_strategy per state — ~100 min / 1000 runs), so the result is pickled
    next to the dataset and reloaded when valid. This unblocks fast retrain
    loops (value-head salvage, iteration-1 tuning) that would otherwise re-pay
    the expansion each time. Cache is keyed on the dataset's mtime + the
    encoding/examples schema versions; a mismatch transparently re-expands."""
    import time

    cache_path = cache_path or (dataset_path + ".examples.pkl")
    ds_mtime = os.path.getmtime(dataset_path)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as fh:
                blob = pickle.load(fh)
            if (
                blob.get("dataset_mtime") == ds_mtime
                and blob.get("encoding_version") == ENCODING_VERSION
                and blob.get("examples_cache_version") == EXAMPLES_CACHE_VERSION
            ):
                print(f"[examples] cache hit: {len(blob['examples'])} examples from {cache_path}",
                      flush=True)
                return blob["examples"]
            print("[examples] cache stale (version/mtime) — re-expanding", flush=True)
        except Exception:  # noqa: BLE001 — a corrupt cache must not block work
            print("[examples] cache unreadable — re-expanding", flush=True)

    rows = [json.loads(l) for l in open(dataset_path, encoding="utf-8") if l.strip()]
    examples: list[TrainingExample] = []
    t0 = time.perf_counter()
    for i, r in enumerate(rows):
        examples.extend(examples_from_capture(RunCapture.from_json_dict(r)))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"[examples] expanded {i+1}/{len(rows)} runs -> {len(examples)} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(
            {
                "dataset_mtime": ds_mtime,
                "encoding_version": ENCODING_VERSION,
                "examples_cache_version": EXAMPLES_CACHE_VERSION,
                "examples": examples,
            },
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(tmp, cache_path)
    print(f"[examples] expanded {len(examples)} examples in {time.perf_counter()-t0:.0f}s; "
          f"cached -> {cache_path}", flush=True)
    return examples


def build_examples(
    seed: str,
    policy: Policy,
    *,
    stake: str = "white",
    max_steps: int = 5000,
) -> list[TrainingExample]:
    """Convenience: capture a run and expand it into training examples."""
    return examples_from_capture(
        capture_run(seed, policy, stake=stake, max_steps=max_steps)
    )
