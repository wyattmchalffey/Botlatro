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
# v2: rich play-surface fusion features (hand_strength, is_discard, draw_odds).
EXAMPLES_CACHE_VERSION = 2

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
    it is flagged, never zero-pretended.

    The rich fusion features (hand_strength/is_discard/draw_odds, schema v2)
    are concentrated on the PLAY surface, because that is where the measured
    headroom is (29.7% of losses have a clearing line the bot didn't find) and
    where the thin v1 features starved the iteration-1 signal. They are FREE:
    `hand_strength` is the made-hand type the Rust batch already returns and v1
    discarded; `draw_odds` is harvested from the single BasicStrategy call
    `_heuristic_choice_info` already makes. SHOP candidates get no extra
    fusion features by design — per-decision shop selection is measured
    near-optimal, so the heuristic_choice flag already routes it."""

    action_type_index: int
    n_cards: float           # len(card_indices) / 8
    amount: float            # (amount or 0) / 20
    has_target: float        # 1.0 if target_id set
    play_score: float        # normalized Rust immediate score (play actions)
    has_play_score: float    # 1.0 when play_score is real
    heuristic_choice: float  # 1.0 if the reference heuristic would take this action
    hand_strength: float = 0.0   # made-hand rank/11 (HighCard=0..FlushFive=1) for plays
    has_hand_type: float = 0.0   # 1.0 when the Rust made-hand type is known
    is_discard: float = 0.0      # 1.0 if a DISCARD action (draw-odds-relevant)
    draw_odds: float = 0.0       # heuristic's completion P for its targeted hand
    has_draw_odds: float = 0.0   # 1.0 when draw_odds is a real heuristic estimate


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
    play_evals = _play_scores_by_position(state, legals)
    heuristic_pos, heuristic_draw_odds = _heuristic_choice_info(state)
    tokens: list[CandidateToken] = []
    for pos, act in enumerate(legals):
        ev = play_evals.get(pos)                     # (norm_score, hand_strength) or None
        is_chosen = pos == heuristic_pos
        tokens.append(
            CandidateToken(
                action_type_index=_ACTION_TYPE_INDEX.get(act.action_type, 0),
                n_cards=min(len(act.card_indices), 8) / 8.0,
                amount=min(abs(act.amount or 0), 20) / 20.0,
                has_target=1.0 if act.target_id else 0.0,
                play_score=ev[0] if ev is not None else 0.0,
                has_play_score=1.0 if ev is not None else 0.0,
                heuristic_choice=1.0 if is_chosen else 0.0,
                hand_strength=ev[1] if ev is not None else 0.0,
                has_hand_type=1.0 if ev is not None else 0.0,
                is_discard=1.0 if act.action_type == ActionType.DISCARD else 0.0,
                # The heuristic's draw-odds estimate is computed for ITS chosen
                # action only (that's the one call we make), so it is fused onto
                # that candidate; absence is flagged, never zero-pretended.
                draw_odds=heuristic_draw_odds if (is_chosen and heuristic_draw_odds is not None) else 0.0,
                has_draw_odds=1.0 if (is_chosen and heuristic_draw_odds is not None) else 0.0,
            )
        )
    return tuple(tokens)


_REF_HEURISTIC = None
_HAND_TYPE_RANK: dict[str, float] | None = None
import re as _re

_DRAW_P_RE = _re.compile(r"\bp=([01](?:\.\d+)?)")


def _hand_type_rank() -> dict[str, float]:
    """value-string -> normalized strength (HighCard=0 .. FlushFive=1). Built
    once; the ordering is the HandType enum's declaration order (ascending)."""
    global _HAND_TYPE_RANK
    if _HAND_TYPE_RANK is None:
        from balatro_ai.bots.basic_strategy.hand_value import HandType

        members = list(HandType)
        denom = max(1, len(members) - 1)
        _HAND_TYPE_RANK = {ht.value: i / denom for i, ht in enumerate(members)}
    return _HAND_TYPE_RANK


def _heuristic_choice_info(state: GameState) -> tuple[int, float | None]:
    """(index of the candidate the reference heuristic would take or -1,
    the heuristic's draw-odds for its target or None). BasicStrategy is the play
    policy every mixture bot shares; `heuristic_choice` is THE load-bearing
    fusion feature (B0: plays were near-chance on the immediate score because it
    doesn't capture min-sufficient+pace; 'would the heuristic pick this' does).
    The draw-odds is harvested from the SAME single choose_action call (parsed
    from the chosen action's reason string `p=…`), so it costs nothing extra.
    Pure function of the state; identical at expansion and inference."""
    global _REF_HEURISTIC
    try:
        if _REF_HEURISTIC is None:
            from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot

            _REF_HEURISTIC = BasicStrategyBot(seed=0)
        action = _REF_HEURISTIC.choose_action(state)
        key = action.stable_key
        pos = next((i for i, a in enumerate(state.legal_actions) if a.stable_key == key), -1)
        draw_odds: float | None = None
        reason = (action.metadata or {}).get("reason") if action.metadata else None
        if isinstance(reason, str):
            m = _DRAW_P_RE.search(reason)
            if m:
                draw_odds = float(m.group(1))
        return pos, draw_odds
    except Exception:  # noqa: BLE001 — feature extraction must never break capture
        return -1, None


def _play_scores_by_position(
    state: GameState, legals: tuple[Action, ...]
) -> dict[int, tuple[float, float]]:
    """Batched Rust eval for the PLAY_HAND candidates: (normalized log-score,
    made-hand strength rank). Returns {} (all has_play_score/has_hand_type=0) if
    the Rust path bails — the missing-feature flags then carry that honestly.
    The made-hand strength is `entry[1]`, which v1 computed and discarded; it is
    the single highest-value play feature (what hand a play makes)."""
    play_positions = [
        i for i, a in enumerate(legals)
        if a.action_type == ActionType.PLAY_HAND and a.card_indices
    ]
    if not play_positions:
        return {}
    try:
        from math import log10

        from balatro_ai.search.rust_bridge import rust_play_action_evals

        ranks = _hand_type_rank()
        play_actions = [legals[i] for i in play_positions]
        evals = rust_play_action_evals(state, play_actions)
        if evals is None:
            return {}
        out: dict[int, tuple[float, float]] = {}
        for pos, entry in zip(play_positions, evals):
            if entry is not None:
                norm_score = min(log10(max(1, int(entry[0]))) / 6.0, 3.0)
                strength = ranks.get(str(entry[1]), 0.0)
                out[pos] = (norm_score, strength)
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
    source: str = ""                              # generating bot/recipe (per-policy AWR baselines)


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
    source: str = ""        # generating bot/recipe name (propagated to examples)

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
            "bot_name": self.source,
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
            source=str(data.get("bot_name", data.get("source", ""))),
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
                source=capture.source,
            )
        )
    return examples


def _expand_row(row: dict) -> list[TrainingExample]:
    """Top-level (picklable) per-run expander for parallel expansion."""
    return examples_from_capture(RunCapture.from_json_dict(row))


def load_or_expand_examples(
    dataset_path: str,
    *,
    cache_path: str | None = None,
    progress_every: int = 100,
    jobs: int | None = None,
) -> list[TrainingExample]:
    """Expand a capture JSONL into examples, CACHED to disk.

    Expansion is expensive (the heuristic_choice fusion feature runs
    basic_strategy per state — ~100 min / 1000 runs single-process), so the
    result is pickled next to the dataset and reloaded when valid, and the
    first expansion can fan out across processes (`jobs`>1; expansion is
    embarrassingly parallel across runs — ~Nx on N cores). Cache is keyed on
    the dataset's mtime + the encoding/examples schema versions; a mismatch
    transparently re-expands. `jobs` defaults to BALATRO_EXPAND_JOBS or 1."""
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

    if jobs is None:
        jobs = int(os.environ.get("BALATRO_EXPAND_JOBS", "1"))
    rows = [json.loads(l) for l in open(dataset_path, encoding="utf-8") if l.strip()]
    examples: list[TrainingExample] = []
    t0 = time.perf_counter()
    if jobs and jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        print(f"[examples] expanding {len(rows)} runs across {jobs} workers...", flush=True)
        done = 0
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            # chunksize batches rows per task to amortize IPC of the big result lists
            for chunk in ex.map(_expand_row, rows, chunksize=4):
                examples.extend(chunk)
                done += 1
                if progress_every and done % progress_every == 0:
                    print(f"[examples] expanded {done}/{len(rows)} runs -> {len(examples)} "
                          f"({time.perf_counter()-t0:.0f}s)", flush=True)
    else:
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


# --------------------------------------------------------------------------- #
# Out-of-core (sharded) example storage. At 50k runs the full example list is
# ~6M examples ~= 300-400GB in RAM (the per-candidate features are bulky), which
# fits no normal box. Expansion writes fixed-run shards to disk; training
# streams them one shard at a time (~0.7GB resident). Benches/gates are
# unaffected — they run the bot on seeds, not stored examples.
# --------------------------------------------------------------------------- #

def _shard_manifest_path(shard_dir: str) -> str:
    return os.path.join(shard_dir, "manifest.json")


def expand_to_shards(
    dataset_path: str,
    shard_dir: str,
    *,
    runs_per_shard: int = 500,
    jobs: int | None = None,
    progress_every: int = 2000,
) -> dict:
    """Expand a capture JSONL into sharded example pickles on disk, holding at
    most ~runs_per_shard runs of examples in RAM at once. Writes
    shard_dir/shardNNNNN.pkl + manifest.json (versions/mtime/shard list/counts),
    and reuses an existing valid manifest. Returns the manifest dict.

    `jobs` defaults to BALATRO_EXPAND_JOBS or 1 (expansion parallelizes across
    runs); the parent buffers at most one shard's worth, so memory stays bounded
    regardless of total dataset size."""
    import time

    os.makedirs(shard_dir, exist_ok=True)
    ds_mtime = os.path.getmtime(dataset_path)
    manifest_path = _shard_manifest_path(shard_dir)
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as fh:
                m = json.load(fh)
            if (
                m.get("dataset_mtime") == ds_mtime
                and m.get("encoding_version") == ENCODING_VERSION
                and m.get("examples_cache_version") == EXAMPLES_CACHE_VERSION
                and all(os.path.exists(os.path.join(shard_dir, s)) for s in m.get("shards", []))
            ):
                print(f"[shards] manifest hit: {m['n_examples']} examples in "
                      f"{len(m['shards'])} shards from {shard_dir}", flush=True)
                return m
            print("[shards] manifest stale — re-expanding", flush=True)
        except Exception:  # noqa: BLE001
            print("[shards] manifest unreadable — re-expanding", flush=True)

    if jobs is None:
        jobs = int(os.environ.get("BALATRO_EXPAND_JOBS", "1"))
    rows = [json.loads(l) for l in open(dataset_path, encoding="utf-8") if l.strip()]
    shards: list[str] = []
    state = {"n_examples": 0}
    buf: list[TrainingExample] = []
    runs_in_buf = 0
    t0 = time.perf_counter()

    def _flush() -> None:
        nonlocal buf, runs_in_buf
        if not buf:
            return
        name = f"shard{len(shards):05d}.pkl"
        tmp = os.path.join(shard_dir, name + ".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(buf, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, os.path.join(shard_dir, name))
        shards.append(name)
        state["n_examples"] += len(buf)
        buf = []
        runs_in_buf = 0

    def _consume(run_examples: list[TrainingExample], done: int) -> None:
        nonlocal runs_in_buf
        buf.extend(run_examples)
        runs_in_buf += 1
        if runs_in_buf >= runs_per_shard:
            _flush()
        if progress_every and done % progress_every == 0:
            print(f"[shards] expanded {done}/{len(rows)} runs -> "
                  f"{state['n_examples'] + len(buf)} ex, {len(shards)} shards "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    if jobs and jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        print(f"[shards] expanding {len(rows)} runs across {jobs} workers "
              f"(runs_per_shard={runs_per_shard})...", flush=True)
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for done, chunk in enumerate(ex.map(_expand_row, rows, chunksize=4), start=1):
                _consume(chunk, done)
    else:
        for done, r in enumerate(rows, start=1):
            _consume(_expand_row(r), done)
    _flush()

    manifest = {
        "dataset_mtime": ds_mtime,
        "encoding_version": ENCODING_VERSION,
        "examples_cache_version": EXAMPLES_CACHE_VERSION,
        "runs_per_shard": runs_per_shard,
        "shards": shards,
        "n_examples": state["n_examples"],
        "n_runs": len(rows),
    }
    tmp = manifest_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    os.replace(tmp, manifest_path)
    print(f"[shards] DONE: {state['n_examples']} examples in {len(shards)} shards "
          f"({time.perf_counter()-t0:.0f}s) -> {shard_dir}", flush=True)
    return manifest


class ShardedExampleStore:
    """Streaming reader over expand_to_shards output. `iter_shards` yields one
    shard's example list at a time (shuffle the shard order per epoch); the
    trainer shuffles within each shard, so SGD sees a shard-local shuffle —
    standard for large-data training (cf. webdataset). `load_all` materializes
    everything (only for small sets that fit in RAM, e.g. held-out)."""

    def __init__(self, shard_dir) -> None:
        # Accept one dir or several — iter1 streams mixture+on-policy as one
        # store. Shard files concatenate; the global index is the weight key.
        dirs = [shard_dir] if isinstance(shard_dir, str) else list(shard_dir)
        self.shard_dirs = dirs
        self.shard_files: list[str] = []
        self.n_examples = 0
        self.manifests: list[dict] = []
        for d in dirs:
            with open(_shard_manifest_path(d), encoding="utf-8") as fh:
                m = json.load(fh)
            self.manifests.append(m)
            self.shard_files.extend(os.path.join(d, s) for s in m["shards"])
            self.n_examples += int(m["n_examples"])

    def __len__(self) -> int:
        return self.n_examples

    def iter_shards(self, *, shuffle: bool = False, rng=None):
        order = list(range(len(self.shard_files)))
        if shuffle and rng is not None:
            rng.shuffle(order)
        for i in order:
            with open(self.shard_files[i], "rb") as fh:
                yield i, pickle.load(fh)

    def load_all(self) -> list[TrainingExample]:
        out: list[TrainingExample] = []
        for _, ex in self.iter_shards():
            out.extend(ex)
        return out


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
