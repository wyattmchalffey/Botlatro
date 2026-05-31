# Botlatro

Botlatro is a local research project for building a Balatro-playing AI agent.
The first target is reliability: read structured game state, choose legal
actions, complete runs, and measure progress across fixed seed sets.

This project is for offline/local research using an owned copy of Balatro. Do
not use it to cheat leaderboards, competitions, or online/shared systems.

## Current Status

The repository is in Phase 7, but the framing pivoted on 2026-05-24 from
"make the live bot stronger" to **"build an offline expert solver to
generate Phase 8 training data."** Full plan and rationale in
[`PHASE7_OFFLINE_SOLVER_PLAN.md`](PHASE7_OFFLINE_SOLVER_PLAN.md).

Verified building blocks for the new path:

- **`forward_sim` is 99.9% exact** across play/discard/sell/reroll/end_shop
  on 5,074 audited transitions from 241 BalatroBench runs. Diagnostic at
  `src/balatro_ai/eval/sim_divergence_audit.py`.
- **Per-hand scoring is essentially perfect** — 14 misses across 2,148
  records, all labeled-known-uncertainties (Space Joker, Ramen, The Mouth
  round-history).
- **Seed-faithful RNG coverage now spans the major solver surfaces**:
  initial deck shuffle, boss blind, current voucher, Small/Big skip tags,
  shop cards, booster slots, pack contents, edition/sticker polls, and
  per-card spectral helpers. The canonical first-shop fixtures match on
  boss/tags/voucher/shop cards/boosters; deck fixtures still match 4/4.
  Opened-pack fixtures match 24/24, including Omen Globe, Telescope, and
  Glow Up voucher paths. No-purchase shop-sequence fixtures match 51/51
  through the first ante-3 shop across White and Gold Stakes plus
  Magic Trick/Illusion voucher-rate paths, including eternal/perishable/rental
  sticker polls.
- **Rust core (`botlatro-core/`)** ports the hottest paths to a native
  PyO3 extension. Phase 1 (state types), Phase 2 (hand evaluation: 75×
  per-call speedup, ~80 jokers covered), Phase 3 (forward-sim helpers +
  `simulate_play_simple` orchestration), and Phase 4a-g (batched action
  scorer wire-ins across beam-rollout, shop build, scoring helpers) +
  **Phase 4d.1 (native rollout loop in `clear_probability_native` —
  the first architectural Phase-4d piece)** all landed. **Solver
  trajectory on AAAAAAA: 49.4s vs 236s baseline → 79% speedup
  (4.78×)**. Full roadmap and status in
  [`RUST_PORT_PLAN.md`](RUST_PORT_PLAN.md); 226 Python-side tests +
  97 cargo tests green.

  **Correction (2026-05-28):** the earlier "parity preserved (130
  steps on AAAAAAA)" claims were measured under a since-fixed
  nondeterminism bug — an `id()`-keyed leaf-value memo in
  `solver/search_v2/play.py` collided after GC, so trajectories
  varied run-to-run (and dataset generation was silently corrupted
  whenever a worker handled >1 seed). Fixed with a tuple-ref guard;
  the solver is now deterministic and Rust-on vs Rust-off match
  step-for-step (40/40 steps, 0 score mismatches on AAAAAAA). With
  the fix, AAAAAAA is a 70-step run to ante 3. **Validation finding:**
  search depth (d3→d5) and leaf reweighting do NOT move winrate
  (flat at ante ~3.1) — the value function (clear-probability
  dominated, economy/build-blind, one-blind horizon) is the ceiling,
  not search depth. Profiling shows shop search is still ~50% of
  trajectory time and ~46% of hand evaluations still fall back to
  Python — the largest remaining Rust-port opportunities.

  **Opt-in best-play fast path (2026-05-29):** `best_play_from_hand` (the
  live bot's hottest loop, ~333K Python hand-evals/game) can route per-subset
  scoring through `score_play_actions_batch` via
  `search/rust_bridge.rust_best_play_scores` — **2.1× faster (26.6→11.0
  s/game)**. Gated `BALATRO_RUST_BESTPLAY` (default OFF) because the Rust
  simple-evaluator diverges from the Python evaluator on a few stateful
  jokers (Ride the Bus / Bull / Banner / Blue Joker / The Family), shifting
  ~1.5% of decisions; the canonical bot stays bit-for-bit pure-Python. Parity
  tool: `scripts/bestplay_parity_check.py` (`BALATRO_BESTPLAY_PARITY=1`).

- **The offline solver (`solver/policy.py`) is built and generating data.**
  `SolverPolicy` runs a whole-blind beam play search plus a shop beam over a
  build-aware value function. Its data-gen winrate was ~1%; it is now **~8%**
  after two systematic value-function bug fixes — a play-value bug (a cleared
  blind was valued *below* an almost-cleared state, ec9d0b7) and a shop
  joker-churn bug (the beam sold a good joker then re-bought into the freed
  slot, gaming a state-relative buy heuristic, f2944d8) — plus a first-shop
  Buffoon-pack fix (every data-gen game had been denied its guaranteed early
  joker). Raising it toward the ~23% `basic_strategy_bot` is the active
  priority; method and findings in [`PROGRESS.md`](PROGRESS.md). NOTE: the
  data-gen harness runs the sim with the shop *sampler* (approximate shops),
  not the seed-faithful path (that activates only when a `balatro_seed` string
  is passed).

Live-bot historical context (no longer the target metric — the offline
solver is the data generator now):

- `basic_strategy_bot` is the confirmed rule-bot baseline. Most recent
  1000-seed white-stake benchmark: **74/1000 wins (7.4%)**, avg ante 4.86.
- `search_bot_v2` and earlier search variants tie `basic_strategy_bot` at
  ~5–7% across same-seed comparisons; leaf-tuning has stopped moving the
  number, which is what motivated the pivot.
- **Tuning win (2026-05-29):** a causal config sweep raised
  `shop_target_safety_base` 1.15→1.30 (new `bots/config.py` default), which
  lifted the small-seed-set white-stake winrate from **12.5% → 17.0%**
  (34/200 vs 25/200, stable across both seed halves). It came from a Phase 8
  value-model *probe* — the learned value function itself was redundant with
  the heuristic and regressed winrate, but its calibration diagnostic showed
  the bot was over-optimistic / under-building in antes 1–2, which this knob
  fixes. A fresh 1000-seed confirmation is still pending. Tooling:
  `scripts/winrate_bench_config.py` (causal `BotConfig`-override A/Bs).
- Phase 8 neural training still gates on a stronger teacher (~40–50%
  white-stake winrate equivalent on trajectory quality), but the teacher
  is now the offline solver, not the live bot.

The repository contains:

- Long-term project plan in `PLAN.md` (phases 0–15).
- **Active Phase 7 → Phase 8 plan in `PHASE7_OFFLINE_SOLVER_PLAN.md`.**
- **Rust port plan and status in `RUST_PORT_PLAN.md` (Phases 1-4a complete).**
- Python package scaffold in `src/balatro_ai`.
- **Native Rust extension in `botlatro-core/` (PyO3 + maturin).**
- Core state and action models.
- A JSON-RPC client for a local Balatro bridge.
- BalatroBot API notes in `docs/BALATROBOT_API_NOTES.md`.
- Local setup notes in `SETUP.md`.
- A Gym-like environment wrapper.
- Random, greedy, basic-strategy, and experimental search bots.
- Benchmark metric utilities.
- Deterministic benchmark seed generation.
- Replay logging helpers.
- Pure-Python local simulator and replay validator.
- **Seed-faithful RNG predictors in `src/balatro_ai/rng/` (pseudohash,
  pseudoseed, LuaJIT TW223, deck/shop/setup/pack/spectral prediction).**
- **Sim-vs-game divergence audit in `src/balatro_ai/eval/sim_divergence_audit.py`.**
- Standard-library tests.

## Run Tests

The tests use Python's standard-library `unittest` runner:

```bash
python -m unittest discover -s tests
```

## Run Local CLI Commands Without Installing

Until the package is installed in editable mode, set `PYTHONPATH` to `src`:

```powershell
$env:PYTHONPATH = "src"
python -m balatro_ai.eval.benchmark --bot random_bot --seeds 5 --stake white
```

When a local Balatro JSON-RPC bridge is running, run one seeded game with:

```powershell
$env:PYTHONPATH = "src"
python -m balatro_ai.eval.run_seed --bot random_bot --seed 123 --stake white --print-states
```

The default endpoint matches BalatroBot's documented default:
`http://127.0.0.1:12346`.

Run a live 10-seed smoke benchmark with:

```powershell
$env:PYTHONPATH = "src"
python -m balatro_ai.eval.benchmark --bot random_bot --seeds 10 --stake white --execute --replay-dir .data\replays
```

Run a bridge-free local-simulator benchmark with:

```powershell
$env:PYTHONPATH = "src"
python -m balatro_ai.eval.local_benchmark --bot basic_strategy_bot --seeds 200 --stake white --workers 8 --jsonl-out .data\local-basic-200.jsonl
```

Use `search_bot_v2` only as an experiment lane until a same-seed comparison
shows it beating `basic_strategy_bot`. The live search-bot iteration loop is
no longer the primary winrate path — see
[`PHASE7_OFFLINE_SOLVER_PLAN.md`](PHASE7_OFFLINE_SOLVER_PLAN.md) for why.

## RNG matching commands

See [`docs/LIVE_BRIDGE_TESTS.md`](docs/LIVE_BRIDGE_TESTS.md) for the bridge
launch command, health check, and full live capture workflow.

With the bridge running, capture seed-faithful ground truth:

```powershell
$env:PYTHONPATH = "src"
python -m balatro_ai.rng.capture --all         # initial deck order per seed
python -m balatro_ai.rng.capture_shop --all    # first-shop contents per seed
python -m balatro_ai.rng.capture_shop_sequence --all --shops 6
python -m balatro_ai.rng.capture_shop_sequence --all --shops 6 --stake gold
python -m balatro_ai.rng.capture_shop_sequence --seed 0000003 --shops 1 --used-voucher v_magic_trick
python -m balatro_ai.rng.capture_surfaces --all --all-pack-kinds
python -m balatro_ai.rng.capture_surfaces --seed BBBBBBB --pack-key p_arcana_normal_1 --used-voucher v_omen_globe
python -m balatro_ai.rng.capture_surfaces --seed AAAAAAA --pack-key p_celestial_normal_1 --used-voucher v_telescope --played-hand "High Card=3"
python -m balatro_ai.rng.capture_spectral_helpers --all-helpers
```

Then validate predictions offline:

```powershell
$env:PYTHONPATH = "src"
python -m balatro_ai.rng.validate --all
python -m balatro_ai.rng.validate_shop_sequence --all
python -m balatro_ai.rng.validate_surfaces --all
python -m balatro_ai.rng.validate_spectral_helpers --all
python -m unittest discover -s tests -p "test_rng*.py"
```

Audit `forward_sim` correctness against captured game transitions:

```powershell
$env:PYTHONPATH = "src"
python -m balatro_ai.eval.sim_divergence_audit .data\balatrobench_raw_subset
```

## Benchmark Assumptions

Bridge-backed benchmarks use Balatro profile `P1`, which has all unlocks
available. Bridge benchmark output records this as `Profile: P1` and
`Unlocks: all` alongside the active deck and stake. The default bridge deck is
`RED`; override it with `--deck` or the GUI `Deck` field if a run should use
another deck.

Pure-Python local-simulator benchmarks are bridge-free and currently report
unknown profile/deck metadata unless those fields are supplied by the simulator
options. Use them for fast iteration, then verify suspicious behavior with fresh
bridge replay data.

## Benchmark GUI

Launch the simple benchmark GUI with:

```powershell
$env:PYTHONPATH = "src"
python -m balatro_ai.gui.benchmark_app
```

The GUI lets you configure bot, stake, seed count, max steps, replay/output
paths, worker count, bridge ports, deck, profile/unlock labels, failed-seed
retries, and BalatroBot launch options. To use
parallel workers, set `Workers` above 1 and enable `Launch bridge workers`; the
GUI will start one BalatroBot bridge per consecutive port. Leave `Stop existing
first` enabled when launching workers so stale bridge processes cannot occupy
the requested ports. Put exact seeds in `Seed list` for one-off or hand-picked
runs; comma, spaces, and newlines all work, and this overrides the `Seeds`
count. Use `Use Benchmark Seeds` to load the current 100-seed comparison set
from `.data/current-light-100-seeds.txt`; use `Generated Seeds` to go back to
deterministic label-based seed generation. Use `Stop Run` to cancel the active benchmark; it stops owned workers and
prevents new seeds from being scheduled. Use the `Benchmark Speed` preset for
fast headless sweeps; it sets headless mode, fast mode, no shaders, quiet bridge
logs, summary replay mode, `fps-cap` 2000, `gamespeed` 32, and animation FPS 1.
Use `Replay mode` to choose no replay logging, summary-only JSONL, lightweight
JSONL, or full score-audit replay details. Summary replay writes one tiny final
result row per seed, which is useful for fast sweeps that still need replay
analyzer win/ante counts. Benchmark runs retire unhealthy worker endpoints after
bridge/client errors and retry failed seeds once by default, replacing that
seed's replay file so analyzer results stay clean. Use `Bridge logs` to choose quiet logs, disposable
off logs, cleaned normal logs, or untouched normal logs. BalatroBot treats headless and
render-on-API as mutually
exclusive, so the GUI keeps only one of those options enabled at a time. The
`Tiny startup` option creates a patched Balatro copy under `.balatro-headless`
for headless workers, which prevents the brief fullscreen-sized window flash
before BalatroBot minimizes the instance.

Run preflight checks with:

```powershell
$env:PYTHONPATH = "src"
python -m balatro_ai.tools.preflight
```

Clean existing bridge logs with:

```powershell
$env:PYTHONPATH = "src"
python -m balatro_ai.tools.clean_bridge_logs --logs-root .logs --replace
```

Audit replay score predictions with:

```powershell
$env:PYTHONPATH = "src"
python -m balatro_ai.eval.score_audit --replay-dir .data\replays
```

Explain the largest current evaluator misses with:

```powershell
$env:PYTHONPATH = "src"
python -m balatro_ai.eval.explain_score_misses --replay-dir .data\replays --worst 20
```

Validate the deterministic score-edge fixtures and recomputed replay oracle rows
together with:

```powershell
$env:PYTHONPATH = "src"
python -m balatro_ai.eval.score_dataset --fixtures tests\fixtures\score_edges --replay-dir .data\replays
```

Collect opt-in human gameplay logs with the repo-local Steamodded mod in
`mods\botlatro_user_logger`; see `docs\USER_GAMEPLAY_LOGGER.md` for install and
import instructions.

Score a small deterministic evaluator scenario with:

```powershell
$env:PYTHONPATH = "src"
python -m balatro_ai.eval.scenario_score --cards "KS" --jokers "Hanging Chad,Photograph"
```

Money-scaled joker scenarios can include current money:

```powershell
$env:PYTHONPATH = "src"
python -m balatro_ai.eval.scenario_score --cards "AS" --jokers "Bull,Bootstraps" --money 11
```

## Next Target

The near-term track is **raising the offline solver's data-gen winrate** so its
generated trajectories are strong enough to train on. The `SolverPolicy` is
built (see [`SOLVER_PLAN.md`](SOLVER_PLAN.md)) and generating data at ~8%
winrate; the gap to the ~23% `basic_strategy_bot` is build *scaling*. The method
that has produced the wins is tracing systematic shop/play value-function
mis-valuations with `scripts/shop_decision_trace.py` and validating fixes with
paired same-seed A/Bs (`scripts/shop_paired_ab.py`, ≥80 seeds). Leveling-term
tuning and deeper shop search were both measured inert/negative — do not chase
them. Once the winrate is acceptable, generate a 10k-50k trajectory dataset for
Phase 8 imitation training. (One narrow seed-faithful RNG edge remains —
Illusion shop playing-card generation perturbs the global `math.random` used by
the first Buffoon pack — but it does not block the sampler-based data-gen path.)
