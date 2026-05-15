# Botlatro

Botlatro is a local research project for building a Balatro-playing AI agent.
The first target is reliability: read structured game state, choose legal
actions, complete runs, and measure progress across fixed seed sets.

This project is for offline/local research using an owned copy of Balatro. Do
not use it to cheat leaderboards, competitions, or online/shared systems.

## Current Status

The repository is currently in Phase 7: local-simulator/search-bot iteration.
The project has moved beyond the initial scaffold and now has a source-aligned
pure-Python simulator, bridge replay validation, rule bots, and experimental
search bots.

Current behavior to keep in mind:

- `basic_strategy_bot` is the confirmed rule-bot baseline. On the current
  strict 200-seed local-sim set it is confirmed at `23/200` White Stake wins
  (`11.5%`), with average ante about `5.68`.
- `search_bot_v2` is an experimental wrapper around Basic Strategy plus
  shop/pack/consumable/hand search. It is not currently confirmed stronger than
  `basic_strategy_bot`.
- Phase 8 neural training should wait until a rule/search bot is much stronger
  on same-seed comparisons. Treat `40%` White Stake as a serious data-collection
  threshold and `50%+` as the practical Phase 8 gate.
- The broad all-hand discard evaluator experiment was reverted after a
  regression. The current retained blind-play improvement is the narrower
  straight-draw preferred-hand hunt.

The repository contains:

- Project plan in `PLAN.md`.
- Python package scaffold in `src/balatro_ai`.
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
shows it beating `basic_strategy_bot`.

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

Improve the confirmed rule/search win rate on fixed local-sim White Stake seed
sets. The near-term target is to beat the current `basic_strategy_bot`
`23/200` strict-set baseline without regressing runtime or early-ante survival.
