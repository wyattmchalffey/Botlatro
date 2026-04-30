# Botlatro

Botlatro is a local research project for building a Balatro-playing AI agent.
The first target is reliability: read structured game state, choose legal
actions, complete runs, and measure progress across fixed seed sets.

This project is for offline/local research using an owned copy of Balatro. Do
not use it to cheat leaderboards, competitions, or online/shared systems.

## Current Step

The repository currently contains the Phase 0/1 foundation:

- Project plan in `PLAN.md`.
- Python package scaffold in `src/balatro_ai`.
- Core state and action models.
- A JSON-RPC client skeleton for a local Balatro bridge.
- BalatroBot API notes in `docs/BALATROBOT_API_NOTES.md`.
- Local setup notes in `SETUP.md`.
- A Gym-like environment wrapper.
- A random legal-action bot.
- Benchmark metric utilities.
- Deterministic benchmark seed generation.
- Replay logging helpers.
- Standard-library tests.

## Run Tests

The initial tests use only the Python standard library:

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

## Benchmark Assumptions

Current local benchmarks use Balatro profile `P1`, which has all unlocks
available. Benchmark output records this as `Profile: P1` and `Unlocks: all`
alongside the active deck and stake. The default deck is `RED`; override it with
`--deck` or the GUI `Deck` field if a run should use another deck.

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

Connect `JsonRpcBalatroClient` to the actual local BalatroBot API shape, then
run the random bot through 10 complete local game runs without crashing.
