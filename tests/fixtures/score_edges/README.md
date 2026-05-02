# Score Edge Fixtures

These fixtures cover scoring and transition states that normal Basic Strategy
runs may not reach often enough to validate Phase 7 search safely.

Fixture kinds:

- `score`: calls the local hand evaluator and compares the expected score pieces.
- `play_transition` / `discard_transition`: applies the deterministic forward
  simulator to a pre-action state and compares selected post-state fields.
- `known_gap`: records an edge case that still needs a fresh Balatro oracle
  capture before it can become an exact fixture.

When a real bridge-captured row is available, keep the raw source data in the
fixture and set `source` to the replay path or capture note.

The current rare-edge packs intentionally cover cases that normal Basic
Strategy runs may not reach often: rare hand types, Four Fingers / Shortcut /
Splash scoring selection, copy jokers, disabled jokers, stacked retriggers,
held-card XMult, card edition/enhancement ordering, boss restrictions,
specific-card XMult jokers, and visible-current counters. Known-gap cases are
not considered exact oracle fixtures yet; they are reminders of the next rows
to capture from real Balatro.

Run the fixtures alone with:

```powershell
$env:PYTHONPATH = "src"
python -m balatro_ai.eval.score_edge_fixtures
```

Run the fixtures plus replay score-audit rows with:

```powershell
$env:PYTHONPATH = "src"
python -m balatro_ai.eval.score_dataset --fixtures tests\fixtures\score_edges --replay-dir .data\primary-score-audit-100
```
