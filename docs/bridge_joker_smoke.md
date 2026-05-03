# Bridge Joker Smoke Oracle

This smoke harness uses the real Balatro bridge as an oracle for targeted joker
scenarios. It avoids waiting for normal seeded runs to naturally find rare
jokers.

## Bridge Setup

The Lua endpoint lives in:

```text
mods/balatrobot_scenario/scenario.lua
```

It has also been installed into the local BalatroBot mod at:

```text
C:/Users/Wyatt/AppData/Roaming/Balatro/Mods/balatrobot/src/lua/endpoints/scenario.lua
```

and registered from:

```text
C:/Users/Wyatt/AppData/Roaming/Balatro/Mods/balatrobot/balatrobot.lua
```

Restart any already-running bridge workers before using it, because BalatroBot
loads endpoint files at mod startup.

## Generate the Manifest

```powershell
python -m balatro_ai.sim.bridge_joker_smoke --dry-run --manifest .data/bridge-joker-smoke/manifest.json
```

The generated manifest contains one primary scenario for every joker in
`src/balatro_ai/data/shop_pools.json`, plus targeted rare-combo, shop-event,
boss-blind, and forced-stochastic scenarios.

## Run Against a Live Bridge

Start one bridge worker, then run:

```powershell
python -m balatro_ai.sim.bridge_joker_smoke `
  --endpoint http://127.0.0.1:12346 `
  --output .data/bridge-joker-smoke/joker_smoke.jsonl
```

By default the command records the bridge oracle rows and then immediately runs
`balatro_ai.sim.replay_validator` over that JSONL.

Useful smaller runs:

```powershell
python -m balatro_ai.sim.bridge_joker_smoke --only Photograph --skip-validate
python -m balatro_ai.sim.bridge_joker_smoke --limit 10
```

## What This Validates

Each scenario creates a controlled hand and owned joker lineup using the real
game, runs one real bridge action, and writes the pre/post states as replay rows.
The local simulator validator then diffs the Python transition against the game
transition.

The harness can force selected probability outcomes by setting the game's
`G.GAME.probabilities.normal` value through the scenario endpoint, then passing
the matching stochastic outcome metadata to replay validation. This validates
the mechanics for rare procs without pretending that one row proves the odds
distribution.

Current expanded smoke coverage is 185 scenarios:

- 150 one-joker scenarios from the source-derived shop pool.
- Rare interaction stacks such as Photograph/retrigger order, Mime/Baron held
  cards, Blueprint/Brainstorm copies, Smeared suit checks, and forced copied
  Bloodstone triggers.
- Shop-event cases for Flash Card, Campfire, Red Card, Hologram,
  Constellation, Fortune Teller, and Perkeo.
- Boss-blind cases for The Flint, The Psychic, The Hook, The Head, The Plant,
  The Tooth, The Water, and The Needle.
- Forced stochastic cases for Bloodstone, Space Joker, Lucky Card/Lucky Cat,
  Glass Card shatter, Business Card, and Reserved Parking.
