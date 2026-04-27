# Local BalatroBot Setup

Botlatro needs a running BalatroBot JSON-RPC bridge before it can execute real
game runs.

## Current Local Discovery

On this machine, preflight discovery found:

- Balatro install: `F:\SteamLibrary\steamapps\common\Balatro`
- BalatroBot endpoint: `http://127.0.0.1:12346`
- `uv`: installed in the Python user Scripts folder
- `uvx`: installed in the Python user Scripts folder
- Lovely Injector: installed at `F:\SteamLibrary\steamapps\common\Balatro\version.dll`
- Steamodded: installed at `%AppData%\Balatro\Mods\smods`
- BalatroBot mod: installed at `%AppData%\Balatro\Mods\balatrobot`

## Local Compatibility Patch

The current BalatroBot release declares a dependency string that did not match
the installed Steamodded metadata, even though Steamodded loaded successfully.
The local BalatroBot manifest was patched to remove that stale dependency gate:

```text
%AppData%\Balatro\Mods\balatrobot\balatrobot.json
```

The original manifest backup is stored at:

```text
.backups\balatrobot\balatrobot.original.json
```

## Preflight Command

```powershell
$env:PYTHONPATH = "src"
python -m balatro_ai.tools.preflight
```

## Required External Setup

The official BalatroBot docs list these prerequisites:

- Balatro v1.0.1+
- Lovely Injector
- Steamodded
- uv
- BalatroBot mod files copied into `%AppData%\Balatro\Mods\balatrobot`

The docs also show launching the bridge with:

```powershell
uvx balatrobot serve
```

That command can download and run software, so Botlatro should only run it after
explicit confirmation.

## Once The Bridge Is Running

Run a one-seed smoke test:

```powershell
$env:PYTHONPATH = "src"
python -m balatro_ai.eval.run_seed --bot random_bot --seed 123 --stake white --print-states
```

Then run a 10-seed smoke test:

```powershell
$env:PYTHONPATH = "src"
python -m balatro_ai.eval.benchmark --bot random_bot --seeds 10 --stake white
```
