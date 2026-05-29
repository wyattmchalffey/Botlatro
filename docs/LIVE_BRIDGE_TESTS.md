# Live Bridge Tests

This is the repeatable local workflow for running bridge-backed tests against
BalatroBot. Run commands from the project root in PowerShell.

## 1. Start One Bridge Worker

Use one terminal for the bridge process. It stays running until you press
`Ctrl+C`.

```powershell
$env:PYTHONPATH = "src"
$env:BALATROBOT_LOG_LEVEL = "quiet"

$uvx = "$env:APPDATA\Python\Python312\Scripts\uvx.exe"
$balatroExe = "F:\SteamLibrary\steamapps\common\Balatro\Balatro.exe"
$lovelyDll = "F:\SteamLibrary\steamapps\common\Balatro\version.dll"

& $uvx balatrobot serve `
  --host 127.0.0.1 `
  --port 12346 `
  --fps-cap 2000 `
  --gamespeed 32 `
  --animation-fps 1 `
  --logs-path logs\bridge-tests `
  --love-path $balatroExe `
  --lovely-path $lovelyDll `
  --headless `
  --fast `
  --no-debug `
  --no-shaders `
  --no-audio
```

If `uvx` is already on `PATH`, `uvx balatrobot serve ...` is equivalent. If
Balatro is installed somewhere else, update `$balatroExe` and `$lovelyDll`.

## 2. Check Bridge Health

Use a second terminal for test/capture commands.

```powershell
$env:PYTHONPATH = "src"
python -m balatro_ai.tools.preflight
```

The important line is `balatrobot health`, which should report
`http://127.0.0.1:12346 returned ok`.

## 3. Capture RNG Fixtures

These commands drive the live bridge and write seed-faithful fixtures under the
project data directories.

```powershell
$env:PYTHONPATH = "src"

python -m balatro_ai.rng.capture --all
python -m balatro_ai.rng.capture_shop --all
python -m balatro_ai.rng.capture_shop_sequence --all --shops 6
python -m balatro_ai.rng.capture_shop_sequence --all --shops 6 --stake gold
python -m balatro_ai.rng.capture_shop_sequence --seed 0000003 --shops 1 --used-voucher v_magic_trick
python -m balatro_ai.rng.capture_surfaces --all --all-pack-kinds
python -m balatro_ai.rng.capture_surfaces --seed BBBBBBB --pack-key p_arcana_normal_1 --used-voucher v_omen_globe
python -m balatro_ai.rng.capture_surfaces --seed AAAAAAA --pack-key p_celestial_normal_1 --used-voucher v_telescope --played-hand "High Card=3"
python -m balatro_ai.rng.capture_spectral_helpers --all-helpers
```

## 4. Validate Offline Predictions

After fixtures are captured, the validators and unit tests run offline.

```powershell
$env:PYTHONPATH = "src"

python -m balatro_ai.rng.validate --all
python -m balatro_ai.rng.validate_shop_sequence --all
python -m balatro_ai.rng.validate_surfaces --all
python -m balatro_ai.rng.validate_spectral_helpers --all
python -m unittest discover -s tests -p "test_rng*.py"
```

## 5. Run Joker Scenario Smoke Tests

The scenario endpoint is loaded at BalatroBot startup, so restart the bridge
after changing any Lua endpoint file.

```powershell
$env:PYTHONPATH = "src"

python -m balatro_ai.sim.bridge_joker_smoke `
  --endpoint http://127.0.0.1:12346 `
  --output .data\bridge-joker-smoke\joker_smoke.jsonl
```

Useful smaller runs:

```powershell
python -m balatro_ai.sim.bridge_joker_smoke --only Photograph --skip-validate
python -m balatro_ai.sim.bridge_joker_smoke --limit 10
```

## Troubleshooting

- If health fails, stop stale bridge/BalatroBot processes and start the bridge
  again.
- If a scenario endpoint change does not appear to take effect, restart the
  bridge. BalatroBot loads endpoint files at mod startup.
- If captures hang on an old worker, use a fresh `logs\bridge-tests-*`
  directory so the current run's logs are easy to inspect.
- The benchmark GUI can also launch workers with the same defaults: headless,
  fast mode, no shaders, quiet logs, `fps-cap=2000`, `gamespeed=32`, and
  `animation-fps=1`.
