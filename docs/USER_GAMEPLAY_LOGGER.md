# User Gameplay Logger

`mods/botlatro_user_logger` is an opt-in Steamodded mod that records human
gameplay as JSONL rows. It is meant for expanding Botlatro's replay dataset
with your own games or with games from people who explicitly agree to share
their logs.

For useful state rows, run it alongside BalatroBot. The logger piggybacks on
BalatroBot's state serializer but does not require the bridge server to be
running.

## What It Logs

The logger records one row before each major decision:

- blind select / skip
- play hand / discard
- cash out
- buy / sell / reroll / leave shop
- open pack / choose pack card / skip pack
- use consumable

Each row includes:

- a compact `state` string compatible with `replay_analyzer`
- normalized `state_detail`
- the raw BalatroBot state as `raw_state` when BalatroBot is installed
- `chosen_action`
- `chosen_item` for shop, pack, sell, and consumable choices

At the end of a run it writes a `run_summary` row.

The mod does not send data over the network. It writes local files under the
Balatro save directory:

```text
%AppData%\Balatro\BotlatroUserLogs
```

## Install Later

Do not install this while a long benchmark is running. The running Balatro
instances should not be touched mid-sweep.

After the sweep finishes, copy this folder:

```text
mods\botlatro_user_logger
```

into:

```text
%AppData%\Balatro\Mods\botlatro_user_logger
```

Then start Balatro with Steamodded enabled and play normally.

## Import Logs

Import local logs into the repo dataset with:

```powershell
python -m balatro_ai.eval.import_user_logs --source "$env:APPDATA\Balatro\BotlatroUserLogs" --dest .data\human-replays --player-id wyatt
```

For logs shared by someone else, put their `.jsonl` files in any folder and run
the same command with a different `--player-id`.

## Consent And Data Hygiene

Only import logs from people who agreed to share them. The logger is designed to
record game state and actions, not personal information, but filenames and
player IDs should still be treated as dataset metadata.
