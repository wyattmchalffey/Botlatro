# BalatroBot API Notes

Source: <https://coder.github.io/balatrobot/api/>

The local BalatroBot bridge uses JSON-RPC 2.0 over HTTP. The documented default
endpoint is:

```text
http://127.0.0.1:12346
```

Core methods used by Botlatro:

- `menu`: return to the main menu.
- `start`: start a new run with `deck`, `stake`, and optional `seed`.
- `gamestate`: fetch current state.
- `select`: select current blind.
- `skip`: skip current blind.
- `play`: play hand cards with `cards`.
- `discard`: discard hand cards with `cards`.
- `cash_out`: collect round rewards.
- `next_round`: leave shop.
- `buy`: buy a shop card, voucher, or pack.
- `pack`: select or skip a booster pack card.
- `sell`: sell a joker or consumable.
- `reroll`: reroll shop.
- `use`: use a consumable.

Documented game states:

- `MENU`
- `BLIND_SELECT`
- `SELECTING_HAND`
- `ROUND_EVAL`
- `SHOP`
- `SMODS_BOOSTER_OPENED`
- `GAME_OVER`

