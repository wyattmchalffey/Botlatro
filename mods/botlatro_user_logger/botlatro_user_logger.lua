-- Botlatro User Logger
--
-- Opt-in local gameplay logger. It writes JSONL rows to:
--   <Balatro save dir>/BotlatroUserLogs/<seed>_<timestamp>.jsonl
--
-- If BalatroBot is installed, the logger also stores BB_GAMESTATE.get_gamestate()
-- as raw_state so Python can parse the exact same structured game surface.

local json = require("json")

BOTLATRO_USER_LOGGER = BOTLATRO_USER_LOGGER or {}

local BUL = BOTLATRO_USER_LOGGER
BUL.version = "0.1.0"
BUL.enabled = BUL.enabled ~= false
BUL.log_dir = BUL.log_dir or "BotlatroUserLogs"
BUL.current_game = nil
BUL.current_run_id = nil
BUL.summary_logged = false
BUL.wrappers_installed = false

local PHASE_MAP = {
  MENU = "menu",
  BLIND_SELECT = "blind_select",
  SELECTING_HAND = "selecting_hand",
  HAND_PLAYED = "playing_blind",
  DRAW_TO_HAND = "playing_blind",
  ROUND_EVAL = "round_eval",
  SHOP = "shop",
  SMODS_BOOSTER_OPENED = "booster_opened",
  GAME_OVER = "run_over",
}

local SUIT_MAP = { Spades = "S", Hearts = "H", Diamonds = "D", Clubs = "C" }
local RANK_MAP = { ["10"] = "T", Jack = "J", Queen = "Q", King = "K", Ace = "A" }

local function safe_call(fn, fallback)
  local ok, result = pcall(fn)
  if ok then return result end
  return fallback
end

local function state_name()
  if not G or not G.STATES then return "UNKNOWN" end
  for name, value in pairs(G.STATES) do
    if value == G.STATE then return name end
  end
  return "UNKNOWN"
end

local function phase_name(raw)
  return PHASE_MAP[tostring(raw or state_name())] or "unknown"
end

local function timestamp()
  return os.date("!%Y%m%dT%H%M%SZ")
end

local function sanitize_filename(value)
  local text = tostring(value or "unknown"):gsub("[^%w%-_%.]", "_")
  if text == "" then return "unknown" end
  return text
end

local function raw_state()
  if BB_GAMESTATE and type(BB_GAMESTATE.get_gamestate) == "function" then
    return safe_call(function() return BB_GAMESTATE.get_gamestate() end, nil)
  end
  return nil
end

local function raw_blinds()
  if BB_GAMESTATE and type(BB_GAMESTATE.get_blinds_info) == "function" then
    return safe_call(function() return BB_GAMESTATE.get_blinds_info() end, nil)
  end
  return nil
end

local function bb_cards(state, key)
  if not state or type(state[key]) ~= "table" or type(state[key].cards) ~= "table" then
    return {}
  end
  return state[key].cards
end

local function card_rank(card)
  if card and card.value and card.value.rank then return tostring(card.value.rank) end
  if card and card.config and card.config.card and card.config.card.value then
    local value = tostring(card.config.card.value)
    return RANK_MAP[value] or value
  end
  if card and card.base and card.base.value then
    local value = tostring(card.base.value)
    return RANK_MAP[value] or value
  end
  return ""
end

local function card_suit(card)
  if card and card.value and card.value.suit then return tostring(card.value.suit) end
  if card and card.config and card.config.card and card.config.card.suit then
    return SUIT_MAP[tostring(card.config.card.suit)] or tostring(card.config.card.suit)
  end
  if card and card.base and card.base.suit then
    return SUIT_MAP[tostring(card.base.suit)] or tostring(card.base.suit)
  end
  return ""
end

local function card_name(card)
  local rank, suit = card_rank(card), card_suit(card)
  if rank ~= "" and suit ~= "" then return rank .. suit end
  return tostring((card and (card.label or card.name or card.key)) or "unknown")
end

local function normalize_playing_card(card)
  local modifier = card and card.modifier or {}
  local state = card and card.state or {}
  return {
    rank = card_rank(card),
    suit = card_suit(card),
    name = card_name(card),
    enhancement = modifier.enhancement,
    seal = modifier.seal,
    edition = modifier.edition,
    debuffed = state.debuff == true,
    metadata = card or {},
  }
end

local function normalize_item(card)
  local modifier = card and card.modifier or {}
  return {
    name = tostring((card and (card.label or card.name or card.key)) or "unknown"),
    key = card and card.key,
    set = card and card.set,
    cost = card and card.cost or {},
    edition = modifier.edition,
    rarity = card and card.rarity,
    metadata = card or {},
  }
end

local function normalize_joker(card)
  local item = normalize_item(card)
  return {
    name = item.name,
    edition = item.edition,
    sell_value = item.cost and item.cost.sell or nil,
    key = item.key,
    set = item.set,
    rarity = item.rarity,
    metadata = item.metadata,
  }
end

local function map_cards(cards, mapper)
  local result = {}
  for i, card in ipairs(cards or {}) do result[i] = mapper(card) end
  return result
end

local function active_blind(raw)
  local blinds = raw and raw.blinds or raw_blinds() or {}
  for _, key in ipairs({ "small", "big", "boss", "current", "selected" }) do
    local blind = blinds[key]
    if type(blind) == "table" and (blind.status == "CURRENT" or blind.status == "SELECT") then
      return blind
    end
  end
  if G and G.GAME and G.GAME.blind and G.GAME.blind.name then
    return { name = G.GAME.blind.name, score = G.GAME.blind.chips or 0 }
  end
  return {}
end

local function hand_levels(raw)
  local levels = {}
  if raw and type(raw.hands) == "table" then
    for name, info in pairs(raw.hands) do
      if type(info) == "table" then levels[name] = info.level or 1 end
    end
  end
  return levels
end

local function normalized_state(raw)
  raw = raw or raw_state() or {}
  local blind = active_blind(raw)
  local round = raw.round or {}
  return {
    phase = phase_name(raw.state),
    ante = raw.ante_num or 0,
    blind = blind.name or "",
    required_score = blind.score or 0,
    current_blind = blind,
    current_score = round.chips or 0,
    hands_remaining = round.hands_left or 0,
    discards_remaining = round.discards_left or 0,
    money = raw.money or 0,
    deck_size = raw.cards and raw.cards.count or 0,
    hand = map_cards(bb_cards(raw, "hand"), normalize_playing_card),
    known_deck = map_cards(bb_cards(raw, "cards"), normalize_playing_card),
    hand_levels = hand_levels(raw),
    hands = raw.hands or {},
    jokers = map_cards(bb_cards(raw, "jokers"), normalize_joker),
    consumables = map_cards(bb_cards(raw, "consumables"), normalize_item),
    owned_vouchers = raw.used_vouchers or {},
    vouchers = raw.used_vouchers or {},
    shop = map_cards(bb_cards(raw, "shop"), normalize_item),
    voucher_shop = map_cards(bb_cards(raw, "vouchers"), normalize_item),
    booster_packs = map_cards(bb_cards(raw, "packs"), normalize_item),
    pack = map_cards(bb_cards(raw, "pack"), normalize_item),
  }
end

local function debug_summary(state)
  local hand, jokers = {}, {}
  for _, card in ipairs(state.hand or {}) do hand[#hand + 1] = card.name or "?" end
  for _, joker in ipairs(state.jokers or {}) do jokers[#jokers + 1] = joker.name or "?" end
  return string.format(
    "phase=%s ante=%d blind=%s score=%d/%d money=%d hands=%d discards=%d hand=[%s] jokers=[%s]",
    state.phase or "unknown",
    state.ante or 0,
    state.blind ~= "" and state.blind or "-",
    state.current_score or 0,
    state.required_score or 0,
    state.money or 0,
    state.hands_remaining or 0,
    state.discards_remaining or 0,
    #hand > 0 and table.concat(hand, " ") or "-",
    #jokers > 0 and table.concat(jokers, ", ") or "-"
  )
end

local function ensure_run_context(raw)
  local game_ref = G and G.GAME or nil
  if BUL.current_game ~= game_ref then
    BUL.current_game = game_ref
    BUL.summary_logged = false
    local seed = raw and raw.seed or (G and G.GAME and G.GAME.pseudorandom and G.GAME.pseudorandom.seed) or "unknown"
    BUL.current_run_id = sanitize_filename(seed) .. "_" .. timestamp()
  end
  if not BUL.current_run_id then BUL.current_run_id = "unknown_" .. timestamp() end
end

local function log_path()
  return BUL.log_dir .. "/" .. BUL.current_run_id .. ".jsonl"
end

local function append_row(row)
  if not BUL.enabled then return end
  love.filesystem.createDirectory(BUL.log_dir)
  love.filesystem.append(log_path(), json.encode(row) .. "\n")
end

local function index_in_area(card, area)
  if not card or not area or not area.cards then return nil end
  for i, candidate in ipairs(area.cards) do
    if candidate == card then return i - 1 end
  end
  return nil
end

local function highlighted_hand_indices()
  local indices = {}
  if not G or not G.hand or not G.hand.cards then return indices end
  for i, card in ipairs(G.hand.cards) do
    if card.highlighted then indices[#indices + 1] = i - 1 end
  end
  return indices
end

local function ui_card(e)
  if not e or type(e) ~= "table" or not e.config then return nil end
  return e.config.ref_table or e.config.card or e.config.object
end

local function metadata(kind, index)
  return { kind = kind, index = index }
end

local function action_from_use_card(e)
  local card = ui_card(e)
  local index = index_in_area(card, G and G.shop_vouchers)
  if index ~= nil then
    return { type = "buy", card_indices = {}, target_id = "voucher", amount = index, metadata = metadata("voucher", index) }
  end
  index = index_in_area(card, G and G.shop_booster)
  if index ~= nil then
    return { type = "open_pack", card_indices = {}, target_id = "pack", amount = index, metadata = metadata("pack", index) }
  end
  index = index_in_area(card, G and G.pack_cards)
  if index ~= nil then
    return { type = "choose_pack_card", card_indices = highlighted_hand_indices(), target_id = "card", amount = index, metadata = metadata("card", index) }
  end
  index = index_in_area(card, G and G.consumeables)
  if index ~= nil then
    return { type = "use_consumable", card_indices = highlighted_hand_indices(), target_id = "consumable", amount = index, metadata = metadata("consumable", index) }
  end
  return nil
end

local function action_from_buy(e)
  local index = index_in_area(ui_card(e), G and G.shop_jokers)
  if index == nil then return nil end
  return { type = "buy", card_indices = {}, target_id = "card", amount = index, metadata = metadata("card", index) }
end

local function action_from_sell(e)
  local card = ui_card(e)
  local index = index_in_area(card, G and G.jokers)
  if index ~= nil then
    return { type = "sell", card_indices = {}, target_id = "joker", amount = index, metadata = metadata("joker", index) }
  end
  index = index_in_area(card, G and G.consumeables)
  if index ~= nil then
    return { type = "sell", card_indices = {}, target_id = "consumable", amount = index, metadata = metadata("consumable", index) }
  end
  return nil
end

local function chosen_item(state, action)
  if not action or action.amount == nil then return nil end
  local pos = action.amount + 1
  if action.type == "buy" and action.target_id == "card" then return state.shop and state.shop[pos] end
  if action.type == "buy" and action.target_id == "voucher" then return state.voucher_shop and state.voucher_shop[pos] end
  if action.type == "open_pack" then return state.booster_packs and state.booster_packs[pos] end
  if action.type == "choose_pack_card" then return state.pack and state.pack[pos] end
  if action.type == "sell" and action.target_id == "joker" then return state.jokers and state.jokers[pos] end
  if action.type == "sell" and action.target_id == "consumable" then return state.consumables and state.consumables[pos] end
  if action.type == "use_consumable" then return state.consumables and state.consumables[pos] end
  return nil
end

local function log_action(action)
  if not BUL.enabled or not action then return end
  local raw = raw_state()
  ensure_run_context(raw)
  local state = normalized_state(raw)
  append_row({
    record_type = "user_step",
    source = "botlatro_user_logger",
    logger_version = BUL.version,
    actor = "human",
    bot_version = "human",
    run_id = BUL.current_run_id,
    seed = raw and raw.seed or nil,
    state = debug_summary(state),
    state_detail = state,
    raw_state = raw,
    legal_actions = {},
    chosen_action = action,
    chosen_item = chosen_item(state, action),
    reward = 0,
    extra = {},
    logged_at = timestamp(),
  })
end

local function maybe_log_summary()
  if not BUL.enabled or BUL.summary_logged or not G or not G.GAME then return end
  local raw = raw_state()
  ensure_run_context(raw)
  local state = normalized_state(raw)
  local won = (raw and raw.won == true) or (G.GAME and G.GAME.won == true)
  local terminal = won or state.phase == "run_over" or state_name() == "GAME_OVER"
  if not terminal then return end
  BUL.summary_logged = true
  append_row({
    record_type = "run_summary",
    source = "botlatro_user_logger",
    logger_version = BUL.version,
    actor = "human",
    bot_version = "human",
    run_id = BUL.current_run_id,
    seed = raw and raw.seed or nil,
    stake = raw and raw.stake or "unknown",
    won = won,
    outcome = won and "win" or "loss",
    ante = state.ante or 0,
    final_score = state.current_score or 0,
    final_money = state.money or 0,
    death_reason = won and nil or state.blind,
    final_state = debug_summary(state),
    final_state_detail = state,
    raw_final_state = raw,
    runtime_seconds = 0,
    logged_at = timestamp(),
  })
end

local function wrap_func(name, make_action)
  if not G or not G.FUNCS or type(G.FUNCS[name]) ~= "function" then return end
  local original = G.FUNCS[name]
  G.FUNCS[name] = function(e, ...)
    log_action(make_action(e))
    return original(e, ...)
  end
end

local function install_wrappers()
  if BUL.wrappers_installed or not G or not G.FUNCS then return end
  BUL.wrappers_installed = true
  wrap_func("play_cards_from_highlighted", function() return { type = "play_hand", card_indices = highlighted_hand_indices() } end)
  wrap_func("discard_cards_from_highlighted", function() return { type = "discard", card_indices = highlighted_hand_indices() } end)
  wrap_func("select_blind", function() return { type = "select_blind", card_indices = {} } end)
  wrap_func("skip_blind", function() return { type = "skip_blind", card_indices = {} } end)
  wrap_func("cash_out", function() return { type = "cash_out", card_indices = {} } end)
  wrap_func("toggle_shop", function() return { type = "end_shop", card_indices = {} } end)
  wrap_func("reroll_shop", function() return { type = "reroll", card_indices = {} } end)
  wrap_func("skip_booster", function() return { type = "choose_pack_card", card_indices = {}, target_id = "skip", metadata = metadata("skip", true) } end)
  wrap_func("buy_from_shop", action_from_buy)
  wrap_func("sell_card", action_from_sell)
  wrap_func("use_card", action_from_use_card)
end

local old_love_update = love.update
love.update = function(dt)
  if old_love_update then old_love_update(dt) end
  install_wrappers()
  maybe_log_summary()
end

sendInfoMessage("Botlatro User Logger loaded - version " .. BUL.version, "BOTLATRO.USER_LOGGER")
