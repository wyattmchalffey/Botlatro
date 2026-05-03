-- Dev-only scenario setup endpoint for simulator oracle tests.
--
-- This endpoint is intentionally small and declarative: Python gives it a
-- controlled hand/joker setup, then normal BalatroBot action endpoints execute
-- the actual play/discard/cash-out/select transition.

local SUIT_MAP = {
  H = "Hearts",
  D = "Diamonds",
  C = "Clubs",
  S = "Spades",
}

local RANK_MAP = {
  ["2"] = "2",
  ["3"] = "3",
  ["4"] = "4",
  ["5"] = "5",
  ["6"] = "6",
  ["7"] = "7",
  ["8"] = "8",
  ["9"] = "9",
  T = "10",
  J = "Jack",
  Q = "Queen",
  K = "King",
  A = "Ace",
}

local SEAL_MAP = {
  RED = "Red",
  BLUE = "Blue",
  GOLD = "Gold",
  PURPLE = "Purple",
}

local EDITION_MAP = {
  HOLO = "e_holo",
  FOIL = "e_foil",
  POLYCHROME = "e_polychrome",
  NEGATIVE = "e_negative",
}

local ENHANCEMENT_MAP = {
  BONUS = "m_bonus",
  MULT = "m_mult",
  WILD = "m_wild",
  GLASS = "m_glass",
  STEEL = "m_steel",
  STONE = "m_stone",
  GOLD = "m_gold",
  LUCKY = "m_lucky",
}

local function clear_area(area)
  if not area or not area.cards then
    return
  end
  for i = #area.cards, 1, -1 do
    local card = area.cards[i]
    if card and card.remove then
      card:remove()
    end
    area.cards[i] = nil
  end
  area.cards = {}
  if area.config then
    area.config.card_count = 0
  end
end

local function normalize_key(spec)
  if type(spec) == "string" then
    return spec
  end
  if type(spec) == "table" then
    return spec.key
  end
  return nil
end

local function parse_playing_card_key(key)
  if type(key) ~= "string" or not key:match("^[HDCS]_[2-9TJQKA]$") then
    return nil, nil
  end
  return RANK_MAP[key:sub(3, 3)], SUIT_MAP[key:sub(1, 1)]
end

local function add_playing_card_to_area(spec, area)
  local key = normalize_key(spec)
  local rank, suit = parse_playing_card_key(key)
  if not rank or not suit then
    return false, "Invalid playing card key: " .. tostring(key)
  end

  local params = {
    rank = rank,
    suit = suit,
    set = "Base",
    no_edition = true,
    skip_materialize = true,
  }
  if area then
    params.area = area
  end
  if type(spec) == "table" then
    if spec.seal then
      params.seal = SEAL_MAP[spec.seal]
    end
    if spec.edition then
      params.edition = EDITION_MAP[spec.edition]
      params.no_edition = nil
    end
    if spec.enhancement then
      params.enhancement = ENHANCEMENT_MAP[spec.enhancement]
      params.set = nil
    end
  end

  local success, result = pcall(SMODS.add_card, params)
  if not success or not result then
    return false, "Failed to add playing card: " .. tostring(key)
  end
  return true, nil, result
end

local function add_playing_card(spec)
  return add_playing_card_to_area(spec, nil)
end

local function add_joker_to_area(spec, area)
  local key = normalize_key(spec)
  if type(key) ~= "string" or key:sub(1, 2) ~= "j_" then
    return false, "Invalid joker key: " .. tostring(key)
  end

  local params = {
    key = key,
    skip_materialize = true,
    no_edition = true,
    stickers = {},
    force_stickers = true,
  }
  if area then
    params.area = area
  end
  if type(spec) == "table" then
    if spec.edition then
      params.edition = EDITION_MAP[spec.edition]
      params.no_edition = nil
    end
    if spec.eternal then
      params.stickers[#params.stickers + 1] = "eternal"
    end
    if spec.perishable then
      params.stickers[#params.stickers + 1] = "perishable"
    end
    if spec.rental then
      params.stickers[#params.stickers + 1] = "rental"
    end
  end

  local success, result = pcall(SMODS.add_card, params)
  if not success or not result then
    return false, "Failed to add joker: " .. tostring(key)
  end
  if type(spec) == "table" and spec.perishable and result.ability then
    result.ability.perish_tally = spec.perishable
  end
  if type(spec) == "table" and type(spec.ability) == "table" then
    result.ability = result.ability or {}
    for k, v in pairs(spec.ability) do
      result.ability[k] = v
    end
  end
  return true, nil, result
end

local function add_joker(spec)
  return add_joker_to_area(spec, nil)
end

local function shop_ui_type(card)
  if not card or not card.ability then
    return nil
  end
  if card.ability.set == "Default" then
    return "Base"
  end
  return card.ability.set
end

local function add_center_card_to_area(spec, area)
  local key = normalize_key(spec)
  if type(key) ~= "string" then
    return false, "Invalid card key: " .. tostring(key)
  end
  if key:match("^[HDCS]_[2-9TJQKA]$") then
    return add_playing_card_to_area(spec, area)
  end
  if key:sub(1, 2) == "j_" then
    return add_joker_to_area(spec, area)
  end
  local params = {
    key = key,
    area = area,
    skip_materialize = true,
    no_edition = true,
  }
  if type(spec) == "table" and spec.edition then
    params.edition = EDITION_MAP[spec.edition]
    params.no_edition = nil
  end
  local success, result = pcall(SMODS.add_card, params)
  if not success or not result then
    return false, "Failed to add card: " .. tostring(key)
  end
  return true, nil, result
end

local function add_booster_pack_to_shop(spec)
  local key = normalize_key(spec)
  if type(key) ~= "string" or key:sub(1, 2) ~= "p_" then
    return false, "Invalid booster key: " .. tostring(key)
  end
  local success, result = pcall(SMODS.add_booster_to_shop, key)
  if not success or not result then
    return false, "Failed to add booster: " .. tostring(key)
  end
  return true, nil
end

local function apply_scalar_state(args)
  if args.money ~= nil then
    G.GAME.dollars = args.money
  end
  if args.chips ~= nil then
    G.GAME.chips = args.chips
  end
  if args.hands ~= nil then
    G.GAME.current_round.hands_left = args.hands
  end
  if args.discards ~= nil then
    G.GAME.current_round.discards_left = args.discards
  end
end

local function apply_probability_state(args)
  if args.probability_normal ~= nil then
    G.GAME.probabilities = G.GAME.probabilities or {}
    G.GAME.probabilities.normal = args.probability_normal
  end
end

local function apply_blind_state(args)
  if not args.blind_key then
    return true, nil
  end
  local blind = G.P_BLINDS and G.P_BLINDS[args.blind_key]
  if not blind then
    return false, "Invalid blind key: " .. tostring(args.blind_key)
  end
  if not G.GAME.blind or not G.GAME.blind.set_blind then
    return false, "Current run has no active blind object"
  end

  G.GAME.round_resets.blind = blind
  G.GAME.round_resets.blind_choices = G.GAME.round_resets.blind_choices or {}
  G.GAME.round_resets.blind_states = G.GAME.round_resets.blind_states or {}
  if blind.boss then
    G.GAME.round_resets.blind_choices.Boss = args.blind_key
    G.GAME.round_resets.blind_states.Small = "Defeated"
    G.GAME.round_resets.blind_states.Big = "Defeated"
    G.GAME.round_resets.blind_states.Boss = "Current"
    G.GAME.blind_on_deck = "Boss"
  elseif args.blind_key == "bl_big" then
    G.GAME.round_resets.blind_choices.Big = args.blind_key
    G.GAME.round_resets.blind_states.Small = "Defeated"
    G.GAME.round_resets.blind_states.Big = "Current"
    G.GAME.round_resets.blind_states.Boss = "Upcoming"
    G.GAME.blind_on_deck = "Big"
  else
    G.GAME.round_resets.blind_choices.Small = args.blind_key
    G.GAME.round_resets.blind_states.Small = "Current"
    G.GAME.round_resets.blind_states.Big = "Upcoming"
    G.GAME.round_resets.blind_states.Boss = "Upcoming"
    G.GAME.blind_on_deck = "Small"
  end

  G.GAME.blind:set_blind(blind, nil, true)
  if args.blind_score ~= nil then
    G.GAME.blind.chips = args.blind_score
    G.GAME.blind.chip_text = number_format(args.blind_score)
  end
  return true, nil
end

local function refresh_blind_state(args)
  if not args.blind_key or not G.GAME.blind or not G.GAME.blind.set_blind then
    return
  end
  G.GAME.blind:set_blind(nil, true, true)
  if args.blind_score ~= nil then
    G.GAME.blind.chips = args.blind_score
    G.GAME.blind.chip_text = number_format(args.blind_score)
  end
end

---@type Endpoint
return {
  name = "scenario",
  description = "Set up a deterministic dev scenario for bridge oracle tests",
  schema = {
    money = { type = "integer", required = false },
    chips = { type = "integer", required = false },
    hands = { type = "integer", required = false },
    discards = { type = "integer", required = false },
    probability_normal = { type = "integer", required = false },
    blind_key = { type = "string", required = false },
    blind_score = { type = "integer", required = false },
    clear_hand = { type = "boolean", required = false },
    clear_jokers = { type = "boolean", required = false },
    clear_shop = { type = "boolean", required = false },
    clear_pack = { type = "boolean", required = false },
    clear_consumables = { type = "boolean", required = false },
    hand = { type = "array", required = false, items = "table" },
    jokers = { type = "array", required = false, items = "table" },
    shop_cards = { type = "array", required = false, items = "table" },
    booster_packs = { type = "array", required = false, items = "table" },
    pack_cards = { type = "array", required = false, items = "table" },
    consumables = { type = "array", required = false, items = "table" },
  },
  requires_state = { G.STATES.SELECTING_HAND, G.STATES.SHOP, G.STATES.ROUND_EVAL, G.STATES.SMODS_BOOSTER_OPENED },

  execute = function(args, send_response)
    sendDebugMessage("Init scenario()", "BB.ENDPOINTS")

    if not G.STAGE or G.STAGE ~= G.STAGES.RUN then
      send_response({
        message = "Can only set scenarios during an active run",
        name = BB_ERROR_NAMES.INVALID_STATE,
      })
      return
    end

    apply_scalar_state(args)
    apply_probability_state(args)
    local blind_ok, blind_err = apply_blind_state(args)
    if not blind_ok then
      send_response({ message = blind_err, name = BB_ERROR_NAMES.BAD_REQUEST })
      return
    end

    if args.clear_hand then
      clear_area(G.hand)
    end
    if args.clear_jokers then
      clear_area(G.jokers)
    end
    if args.clear_shop then
      clear_area(G.shop_jokers)
      clear_area(G.shop_booster)
    end
    if args.clear_pack then
      clear_area(G.pack_cards)
    end
    if args.clear_consumables then
      clear_area(G.consumeables)
    end

    if args.jokers then
      for _, joker_spec in ipairs(args.jokers) do
        local ok, err = add_joker(joker_spec)
        if not ok then
          send_response({ message = err, name = BB_ERROR_NAMES.BAD_REQUEST })
          return
        end
      end
    end

    if args.hand then
      if G.STATE ~= G.STATES.SELECTING_HAND then
        send_response({
          message = "Scenario hand replacement is only allowed while selecting a hand",
          name = BB_ERROR_NAMES.INVALID_STATE,
        })
        return
      end
      for _, card_spec in ipairs(args.hand) do
        local ok, err = add_playing_card(card_spec)
        if not ok then
          send_response({ message = err, name = BB_ERROR_NAMES.BAD_REQUEST })
          return
        end
      end
    end

    if args.shop_cards then
      if G.STATE ~= G.STATES.SHOP then
        send_response({
          message = "Scenario shop replacement is only allowed in the shop",
          name = BB_ERROR_NAMES.INVALID_STATE,
        })
        return
      end
      for _, card_spec in ipairs(args.shop_cards) do
        local ok, err, card = add_center_card_to_area(card_spec, G.shop_jokers)
        if not ok then
          send_response({ message = err, name = BB_ERROR_NAMES.BAD_REQUEST })
          return
        end
        if card and create_shop_card_ui then
          create_shop_card_ui(card, shop_ui_type(card), G.shop_jokers)
        end
      end
    end

    if args.booster_packs then
      if G.STATE ~= G.STATES.SHOP then
        send_response({
          message = "Scenario booster replacement is only allowed in the shop",
          name = BB_ERROR_NAMES.INVALID_STATE,
        })
        return
      end
      for _, pack_spec in ipairs(args.booster_packs) do
        local ok, err = add_booster_pack_to_shop(pack_spec)
        if not ok then
          send_response({ message = err, name = BB_ERROR_NAMES.BAD_REQUEST })
          return
        end
      end
    end

    if args.pack_cards then
      if G.STATE ~= G.STATES.SMODS_BOOSTER_OPENED then
        send_response({
          message = "Scenario pack replacement is only allowed while a booster is open",
          name = BB_ERROR_NAMES.INVALID_STATE,
        })
        return
      end
      for _, card_spec in ipairs(args.pack_cards) do
        local ok, err = add_center_card_to_area(card_spec, G.pack_cards)
        if not ok then
          send_response({ message = err, name = BB_ERROR_NAMES.BAD_REQUEST })
          return
        end
      end
    end

    if args.consumables then
      for _, card_spec in ipairs(args.consumables) do
        local ok, err = add_center_card_to_area(card_spec, G.consumeables)
        if not ok then
          send_response({ message = err, name = BB_ERROR_NAMES.BAD_REQUEST })
          return
        end
      end
    end

    refresh_blind_state(args)

    local expected_hand_count = args.hand and #args.hand or nil
    local expected_joker_count = args.jokers and #args.jokers or nil
    local expected_shop_count = args.shop_cards and #args.shop_cards or nil
    local expected_booster_count = args.booster_packs and #args.booster_packs or nil
    local expected_pack_count = args.pack_cards and #args.pack_cards or nil
    local expected_consumable_count = args.consumables and #args.consumables or nil

    G.E_MANAGER:add_event(Event({
      trigger = "condition",
      blocking = false,
      func = function()
        local hand_done = true
        local joker_done = true
        if expected_hand_count then
          hand_done = G.hand and G.hand.cards and #G.hand.cards == expected_hand_count
        end
        if expected_joker_count then
          joker_done = G.jokers and G.jokers.cards and #G.jokers.cards == expected_joker_count
        end
        local shop_done = true
        local booster_done = true
        local pack_done = true
        local consumable_done = true
        if expected_shop_count then
          shop_done = G.shop_jokers and G.shop_jokers.cards and #G.shop_jokers.cards == expected_shop_count
        end
        if expected_booster_count then
          booster_done = G.shop_booster and G.shop_booster.cards and #G.shop_booster.cards == expected_booster_count
        end
        if expected_pack_count then
          pack_done = G.pack_cards and G.pack_cards.cards and #G.pack_cards.cards == expected_pack_count
        end
        if expected_consumable_count then
          consumable_done = G.consumeables and G.consumeables.cards and #G.consumeables.cards == expected_consumable_count
        end

        if hand_done and joker_done and shop_done and booster_done and pack_done and consumable_done and G.STATE_COMPLETE == true and not G.CONTROLLER.locked then
          sendDebugMessage("Return scenario()", "BB.ENDPOINTS")
          send_response(BB_GAMESTATE.get_gamestate())
          return true
        end
        return false
      end,
    }))
  end,
}
