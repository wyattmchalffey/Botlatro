//! Joker effects + per-card enhancements (Phase 2d).
//!
//! Two-pass evaluation matching Python `_effect_adjustments`:
//! 1. **Per-card pass**: for each scoring card, apply enhancement
//!    effects then per-card joker effects (Fibonacci, Triboulet, etc).
//!    Multiplicative xmult here applies BEFORE additive mult from
//!    ability-pass jokers â€” order matters.
//! 2. **Ability pass**: apply ability-joker effects (Joker, Half
//!    Joker, hand-shape-gated jokers, ctx-dependent jokers).
//!
//! Why two passes: Python's hot loop interleaves card-level processing
//! (`add_chips`, `add_mult`, `multiply_mult` from each card and
//! per-card jokers) with later joker iteration. A combined "sum
//! additive, multiply xmult once at end" approach gets the wrong
//! answer when a Joker (+4 mult) is paired with a per-card xmult
//! like Triboulet (x2 per K/Q) â€” Python does (mult * xmult) + add,
//! not (mult + add) * xmult. The split below mirrors Python's
//! ordering exactly.
//!
//! Unsupported joker handling: each pass returns `None` (via its
//! per-name match arm) for jokers not in its category. The caller
//! (`evaluate_simple`) checks BOTH passes for each joker â€” if
//! neither has a match, the joker is unsupported and we bail to
//! Python. This prevents silent omission of joker contributions.

use crate::hand_eval::hand_type::HandType;
use crate::state::card::{Card, Enhancement, Rank, Seal, Suit};

/// Joker effect: additive chips/mult + multiplicative mult contributions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JokerEffect {
    pub chip_delta: i64,
    pub mult_delta: i64,
    pub xmult: f64,
}

impl Default for JokerEffect {
    fn default() -> Self {
        Self { chip_delta: 0, mult_delta: 0, xmult: 1.0 }
    }
}

impl JokerEffect {
    pub fn combine(self, other: JokerEffect) -> JokerEffect {
        JokerEffect {
            chip_delta: self.chip_delta + other.chip_delta,
            mult_delta: self.mult_delta + other.mult_delta,
            xmult: self.xmult * other.xmult,
        }
    }
}

/// Hand-state context for jokers that depend on run-level fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct HandContext {
    pub money: i32,
    pub joker_count: u32,
    pub joker_slot_limit: u32,
    pub discards_remaining: u32,
    pub hands_remaining: u32,
    /// Number of times THIS hand_type has been played this round
    /// (BEFORE the current play). Used by Supernova.
    pub played_count_this_hand_type: u32,
    /// True if THIS hand_type was played at least once already this
    /// round. Used by Card Sharp.
    pub hand_type_played_before: bool,
    /// Cards remaining in the deck (draw pile). Used by Blue Joker
    /// fallback formula (2*deck_size when joker metadata is absent).
    pub deck_size: u32,
    /// Highest play-count among hand_types OTHER than the current one.
    /// Used by Obelisk's "no dominant hand type" gate: scale only when
    /// some other hand_type has been played at least as many times as
    /// THIS hand type would be after the current play.
    pub played_count_max_other_hand_type: u32,
    /// True when Pareidolia is in the joker set — every card counts
    /// as a face card for face-detection effects. Used by Photograph,
    /// Smiley Face, Scary Face, Sock and Buskin retrigger, Ride the
    /// Bus's no-face gate.
    pub pareidolia_active: bool,
}

/// PER-CARD joker effect: applies to one scoring card at a time during
/// the card-scoring loop. Returns `None` if this joker isn't a per-card
/// joker (caller checks the ability_joker_effect path next).
///
/// Per-card jokers (from Python `_SCORED_CARD_JOKER_NAMES`):
/// - Fibonacci, Scholar, Scary Face, Arrowhead, Even Steven, Odd Todd,
///   Smiley Face, Walkie Talkie, Onyx Agate, Triboulet, Photograph
/// - SUIT_MULT_JOKERS: Greedy, Lusty, Wrathful, Gluttonous
///
/// `is_first_face` is true iff `card` is the first face card encountered
/// in the scoring iteration order â€” used by Photograph (x2 on first
/// scored face only).
/// True if `card` counts as a face card. Pareidolia turns ALL cards
/// into face cards for face-detection purposes (Photograph, Smiley
/// Face, Scary Face, Sock and Buskin retrigger, Ride the Bus's
/// no-face gate).
#[inline]
pub fn is_face_with_pareidolia(card: Card, pareidolia_active: bool) -> bool {
    if pareidolia_active {
        return true;
    }
    matches!(card.rank, Rank::Jack | Rank::Queen | Rank::King)
}

pub fn per_card_joker_effect(
    joker_name: &str,
    card: Card,
    is_first_face: bool,
    meta: JokerMetadata,
    pareidolia_active: bool,
) -> Option<JokerEffect> {
    let chips = |n: i64| JokerEffect { chip_delta: n, ..Default::default() };
    let mult = |n: i64| JokerEffect { mult_delta: n, ..Default::default() };
    let chip_mult = |c: i64, m: i64| JokerEffect {
        chip_delta: c, mult_delta: m, ..Default::default()
    };
    let xmult = |x: f64| JokerEffect { xmult: x, ..Default::default() };

    let r = card.rank;
    let s = card.suit;
    let is_face = is_face_with_pareidolia(card, pareidolia_active);
    match joker_name {
        // Rank-set jokers
        "Fibonacci" if matches!(r, Rank::Ace | Rank::Two | Rank::Three | Rank::Five | Rank::Eight) => Some(mult(8)),
        "Scholar" if r == Rank::Ace => Some(chip_mult(20, 4)),
        "Smiley Face" if is_face => Some(mult(5)),
        "Scary Face" if is_face => Some(chips(30)),
        "Walkie Talkie" if matches!(r, Rank::Ten | Rank::Four) => Some(chip_mult(10, 4)),
        "Even Steven" if matches!(r, Rank::Two | Rank::Four | Rank::Six | Rank::Eight | Rank::Ten) => Some(mult(4)),
        "Odd Todd" if matches!(r, Rank::Ace | Rank::Three | Rank::Five | Rank::Seven | Rank::Nine) => Some(chips(31)),
        // Suit jokers
        "Greedy Joker" if s == Suit::Diamonds => Some(mult(3)),
        "Lusty Joker" if s == Suit::Hearts => Some(mult(3)),
        "Wrathful Joker" if s == Suit::Spades => Some(mult(3)),
        "Gluttonous Joker" if s == Suit::Clubs => Some(mult(3)),
        "Onyx Agate" if s == Suit::Clubs => Some(mult(7)),
        "Arrowhead" if s == Suit::Spades => Some(chips(50)),
        // Xmult per-card
        "Triboulet" if matches!(r, Rank::King | Rank::Queen) => Some(xmult(2.0)),
        // Photograph: x2 mult only on the FIRST scored face. Subsequent
        // faces (including the one immediately after the first) do NOT
        // trigger â€” Python tracks `first_face_index` and gates the
        // multiply_mult on `index == first_face_index`.
        "Photograph" if is_first_face => Some(xmult(2.0)),
        // Ancient Joker â€” x1.5 mult per scored card matching target_suit.
        // Suit is metadata-derived; uninitialized = no contribution.
        "Ancient Joker" if meta.has_target_suit && s as u8 == meta.target_suit =>
            Some(xmult(1.5)),
        // The Idol â€” x2 mult per scored card matching target_rank AND
        // target_suit. Both must be set.
        "The Idol" if meta.has_target_rank && meta.has_target_suit
                && r as u8 == meta.target_rank
                && s as u8 == meta.target_suit =>
            Some(xmult(2.0)),

        // Per-card jokers that didn't match the rank/suit condition â†’
        // 0 contribution for this card (NOT None â€” we DO know the joker).
        "Fibonacci" | "Scholar" | "Smiley Face" | "Scary Face"
        | "Walkie Talkie" | "Even Steven" | "Odd Todd"
        | "Greedy Joker" | "Lusty Joker" | "Wrathful Joker" | "Gluttonous Joker"
        | "Onyx Agate" | "Arrowhead" | "Triboulet" | "Photograph"
        | "Ancient Joker" | "The Idol" => Some(JokerEffect::default()),

        // Not a per-card joker â€” caller will check ability_joker_effect.
        _ => None,
    }
}

/// Joker-specific metadata snapshot. One per held joker, computed
/// by the Python wire-in at FFI conversion time from
/// `_joker_current_plus(joker, "mult"/"chips")` and
/// `_joker_current_xmult(joker)`. For synthetic jokers (no bridge
/// metadata) these default to 0/0/1.0 â€” which is what scaling
/// jokers contribute at "zero scaling" anyway.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct JokerMetadata {
    pub current_plus_mult: i32,
    pub current_plus_chips: i32,
    pub current_xmult: f64,
    /// True when Loyalty Card's "remaining == 0" cycle gate fires
    /// this hand. Python: `_loyalty_card_ready(joker)`.
    pub loyalty_ready: bool,
    /// True when Driver's License has â‰¥16 enhanced cards in the deck.
    /// Python: `_drivers_license_active(joker)`.
    pub drivers_active: bool,
    /// Starting / leading value of a decaying scaling joker. Python
    /// uses this as the fallback for Ice Cream (chips) / Popcorn (mult)
    /// when `current_plus` is zero â€” i.e. fresh joker or first hand.
    pub leading_plus_mult: i32,
    pub leading_plus_chips: i32,
    /// Sell value of THIS joker, used by Swashbuckler (which adds the
    /// sum of OTHER jokers' sell values as +mult).
    pub sell_value: i32,
    /// Joker rarity, encoded as: 0 = common, 1 = uncommon, 2 = rare,
    /// 3 = legendary. Baseball Card uses `rarity == Uncommon` to gate
    /// its x1.5 mult bonus per uncommon other joker.
    pub rarity: u8,
    /// Target suit (Ancient Joker, The Idol). `target_suit` is the
    /// `Suit as u8` value (Clubs=0..Spades=3); only consulted when
    /// `has_target_suit` is true.
    pub has_target_suit: bool,
    pub target_suit: u8,
    /// Target rank (The Idol). `target_rank` is the `Rank as u8`
    /// value (Two=0..Ace=12); only consulted when `has_target_rank`
    /// is true.
    pub has_target_rank: bool,
    pub target_rank: u8,
    /// Obelisk-specific: precomputed `_obelisk_xmult_gain(joker)`.
    /// Pulled from joker metadata `xmult_gain` / `x_mult_gain` /
    /// `Xmult_mod` / `extra` with default 0.2.
    pub obelisk_gain: f64,
}

/// ABILITY joker effect: applies once after the card-scoring loop.
/// Returns `None` if this joker isn't an ability joker.
///
/// `scoring_cards` is the slice of cards that scored â€” needed for
/// jokers that look across all scored cards as a SET (Flower Pot
/// checks for one card of each suit, Seeing Double looks for
/// Club+non-Club).
///
/// `held_cards` is the slice of cards still in hand â€” needed for
/// jokers that condition on the held set (Blackboard checks all
/// black-suited).
///
/// `meta` is the joker's per-instance scaling metadata. Used by
/// scaling jokers (Vampire, Hologram, etc.) to read accumulated
/// chip/mult/xmult contributions. Default values (0/0/1.0) mean
/// "no scaling yet" â€” these match the synthetic test case where
/// joker metadata is not populated.
pub fn ability_joker_effect(
    joker_name: &str,
    hand_type: HandType,
    total_played: usize,
    scoring_cards: &[Card],
    held_cards: &[Card],
    ctx: HandContext,
    meta: JokerMetadata,
) -> Option<JokerEffect> {
    let chips = |n: i64| JokerEffect { chip_delta: n, ..Default::default() };
    let mult = |n: i64| JokerEffect { mult_delta: n, ..Default::default() };
    let xmult = |x: f64| JokerEffect { xmult: x, ..Default::default() };

    match joker_name {
        // Flat-effect ability jokers
        "Joker" => Some(mult(4)),
        "Stuntman" => Some(chips(250)),
        "Gros Michel" => Some(mult(15)),

        // Hand-shape gated jokers
        "Jolly Joker" if hand_contains_pair(hand_type) => Some(mult(8)),
        "Zany Joker" if hand_contains_three_of_a_kind(hand_type) => Some(mult(12)),
        "Mad Joker" if hand_contains_two_pair(hand_type) => Some(mult(10)),
        "Crazy Joker" if hand_type == HandType::Straight || hand_type == HandType::StraightFlush => Some(mult(12)),
        "Droll Joker" if hand_contains_flush(hand_type) => Some(mult(10)),
        "Sly Joker" if hand_contains_pair(hand_type) => Some(chips(50)),
        "Wily Joker" if hand_contains_three_of_a_kind(hand_type) => Some(chips(100)),
        "Clever Joker" if hand_contains_two_pair(hand_type) => Some(chips(80)),
        "Devious Joker" if hand_type == HandType::Straight || hand_type == HandType::StraightFlush => Some(chips(100)),
        "Crafty Joker" if hand_contains_flush(hand_type) => Some(chips(80)),
        "Jolly Joker" | "Zany Joker" | "Mad Joker" | "Crazy Joker"
        | "Droll Joker" | "Sly Joker" | "Wily Joker" | "Clever Joker"
        | "Devious Joker" | "Crafty Joker" => Some(JokerEffect::default()),

        // Hand-size jokers
        "Half Joker" if total_played <= 3 => Some(mult(20)),
        "Half Joker" => Some(JokerEffect::default()),

        // Ctx-dependent jokers
        "Bull" => Some(chips((2 * ctx.money.max(0)) as i64)),
        "Banner" => Some(chips((30 * ctx.discards_remaining) as i64)),
        "Mystic Summit" if ctx.discards_remaining == 0 => Some(mult(15)),
        "Mystic Summit" => Some(JokerEffect::default()),
        "Abstract Joker" => Some(mult((3 * ctx.joker_count) as i64)),
        "Bootstraps" => Some(mult((2 * (ctx.money.max(0) / 5)) as i64)),

        // Ability xmult jokers (gated on hand shape)
        "The Duo" if hand_contains_pair(hand_type) => Some(xmult(2.0)),
        "The Trio" if hand_contains_three_of_a_kind(hand_type) => Some(xmult(3.0)),
        "The Family" if matches!(hand_type, HandType::FourOfAKind | HandType::FiveOfAKind
                                  | HandType::FlushFive) => Some(xmult(4.0)),
        "The Order" if hand_type == HandType::Straight || hand_type == HandType::StraightFlush
            => Some(xmult(3.0)),
        "The Tribe" if hand_contains_flush(hand_type) => Some(xmult(2.0)),
        "The Duo" | "The Trio" | "The Family" | "The Order" | "The Tribe"
            => Some(JokerEffect::default()),

        // Acrobat: x3 mult on the LAST hand played (hands_remaining will
        // be 1 when the hand-to-play is about to consume the final
        // hands_remaining slot in the round). Python's check is `==1`.
        "Acrobat" if ctx.hands_remaining == 1 => Some(xmult(3.0)),
        "Acrobat" => Some(JokerEffect::default()),

        // Flower Pot: x3 mult if scored cards span all 4 suits.
        // Note: simple-case implementation. Python uses joker-aware
        // suit detection (Wild cards count as any suit, Smeared Joker
        // makes red=red/black=black). Wild card handling is already
        // blocked by identify_hand_type bailing, and Smeared isn't in
        // our supported set, so this simple check matches the Rust
        // fast-path conditions.
        "Flower Pot" if has_all_four_suits(scoring_cards) => Some(xmult(3.0)),
        "Flower Pot" => Some(JokerEffect::default()),

        // Cavendish: always x3 mult. (The 1-in-1000 death is a separate
        // state effect, not part of the scoring computation.)
        "Cavendish" => Some(xmult(3.0)),

        // Blackboard: x3 mult if all held cards are black (Spades/Clubs).
        // Stone cards disqualify. Empty held â†’ inactive.
        "Blackboard" if blackboard_active(held_cards) => Some(xmult(3.0)),
        "Blackboard" => Some(JokerEffect::default()),

        // Card Sharp: x3 mult if THIS hand_type was already played
        // earlier in the round. The ctx field is computed by the
        // caller (Python wire-in reads from
        // played_hand_types_this_round).
        "Card Sharp" if ctx.hand_type_played_before => Some(xmult(3.0)),
        "Card Sharp" => Some(JokerEffect::default()),

        // Supernova: +N mult where N = count of THIS hand_type already
        // played (this hand included). Python: count_before + 1.
        // Python falls back to this when joker.effect.current_plus has
        // no value â€” which is the synthetic-joker case. For real
        // bridge-state jokers, Python uses the metadata value; we'd
        // bail then. Since the synthetic case is dominant for the
        // solver's hot path, this matches in practice.
        "Supernova" if meta.current_plus_mult > 0 => Some(mult(meta.current_plus_mult as i64)),
        "Supernova" => Some(mult((ctx.played_count_this_hand_type + 1) as i64)),

        // === Scaling jokers â€” read from joker metadata ===
        // Most read the scaled value AND add a per-trigger bonus
        // when the trigger condition is met. Synthetic jokers (no
        // metadata) get 0 from the metadata read; the trigger bonus
        // still fires.

        // Scaled additive-mult, no trigger bonus.
        "Erosion" | "Fortune Teller" | "Red Card" | "Flash Card"
        | "Ceremonial Dagger" =>
            Some(mult(meta.current_plus_mult as i64)),

        // Scaled additive-mult + +1 ALWAYS.
        "Green Joker" => Some(mult(meta.current_plus_mult as i64 + 1)),

        // Scaled additive-mult, fires ONLY when no scored face.
        // (Python wraps the whole block in the no-face check.)
        // Pareidolia turns every card into a face → no card passes
        // the no-face check → joker never fires.
        "Ride the Bus" if !ctx.pareidolia_active && scoring_cards.iter().all(|c| !matches!(
            c.rank, Rank::Jack | Rank::Queen | Rank::King
        )) => Some(mult(meta.current_plus_mult as i64 + 1)),
        "Ride the Bus" => Some(JokerEffect::default()),

        // Scaled additive-mult + 2 if hand contains two-pair.
        "Spare Trousers" if hand_contains_two_pair(hand_type) =>
            Some(mult(meta.current_plus_mult as i64 + 2)),
        "Spare Trousers" => Some(mult(meta.current_plus_mult as i64)),

        // Scaled additive-chips, no trigger.
        "Castle" | "Stone Joker" => Some(chips(meta.current_plus_chips as i64)),

        // Scaled additive-chips + 4 if hand is EXACTLY 4 cards.
        "Square Joker" if total_played == 4 =>
            Some(chips(meta.current_plus_chips as i64 + 4)),
        "Square Joker" => Some(chips(meta.current_plus_chips as i64)),

        // Scaled additive-chips + 15 if hand is a straight.
        "Runner" if hand_type == HandType::Straight || hand_type == HandType::StraightFlush =>
            Some(chips(meta.current_plus_chips as i64 + 15)),
        "Runner" => Some(chips(meta.current_plus_chips as i64)),

        // Wee Joker: +current_plus_chips + 8 * scored 2s with
        // retrigger counts. The retrigger awareness needs deeper
        // hooking â€” bail to Python for now. (Listed in
        // is_supported_joker so the pre-check passes; the None here
        // means the dispatch arm rejects, falling through to the
        // catch-all `_ => None` which makes evaluate_simple's
        // is_supported pre-check moot. Force the bail via the
        // ability_joker_effect path returning None.)
        // Wee Joker's contribution is computed inline in
        // evaluate_simple (it needs total_triggers_on_2s). We never
        // dispatch here for Wee Joker; this arm returns default so
        // is_supported_joker recognizes the name.
        "Wee Joker" => Some(JokerEffect::default()),

        // Ice Cream / Popcorn â€” decaying scaling jokers. Python uses
        // `current_plus or leading_plus` â€” meaning if `current_plus`
        // is zero (e.g. fresh joker, never played), fall back to
        // `leading_plus`. We mirror with explicit if/else.
        "Ice Cream" => {
            let n = if meta.current_plus_chips != 0 {
                meta.current_plus_chips
            } else {
                meta.leading_plus_chips
            };
            Some(chips(n as i64))
        }
        "Popcorn" => {
            let n = if meta.current_plus_mult != 0 {
                meta.current_plus_mult
            } else {
                meta.leading_plus_mult
            };
            Some(mult(n as i64))
        }

        // Obelisk â€” xmult scales when some other hand_type has been
        // played at least as many times as THIS hand_type (after the
        // current play). Otherwise contributes x1 (no scaling).
        // `meta.obelisk_gain` is the per-hand growth (typically 0.2).
        "Obelisk" => {
            let this_count_after = ctx.played_count_this_hand_type + 1;
            if ctx.played_count_max_other_hand_type >= this_count_after {
                Some(xmult(meta.current_xmult + meta.obelisk_gain))
            } else {
                Some(JokerEffect::default())
            }
        }

        // Swashbuckler â€” +mult equal to sum of other jokers' sell
        // values. The sum is precomputed Python-side and stored in
        // `current_plus_mult` for the active Swashbuckler (Python
        // pattern: `if physical_joker is ability_joker: add_mult(sum)`
        // else `add_mult(current_plus_mult)`). Either way reading
        // current_plus_mult gives the right answer.
        "Swashbuckler" => Some(mult(meta.current_plus_mult as i64)),

        // Xmult scaling jokers (read `current_xmult`). The xmult
        // metadata value already incorporates any scaling â€” we just
        // multiply through. Joker Stencil's "+1 per empty slot" math
        // is performed bridge-side and stored in current_xmult, so
        // we honor it correctly via metadata. Constellation and Ramen
        // also work here because the Python `_joker_current_xmult`
        // helper handles their custom internal_xmult_from_visible
        // computation before returning the value we read.
        "Vampire" | "Hologram" | "Joker Stencil" | "Lucky Cat"
        | "Hit the Road" | "Steel Joker" | "Glass Joker"
        | "Madness" | "Throwback" | "Yorick" | "Campfire"
        | "Constellation" | "Ramen" | "Caino" =>
            Some(xmult(meta.current_xmult)),

        // === Score-neutral jokers ===
        // The following jokers either have no scoring contribution
        // (pure state effects) or rely on stochastic outcomes that
        // are empty in the solver's deterministic search context.
        // They're listed here so is_supported_joker recognizes them
        // and evaluate_simple doesn't bail; the JokerEffect::default()
        // return means they contribute 0 chips, 0 mult, x1.0 xmult
        // â€” matching Python's behavior in our case.
        //
        // State-only (affect deck / jokers / hand size / money but
        // not the score being computed):
        "Astronomer" | "Burglar" | "Burnt Joker" | "Cartomancer" | "Certificate"
        | "Chaos the Clown" | "Chicot" | "Cloud 9" | "Credit Card"
        | "Delayed Gratification" | "Diet Cola" | "DNA" | "Drunkard" | "Egg"
        | "Faceless Joker" | "Gift Card" | "Golden Joker"
        | "Golden Ticket" | "Hallucination" | "Hiker"
        | "Invisible Joker" | "Juggler" | "Luchador"
        | "Mail-In Rebate" | "Marble Joker" | "Matador"
        | "Merry Andy" | "Midas Mask" | "Mr. Bones" | "Oops! All 6s"
        | "Perkeo" | "Riff-raff" | "Rough Gem" | "Satellite"
        | "Seance" | "Showman" | "Sixth Sense" | "Superposition"
        | "To Do List" | "To the Moon" | "Trading Card"
        | "Troubadour" | "Turtle Bean"
        // RNG jokers â€” fire only when stochastic_outcomes is non-empty;
        // in the solver's deterministic search context that's the
        // default (empty), so 0 contribution.
        | "Misprint" | "Bloodstone" | "Business Card"
        | "Space Joker" | "Reserved Parking" | "Vagabond"
        | "8 Ball" | "Rocket"
        // Retrigger jokers â€” their effect is to add triggers to scored
        // cards (handled by `scored_card_trigger_count` in the per-card
        // pass). Their ability-pass contribution is zero.
        | "Hack" | "Sock and Buskin" | "Dusk" | "Seltzer"
        | "Hanging Chad"
        // Splash â€” its effect (force all cards to score) is handled
        // in scoring_indices_simple, not here.
        | "Splash"
        // Baseball Card â€” its multiplicative effect (x1.5 mult on
        // uncommon-rarity other jokers) is applied OUTSIDE this
        // function in evaluate_simple's ability loop. Here it
        // contributes nothing on its own.
        | "Baseball Card"
        // Pareidolia: sets ctx.pareidolia_active so face-gated jokers
        // fire on every card. The joker itself adds nothing to score.
        | "Pareidolia"
        // Identification modifiers: their effect is consumed by
        // identify_hand_type_with_jokers / scoring_indices_with_jokers
        // before reaching this dispatch. Score contribution is 0.
        | "Smeared Joker" | "Four Fingers" | "Shortcut"
        // Blueprint / Brainstorm — unresolved copies (no compatible
        // target) score zero.
        | "Blueprint" | "Brainstorm" =>
            Some(JokerEffect::default()),

        // Gating-based xmult jokers. Python checks the joker's
        // ready/active state from metadata; we pass it as a flag on
        // the JokerMetadata struct.
        "Loyalty Card" if meta.loyalty_ready => Some(xmult(4.0)),
        "Loyalty Card" => Some(JokerEffect::default()),
        "Driver's License" if meta.drivers_active => Some(xmult(3.0)),
        "Driver's License" => Some(JokerEffect::default()),

        // Blue Joker: +current_plus_chips OR 2*deck_size as fallback
        // when metadata is absent. Synthetic jokers have
        // current_plus_chips=0, so we always fall to the deck_size
        // formula. Bridge-real jokers fill current_plus_chips with
        // the scaled value AND we never hit the fallback. Python:
        //   add_chips(current_plus or 2*max(0, deck_size))
        "Blue Joker" if meta.current_plus_chips > 0 =>
            Some(chips(meta.current_plus_chips as i64)),
        "Blue Joker" => Some(chips((2 * ctx.deck_size) as i64)),

        // Seeing Double: x2 mult if scored cards include at least one
        // Club AND at least one non-Club. Python's check uses joker-
        // aware suit detection; in the no-Wild/no-Smeared fast path it
        // reduces to the simple count.
        "Seeing Double" if has_club_and_non_club(scoring_cards) => Some(xmult(2.0)),
        "Seeing Double" => Some(JokerEffect::default()),

        // Not an ability joker â€” caller will have already checked per_card.
        _ => None,
    }
}

/// True if `scoring_cards` includes at least one card of each suit.
fn has_all_four_suits(cards: &[Card]) -> bool {
    let mut seen = [false; 4];
    for c in cards {
        seen[c.suit as usize] = true;
    }
    seen.iter().all(|&s| s)
}

/// True if `scoring_cards` includes at least one Club AND at least
/// one non-Club. Used by Seeing Double.
fn has_club_and_non_club(cards: &[Card]) -> bool {
    let mut has_club = false;
    let mut has_other = false;
    for c in cards {
        if c.suit == Suit::Clubs {
            has_club = true;
        } else {
            has_other = true;
        }
        if has_club && has_other {
            return true;
        }
    }
    false
}

/// True if `name` is a joker the Rust port knows about (either pass).
/// `evaluate_simple` uses this to decide whether to bail to Python.
pub fn is_supported_joker(name: &str) -> bool {
    matches!(
        name,
        // Per-card
        "Fibonacci" | "Scholar" | "Smiley Face" | "Scary Face"
        | "Walkie Talkie" | "Even Steven" | "Odd Todd"
        | "Greedy Joker" | "Lusty Joker" | "Wrathful Joker" | "Gluttonous Joker"
        | "Onyx Agate" | "Arrowhead" | "Triboulet"
        // Ability â€” flat
        | "Joker" | "Stuntman" | "Gros Michel"
        // Ability â€” hand-shape additive
        | "Jolly Joker" | "Zany Joker" | "Mad Joker" | "Crazy Joker" | "Droll Joker"
        | "Sly Joker" | "Wily Joker" | "Clever Joker" | "Devious Joker" | "Crafty Joker"
        // Ability â€” ctx
        | "Half Joker"
        | "Bull" | "Banner" | "Mystic Summit" | "Abstract Joker" | "Bootstraps"
        // Ability â€” xmult
        | "The Duo" | "The Trio" | "The Family" | "The Order" | "The Tribe"
        | "Acrobat" | "Flower Pot"
        | "Cavendish" | "Seeing Double"
        | "Blackboard"
        | "Card Sharp" | "Supernova"
        // Scaling additive-mult jokers (read current_plus_mult)
        | "Ride the Bus" | "Green Joker" | "Spare Trousers"
        | "Castle" | "Erosion"
        // Scaling additive-chip jokers (read current_plus_chips)
        | "Square Joker" | "Runner"
        // Scaling xmult jokers (read current_xmult â€” pure
        // multiply_mult(current_xmult), no gating beyond what
        // bridge metadata captures). Constellation and Ramen work
        // because Python wire-in's _joker_current_xmult helper
        // already runs their custom internal_xmult formula before
        // we read the value.
        | "Vampire" | "Hologram" | "Joker Stencil" | "Lucky Cat"
        | "Hit the Road" | "Steel Joker" | "Glass Joker"
        | "Madness" | "Throwback" | "Yorick" | "Campfire"
        | "Constellation" | "Ramen" | "Caino"
        // Additional scaling-mult jokers
        | "Fortune Teller" | "Red Card" | "Flash Card"
        | "Ceremonial Dagger"
        // Additional scaling-chips jokers
        | "Stone Joker"
        // Deck-size dependent
        | "Blue Joker"
        // Score-neutral jokers â€” state effects, money effects, or
        // RNG jokers that contribute 0 when stochastic_outcomes
        // is empty (our solver's deterministic-search context).
        | "Astronomer" | "Burglar" | "Cartomancer" | "Certificate"
        | "Chaos the Clown" | "Chicot" | "Cloud 9" | "Credit Card"
        | "Delayed Gratification" | "DNA" | "Drunkard" | "Egg"
        | "Faceless Joker" | "Gift Card" | "Golden Joker"
        | "Golden Ticket" | "Hallucination" | "Hiker"
        | "Invisible Joker" | "Juggler" | "Luchador"
        | "Mail-In Rebate" | "Marble Joker" | "Matador"
        | "Merry Andy" | "Midas Mask" | "Mr. Bones" | "Oops! All 6s"
        | "Perkeo" | "Riff-raff" | "Rough Gem" | "Satellite"
        | "Seance" | "Showman" | "Sixth Sense" | "Superposition"
        | "To Do List" | "To the Moon" | "Trading Card"
        | "Troubadour" | "Turtle Bean"
        | "Misprint" | "Bloodstone" | "Business Card"
        | "Space Joker" | "Reserved Parking" | "Vagabond"
        | "8 Ball" | "Rocket"
        // Retrigger jokers â€” their ability-pass effect is zero;
        // their per-card retrigger is handled in evaluate_simple's
        // scored_card_trigger_count + inner loop.
        | "Hack" | "Sock and Buskin" | "Dusk" | "Seltzer"
        | "Hanging Chad"
        // Splash
        | "Splash"
        // Gating-based xmult jokers (need extra metadata flags)
        | "Loyalty Card" | "Driver's License"
        // Scaling jokers needing leading_plus or per-hand-shape gating
        | "Ice Cream" | "Popcorn" | "Obelisk"
        // Multi-joker interactions
        | "Swashbuckler" | "Baseball Card"
        // Target-suit / target-rank jokers
        | "Ancient Joker" | "The Idol"
        // Score-neutral additions (sold/discard-triggered)
        | "Burnt Joker" | "Diet Cola"
        // Identification modifier — Pareidolia turns every card into
        // a face card for face-detection effects. Effects applied via
        // ctx.pareidolia_active in face-gated joker arms; the joker
        // itself contributes 0 score.
        | "Pareidolia"
        // Wee Joker — handled inline in evaluate_simple (needs
        // total_triggers_on_2s). ability_joker_effect returns default
        // for it.
        | "Wee Joker"
        // Identification modifiers — set IdJokers flags consumed by
        // identify_hand_type_with_jokers / scoring_indices_with_jokers.
        // The joker itself contributes 0 to score.
        | "Smeared Joker" | "Four Fingers" | "Shortcut"
        // Blueprint / Brainstorm — Python wire-in resolves these,
        // substituting the COPIED joker's name + metadata at the
        // physical slot. If they reach Rust unresolved (no copy
        // target / incompatible target), the slot scores zero.
        | "Blueprint" | "Brainstorm"
        // Per-card (late addition — kept here rather than at the
        // top per-card block to preserve historical batch ordering).
        | "Photograph"
        // Held-card pass (acted on by evaluate_simple's held loop +
        // the held-card-effect function above, NOT by per_card/ability
        // dispatch. Listed here so is_supported_joker recognizes them
        // and evaluate_simple doesn't bail.)
        | "Shoot the Moon" | "Baron" | "Raised Fist" | "Mime"
    )
}

/// Per-scored-card enhancement effect (chips/mult).
pub fn enhancement_effect(enhancement: Enhancement) -> JokerEffect {
    match enhancement {
        Enhancement::Mult => JokerEffect { mult_delta: 4, ..Default::default() },
        // Bonus chip handled in card_chip_value, not here.
        // Glass/Steel/Stone/Gold/Lucky deferred (need RNG / held-card).
        _ => JokerEffect::default(),
    }
}

/// Effect of a HELD card (not scored): runs in the held-card pass
/// after the ability pass. Combines:
/// - Steel enhancement â†’ x1.5 mult
/// - Per-held-card joker effects (Shoot the Moon, Baron, Raised Fist)
///
/// `lowest_held` is the card identified as the "raised fist" target
/// (lowest-rank held card, ignoring Stone). When `Some` and matches
/// `card`, Raised Fist adds 2 * rank_value mult.
///
/// Returns a SINGLE accumulated JokerEffect per held card. The caller
/// (held pass in evaluate_simple) is responsible for retriggering
/// this effect Mime times (+ 1 if Red Seal).
pub fn held_card_effect(
    card: Card,
    joker_names: &[String],
    debuffed_suits: &[Suit],
    is_raised_fist_target: bool,
) -> JokerEffect {
    let mut acc = JokerEffect::default();

    // Stone cards and debuffed cards contribute nothing.
    if card.enhancement == Enhancement::Stone {
        return acc;
    }
    if card.debuffed {
        return acc;
    }
    if !debuffed_suits.is_empty() && debuffed_suits.contains(&card.suit) {
        return acc;
    }

    // Steel enhancement: x1.5 mult per held Steel card.
    // (Steel scored cards also score chips per `_card_chip_value` â€”
    // but Steel doesn't add chips. The chips=0 base from
    // `card_chip_value` handles non-Steel cases.)
    if card.enhancement == Enhancement::Steel {
        acc.xmult *= 1.5;
    }

    // Per-card jokers that fire on held cards.
    for name in joker_names {
        match name.as_str() {
            "Shoot the Moon" if card.rank == Rank::Queen => {
                acc.mult_delta += 13;
            }
            "Baron" if card.rank == Rank::King => {
                acc.xmult *= 1.5;
            }
            "Raised Fist" if is_raised_fist_target => {
                acc.mult_delta += 2 * card.rank.chip_value() as i64;
            }
            _ => {}
        }
    }

    acc
}

/// Number of times each scored card's effect should fire. Mirrors
/// Python `_scored_card_trigger_counts`. Base count is 1; retrigger
/// jokers add extras per matching card.
///
/// Retrigger contributors:
/// - Red Seal: +1 per card with red seal
/// - Hack: +1 per scored 2/3/4/5 (per Hack held)
/// - Sock and Buskin: +1 per scored face card (per Sock and Buskin held)
/// - Dusk: +1 per scored card (only if hands_remaining == 1)
/// - Seltzer: +1 per scored card (per Seltzer)
/// - Hanging Chad: +2 ONLY for the first scored card (per Hanging Chad)
pub fn scored_card_trigger_count(
    card: Card,
    is_first_scored: bool,
    joker_names: &[String],
    hands_remaining: u32,
    pareidolia_active: bool,
) -> u32 {
    let mut count: u32 = 1;
    if card.seal == Seal::Red {
        count += 1;
    }
    let is_2_to_5 = matches!(card.rank, Rank::Two | Rank::Three | Rank::Four | Rank::Five);
    let is_face = is_face_with_pareidolia(card, pareidolia_active);
    let dusk_active = hands_remaining == 1;
    for n in joker_names {
        match n.as_str() {
            "Hack" if is_2_to_5 => count += 1,
            "Sock and Buskin" if is_face => count += 1,
            "Dusk" if dusk_active => count += 1,
            "Seltzer" => count += 1,
            "Hanging Chad" if is_first_scored => count += 2,
            _ => {}
        }
    }
    count
}

/// Number of times each held card's effect should fire.
/// = 1 (base) + count(Mime in jokers) + (1 if red_seal else 0).
#[inline]
pub fn held_retrigger_count(joker_names: &[String], red_seal: bool) -> u32 {
    let mut count: u32 = 1;
    for n in joker_names {
        if n == "Mime" {
            count += 1;
        }
    }
    if red_seal {
        count += 1;
    }
    count
}

/// Fast-skip predicate: returns true iff the held-card pass might
/// produce any non-default effect. Lets evaluate_simple bypass the
/// held loop entirely on the common case (no Steel cards in hand,
/// no held-card joker, no red seals on held).
pub fn held_pass_possible(held_cards: &[Card], joker_names: &[String]) -> bool {
    if held_cards.is_empty() {
        return false;
    }
    // Any Steel-enhanced held card â†’ x1.5 mult contribution.
    if held_cards.iter().any(|c| c.enhancement == Enhancement::Steel) {
        return true;
    }
    // Any red-sealed held card â†’ retriggers (relevant if there's any
    // held-card joker, see next check).
    let any_red_seal = held_cards.iter().any(|c| c.seal == Seal::Red);
    // Any held-card-affecting joker.
    let has_held_joker = joker_names.iter().any(|n| matches!(
        n.as_str(),
        "Shoot the Moon" | "Baron" | "Raised Fist" | "Mime"
    ));
    has_held_joker || (any_red_seal && false)
    // (Red seals alone don't add an effect; they only multiply
    // existing held effects. So we don't need to run the pass
    // unless there's a joker too.)
}

/// Identify the Raised Fist target â€” the LAST card in the held order
/// whose straight-value equals the minimum among non-Stone held cards.
/// Returns `None` if no candidates (all Stone). Matches Python
/// `_raised_fist_card`.
pub fn raised_fist_target(held_cards: &[Card]) -> Option<usize> {
    let mut candidates: Vec<(usize, u8)> = held_cards.iter().enumerate()
        .filter(|(_, c)| c.enhancement != Enhancement::Stone)
        .map(|(i, c)| (i, c.rank.straight_value()))
        .collect();
    if candidates.is_empty() {
        return None;
    }
    let min_value = candidates.iter().map(|(_, v)| *v).min().unwrap();
    candidates.retain(|(_, v)| *v == min_value);
    // Python uses `next(c for c in reversed(candidates) if ...)` â€” the
    // LAST candidate with min value in iteration order.
    Some(candidates.last().unwrap().0)
}

/// Blackboard: x3 mult if ALL held cards are black (Spades or Clubs).
/// Stone cards are not black. An empty held hand counts as inactive.
pub fn blackboard_active(held_cards: &[Card]) -> bool {
    if held_cards.is_empty() {
        return false;
    }
    held_cards.iter().all(|c| {
        c.enhancement != Enhancement::Stone
            && (c.suit == Suit::Spades || c.suit == Suit::Clubs)
    })
}

// --- Hand-type predicates -------------------------------------------------

#[inline]
fn hand_contains_pair(ht: HandType) -> bool {
    matches!(ht, HandType::Pair | HandType::TwoPair | HandType::ThreeOfAKind
              | HandType::FullHouse | HandType::FourOfAKind | HandType::FiveOfAKind
              | HandType::FlushHouse | HandType::FlushFive)
}

#[inline]
fn hand_contains_two_pair(ht: HandType) -> bool {
    matches!(ht, HandType::TwoPair | HandType::FullHouse | HandType::FlushHouse)
}

#[inline]
fn hand_contains_three_of_a_kind(ht: HandType) -> bool {
    matches!(ht, HandType::ThreeOfAKind | HandType::FullHouse | HandType::FourOfAKind
              | HandType::FiveOfAKind | HandType::FlushHouse | HandType::FlushFive)
}

#[inline]
fn hand_contains_flush(ht: HandType) -> bool {
    matches!(ht, HandType::Flush | HandType::StraightFlush | HandType::FlushHouse
              | HandType::FlushFive)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::card::{Edition, Seal};

    fn card(rank: Rank, suit: Suit) -> Card {
        Card { rank, suit, enhancement: Enhancement::None,
               edition: Edition::None, seal: Seal::None, debuffed: false }
    }

    fn ctx() -> HandContext { HandContext::default() }

    // Ability-pass jokers
    #[test]
    fn plain_joker_ability_only() {
        let e = ability_joker_effect("Joker", HandType::HighCard, 5, &[], &[], ctx(), JokerMetadata::default()).unwrap();
        assert_eq!(e.mult_delta, 4);
        assert!(per_card_joker_effect("Joker", card(Rank::Ace, Suit::Hearts), false, JokerMetadata::default(), false).is_none());
    }

    #[test]
    fn jolly_joker_on_pair() {
        assert_eq!(
            ability_joker_effect("Jolly Joker", HandType::Pair, 5, &[], &[], ctx(), JokerMetadata::default()).unwrap().mult_delta,
            8,
        );
    }

    #[test]
    fn jolly_joker_high_card_no_effect() {
        assert_eq!(
            ability_joker_effect("Jolly Joker", HandType::HighCard, 5, &[], &[], ctx(), JokerMetadata::default()).unwrap(),
            JokerEffect::default(),
        );
    }

    #[test]
    fn bull_chips_per_dollar() {
        let c = HandContext { money: 25, ..ctx() };
        assert_eq!(ability_joker_effect("Bull", HandType::HighCard, 5, &[], &[], c, JokerMetadata::default()).unwrap().chip_delta, 50);
    }

    #[test]
    fn abstract_joker_3_per_joker() {
        let c = HandContext { joker_count: 3, ..ctx() };
        assert_eq!(ability_joker_effect("Abstract Joker", HandType::HighCard, 5, &[], &[], c, JokerMetadata::default()).unwrap().mult_delta, 9);
    }

    // Per-card jokers
    #[test]
    fn greedy_per_diamond() {
        let m = JokerMetadata::default();
        let e = per_card_joker_effect("Greedy Joker", card(Rank::Ace, Suit::Diamonds), false, m, false).unwrap();
        assert_eq!(e.mult_delta, 3);
        let e_h = per_card_joker_effect("Greedy Joker", card(Rank::Ace, Suit::Hearts), false, m, false).unwrap();
        assert_eq!(e_h.mult_delta, 0);
    }

    #[test]
    fn fibonacci_per_match() {
        let m = JokerMetadata::default();
        let e = per_card_joker_effect("Fibonacci", card(Rank::Five, Suit::Hearts), false, m, false).unwrap();
        assert_eq!(e.mult_delta, 8);
        let e_no = per_card_joker_effect("Fibonacci", card(Rank::Four, Suit::Hearts), false, m, false).unwrap();
        assert_eq!(e_no.mult_delta, 0);
    }

    #[test]
    fn scholar_per_ace() {
        let m = JokerMetadata::default();
        let e = per_card_joker_effect("Scholar", card(Rank::Ace, Suit::Spades), false, m, false).unwrap();
        assert_eq!(e.chip_delta, 20);
        assert_eq!(e.mult_delta, 4);
    }

    #[test]
    fn triboulet_per_kq() {
        let m = JokerMetadata::default();
        let e = per_card_joker_effect("Triboulet", card(Rank::King, Suit::Hearts), false, m, false).unwrap();
        assert_eq!(e.xmult, 2.0);
        let e_no = per_card_joker_effect("Triboulet", card(Rank::Five, Suit::Hearts), false, m, false).unwrap();
        assert_eq!(e_no.xmult, 1.0);
    }

    #[test]
    fn photograph_only_first_face() {
        let m = JokerMetadata::default();
        let e_first = per_card_joker_effect("Photograph", card(Rank::Jack, Suit::Hearts), true, m, false).unwrap();
        assert_eq!(e_first.xmult, 2.0);
        let e_not_first = per_card_joker_effect("Photograph", card(Rank::Jack, Suit::Hearts), false, m, false).unwrap();
        assert_eq!(e_not_first.xmult, 1.0);
        // Even when is_first_face=true, non-face cards don't trigger
        // (Photograph requires "first face" â€” non-face wouldn't be
        // the first face, but if caller mistakenly passes true the
        // effect SHOULD be neutral. Per Python, the check is the
        // join of "index matches first_face_index" â€” non-face never
        // is the first face by construction. We mirror that: gate
        // only on is_first_face.)
        // (Implementation note: the gating is `is_first_face` alone,
        // so technically a misuse would fire. The caller must compute
        // is_first_face correctly, which evaluate_simple does by
        // scanning for face cards in scoring order.)
    }

    #[test]
    fn unknown_joker_returns_none_from_both_passes() {
        let c = card(Rank::Ace, Suit::Hearts);
        assert!(per_card_joker_effect("Future Joker", c, false, JokerMetadata::default(), false).is_none());
        assert!(ability_joker_effect("Future Joker", HandType::Pair, 5, &[], &[], ctx(), JokerMetadata::default()).is_none());
        assert!(!is_supported_joker("Future Joker"));
    }

    #[test]
    fn mult_enhancement_adds_4() {
        let e = enhancement_effect(Enhancement::Mult);
        assert_eq!(e.mult_delta, 4);
    }

    // === Batch 3 jokers ===

    #[test]
    fn the_duo_x2_on_pair() {
        assert_eq!(
            ability_joker_effect("The Duo", HandType::Pair, 5, &[], &[], ctx(), JokerMetadata::default()).unwrap().xmult,
            2.0,
        );
        assert_eq!(
            ability_joker_effect("The Duo", HandType::HighCard, 5, &[], &[], ctx(), JokerMetadata::default()).unwrap().xmult,
            1.0,
        );
    }

    #[test]
    fn the_trio_x3_on_3oak() {
        assert_eq!(
            ability_joker_effect("The Trio", HandType::ThreeOfAKind, 5, &[], &[], ctx(), JokerMetadata::default()).unwrap().xmult,
            3.0,
        );
        // Also fires on Full House (contains 3OAK)
        assert_eq!(
            ability_joker_effect("The Trio", HandType::FullHouse, 5, &[], &[], ctx(), JokerMetadata::default()).unwrap().xmult,
            3.0,
        );
    }

    #[test]
    fn the_family_x4_on_4oak_only() {
        assert_eq!(
            ability_joker_effect("The Family", HandType::FourOfAKind, 5, &[], &[], ctx(), JokerMetadata::default()).unwrap().xmult,
            4.0,
        );
        // Three of a kind doesn't trigger Family.
        assert_eq!(
            ability_joker_effect("The Family", HandType::ThreeOfAKind, 5, &[], &[], ctx(), JokerMetadata::default()).unwrap().xmult,
            1.0,
        );
    }

    #[test]
    fn the_order_x3_on_straight() {
        assert_eq!(
            ability_joker_effect("The Order", HandType::Straight, 5, &[], &[], ctx(), JokerMetadata::default()).unwrap().xmult,
            3.0,
        );
    }

    #[test]
    fn the_tribe_x2_on_flush() {
        assert_eq!(
            ability_joker_effect("The Tribe", HandType::Flush, 5, &[], &[], ctx(), JokerMetadata::default()).unwrap().xmult,
            2.0,
        );
    }

    #[test]
    fn acrobat_x3_on_last_hand() {
        let last = HandContext { hands_remaining: 1, ..ctx() };
        let not_last = HandContext { hands_remaining: 2, ..ctx() };
        assert_eq!(
            ability_joker_effect("Acrobat", HandType::HighCard, 5, &[], &[], last, JokerMetadata::default()).unwrap().xmult,
            3.0,
        );
        assert_eq!(
            ability_joker_effect("Acrobat", HandType::HighCard, 5, &[], &[], not_last, JokerMetadata::default()).unwrap().xmult,
            1.0,
        );
    }

    #[test]
    fn card_sharp_x3_if_hand_type_seen() {
        let seen = HandContext { hand_type_played_before: true, ..ctx() };
        let not_seen = HandContext { hand_type_played_before: false, ..ctx() };
        assert_eq!(
            ability_joker_effect("Card Sharp", HandType::Pair, 5, &[], &[], seen, JokerMetadata::default()).unwrap().xmult,
            3.0,
        );
        assert_eq!(
            ability_joker_effect("Card Sharp", HandType::Pair, 5, &[], &[], not_seen, JokerMetadata::default()).unwrap().xmult,
            1.0,
        );
    }

    #[test]
    fn supernova_mult_per_count() {
        let c = HandContext { played_count_this_hand_type: 2, ..ctx() };
        // Count 2 + 1 (this play) = 3 â†’ +3 mult.
        assert_eq!(
            ability_joker_effect("Supernova", HandType::Pair, 5, &[], &[], c, JokerMetadata::default()).unwrap().mult_delta,
            3,
        );
    }

    #[test]
    fn flower_pot_x3_on_all_four_suits() {
        let four_suits = [
            card(Rank::Ace, Suit::Clubs),
            card(Rank::Two, Suit::Diamonds),
            card(Rank::Three, Suit::Hearts),
            card(Rank::Four, Suit::Spades),
        ];
        let three_suits = [
            card(Rank::Ace, Suit::Clubs),
            card(Rank::Two, Suit::Diamonds),
            card(Rank::Three, Suit::Hearts),
        ];
        assert_eq!(
            ability_joker_effect("Flower Pot", HandType::HighCard, 4, &four_suits, &[], ctx(), JokerMetadata::default()).unwrap().xmult,
            3.0,
        );
        assert_eq!(
            ability_joker_effect("Flower Pot", HandType::HighCard, 3, &three_suits, &[], ctx(), JokerMetadata::default()).unwrap().xmult,
            1.0,
        );
    }

    #[test]
    fn is_supported_joker_covers_all_ports() {
        // Every joker mentioned in either dispatch must be supported.
        let names = [
            "Joker", "Jolly Joker", "Greedy Joker", "Even Steven",
            "Scary Face", "Half Joker", "Fibonacci", "Triboulet",
            "Bull", "Banner", "Mystic Summit", "Abstract Joker",
            "Stuntman", "Bootstraps", "Gros Michel",
        ];
        for n in &names {
            assert!(is_supported_joker(n), "should support {}", n);
        }
    }
}

