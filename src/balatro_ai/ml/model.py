"""Value network + tensorization for the neural planner (Stage 0.3).

Turns the dependency-free `EncodedState` (Step 0.1) into padded tensors and runs
a small **set-encoder** value net: embeddings for the identity vocabularies
(jokers, cards, shop items, boss, consumables, vouchers), per-token MLPs with
masked mean-pooling over the variable-length sets, concatenated with the fixed
scalar/one-hot/level/deck blocks, then an MLP trunk to a win-probability logit.

This is the NNUE-analog leaf the plan calls for — cheap to evaluate, learns from
the raw inputs the heuristic discards. It is the **only** part of the ml layer
that imports torch; `encoding.py`/`dataset.py` stay torch-free so they test
without it.

Sizes come from `encoding.encoding_spec()`, so the net always matches the
encoder version it was built against (the version is recorded in checkpoints).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from balatro_ai.api.actions import ActionType
from balatro_ai.ml.encoding import EncodedState, encoding_spec

# Action-type vocabulary for the policy head.
POLICY_ACTION_TYPES = tuple(a.value for a in ActionType)
ACTION_TYPE_INDEX = {t: i for i, t in enumerate(POLICY_ACTION_TYPES)}
PLAY_ACTION_TYPES = frozenset({ActionType.PLAY_HAND.value, ActionType.DISCARD.value})

# Per-token feature layout (must match the column order built in `collate_states`).
_JOKER_NCAT, _JOKER_NNUM = 2, 6   # cat: key, edition | num: eternal,perishable,rental,debuffed,counter,sell
_CARD_NCAT, _CARD_NNUM = 5, 1     # cat: rank,suit,enh,edition,seal | num: debuffed
_SHOP_NCAT, _SHOP_NNUM = 3, 1     # cat: item_type,key,edition | num: cost


@dataclass
class Batch:
    """A padded minibatch of encoded states (all tensors share batch dim 0)."""

    scalars: torch.Tensor       # [B, scalar_dim]
    phase: torch.Tensor         # [B, phase_dim]
    blind: torch.Tensor         # [B, blind_type_dim]
    hand_levels: torch.Tensor   # [B, hand_levels_dim]
    deck_counts: torch.Tensor   # [B, deck_counts_dim]
    boss: torch.Tensor          # [B] long
    joker_cat: torch.Tensor     # [B, J, 2] long
    joker_num: torch.Tensor     # [B, J, 6]
    joker_mask: torch.Tensor    # [B, J] bool
    card_cat: torch.Tensor      # [B, H, 5] long
    card_num: torch.Tensor      # [B, H, 1]
    card_mask: torch.Tensor     # [B, H] bool
    shop_cat: torch.Tensor      # [B, S, 3] long
    shop_num: torch.Tensor      # [B, S, 1]
    shop_mask: torch.Tensor     # [B, S] bool
    consumable_idx: torch.Tensor  # [B, C] long
    consumable_mask: torch.Tensor  # [B, C] bool
    voucher_idx: torch.Tensor   # [B, V] long
    voucher_mask: torch.Tensor  # [B, V] bool

    @property
    def size(self) -> int:
        return self.scalars.shape[0]


def _pad_tokens(
    cat_rows: list[list[tuple[int, ...]]],
    num_rows: list[list[tuple[float, ...]]],
    ncat: int,
    nnum: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b = len(cat_rows)
    length = max(1, max((len(r) for r in cat_rows), default=1))
    cat = torch.zeros(b, length, ncat, dtype=torch.long)
    num = torch.zeros(b, length, nnum, dtype=torch.float32)
    mask = torch.zeros(b, length, dtype=torch.bool)
    for i, (crow, nrow) in enumerate(zip(cat_rows, num_rows)):
        for j, (c, n) in enumerate(zip(crow, nrow)):
            cat[i, j] = torch.tensor(c, dtype=torch.long)
            num[i, j] = torch.tensor(n, dtype=torch.float32)
            mask[i, j] = True
    return cat, num, mask


def _pad_index(rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    b = len(rows)
    length = max(1, max((len(r) for r in rows), default=1))
    idx = torch.zeros(b, length, dtype=torch.long)
    mask = torch.zeros(b, length, dtype=torch.bool)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            idx[i, j] = int(v)
            mask[i, j] = True
    return idx, mask


def collate_states(states: Sequence[EncodedState]) -> Batch:
    """Pad a list of `EncodedState` into a `Batch` of tensors."""
    jc, jn, jm = _pad_tokens(
        [[(t.key_index, t.edition_index) for t in s.jokers] for s in states],
        [[(t.eternal, t.perishable, t.rental, t.debuffed, t.counter, t.sell_value)
          for t in s.jokers] for s in states],
        _JOKER_NCAT, _JOKER_NNUM,
    )
    cc_, cn_, cm_ = _pad_tokens(
        [[(t.rank_index, t.suit_index, t.enhancement_index, t.edition_index, t.seal_index)
          for t in s.hand] for s in states],
        [[(t.debuffed,) for t in s.hand] for s in states],
        _CARD_NCAT, _CARD_NNUM,
    )
    sc_, sn_, sm_ = _pad_tokens(
        [[(t.item_type_index, t.key_index, t.edition_index) for t in s.shop] for s in states],
        [[(t.cost,) for t in s.shop] for s in states],
        _SHOP_NCAT, _SHOP_NNUM,
    )
    cons_idx, cons_mask = _pad_index([list(s.consumables) for s in states])
    vou_idx, vou_mask = _pad_index([list(s.vouchers) for s in states])
    return Batch(
        scalars=torch.tensor([s.scalars for s in states], dtype=torch.float32),
        phase=torch.tensor([s.phase_onehot for s in states], dtype=torch.float32),
        blind=torch.tensor([s.blind_type_onehot for s in states], dtype=torch.float32),
        hand_levels=torch.tensor([s.hand_levels for s in states], dtype=torch.float32),
        deck_counts=torch.tensor([s.deck_counts for s in states], dtype=torch.float32),
        boss=torch.tensor([s.boss_index for s in states], dtype=torch.long),
        joker_cat=jc, joker_num=jn, joker_mask=jm,
        card_cat=cc_, card_num=cn_, card_mask=cm_,
        shop_cat=sc_, shop_num=sn_, shop_mask=sm_,
        consumable_idx=cons_idx, consumable_mask=cons_mask,
        voucher_idx=vou_idx, voucher_mask=vou_mask,
    )


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over dim 1 using a [B, L] bool mask; empty sets pool to zero."""
    m = mask.unsqueeze(-1).to(x.dtype)
    summed = (x * m).sum(dim=1)
    denom = m.sum(dim=1).clamp(min=1.0)
    return summed / denom


class ValueNet(nn.Module):
    """Small set-encoder → win-probability logit. Sized from `encoding_spec()`."""

    def __init__(
        self,
        spec: dict | None = None,
        *,
        d_id: int = 16,
        d_small: int = 4,
        d_token: int = 32,
        d_trunk: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        spec = spec or encoding_spec()
        self.encoding_version = spec["version"]
        self.spec = spec
        # Recorded so a checkpoint can rebuild the exact architecture.
        self.hparams = {
            "d_id": d_id, "d_small": d_small, "d_token": d_token, "d_trunk": d_trunk,
            "dropout": dropout,
        }

        # Identity / categorical embeddings.
        self.joker_key = nn.Embedding(spec["joker_vocab_size"], d_id)
        self.joker_edition = nn.Embedding(spec["edition_vocab_size"], d_small)
        self.card_rank = nn.Embedding(spec["rank_vocab_size"], d_small)
        self.card_suit = nn.Embedding(spec["suit_vocab_size"], d_small)
        self.card_enh = nn.Embedding(spec["enhancement_vocab_size"], d_small)
        self.card_edition = nn.Embedding(spec["edition_vocab_size"], d_small)
        self.card_seal = nn.Embedding(spec["seal_vocab_size"], d_small)
        self.shop_type = nn.Embedding(spec["item_type_vocab_size"], d_small)
        self.shop_key = nn.Embedding(spec["item_vocab_size"], d_id)
        self.shop_edition = nn.Embedding(spec["edition_vocab_size"], d_small)
        self.boss_emb = nn.Embedding(spec["boss_vocab_size"], d_small)
        self.consumable_emb = nn.Embedding(spec["consumable_vocab_size"], d_small)
        self.voucher_emb = nn.Embedding(spec["voucher_vocab_size"], d_small)

        # Per-token encoders (concat embeddings + numeric features -> d_token).
        self.joker_mlp = nn.Sequential(
            nn.Linear(d_id + d_small + _JOKER_NNUM, d_token), nn.ReLU())
        self.card_mlp = nn.Sequential(
            nn.Linear(d_small * 5 + _CARD_NNUM, d_token), nn.ReLU())
        self.shop_mlp = nn.Sequential(
            nn.Linear(d_small * 2 + d_id + _SHOP_NNUM, d_token), nn.ReLU())

        fixed = (spec["scalar_dim"] + spec["phase_dim"] + spec["blind_type_dim"]
                 + spec["hand_levels_dim"] + spec["deck_counts_dim"])
        pooled = d_token * 3 + d_small * 3  # jokers, cards, shop + consumable, voucher, boss
        trunk_in = fixed + pooled
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, d_trunk), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_trunk, d_trunk), nn.ReLU(), nn.Dropout(dropout),
        )
        self.win_head = nn.Linear(d_trunk, 1)
        self.ante_head = nn.Linear(d_trunk, 1)   # predicts final ante / 8 in [0,1]
        self.clear_head = nn.Linear(d_trunk, 1)  # distilled rollout leaf value in [0,2]
        # Policy heads (Stage 2): action-type distribution + a per-card pointer
        # for play/discard (scores each hand card, since the trunk pools away
        # positional identity).
        self.type_head = nn.Linear(d_trunk, len(POLICY_ACTION_TYPES))
        self.hand_type_head = nn.Linear(d_trunk, 12)  # poker hand played (play steps)
        self.card_policy = nn.Sequential(
            nn.Linear(d_token + d_trunk, d_token), nn.ReLU(),
            nn.Linear(d_token, 1))
        # Candidate-subset play policy: score each enumerated candidate play from
        # its pooled subset card-embeddings + global context + hand-type + size.
        self.play_candidate_mlp = nn.Sequential(
            nn.Linear(d_token + d_trunk + 12 + 1, d_trunk), nn.ReLU(),
            nn.Linear(d_trunk, 1))

    def _features(self, batch: Batch) -> torch.Tensor:
        joker_tok = torch.cat(
            [self.joker_key(batch.joker_cat[..., 0]),
             self.joker_edition(batch.joker_cat[..., 1]),
             batch.joker_num], dim=-1)
        joker_pool = _masked_mean(self.joker_mlp(joker_tok), batch.joker_mask)

        card_tok = torch.cat(
            [self.card_rank(batch.card_cat[..., 0]),
             self.card_suit(batch.card_cat[..., 1]),
             self.card_enh(batch.card_cat[..., 2]),
             self.card_edition(batch.card_cat[..., 3]),
             self.card_seal(batch.card_cat[..., 4]),
             batch.card_num], dim=-1)
        card_pool = _masked_mean(self.card_mlp(card_tok), batch.card_mask)

        shop_tok = torch.cat(
            [self.shop_type(batch.shop_cat[..., 0]),
             self.shop_key(batch.shop_cat[..., 1]),
             self.shop_edition(batch.shop_cat[..., 2]),
             batch.shop_num], dim=-1)
        shop_pool = _masked_mean(self.shop_mlp(shop_tok), batch.shop_mask)

        consumable_pool = _masked_mean(
            self.consumable_emb(batch.consumable_idx), batch.consumable_mask)
        voucher_pool = _masked_mean(
            self.voucher_emb(batch.voucher_idx), batch.voucher_mask)
        boss_vec = self.boss_emb(batch.boss)

        return torch.cat(
            [batch.scalars, batch.phase, batch.blind, batch.hand_levels,
             batch.deck_counts, joker_pool, card_pool, shop_pool,
             consumable_pool, voucher_pool, boss_vec], dim=-1)

    def _trunk(self, batch: Batch) -> torch.Tensor:
        return self.trunk(self._features(batch))

    def forward(self, batch: Batch) -> torch.Tensor:
        """Win-probability logit (back-compat default head)."""
        return self.win_head(self._trunk(batch)).squeeze(-1)

    def win_prob(self, batch: Batch) -> torch.Tensor:
        return torch.sigmoid(self.forward(batch))

    def ante_value(self, batch: Batch) -> torch.Tensor:
        """Predicted final ante as a fraction of ante 8, in [0, 1]."""
        return torch.sigmoid(self.ante_head(self._trunk(batch))).squeeze(-1)

    def clear_value(self, batch: Batch) -> torch.Tensor:
        """Distilled rollout-leaf value in [0, 2] (matches the leaf convention)."""
        return (2.0 * torch.sigmoid(self.clear_head(self._trunk(batch)))).squeeze(-1)

    def _card_embeddings(self, batch: Batch) -> torch.Tensor:
        """Per-card token embeddings [B, H, d_token] (pre-pool)."""
        card_tok = torch.cat(
            [self.card_rank(batch.card_cat[..., 0]),
             self.card_suit(batch.card_cat[..., 1]),
             self.card_enh(batch.card_cat[..., 2]),
             self.card_edition(batch.card_cat[..., 3]),
             self.card_seal(batch.card_cat[..., 4]),
             batch.card_num], dim=-1)
        return self.card_mlp(card_tok)

    def policy(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Imitation policy: (action_type_logits [B, T], per-card play logits [B, H]).

        `card_logits` scores each hand position for inclusion in a play/discard;
        mask by `batch.card_mask` before use. The trunk supplies global context to
        each per-card score.
        """
        ctx = self._trunk(batch)                       # [B, d_trunk]
        type_logits = self.type_head(ctx)              # [B, T]
        card_emb = self._card_embeddings(batch)        # [B, H, d_token]
        h = card_emb.shape[1]
        per_card = torch.cat(
            [card_emb, ctx.unsqueeze(1).expand(-1, h, -1)], dim=-1)
        card_logits = self.card_policy(per_card).squeeze(-1)  # [B, H]
        return type_logits, card_logits

    def hand_type_logits(self, batch: Batch) -> torch.Tensor:
        """Logits over the 12 poker hands the teacher plays (play steps)."""
        return self.hand_type_head(self._trunk(batch))

    def play_candidate_scores(
        self,
        batch: Batch,
        cand_masks: torch.Tensor,   # [B, C, H] hand positions each candidate plays
        cand_htypes: torch.Tensor,  # [B, C] long poker hand-type index
        cand_sizes: torch.Tensor,   # [B, C] subset size
    ) -> torch.Tensor:
        """Score each candidate play subset; logits [B, C] (softmax over C)."""
        ctx = self._trunk(batch)                          # [B, d_trunk]
        card_emb = self._card_embeddings(batch)           # [B, H, d_token]
        _, c, _ = cand_masks.shape
        m = cand_masks.unsqueeze(-1)                      # [B, C, H, 1]
        pooled = (card_emb.unsqueeze(1) * m).sum(dim=2) / m.sum(dim=2).clamp(min=1.0)
        feat = torch.cat([
            pooled,                                        # [B, C, d_token]
            ctx.unsqueeze(1).expand(-1, c, -1),           # [B, C, d_trunk]
            F.one_hot(cand_htypes, 12).float(),           # [B, C, 12]
            (cand_sizes / 5.0).unsqueeze(-1),             # [B, C, 1]
        ], dim=-1)
        return self.play_candidate_mlp(feat).squeeze(-1)  # [B, C]

    def heads(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """(win_logit, ante_value) from a single shared trunk pass."""
        t = self._trunk(batch)
        return self.win_head(t).squeeze(-1), torch.sigmoid(self.ante_head(t)).squeeze(-1)
