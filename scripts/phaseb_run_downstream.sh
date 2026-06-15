#!/usr/bin/env bash
# Downstream Phase-B iteration-1 pipeline (run AFTER B0-rich + fork labels exist).
#   1. V0-rich value salvage   (rich-schema value head = AWR baseline)
#   2. On-policy generation     (from B0-rich, gated exploration + 20% recipes)
#   3. Full-fidelity iteration 1 + powered 1024-seed gate vs B0-rich
#
# Seed ranges are disjoint: mixture 5.10M, on-policy 5.20M, gate 5.30M.
set -euo pipefail
cd "$(dirname "$0")/.."

export BALATRO_NO_FORESIGHT=shuffle
export BALATRO_EXPAND_JOBS=12
export PYTHONPATH=src

B0=.data/phaseb_policy_b0rich.pt
V0=.data/phaseb_policy_v0rich.pt
FORK=.data/phaseb_forkaudit_labels.pkl
ONPOLICY=.data/phaseb_onpolicy_1500.jsonl
ITER=.data/phaseb_policy_iter1full.pt
RESULT=.data/phaseb_iter1full_result.json

echo "=========================================================="
echo "[downstream] STEP 1/3 — V0-rich value salvage"
echo "=========================================================="
python scripts/phaseb_value_salvage.py \
    --dataset .data/phaseb_mix_1000.jsonl --heldout .data/phaseb_heldout_200.jsonl \
    --epochs 25 --weight-decay 1e-3 --dropout 0.2 --ckpt "$V0"

echo "=========================================================="
echo "[downstream] STEP 2/3 — on-policy generation from B0-rich (1500 episodes)"
echo "=========================================================="
python scripts/phaseb_onpolicy_gen.py \
    --ckpt "$B0" --n 1500 --seed-offset 5200000 \
    --recipe-frac 0.2 --margin 0.2 --temp 1.0 --topk 4 --jobs 12 \
    --out "$ONPOLICY"

echo "=========================================================="
echo "[downstream] STEP 3/3 — full-fidelity iteration 1 + powered gate"
echo "=========================================================="
python scripts/phaseb_iter1_full.py \
    --mixture .data/phaseb_mix_1000.jsonl --onpolicy "$ONPOLICY" \
    --forkaudit "$FORK" --fork-weight 5.0 \
    --heldout .data/phaseb_heldout_200.jsonl --baseline-value "$V0" \
    --b0 "$B0" --beta 2.0 --w-max 5.0 --per-policy-blend 0.5 \
    --epochs 15 --eval-seeds 1024 --eval-offset 5300000 --jobs 12 \
    --out "$ITER" --result "$RESULT"

echo "=========================================================="
echo "[downstream] PIPELINE COMPLETE — result: $RESULT"
echo "=========================================================="
