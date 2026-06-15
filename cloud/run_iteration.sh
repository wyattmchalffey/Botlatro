#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Full plan-faithful Phase-B iteration on one cloud box:
#   1. generate the iteration-0 mixture (deployed 55% / recipe 35% / perturb 10%)
#   2. generate a held-out set (value early-stop + gates)
#   3. B0  — BC on the mixture + plumbing gate
#   4. V0  — value-head salvage (AWR baseline)
#   5. fork-audit dense play labels (search-improved targets at death blinds)
#   6. on-policy generation from B0 (gated exploration + 20% recipes)
#   7. iteration 1 — per-policy AWR on mixture+on-policy+fork labels
#                    + POWERED paired gate vs B0
#
# Parameterized by --n-runs so you can validate the WHOLE chain cheaply
# (--n-runs 2000, ~30 min, a few cents) before committing the full 50k.
#
#   bash cloud/run_iteration.sh --n-runs 50000 --eval-seeds 2048
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

N_RUNS=50000
HELDOUT=2000
ONPOLICY=15000
EVAL_SEEDS=2048
JOBS="$(nproc 2>/dev/null || echo 8)"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --n-runs) N_RUNS="$2"; shift 2;;
    --heldout) HELDOUT="$2"; shift 2;;
    --onpolicy) ONPOLICY="$2"; shift 2;;
    --eval-seeds) EVAL_SEEDS="$2"; shift 2;;
    --jobs) JOBS="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

export BALATRO_NO_FORESIGHT=shuffle
export BALATRO_EXPAND_JOBS="$JOBS"
export PYTHONPATH=src
D=.data
mkdir -p "$D"
echo "[run] n_runs=$N_RUNS heldout=$HELDOUT onpolicy=$ONPOLICY eval_seeds=$EVAL_SEEDS jobs=$JOBS"
echo "[run] seed map: mix 5.10M+, heldout 5.50M+, onpolicy 5.20M+, gate 5.30M+ (disjoint)"

step() { echo; echo "===== $* ====="; }

step "1/7 generate iteration-0 mixture ($N_RUNS runs)"
python cloud/gen_mixture.py --n-runs "$N_RUNS" --seed-offset 5100000 \
  --shards 1 --shard-id 0 --jobs "$JOBS" --out-dir "$D/cloud_mix"
cat "$D"/cloud_mix/shard*.jsonl > "$D/cloud_mix.jsonl"
echo "[run] mixture: $(wc -l < "$D/cloud_mix.jsonl") runs"

step "2/7 generate held-out set ($HELDOUT runs)"
python cloud/gen_mixture.py --n-runs "$HELDOUT" --seed-offset 5500000 \
  --shards 1 --shard-id 0 --jobs "$JOBS" --out-dir "$D/cloud_heldout"
cat "$D"/cloud_heldout/shard*.jsonl > "$D/cloud_heldout.jsonl"

step "3/7 B0 — BC + plumbing gate"
python scripts/phaseb_gate_b0.py --dataset "$D/cloud_mix.jsonl" \
  --epochs 20 --eval-seeds 512 --eval-offset 5100000 --workers "$JOBS" \
  --ckpt "$D/cloud_b0.pt"

step "4/7 V0 — value-head salvage"
python scripts/phaseb_value_salvage.py --dataset "$D/cloud_mix.jsonl" \
  --heldout "$D/cloud_heldout.jsonl" --epochs 25 --ckpt "$D/cloud_v0.pt"

step "5/7 fork-audit dense play labels"
python scripts/phaseb_forkaudit_labels.py --caps "$D/cloud_mix.jsonl" \
  --max-losses 100000 --depth 6 --width 6 --jobs "$JOBS" \
  --out "$D/cloud_forkaudit.pkl"

step "6/7 on-policy generation from B0 ($ONPOLICY episodes)"
python scripts/phaseb_onpolicy_gen.py --ckpt "$D/cloud_b0.pt" \
  --n "$ONPOLICY" --seed-offset 5200000 --recipe-frac 0.2 \
  --margin 0.2 --temp 1.0 --topk 4 --jobs "$JOBS" \
  --out "$D/cloud_onpolicy.jsonl"

step "7/7 iteration 1 — AWR + powered $EVAL_SEEDS-seed gate"
python scripts/phaseb_iter1_full.py \
  --mixture "$D/cloud_mix.jsonl" --onpolicy "$D/cloud_onpolicy.jsonl" \
  --forkaudit "$D/cloud_forkaudit.pkl" --fork-weight 5.0 \
  --heldout "$D/cloud_heldout.jsonl" --baseline-value "$D/cloud_v0.pt" \
  --b0 "$D/cloud_b0.pt" --beta 2.0 --w-max 5.0 --per-policy-blend 0.5 \
  --epochs 15 --eval-seeds "$EVAL_SEEDS" --eval-offset 5300000 --jobs "$JOBS" \
  --out "$D/cloud_iter1.pt" --result "$D/cloud_iter1_result.json"

echo
echo "===== ITERATION COMPLETE ====="
echo "[run] verdict -> $D/cloud_iter1_result.json"
cat "$D/cloud_iter1_result.json"
echo
echo "[run] retrieve these: $D/cloud_iter1_result.json $D/cloud_b0.pt $D/cloud_v0.pt $D/cloud_iter1.pt"
