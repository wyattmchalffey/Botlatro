#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# PHASE B (GPU box, e.g. a RunPod L4 pod): everything that wants the GPU, plus
# the policy-DEPENDENT CPU steps (on-policy generation needs the trained B0, so
# it runs here on this box's vCPUs). Consumes Phase-A output (shards + jsonl +
# fork labels), produces the powered iteration gate.
#
# Training runs on the GPU (BALATRO_DEVICE=cuda); generation/expansion/benches
# use the box's vCPUs. Epochs are tuned DOWN vs the 1k-run defaults — at 50k the
# dataset is ~6M examples, so few passes suffice (held-out early-stop keeps the
# best checkpoint regardless).
#
#   bash cloud/phase_b_train.sh            # full: 2048-seed gate, 15k on-policy
#   bash cloud/phase_b_train.sh 512 2000   # quick validation pass
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${BOTLATRO_VENV:-$HOME/botlatro-venv}"
# shellcheck disable=SC1091
[[ -f "$VENV/bin/activate" ]] && source "$VENV/bin/activate"

EVAL_SEEDS="${1:-2048}"
ONPOLICY="${2:-15000}"
JOBS="$(nproc 2>/dev/null || echo 8)"
B0_EP="${B0_EP:-10}"; V0_EP="${V0_EP:-12}"; ITER_EP="${ITER_EP:-10}"

export BALATRO_NO_FORESIGHT=shuffle
export BALATRO_EXPAND_JOBS="$JOBS"
export PYTHONPATH=src
# Reproducible eval gate: stable hash ordering + single-thread torch per worker
# (see _eval_seed) — kills the multi-threaded FP nondeterminism that flipped a
# FIXED policy's win count run-to-run (38 vs 42 / 1024). Single-process back-to-
# back is reproducible (0/24), so any residual is in the parallel-eval layer, NOT
# the sim (default sim seeds its RNG per game-seed and is deterministic).
export PYTHONHASHSEED=0
export BALATRO_DEVICE="${BALATRO_DEVICE:-cuda}"   # GPU training; falls back to cpu if no CUDA
# Parallel collation so the GPU never starves on single-threaded data prep
# (measured: collate is ~3x the compute on CPU -> ~97% GPU idle without this).
export BALATRO_LOADER_WORKERS="${BALATRO_LOADER_WORKERS:-$(( JOBS > 6 ? JOBS - 2 : 4 ))}"
D=.data
SHARDS="$D/shards"
echo "[phaseB] device=$BALATRO_DEVICE jobs=$JOBS eval_seeds=$EVAL_SEEDS onpolicy=$ONPOLICY epochs=B0:$B0_EP/V0:$V0_EP/iter1:$ITER_EP"
python -c "import torch;print('[phaseB] torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

step() { echo; echo "===== $* ====="; }

step "1/4 B0 — BC (GPU) + plumbing gate (reuses Phase-A mix shards)"
python scripts/phaseb_gate_b0.py --dataset "$D/cloud_mix.jsonl" \
  --shard-dir "$SHARDS/mix" --runs-per-shard 500 \
  --epochs "$B0_EP" --eval-seeds 512 --eval-offset 5100000 --workers "$JOBS" \
  --ckpt "$D/cloud_b0.pt"

step "2/4 V0 — value-head salvage (GPU, reuses mix shards)"
python scripts/phaseb_value_salvage.py --dataset "$D/cloud_mix.jsonl" \
  --shard-dir "$SHARDS/mix" --heldout "$D/cloud_heldout.jsonl" --epochs "$V0_EP" \
  --ckpt "$D/cloud_v0.pt"

step "3/4 on-policy generation from B0 ($ONPOLICY, CPU on this box)"
python scripts/phaseb_onpolicy_gen.py --ckpt "$D/cloud_b0.pt" \
  --n "$ONPOLICY" --seed-offset 5200000 --recipe-frac 0.2 \
  --margin 0.2 --temp 1.0 --topk 4 --jobs "$JOBS" \
  --out "$D/cloud_onpolicy.jsonl"

step "4/4 iteration 1 — AWR (GPU) + powered $EVAL_SEEDS-seed gate"
python scripts/phaseb_iter1_full.py \
  --mixture "$D/cloud_mix.jsonl" --onpolicy "$D/cloud_onpolicy.jsonl" \
  --shard-dir "$SHARDS" --runs-per-shard 500 \
  --forkaudit "$D/cloud_forkaudit.pkl" --fork-weight 5.0 \
  --heldout "$D/cloud_heldout.jsonl" --baseline-value "$D/cloud_v0.pt" \
  --b0 "$D/cloud_b0.pt" --beta 2.0 --w-max 5.0 --per-policy-blend 0.5 \
  --epochs "$ITER_EP" --eval-seeds "$EVAL_SEEDS" --eval-offset 5300000 --jobs "$JOBS" \
  --out "$D/cloud_iter1.pt" --result "$D/cloud_iter1_result.json"

echo
echo "===== PHASE B COMPLETE — VERDICT ====="
cat "$D/cloud_iter1_result.json"
echo
echo "[phaseB] retrieve: $D/cloud_iter1_result.json $D/cloud_b0.pt $D/cloud_v0.pt $D/cloud_iter1.pt"
