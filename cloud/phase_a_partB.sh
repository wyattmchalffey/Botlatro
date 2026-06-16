#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# PHASE A — PART B (a SECOND parallel generation box): generate a DISJOINT
# mixture seed range to extend an in-progress run toward the plan-scale target,
# then expand it to its OWN shard dir. This is how we reach 50k without waiting:
# box 1 keeps expanding its part-A 35k while this box generates the missing tail
# on its own 16 vCPUs (genuine parallelism — a single box can't speed this up,
# total CPU-seconds is fixed).
#
# Output (cloud_mix_b.jsonl + shards/mix_b) MERGES into the primary box's
# shards/mix at GPU handoff via cloud/merge_shard_dirs.py — the trainer then
# streams the union as one store. Only mixture+shards are produced here; the
# primary box owns held-out + fork-audit (those don't need the part-B tail).
#
# Seeds: --seed-offset O --n-runs N -> seeds O+1 .. O+N. Choose O = primary's
# MAX seed so the ranges are strictly disjoint (per-seed kind is deterministic,
# so the union's mixture composition is identical to one-box generation).
#
#   bash cloud/phase_a_partB.sh 5135713 14304
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${BOTLATRO_VENV:-$HOME/botlatro-venv}"
# shellcheck disable=SC1091
[[ -f "$VENV/bin/activate" ]] && source "$VENV/bin/activate"
SEED_OFFSET="${1:?usage: phase_a_partB.sh <seed-offset> <n-runs>}"
N_RUNS="${2:?usage: phase_a_partB.sh <seed-offset> <n-runs>}"
JOBS="$(nproc 2>/dev/null || echo 8)"

export BALATRO_NO_FORESIGHT=shuffle
export BALATRO_EXPAND_JOBS="$JOBS"
export PYTHONPATH=src
D=.data
echo "[partB] seeds $((SEED_OFFSET + 1))..$((SEED_OFFSET + N_RUNS))  jobs=$JOBS  (parallel generation box)"

echo "[partB] 1/2 mixture generation (n=$N_RUNS, resumes from any existing)"
python cloud/gen_mixture.py --n-runs "$N_RUNS" --seed-offset "$SEED_OFFSET" \
  --shards 1 --shard-id 0 --jobs "$JOBS" --out-dir "$D/cloud_mix_b"
cat "$D"/cloud_mix_b/shard*.jsonl > "$D/cloud_mix_b.jsonl"
echo "[partB] generated $(wc -l < "$D/cloud_mix_b.jsonl") runs"

echo "[partB] 2/2 expand part-B to its own disk shards (out-of-core)"
python -c "from balatro_ai.ml.dataset import expand_to_shards; expand_to_shards('$D/cloud_mix_b.jsonl','$D/shards/mix_b',runs_per_shard=500,jobs=$JOBS)"

echo "[partB] === COMPLETE ==="
du -sh "$D/shards/mix_b" "$D/cloud_mix_b.jsonl" 2>/dev/null
echo "[partB] hand to GPU box / merge target: $D/shards/mix_b  $D/cloud_mix_b.jsonl"
