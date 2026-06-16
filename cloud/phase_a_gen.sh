#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# PHASE A (CPU, runs on the cheap/free generation box, e.g. the GCP free VM):
# everything the GPU box doesn't need a GPU for and that DOESN'T depend on a
# trained policy — mixture + held-out generation, fork-audit search labels, and
# expansion to disk shards. Output (shards + jsonl + fork pkl) is then handed to
# the GPU box for Phase B (training). On-policy generation is NOT here — it
# needs the trained B0, so it lives in Phase B on the GPU box's vCPUs.
#
# Resumable: gen_mixture skips seeds already captured; expand reuses its manifest.
#
#   bash cloud/phase_a_gen.sh 50000
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."
# self-contained: activate the venv built by bootstrap.sh
VENV="${BOTLATRO_VENV:-$HOME/botlatro-venv}"
# shellcheck disable=SC1091
[[ -f "$VENV/bin/activate" ]] && source "$VENV/bin/activate"
N_RUNS="${1:-50000}"
HELDOUT="${2:-2000}"
JOBS="$(nproc 2>/dev/null || echo 8)"

export BALATRO_NO_FORESIGHT=shuffle
export BALATRO_EXPAND_JOBS="$JOBS"
export PYTHONPATH=src
D=.data
echo "[phaseA] n_runs=$N_RUNS heldout=$HELDOUT jobs=$JOBS  (CPU generation box)"

echo "[phaseA] 1/4 mixture generation ($N_RUNS, resumes from any existing)"
python cloud/gen_mixture.py --n-runs "$N_RUNS" --seed-offset 5100000 \
  --shards 1 --shard-id 0 --jobs "$JOBS" --out-dir "$D/cloud_mix"
cat "$D"/cloud_mix/shard*.jsonl > "$D/cloud_mix.jsonl"
echo "[phaseA] mixture: $(wc -l < "$D/cloud_mix.jsonl") runs"

echo "[phaseA] 2/4 held-out generation ($HELDOUT)"
python cloud/gen_mixture.py --n-runs "$HELDOUT" --seed-offset 5500000 \
  --shards 1 --shard-id 0 --jobs "$JOBS" --out-dir "$D/cloud_heldout"
cat "$D"/cloud_heldout/shard*.jsonl > "$D/cloud_heldout.jsonl"

echo "[phaseA] 3/4 fork-audit dense play labels"
python scripts/phaseb_forkaudit_labels.py --caps "$D/cloud_mix.jsonl" \
  --max-losses 5000 --depth 6 --width 6 --jobs "$JOBS" \
  --out "$D/cloud_forkaudit.pkl"

echo "[phaseA] 4/4 expand mixture + held-out to disk shards (out-of-core)"
python -c "from balatro_ai.ml.dataset import expand_to_shards; expand_to_shards('$D/cloud_mix.jsonl','$D/shards/mix',runs_per_shard=500,jobs=$JOBS)"
python -c "from balatro_ai.ml.dataset import load_or_expand_examples; load_or_expand_examples('$D/cloud_heldout.jsonl',jobs=$JOBS)"

echo "[phaseA] === COMPLETE ==="
du -sh "$D/shards/mix" "$D/cloud_forkaudit.pkl" "$D/cloud_mix.jsonl" 2>/dev/null
echo "[phaseA] hand these to the GPU box: $D/shards/mix  $D/cloud_mix.jsonl  $D/cloud_heldout.jsonl  $D/cloud_forkaudit.pkl"
