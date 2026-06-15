# Cloud compute setup — Phase B at plan scale

Run the plan-faithful Phase-B iteration (~50k runs, ~12% mixture) on a rented
CPU box. This restores the data scale my local proxy under-ran by ~50× — the
unremoved confound behind the local iteration-1 null.

**Why CPU, not GPU:** the bottleneck is the Balatro forward *simulator*
(Python+Rust, CPU-only). The net is tiny (0.97M params; ~1–2 GPU-hr for the
whole dataset). Rent **cores, not CUDA**. See the cost model below.

---

## Prerequisite

The Phase-B + cloud work is pushed to the **`phaseb-cloud-pipeline`** branch, so
the box clones *that branch* (not `main`). Do **not** transfer `.data/` (8.6 GB,
regenerated on the box).

---

## Quick start

```bash
# on a fresh Ubuntu 22.04/24.04 box:
git clone -b phaseb-cloud-pipeline https://github.com/wyattmchalffey/Botlatro && cd Botlatro
bash cloud/bootstrap.sh                 # ~5–10 min: rust+maturin, cython, cpu-torch, smoke test
source ~/botlatro-venv/bin/activate

# validate the WHOLE chain cheaply first (~30 min, a few cents):
bash cloud/run_iteration.sh --n-runs 2000 --heldout 400 --onpolicy 800 --eval-seeds 512

# then the real thing:
bash cloud/run_iteration.sh --n-runs 50000 --eval-seeds 2048
```

Always run the **2k smoke first** — it exercises every stage (gen → expand →
B0 → V0 → fork labels → on-policy → iter1 → gate) at toy size, so a bug or a
bad bootstrap fails for cents, not for the full bill.

---

## Recommended box

| Tier | Pick | Why |
|---|---|---|
| **Safe default** | **Hetzner CCX53/CCX63** (48 dedicated vCPU, ~$0.64/hr) | No spot preemption on multi-hour jobs; 3× cheaper than hyperscaler on-demand. **`lscpu` after boot — want Genoa, avoid Milan** (Milan is ~7% slower than your desktop per-core). |
| **Cheapest** | **GCP `c3-highcpu-44` spot** (~$0.38/hr) | ~80% off, but preemptible — fine here because `gen_mixture.py` checkpoints per-shard and resumes. |
| **AWS-native** | **c7a.16xlarge spot** (Zen 4, ~$1.2/hr) | Fastest per-core in the survey. |

The Cython/Rust builds want `build-essential` — `bootstrap.sh` installs it.

---

## Cost & wall-clock (one 50k iteration)

| | Cost | Wall-clock (1 box) |
|---|---|---|
| GCP c3 spot, CPU-only | ~$6–7 | ~12–18 h |
| Hetzner CCX on-demand | ~$9–11 | ~12–18 h |

Generation is ~55% deployed-solver (the slow, strong bot), so the plan-faithful
mix is ~1.6× costlier to *generate* than a recipe-only mix — already priced in
above. A **5-iteration program is still only ~$30–55 total.**

**Cut wall-clock with sharding** (gen is embarrassingly parallel):

```bash
# on each of K nodes (i = 0..K-1), same seed range, disjoint shards:
python cloud/gen_mixture.py --n-runs 50000 --shards K --shard-id i --out-dir mix_out
# then rsync every node's mix_out/shard*.jsonl to ONE box and:
cat mix_out/shard*.jsonl > .data/cloud_mix.jsonl
# run steps 2–7 of run_iteration.sh on that box (train is the only serial part)
```

4 nodes → 50k generated in **~3–4 h** at the same total $.

---

## Retrieve results

Only pull the small artifacts (the dataset stays on the box):

```bash
scp box:~/Botlatro/.data/cloud_iter1_result.json .   # the verdict
scp box:~/Botlatro/.data/cloud_{b0,v0,iter1}.pt .     # checkpoints
```

`cloud_iter1_result.json` has the powered d_winrate CI + McNemar p + mean-ante —
the answer to "does the AWR iteration climb at real data scale?"

---

## What changed vs the local proxy

| | Local proxy | Cloud (plan-faithful) |
|---|---|---|
| Mixture | 1,000 runs, recipe-heavy, **7.6%** | 50,000 runs, **55% deployed**, ~12% |
| Winning trajectories | ~110 | **~5,000–6,000** (the AWR signal) |
| On-policy | 1,500 | 15,000 |
| Gate | 2,048 seeds ✓ | 2,048 seeds ✓ |

If iteration-1 is *still* flat at this scale, the null is real and the pivot to
value-in-search is well-earned. If it climbs, the local null was the data-scale
artifact it looked like.
