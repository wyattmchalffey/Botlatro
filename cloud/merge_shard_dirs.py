#!/usr/bin/env python3
"""Merge a secondary expand_to_shards dir (e.g. shards/mix_b from a parallel
generation box) INTO a primary shard dir (shards/mix), so the trainer streams
the UNION as one store with a single valid manifest — no re-expansion, no
training-script changes.

Why this works: expand_to_shards()'s manifest hit keys ONLY on the JSONL's
mtime (not its content) plus the encoding/cache versions. So we keep the
primary manifest's dataset_mtime UNCHANGED and just append the secondary's
shard pickles (renamed to avoid filename collisions) and example counts. On the
GPU box, expand_to_shards(primary_jsonl, shards/mix) then sees a manifest hit
against the (rsync -t, mtime-preserved) primary JSONL and the store reads every
shard, part-A and part-B alike.

CRITICAL at handoff: the primary JSONL on the GPU box MUST keep the mtime
recorded in the primary manifest's dataset_mtime (use `rsync -t`, or
`touch -d @<dataset_mtime>` it). A mtime MISS would trigger a full re-expansion
that overwrites shardNNNNN.pkl and rewrites the manifest, silently dropping the
part-B shards.

    python cloud/merge_shard_dirs.py .data/shards/mix .data/shards/mix_b
"""
from __future__ import annotations

import json
import os
import shutil
import sys


def merge(primary: str, secondary: str) -> dict:
    pman_path = os.path.join(primary, "manifest.json")
    sman_path = os.path.join(secondary, "manifest.json")
    with open(pman_path, encoding="utf-8") as fh:
        P = json.load(fh)
    with open(sman_path, encoding="utf-8") as fh:
        S = json.load(fh)

    for k in ("encoding_version", "examples_cache_version"):
        if P.get(k) != S.get(k):
            raise SystemExit(
                f"[merge] ABORT: {k} mismatch ({P.get(k)} != {S.get(k)}); "
                "the two shard sets were expanded with incompatible code."
            )

    existing = set(P["shards"])
    moved: list[str] = []
    j = 0
    for s in S["shards"]:
        new = f"shardB{j:05d}.pkl"
        while new in existing:
            j += 1
            new = f"shardB{j:05d}.pkl"
        shutil.move(os.path.join(secondary, s), os.path.join(primary, new))
        existing.add(new)
        moved.append(new)
        j += 1

    P["shards"] += moved
    P["n_examples"] = int(P["n_examples"]) + int(S["n_examples"])
    P["n_runs"] = int(P.get("n_runs", 0)) + int(S.get("n_runs", 0))
    # dataset_mtime DELIBERATELY UNCHANGED — must keep matching the primary JSONL.

    tmp = pman_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(P, fh)
    os.replace(tmp, pman_path)
    print(
        f"[merge] +{len(moved)} shards (+{S['n_examples']} ex) -> "
        f"{P['n_examples']} examples in {len(P['shards'])} shards / "
        f"{P['n_runs']} runs; dataset_mtime preserved ({P['dataset_mtime']})"
    )
    return P


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python cloud/merge_shard_dirs.py <primary_dir> <secondary_dir>")
    merge(sys.argv[1], sys.argv[2])
