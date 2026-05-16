"""Deterministic seed-set generation for fair benchmarks."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path


CANONICAL_200_SEED_COUNT = 200
CANONICAL_200_SOURCE_LABEL = "white:basic-strategy-2026-04-30-1000"
CANONICAL_200_SOURCE_SIZE = 1000
CANONICAL_200_SEED_SET_LABEL = f"{CANONICAL_200_SOURCE_LABEL}:first200"
CANONICAL_200_WINDOW_COUNT = CANONICAL_200_SOURCE_SIZE // CANONICAL_200_SEED_COUNT


@dataclass(frozen=True, slots=True)
class SeedSet:
    label: str
    seeds: tuple[int, ...]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"label": self.label, "seeds": list(self.seeds)}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SeedSet":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(label=str(payload["label"]), seeds=tuple(int(seed) for seed in payload["seeds"]))


def make_seed_set(label: str, size: int) -> SeedSet:
    """Return a deterministic seed set.

    The 200-seed benchmark size is pinned globally so smoke labels cannot
    accidentally create incomparable 200-run baselines. Use
    ``make_explicit_seed_set``/``--seed-list`` for a deliberate custom set.
    """

    return make_benchmark_seed_set(label=label, size=size)


def make_benchmark_seed_set(label: str, size: int, *, seed_window: int = 1) -> SeedSet:
    if size == CANONICAL_200_SEED_COUNT:
        return make_canonical_200_seed_set(seed_window)
    if seed_window != 1:
        raise ValueError("Seed windows are only available for 200-seed benchmarks")
    return _make_hashed_seed_set(label, size)


def make_canonical_200_seed_set(seed_window: int = 1) -> SeedSet:
    if seed_window < 1 or seed_window > CANONICAL_200_WINDOW_COUNT:
        raise ValueError(
            f"Canonical 200-seed window must be between 1 and {CANONICAL_200_WINDOW_COUNT}"
        )
    source = _make_hashed_seed_set(CANONICAL_200_SOURCE_LABEL, CANONICAL_200_SOURCE_SIZE)
    start = (seed_window - 1) * CANONICAL_200_SEED_COUNT
    end = start + CANONICAL_200_SEED_COUNT
    label = (
        CANONICAL_200_SEED_SET_LABEL
        if seed_window == 1
        else f"{CANONICAL_200_SOURCE_LABEL}:{start + 1}-{end}"
    )
    return SeedSet(
        label=label,
        seeds=source.seeds[start:end],
    )


def _make_hashed_seed_set(label: str, size: int) -> SeedSet:
    if size < 0:
        raise ValueError("Seed set size must be non-negative")

    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    seeds: list[int] = []
    seen: set[int] = set()

    while len(seeds) < size:
        seed = rng.randrange(1, 2_147_483_647)
        if seed not in seen:
            seen.add(seed)
            seeds.append(seed)

    return SeedSet(label=label, seeds=tuple(seeds))


def make_explicit_seed_set(label: str, seeds: tuple[int, ...]) -> SeedSet:
    if not seeds:
        raise ValueError("Explicit seed set must contain at least one seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Explicit seed set cannot contain duplicates")
    return SeedSet(label=label, seeds=seeds)


def parse_seed_values(raw: str) -> tuple[int, ...]:
    """Parse comma, space, or newline separated integer seeds."""

    text = raw.strip()
    if not text:
        return ()

    seeds: list[int] = []
    for token in re.split(r"[\s,;]+", text):
        if not token:
            continue
        try:
            seed = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid seed value: {token}") from exc
        if seed < 0:
            raise ValueError("Seed values must be non-negative integers")
        seeds.append(seed)
    return tuple(seeds)
