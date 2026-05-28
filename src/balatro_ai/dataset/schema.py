"""Schema for one row of the dataset JSONL.

`SeedResult` is the per-seed payload. One JSONL row per `SeedResult`,
written atomically (fsync per row) by `JsonlSeedWriter`, parsed back
by `JsonlSeedReader` for resume + analysis.

Schema notes:
- `seed` is the canonical run identifier (string). Numeric seeds are
  stored as the same string representation passed in by the user, so
  resume-deduplication is a simple string match.
- `attempts` is populated only when the policy was multi-archetype
  (lists per-archetype outcomes); otherwise it's an empty list.
- `steps` is populated only when `record_steps=True` was requested at
  generation; otherwise it's an empty list. Per-trajectory size cost
  is significant (~50KB per 100-step trajectory) so we don't always
  emit it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True, slots=True)
class StepRow:
    """One trajectory step. Mirrors `solver.trajectory.StepRecord` 1:1
    so we don't lose information when serializing."""

    step: int
    phase_before: str
    action_type: str
    card_indices: tuple[int, ...]
    money_before: int
    money_after: int
    score_after: int
    ante_after: int
    hands_after: int
    discards_after: int


@dataclass(frozen=True, slots=True)
class ArchetypeAttemptRow:
    """One archetype attempt's outcome inside a multi-archetype solve."""

    archetype_name: str
    won: bool
    final_ante: int
    final_score: int
    final_money: int
    n_steps: int
    wall_seconds: float
    terminated_reason: str


@dataclass(frozen=True, slots=True)
class SeedResult:
    """One row of the dataset JSONL.

    Fields are flat-ish so the file round-trips cleanly through
    `json.dumps` / `json.loads`. Nested attempts/steps lists become
    arrays of dicts.
    """

    seed: str
    stake: str
    policy: str  # e.g. "v2-d3w2-planning", "multi-archetype-v2"
    won: bool
    final_ante: int
    final_score: int
    final_money: int
    n_steps: int
    wall_seconds: float
    terminated_reason: str
    best_archetype: str = ""
    attempts: tuple[ArchetypeAttemptRow, ...] = field(default_factory=tuple)
    steps: tuple[StepRow, ...] = field(default_factory=tuple)
    # When the worker hits an uncaught exception we still want to
    # record the row (for diagnosis) — these fields capture the
    # failure mode. Empty when the seed completed normally.
    error_type: str = ""
    error_message: str = ""

    def to_json_dict(self) -> dict:
        """Serializable form. Tuples become lists; nested dataclasses
        get the same treatment recursively."""

        d = asdict(self)
        # asdict already recurses into nested dataclasses + tuples.
        return d

    @classmethod
    def from_json_dict(cls, data: dict) -> SeedResult:
        """Re-hydrate from `to_json_dict` output."""

        attempts = tuple(
            ArchetypeAttemptRow(**a) for a in data.get("attempts", ())
        )
        steps = tuple(
            StepRow(
                step=s["step"],
                phase_before=s["phase_before"],
                action_type=s["action_type"],
                card_indices=tuple(s["card_indices"]),
                money_before=s["money_before"],
                money_after=s["money_after"],
                score_after=s["score_after"],
                ante_after=s["ante_after"],
                hands_after=s["hands_after"],
                discards_after=s["discards_after"],
            )
            for s in data.get("steps", ())
        )
        return cls(
            seed=str(data["seed"]),
            stake=data.get("stake", "white"),
            policy=data.get("policy", ""),
            won=bool(data.get("won", False)),
            final_ante=int(data.get("final_ante", 0)),
            final_score=int(data.get("final_score", 0)),
            final_money=int(data.get("final_money", 0)),
            n_steps=int(data.get("n_steps", 0)),
            wall_seconds=float(data.get("wall_seconds", 0.0)),
            terminated_reason=str(data.get("terminated_reason", "")),
            best_archetype=str(data.get("best_archetype", "")),
            attempts=attempts,
            steps=steps,
            error_type=str(data.get("error_type", "")),
            error_message=str(data.get("error_message", "")),
        )
