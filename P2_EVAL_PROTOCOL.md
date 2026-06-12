# P2 Evaluation Protocol — seed discipline + iteration gates

*2026-06-12. Locked BEFORE the first self-play iteration, per the pre-flight
review. Self-play loops excel at producing self-confirming metrics; this
protocol is the antidote, and it only works if it is never bent mid-program.*

## Information regime

All training data generation, all evaluation: **BALATRO_NO_FORESIGHT=shuffle**
(honest multiset). The encoder is proven order-blind (231-state probe, 0
order-sensitive encodings) and the capture path routes through the blinded
registry wrappers. Hide-mode and clairvoyant numbers are never comparable to
these and must be labeled as such wherever produced.

## Seed-range ledger

Numeric seeds are `f"{i:07d}"`. Ranges already consumed (tuning, gates,
diagnostics) are BURNED for evaluation purposes — winrates measured on them
are biased by selection/tuning history.

| Range | Status | Used by |
|---|---|---|
| 1 – 1,000 | BURNED (dev) | knob tuning, parity probes, original benches, S0 |
| 1,001 – 2,000 | BURNED | honest-baseline certification (12.4%) |
| 2,001 – 4,512 | BURNED | shop-knob gates 1-3 |
| 6,001 – 7,512 | BURNED | P1.5 gate, honest-draw-probs gate |
| 8,001 – 9,010 | BURNED | harness smokes |
| 5,000,001 – 5,000,300 | BURNED | honest caps: diagnostics, route oracle, ceiling |
| **10,001 – 12,000** | **RESERVED: ITERATION HOLDOUT** | per-iteration paired gates ONLY |
| **20,001 – 22,000** | **RESERVED: FINAL CERTIFICATION** | touched ONCE, at P3 |
| 5,100,000 + | TRAINING | self-play data generation (never evaluated on) |

Rules:
- Training seeds and evaluation seeds are disjoint BY RANGE, not by bookkeeping.
- Nothing tunes on 10,001+. If a knob is ever selected using holdout results,
  the holdout is reburned and the reserve moves (log it here).
- The final-certification range is opened once, for the headline claim, after
  the program declares itself done. No peeking, no reruns.

## Baseline

Frozen comparator: **honest 12.4%** (1,000-seed certification, seeds
1,001-2,000, `.data/no_foresight_paired_1000_holdout.log`), replicated at
12.3% (5M range) and 12.3-13.9% across gate-control arms. Every iteration
gate runs CONTROL = current deployed bot, TREAT = candidate policy, paired
on the same holdout slice.

## Per-iteration gate (pre-registered)

- 512 paired seeds minimum from the iteration holdout (rotating
  non-overlapping slices; the range supports ~4 disjoint 512-slices —
  reuse slices only after a full rotation).
- Adopt a new policy iff McNemar exact p < 0.05 in favor AND d_winrate > 0.
- KILL the program review: two consecutive iterations with no significant
  improvement triggers a design review (not silent continuation).
- Report wins/losses/CI per `env_paired_ab` conventions; persist per-pair
  rows (resumable harness) for flip diagnostics.

## Transfer checks

- Once per program phase: a live-bridge spot-check (~30-50 runs) of the
  current deployed policy, to bound the sim->real gap before it compounds.
- Late-ante RNG certification (P0.4 bridge session, transition-class audit)
  is a PREREQUISITE for trusting training data generated past ante 5.
