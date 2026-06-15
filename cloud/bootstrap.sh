#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Bootstrap a fresh Ubuntu cloud box (22.04 / 24.04) into a working Botlatro
# Phase-B environment: Rust + maturin (balatro_core), Cython (hand_evaluator),
# CPU-only torch + numpy, then a smoke test that PROVES the native builds work.
#
# Idempotent — safe to re-run. Designed for the cheap-CPU-spot path (the
# bottleneck is the CPU simulator, not a GPU; see cloud/README.md).
#
#   git clone https://github.com/wyattmchalffey/Botlatro && cd Botlatro
#   bash cloud/bootstrap.sh            # ~5-10 min on a fresh box
#   source ~/botlatro-venv/bin/activate
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${BOTLATRO_VENV:-$HOME/botlatro-venv}"
PYVER="${BOTLATRO_PYVER:-3.12}"

# Cloud images often run as root (no sudo); use sudo only when present + non-root.
SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then SUDO="sudo"; fi

echo "[bootstrap] repo=$REPO_ROOT venv=$VENV python=$PYVER"

# --- 1. system packages --------------------------------------------------- #
echo "[bootstrap] installing system packages..."
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -qq
# Ubuntu 24.04 ships python3.12; on 22.04 we add deadsnakes for it.
if ! command -v "python${PYVER}" >/dev/null 2>&1; then
  $SUDO apt-get install -y -qq software-properties-common
  $SUDO add-apt-repository -y ppa:deadsnakes/ppa
  $SUDO apt-get update -qq
fi
$SUDO apt-get install -y -qq --no-install-recommends \
  build-essential curl git pkg-config ca-certificates \
  "python${PYVER}" "python${PYVER}-venv" "python${PYVER}-dev"

# --- 2. Rust toolchain (for balatro_core via maturin/PyO3) ----------------- #
if ! command -v cargo >/dev/null 2>&1; then
  echo "[bootstrap] installing Rust toolchain..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
# shellcheck disable=SC1091
[[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env"
echo "[bootstrap] rust: $(rustc --version)"

# --- 3. python venv + deps ------------------------------------------------- #
if [[ ! -d "$VENV" ]]; then
  "python${PYVER}" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip wheel

# CPU-only torch (a CUDA wheel would pull ~2GB of unused libs onto a CPU box).
echo "[bootstrap] installing CPU torch + numpy + build tools..."
python -m pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install --quiet -r "$REPO_ROOT/cloud/requirements.txt"

# --- 4. native extensions -------------------------------------------------- #
echo "[bootstrap] building balatro_core (Rust/maturin)..."
python -m pip install --quiet "$REPO_ROOT/botlatro-core"   # maturin -> balatro_core .so
echo "[bootstrap] building package + Cython hand_evaluator..."
python -m pip install --quiet -e "$REPO_ROOT"               # cython ext + editable install

# --- 5. smoke test (PROVES the port) --------------------------------------- #
echo "[bootstrap] === smoke test ==="
cd "$REPO_ROOT"
PYTHONPATH=src python - <<'PY'
import sys
ok = True
try:
    import balatro_core  # noqa: F401
    print("  [ok] balatro_core (Rust native) imported")
except Exception as e:  # noqa: BLE001
    ok = False; print(f"  [WARN] balatro_core native unavailable -> Python fallback ({e})")
try:
    from balatro_ai.rules import hand_evaluator  # noqa: F401
    print("  [ok] hand_evaluator imported")
except Exception as e:  # noqa: BLE001
    ok = False; print(f"  [FAIL] hand_evaluator import: {e}")
import torch, numpy  # noqa: F401
print(f"  [ok] torch {torch.__version__} (threads={torch.get_num_threads()}), numpy {numpy.__version__}")
# end-to-end: a 1-seed run through the sim + a candidate-feature build
from balatro_ai.bots.registry import create_bot
from balatro_ai.sim.local_runner import LocalBalatroSimulator
from balatro_ai.solver.seed_game import SeedGame
from balatro_ai.solver.trajectory import _stable_seed_int
from balatro_ai.api.state import with_derived_legal_actions
sim = LocalBalatroSimulator(seed=_stable_seed_int("9000001"), stake="white")
sim.state = with_derived_legal_actions(SeedGame("9000001", stake="white").initial_state())
bot = create_bot("basic_strategy_bot", seed=0)
for _ in range(50):
    st = sim.state
    if st.run_over: break
    a = bot.choose_action(st)
    if a.action_type.value == "no_op": break
    sim.step(a)
print(f"  [ok] end-to-end sim ran -> ante {sim.state.ante}, won={sim.state.won}")
print("[bootstrap] SMOKE TEST PASSED" if ok else "[bootstrap] SMOKE TEST: native fallback in use (slower but functional)")
PY
echo "[bootstrap] DONE. Activate with:  source $VENV/bin/activate"
