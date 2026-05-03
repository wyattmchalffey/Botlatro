from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request


SCRIPT_PATH = Path(__file__).resolve()
RUN_ROOT = SCRIPT_PATH.parent
PROJECT_ROOT = RUN_ROOT.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
LOG_ROOT = PROJECT_ROOT / ".logs" / RUN_ROOT.name
REPLAY_DIR = RUN_ROOT / "replays"

HOST = "127.0.0.1"
BASE_PORT = 12346
TARGET_WORKERS = 8
EXTRA_PORTS_TO_TRY = 6
SEED_COUNT = 128
GAME_SPEED = 64
FPS_CAP = 2000
ANIMATION_FPS = 1
TIMEOUT_SECONDS = 30.0
RUN_TIMEOUT_SECONDS = 1800.0
MAX_STEPS = 1000
LABEL = RUN_ROOT.name

UVX_FALLBACK = Path(os.environ.get("APPDATA", "")) / "Python" / "Python312" / "Scripts" / "uvx.exe"
SOURCE_BALATRO_EXE = Path(r"F:\SteamLibrary\steamapps\common\Balatro\Balatro.exe")
SOURCE_LOVELY_DLL = Path(r"F:\SteamLibrary\steamapps\common\Balatro\version.dll")
HEADLESS_DIR = PROJECT_ROOT / ".balatro-headless"
HEADLESS_BALATRO_EXE = HEADLESS_DIR / "Balatro.exe"
HEADLESS_LOVELY_DLL = HEADLESS_DIR / "version.dll"

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

OVERNIGHT_LOG = RUN_ROOT / "overnight.log"
STATUS_PATH = RUN_ROOT / "status.json"
RESULTS_PATH = RUN_ROOT / "benchmark_results.json"
VALIDATION_TEXT_PATH = RUN_ROOT / "validation.txt"
VALIDATION_JSON_PATH = RUN_ROOT / "validation.json"
SMOKE_OUTPUT_PATH = RUN_ROOT / "joker_smoke.jsonl"
SMOKE_MANIFEST_PATH = RUN_ROOT / "joker_smoke_manifest.json"
SMOKE_LOG_PATH = RUN_ROOT / "joker_smoke_result.json"
SMOKE_VALIDATION_TEXT_PATH = RUN_ROOT / "joker_smoke_validation.txt"
SMOKE_VALIDATION_JSON_PATH = RUN_ROOT / "joker_smoke_validation.json"
COMBINED_VALIDATION_TEXT_PATH = RUN_ROOT / "combined_validation.txt"
COMBINED_VALIDATION_JSON_PATH = RUN_ROOT / "combined_validation.json"


class BridgeProcess:
    def __init__(self, port: int, process: subprocess.Popen[bytes], stdout_path: Path, stderr_path: Path) -> None:
        self.port = port
        self.process = process
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path

    @property
    def endpoint(self) -> str:
        return f"http://{HOST}:{self.port}"


bridge_processes: list[BridgeProcess] = []


def main() -> int:
    _bootstrap_pythonpath()
    _ensure_dirs()
    _clear_previous_outputs()
    _log("Phase 7 overnight job starting.")
    _write_status(stage="starting")
    try:
        _stop_existing_bridge_processes()
        _prepare_headless_copy()
        healthy = _launch_bridge_workers()
        if not healthy:
            raise RuntimeError("No bridge workers became healthy.")

        _write_status(stage="benchmark_running", endpoints=[worker.endpoint for worker in healthy])
        benchmark_run = _run_bridge_benchmark(tuple(worker.endpoint for worker in healthy))
        _write_benchmark_results(benchmark_run)

        _write_status(stage="validating_replays", endpoints=[worker.endpoint for worker in healthy])
        replay_validation = _validate_replays((REPLAY_DIR,), VALIDATION_TEXT_PATH, VALIDATION_JSON_PATH)
        _log(
            "Replay validation complete: "
            f"{replay_validation.exact_transitions}/{replay_validation.transitions_checked} exact, "
            f"{replay_validation.comparable_divergences} comparable divergences."
        )

        _write_status(stage="running_joker_smoke", endpoints=[worker.endpoint for worker in healthy])
        smoke_ok = _run_joker_smoke(healthy[0].endpoint)
        if smoke_ok:
            _write_status(stage="validating_combined_corpus", endpoints=[worker.endpoint for worker in healthy])
            combined = _validate_replays(
                (REPLAY_DIR, SMOKE_OUTPUT_PATH),
                COMBINED_VALIDATION_TEXT_PATH,
                COMBINED_VALIDATION_JSON_PATH,
                example_limit=120,
            )
            _log(
                "Combined validation complete: "
                f"{combined.exact_transitions}/{combined.transitions_checked} exact, "
                f"{combined.comparable_divergences} comparable divergences."
            )
        else:
            _log("Skipping combined validation because joker smoke did not produce a clean oracle file.")

        _write_status(stage="finished", endpoints=[worker.endpoint for worker in healthy])
        _log("Phase 7 overnight job finished.")
        return 0
    except Exception as exc:  # noqa: BLE001 - unattended job should preserve all failure context.
        _log(f"ERROR: {type(exc).__name__}: {exc}")
        _log(traceback.format_exc())
        _write_status(stage="failed", error=f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        _stop_launched_bridge_processes()


def _bootstrap_pythonpath() -> None:
    src = str(SRC_ROOT)
    existing = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = src if not existing else src + os.pathsep + existing
    if src not in sys.path:
        sys.path.insert(0, src)


def _ensure_dirs() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)


def _clear_previous_outputs() -> None:
    _clear_dir_contents(REPLAY_DIR)
    _clear_dir_contents(LOG_ROOT)
    for path in (
        OVERNIGHT_LOG,
        STATUS_PATH,
        RESULTS_PATH,
        VALIDATION_TEXT_PATH,
        VALIDATION_JSON_PATH,
        SMOKE_OUTPUT_PATH,
        SMOKE_MANIFEST_PATH,
        SMOKE_LOG_PATH,
        SMOKE_VALIDATION_TEXT_PATH,
        SMOKE_VALIDATION_JSON_PATH,
        COMBINED_VALIDATION_TEXT_PATH,
        COMBINED_VALIDATION_JSON_PATH,
    ):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _clear_dir_contents(path: Path) -> None:
    root = path.resolve()
    if root not in {REPLAY_DIR.resolve(), LOG_ROOT.resolve()}:
        raise RuntimeError(f"Refusing to clear unexpected directory: {path}")
    for child in path.iterdir():
        target = child.resolve()
        if not target.is_relative_to(root):
            raise RuntimeError(f"Refusing to clear path outside {root}: {target}")
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with OVERNIGHT_LOG.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def _write_status(**payload: object) -> None:
    status = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_root": str(RUN_ROOT),
        "log_root": str(LOG_ROOT),
        "replay_dir": str(REPLAY_DIR),
        "seed_count": SEED_COUNT,
        "target_workers": TARGET_WORKERS,
        "game_speed": GAME_SPEED,
        "worker_pids": [{"port": worker.port, "pid": worker.process.pid} for worker in bridge_processes],
        **payload,
    }
    STATUS_PATH.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stop_existing_bridge_processes() -> None:
    _log("Stopping stale Balatro/BalatroBot bridge processes, if any.")
    if os.name != "nt":
        return
    for image_name in ("Balatro.exe", "balatrobot.exe", "uvx.exe", "uv.exe"):
        subprocess.run(
            ["taskkill", "/F", "/T", "/IM", image_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW,
            check=False,
        )
    time.sleep(2)


def _prepare_headless_copy() -> None:
    from balatro_ai.tools.headless_exe import ensure_tiny_startup_copy

    _log(f"Preparing tiny-startup Balatro copy in {HEADLESS_DIR}.")
    copied_exe, copied_lovely = ensure_tiny_startup_copy(SOURCE_BALATRO_EXE, HEADLESS_DIR)
    if not copied_exe.exists():
        raise FileNotFoundError(copied_exe)
    if copied_lovely is None:
        if SOURCE_LOVELY_DLL.exists():
            shutil.copy2(SOURCE_LOVELY_DLL, HEADLESS_LOVELY_DLL)
        else:
            raise FileNotFoundError(SOURCE_LOVELY_DLL)
    _log(f"Using Balatro exe: {copied_exe}")


def _launch_bridge_workers() -> list[BridgeProcess]:
    uvx = shutil.which("uvx") or (str(UVX_FALLBACK) if UVX_FALLBACK.exists() else None)
    if uvx is None:
        raise FileNotFoundError("uvx")

    healthy: list[BridgeProcess] = []
    last_port = BASE_PORT + TARGET_WORKERS + EXTRA_PORTS_TO_TRY
    for port in range(BASE_PORT, last_port):
        if len(healthy) >= TARGET_WORKERS:
            break
        worker = _start_bridge_worker(uvx, port)
        bridge_processes.append(worker)
        _write_status(stage="launching_bridges", endpoints=[item.endpoint for item in healthy])
        if _wait_for_health(port, timeout_seconds=90.0):
            healthy.append(worker)
            _log(f"Bridge {port} is healthy ({len(healthy)}/{TARGET_WORKERS}).")
            time.sleep(1)
            continue
        _log(f"Bridge {port} failed health check; keeping run alive with other workers.")
        _kill_process_tree(worker.process.pid)

    _log(f"Healthy endpoints: {', '.join(worker.endpoint for worker in healthy)}")
    return healthy


def _start_bridge_worker(uvx: str, port: int) -> BridgeProcess:
    worker_log_dir = LOG_ROOT / f"worker-{port}"
    worker_log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = worker_log_dir / "stdout.txt"
    stderr_path = worker_log_dir / "stderr.txt"
    args = [
        uvx,
        "balatrobot",
        "serve",
        "--host",
        HOST,
        "--port",
        str(port),
        "--fps-cap",
        str(FPS_CAP),
        "--gamespeed",
        str(GAME_SPEED),
        "--animation-fps",
        str(ANIMATION_FPS),
        "--logs-path",
        str(worker_log_dir),
        "--love-path",
        str(HEADLESS_BALATRO_EXE),
        "--lovely-path",
        str(HEADLESS_LOVELY_DLL),
        "--headless",
        "--fast",
        "--no-debug",
        "--no-shaders",
        "--no-audio",
    ]
    env = os.environ.copy()
    env["BALATROBOT_LOG_LEVEL"] = "quiet"
    env["BALATROBOT_NO_ANIMATIONS"] = "1"
    _log(f"Starting bridge on port {port}.")
    stdout_file = stdout_path.open("wb")
    stderr_file = stderr_path.open("wb")
    try:
        process = subprocess.Popen(
            args,
            cwd=PROJECT_ROOT,
            stdout=stdout_file,
            stderr=stderr_file,
            env=env,
            creationflags=CREATE_NO_WINDOW,
        )
    finally:
        stdout_file.close()
        stderr_file.close()
    return BridgeProcess(port=port, process=process, stdout_path=stdout_path, stderr_path=stderr_path)


def _wait_for_health(port: int, *, timeout_seconds: float) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _health_ok(port):
            return True
        time.sleep(1)
    return False


def _health_ok(port: int) -> bool:
    payload = {"jsonrpc": "2.0", "method": "health", "id": 1}
    request = urllib.request.Request(
        f"http://{HOST}:{port}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError):
        return False
    return data.get("result", {}).get("status") == "ok"


def _run_bridge_benchmark(endpoints: tuple[str, ...]):
    from balatro_ai.eval.runner import BenchmarkOptions, run_benchmark

    _log(f"Starting {SEED_COUNT}-seed detailed bridge corpus with {len(endpoints)} worker(s).")

    def progress(message: str) -> None:
        _log(message)

    return run_benchmark(
        BenchmarkOptions(
            bot="basic_strategy_bot",
            stake="white",
            deck="RED",
            profile_name="P1",
            unlock_state="all",
            seeds=SEED_COUNT,
            label=LABEL,
            endpoints=endpoints,
            timeout_seconds=TIMEOUT_SECONDS,
            max_steps=MAX_STEPS,
            run_timeout_seconds=RUN_TIMEOUT_SECONDS,
            replay_dir=REPLAY_DIR,
            replay_mode="score_audit",
            start_retries=1,
            retry_failed_seeds=2,
            park_finished_endpoints=True,
        ),
        progress=progress,
    )


def _write_benchmark_results(benchmark_run) -> None:
    payload = {
        "seed_set_label": benchmark_run.seed_set_label,
        "summary": asdict(benchmark_run.summary),
        "results": [asdict(result) for result in benchmark_run.results],
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _log(f"Wrote benchmark results to {RESULTS_PATH}.")


def _validate_replays(paths: tuple[Path, ...], text_path: Path, json_path: Path, *, example_limit: int = 100):
    from balatro_ai.sim.replay_validator import validate_local_sim_replays

    summary = validate_local_sim_replays(paths)
    text_path.write_text(summary.to_text(example_limit=example_limit) + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(summary.to_json_dict(example_limit=example_limit), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _run_joker_smoke(endpoint: str) -> bool:
    from balatro_ai.api.client import JsonRpcBalatroClient
    from balatro_ai.sim.bridge_joker_smoke import iter_joker_smoke_scenarios, run_bridge_joker_smoke, write_manifest
    from balatro_ai.sim.replay_validator import validate_local_sim_replays

    scenarios = list(iter_joker_smoke_scenarios())
    write_manifest(scenarios, SMOKE_MANIFEST_PATH)
    _log(f"Running {len(scenarios)} bridge joker smoke scenario(s) on {endpoint}.")
    client = JsonRpcBalatroClient(endpoint=endpoint, deck="RED", timeout_seconds=TIMEOUT_SECONDS)
    result = run_bridge_joker_smoke(
        client,
        scenarios,
        output_path=SMOKE_OUTPUT_PATH,
        base_seed=900001,
        stake="white",
    )
    SMOKE_LOG_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _log(
        f"Joker smoke wrote {result['rows_written']} row(s) for "
        f"{result['scenarios_run']} scenario(s); failures={len(result['failures'])}."
    )
    if result["failures"]:
        for failure in result["failures"]:
            _log(f"Joker smoke failure: {failure}")

    summary = validate_local_sim_replays((SMOKE_OUTPUT_PATH,))
    SMOKE_VALIDATION_TEXT_PATH.write_text(summary.to_text(example_limit=80) + "\n", encoding="utf-8")
    SMOKE_VALIDATION_JSON_PATH.write_text(
        json.dumps(summary.to_json_dict(example_limit=80), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _log(
        "Joker smoke validation complete: "
        f"{summary.exact_transitions}/{summary.transitions_checked} exact, "
        f"{summary.comparable_divergences} comparable divergences."
    )
    return not result["failures"]


def _stop_launched_bridge_processes() -> None:
    if not bridge_processes:
        return
    _log("Stopping bridge workers launched by this job.")
    for worker in bridge_processes:
        if worker.process.poll() is None:
            _kill_process_tree(worker.process.pid)
    time.sleep(2)


def _kill_process_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW,
            check=False,
        )
    else:
        try:
            os.kill(pid, 15)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
