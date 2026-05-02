"""Tkinter GUI for configuring and running Botlatro benchmarks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from tkinter import BooleanVar, IntVar, StringVar, TclError, Tk, filedialog, messagebox
from tkinter import Text as TkText
from tkinter import ttk

from balatro_ai.eval.runner import BenchmarkOptions, endpoint_urls, run_benchmark
from balatro_ai.eval.seed_sets import parse_seed_values
from balatro_ai.tools.clean_bridge_logs import clean_logs, summarize_results
from balatro_ai.tools.headless_exe import ensure_tiny_startup_copy


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CURRENT_BENCHMARK_SEEDS_PATH = PROJECT_ROOT / ".data" / "current-light-100-seeds.txt"

BOT_NAMES = ("basic_strategy_bot", "greedy_bot", "random_bot")
STAKES = ("white", "red", "green", "black", "blue", "purple", "orange", "gold")
DECKS = (
    "RED",
    "BLUE",
    "YELLOW",
    "GREEN",
    "BLACK",
    "MAGIC",
    "NEBULA",
    "GHOST",
    "ABANDONED",
    "CHECKERED",
    "ZODIAC",
    "PAINTED",
    "ANAGLYPH",
    "PLASMA",
    "ERRATIC",
)
BRIDGE_LOG_MODES = ("quiet", "off", "normal_clean", "normal")
REPLAY_DETAIL_MODES = ("off", "summary", "light", "score_audit")


@dataclass(slots=True)
class BridgeProcess:
    port: int
    process: subprocess.Popen
    log_dir: Path
    clean_logs: bool
    discard_logs: bool


class BenchmarkApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Botlatro Benchmark Runner")
        self.root.geometry("980x760")

        self.messages: Queue[str] = Queue()
        self.worker_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.bridge_processes: list[BridgeProcess] = []
        self.log_lines: list[str] = []
        self.log_lock = threading.Lock()

        self.bot = StringVar(value="basic_strategy_bot")
        self.stake = StringVar(value="white")
        self.deck = StringVar(value="RED")
        self.profile_name = StringVar(value="P1")
        self.unlock_state = StringVar(value="all")
        self.seed_count = IntVar(value=100)
        self.seed_list = StringVar(value="")
        self.label = StringVar(value="primary-score-audit-100")
        self.max_steps = IntVar(value=800)
        self.timeout_seconds = StringVar(value="30")
        self.run_timeout_seconds = StringVar(value="1800")
        self.start_retries = IntVar(value=1)
        self.retry_failed_seeds = IntVar(value=1)
        self.replay_dir = StringVar(value=str(Path(".data") / "gui-replays"))
        self.output_file = StringVar(value=str(Path(".data") / "gui_benchmark.txt"))

        self.host = StringVar(value="127.0.0.1")
        self.base_port = IntVar(value=12346)
        self.workers = IntVar(value=1)
        self.launch_bridges = BooleanVar(value=False)
        self.stop_existing = BooleanVar(value=True)

        self.uvx_path = StringVar(value=_default_uvx_path())
        self.love_path = StringVar(value=r"F:\SteamLibrary\steamapps\common\Balatro\Balatro.exe")
        self.lovely_path = StringVar(value=r"F:\SteamLibrary\steamapps\common\Balatro\version.dll")
        self.logs_root = StringVar(value=str(Path(".logs") / "gui-workers"))
        self.bridge_log_mode = StringVar(value="quiet")

        self.headless = BooleanVar(value=True)
        self.tiny_startup = BooleanVar(value=True)
        self.fast = BooleanVar(value=True)
        self.no_shaders = BooleanVar(value=True)
        self.render_on_api = BooleanVar(value=False)
        self.audio = BooleanVar(value=False)
        self.fps_cap = IntVar(value=2000)
        self.gamespeed = IntVar(value=32)
        self.animation_fps = IntVar(value=1)
        self.replay_mode = StringVar(value="off")

        self._build_ui()
        self._poll_messages()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(4, weight=1)

        benchmark = ttk.LabelFrame(outer, text="Benchmark", padding=10)
        benchmark.grid(row=0, column=0, sticky="ew")
        for col in range(8):
            benchmark.columnconfigure(col, weight=1)

        _combo(benchmark, "Bot", self.bot, BOT_NAMES, 0, 0)
        _combo(benchmark, "Stake", self.stake, STAKES, 0, 2)
        _entry(benchmark, "Seeds", self.seed_count, 0, 4)
        _entry(benchmark, "Label", self.label, 0, 6)
        _entry(benchmark, "Max steps", self.max_steps, 1, 0)
        _entry(benchmark, "Timeout", self.timeout_seconds, 1, 2)
        _entry(benchmark, "Start retries", self.start_retries, 1, 4)
        _entry(benchmark, "Failed retries", self.retry_failed_seeds, 1, 6)
        _path_entry(benchmark, "Replay dir", self.replay_dir, 2, 0, self._browse_replay_dir, column_span=7)
        _path_entry(benchmark, "Output file", self.output_file, 3, 0, self._browse_output_file, column_span=7)
        _wide_entry(benchmark, "Seed list", self.seed_list, 4, 0, column_span=7)
        _combo(benchmark, "Deck", self.deck, DECKS, 5, 0)
        _entry(benchmark, "Profile", self.profile_name, 5, 2)
        _entry(benchmark, "Unlocks", self.unlock_state, 5, 4)

        workers = ttk.LabelFrame(outer, text="Workers", padding=10)
        workers.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        for col in range(8):
            workers.columnconfigure(col, weight=1)

        _entry(workers, "Host", self.host, 0, 0)
        _entry(workers, "Base port", self.base_port, 0, 2)
        _entry(workers, "Workers", self.workers, 0, 4)
        ttk.Checkbutton(workers, text="Launch bridge workers", variable=self.launch_bridges).grid(
            row=0, column=6, columnspan=2, sticky="w"
        )
        ttk.Checkbutton(workers, text="Stop existing first", variable=self.stop_existing).grid(
            row=1, column=6, columnspan=2, sticky="w"
        )

        bridge = ttk.LabelFrame(outer, text="Bridge Launch Options", padding=10)
        bridge.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        for col in range(8):
            bridge.columnconfigure(col, weight=1)

        _path_entry(bridge, "uvx", self.uvx_path, 0, 0, self._browse_uvx, column_span=7)
        _path_entry(bridge, "Balatro exe", self.love_path, 1, 0, self._browse_love, column_span=7)
        _path_entry(bridge, "Lovely dll", self.lovely_path, 2, 0, self._browse_lovely, column_span=7)
        _path_entry(bridge, "Logs root", self.logs_root, 3, 0, self._browse_logs_root, column_span=7)

        ttk.Checkbutton(bridge, text="Headless", variable=self.headless, command=self._headless_toggled).grid(
            row=4, column=0, sticky="w"
        )
        ttk.Checkbutton(bridge, text="Fast", variable=self.fast).grid(row=4, column=1, sticky="w")
        ttk.Checkbutton(bridge, text="No shaders", variable=self.no_shaders).grid(row=4, column=2, sticky="w")
        ttk.Checkbutton(
            bridge,
            text="Render on API",
            variable=self.render_on_api,
            command=self._render_on_api_toggled,
        ).grid(row=4, column=3, sticky="w")
        ttk.Checkbutton(bridge, text="Audio", variable=self.audio).grid(row=4, column=4, sticky="w")
        ttk.Checkbutton(bridge, text="Tiny startup", variable=self.tiny_startup).grid(row=4, column=5, sticky="w")
        _combo(bridge, "Bridge logs", self.bridge_log_mode, BRIDGE_LOG_MODES, 4, 6)
        _entry(bridge, "FPS cap", self.fps_cap, 5, 0)
        _entry(bridge, "Game speed", self.gamespeed, 5, 2)
        _entry(bridge, "Anim FPS", self.animation_fps, 5, 4)
        ttk.Button(bridge, text="Benchmark Speed", command=self._benchmark_speed_preset).grid(
            row=5, column=6, sticky="ew", padx=(8, 0), pady=3
        )
        ttk.Button(bridge, text="Watch Speed", command=self._watch_speed_preset).grid(
            row=5, column=7, sticky="ew", padx=(8, 0), pady=3
        )
        _combo(benchmark, "Replay mode", self.replay_mode, REPLAY_DETAIL_MODES, 6, 0)
        ttk.Button(benchmark, text="Use Benchmark Seeds", command=self._use_benchmark_seeds).grid(
            row=6, column=2, columnspan=2, sticky="ew", padx=(8, 0), pady=3
        )
        ttk.Button(benchmark, text="Generated Seeds", command=self._use_generated_seeds).grid(
            row=6, column=4, columnspan=2, sticky="ew", padx=(8, 0), pady=3
        )
        _entry(benchmark, "Run timeout", self.run_timeout_seconds, 6, 6)

        controls = ttk.Frame(outer)
        controls.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.start_button = ttk.Button(controls, text="Run Benchmark", command=self._start_benchmark)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(controls, text="Stop Run", command=self._stop_run, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="Stop Owned Workers", command=self._stop_owned_workers_clicked).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(controls, text="Stop All Bridges", command=self._stop_all_bridges_clicked).pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="Health Check", command=self._health_check).pack(side="left", padx=(8, 0))

        log_frame = ttk.LabelFrame(outer, text="Output", padding=8)
        log_frame.grid(row=4, column=0, sticky="nsew", pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = TkText(log_frame, wrap="word", height=22)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    def _browse_replay_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.replay_dir.set(path)

    def _browse_output_file(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if path:
            self.output_file.set(path)

    def _browse_uvx(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Executable", "*.exe"), ("All", "*.*")])
        if path:
            self.uvx_path.set(path)

    def _browse_love(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Executable", "*.exe"), ("All", "*.*")])
        if path:
            self.love_path.set(path)

    def _browse_lovely(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("DLL", "*.dll"), ("All", "*.*")])
        if path:
            self.lovely_path.set(path)

    def _browse_logs_root(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.logs_root.set(path)

    def _start_benchmark(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Benchmark Running", "A benchmark is already running.")
            return

        try:
            config = self._read_config()
        except ValueError as exc:
            messagebox.showerror("Invalid Parameters", str(exc))
            return

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.stop_event.clear()
        self._clear_log()
        self._normalize_bridge_options(log=True)
        self.worker_thread = threading.Thread(target=self._run_background, args=(config,), daemon=True)
        self.worker_thread.start()

    def _stop_run(self) -> None:
        self._request_stop()
        self._stop_bridges()

    def _benchmark_speed_preset(self) -> None:
        self.headless.set(True)
        self.tiny_startup.set(True)
        self.fast.set(True)
        self.no_shaders.set(True)
        self.render_on_api.set(False)
        self.audio.set(False)
        self.bridge_log_mode.set("quiet")
        self.replay_mode.set("summary")
        self.start_retries.set(1)
        self.run_timeout_seconds.set("1800")
        self.fps_cap.set(2000)
        self.gamespeed.set(32)
        self.animation_fps.set(1)
        self._log_message("Applied benchmark speed preset: headless, quiet bridge logs, summary replay, gamespeed 32.")

    def _watch_speed_preset(self) -> None:
        self.headless.set(False)
        self.tiny_startup.set(False)
        self.fast.set(False)
        self.no_shaders.set(True)
        self.render_on_api.set(False)
        self.audio.set(False)
        self.bridge_log_mode.set("normal_clean")
        self.replay_mode.set("off")
        self.start_retries.set(1)
        self.run_timeout_seconds.set("0")
        self.fps_cap.set(60)
        self.gamespeed.set(1)
        self.animation_fps.set(30)
        self._log_message("Applied watch speed preset: visible, gamespeed 1, animation FPS 30.")

    def _use_benchmark_seeds(self) -> None:
        if CURRENT_BENCHMARK_SEEDS_PATH.exists():
            seeds = parse_seed_values(CURRENT_BENCHMARK_SEEDS_PATH.read_text(encoding="utf-8"))
            self.seed_list.set(",".join(str(seed) for seed in seeds))
            self.seed_count.set(len(seeds))
            self.label.set("current-light-100")
            self._log_message(f"Loaded {len(seeds)} current benchmark seeds from {CURRENT_BENCHMARK_SEEDS_PATH}.")
            return

        self.seed_list.set("")
        self.seed_count.set(100)
        self.label.set("primary-score-audit-100")
        self._log_message(
            "Current seed file was not found; using generated 100-seed label primary-score-audit-100."
        )

    def _use_generated_seeds(self) -> None:
        self.seed_list.set("")
        try:
            seed_count = int(self.seed_count.get())
        except (TclError, TypeError, ValueError):
            seed_count = 0
        if seed_count < 1:
            self.seed_count.set(100)
        self._log_message(
            f"Using generated deterministic seeds from stake={self.stake.get()} label={self.label.get()}."
        )

    def _headless_toggled(self) -> None:
        if self.headless.get() and self.render_on_api.get():
            self.render_on_api.set(False)

    def _render_on_api_toggled(self) -> None:
        if self.render_on_api.get() and self.headless.get():
            self.headless.set(False)

    def _normalize_bridge_options(self, *, log: bool) -> None:
        if self.headless.get() and self.render_on_api.get():
            self.render_on_api.set(False)
            if log:
                self._log_message("Headless and render-on-API conflict in BalatroBot; disabled render-on-API.")

    def _read_config(self) -> dict[str, object]:
        worker_count = _int_value(self.workers, "Workers")
        if worker_count < 1:
            raise ValueError("Workers must be at least 1.")
        seed_count = _int_value(self.seed_count, "Seeds")
        if seed_count < 0:
            raise ValueError("Seeds must be zero or greater.")
        max_steps = _int_value(self.max_steps, "Max steps")
        if max_steps < 1:
            raise ValueError("Max steps must be at least 1.")
        start_retries = _int_value(self.start_retries, "Start retries")
        if start_retries < 0:
            raise ValueError("Start retries must be zero or greater.")
        failed_retries = _int_value(self.retry_failed_seeds, "Failed retries")
        if failed_retries < 0:
            raise ValueError("Failed retries must be zero or greater.")
        timeout_seconds = _float_value(self.timeout_seconds, "Timeout")
        if timeout_seconds <= 0:
            raise ValueError("Timeout must be greater than zero.")
        run_timeout_seconds = _float_value(self.run_timeout_seconds, "Run timeout")
        if run_timeout_seconds < 0:
            raise ValueError("Run timeout must be zero or greater.")

        seed_values = parse_seed_values(self.seed_list.get())
        replay_dir = Path(self.replay_dir.get()) if self.replay_dir.get().strip() else None
        output_file = Path(self.output_file.get()) if self.output_file.get().strip() else None
        return {
            "options": BenchmarkOptions(
                bot=self.bot.get(),
                stake=self.stake.get(),
                deck=self.deck.get(),
                profile_name=self.profile_name.get(),
                unlock_state=self.unlock_state.get(),
                seeds=seed_count,
                seed_values=seed_values or None,
                label=self.label.get(),
                endpoints=endpoint_urls(self.host.get(), _int_value(self.base_port, "Base port"), worker_count),
                timeout_seconds=timeout_seconds,
                max_steps=max_steps,
                run_timeout_seconds=run_timeout_seconds if run_timeout_seconds > 0 else None,
                replay_dir=replay_dir,
                replay_mode=self.replay_mode.get(),
                start_retries=start_retries,
                retry_failed_seeds=failed_retries,
            ),
            "worker_count": worker_count,
            "output_file": output_file,
        }

    def _run_background(self, config: dict[str, object]) -> None:
        try:
            options = config["options"]
            if not isinstance(options, BenchmarkOptions):
                raise TypeError("Invalid benchmark options")

            if self.launch_bridges.get():
                self._start_bridges(options)
            else:
                self._log_message("Using already-running bridge endpoint(s).")

            if self.stop_event.is_set():
                self._log_message("Run stopped before benchmark started.")
                return

            run_benchmark(options, progress=self._log_message, should_stop=self.stop_event.is_set)
            output_file = config["output_file"]
            if isinstance(output_file, Path):
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with self.log_lock:
                    lines = list(self.log_lines)
                output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
                self._log_message(f"Saved output to {output_file}")
        except Exception as exc:  # noqa: BLE001 - GUI should surface all failures.
            self._log_message(f"ERROR: {exc}")
            if self.launch_bridges.get():
                self._log_message("Stopping partially-started bridge workers.")
                self._stop_bridges()
        finally:
            self.messages.put("__ENABLE_START__")

    def _start_bridges(self, options: BenchmarkOptions) -> None:
        self._stop_bridges()
        if self.stop_existing.get():
            self._stop_existing_bridge_processes()

        logs_root = self._effective_logs_root()
        logs_root.mkdir(parents=True, exist_ok=True)
        love_path, lovely_path = self._effective_balatro_paths()

        for endpoint in options.endpoints:
            if self.stop_event.is_set():
                self._log_message("Bridge launch stopped.")
                return
            port = int(endpoint.rsplit(":", 1)[1])
            log_dir = logs_root / f"worker-{port}-{int(time.time())}"
            log_dir.mkdir(parents=True, exist_ok=True)
            args = self._bridge_args(port=port, log_dir=log_dir, love_path=love_path, lovely_path=lovely_path)
            clean_logs, discard_logs = self._bridge_log_cleanup_flags()
            self._log_message(f"Starting bridge on port {port}...")
            process = subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=self._bridge_env(),
                creationflags=_creation_flags(),
            )
            self.bridge_processes.append(
                BridgeProcess(
                    port=port,
                    process=process,
                    log_dir=log_dir,
                    clean_logs=clean_logs,
                    discard_logs=discard_logs,
                )
            )
            try:
                self._wait_for_health(port)
            except RuntimeError:
                self._log_message(f"Bridge {port} failed. Logs: {log_dir}")
                self._stop_bridges()
                raise
            self._log_message(f"Bridge {port} is healthy.")
            time.sleep(1)

    def _effective_logs_root(self) -> Path:
        if self.bridge_log_mode.get() == "off":
            return Path(tempfile.gettempdir()) / "botlatro-bridge-logs"
        return Path(self.logs_root.get())

    def _bridge_log_cleanup_flags(self) -> tuple[bool, bool]:
        mode = self.bridge_log_mode.get()
        return mode in ("quiet", "normal_clean"), mode == "off"

    def _bridge_env(self) -> dict[str, str]:
        env = os.environ.copy()
        mode = self.bridge_log_mode.get()
        if mode == "off":
            env["BALATROBOT_LOG_LEVEL"] = "off"
        elif mode == "quiet":
            env["BALATROBOT_LOG_LEVEL"] = "quiet"
        else:
            env["BALATROBOT_LOG_LEVEL"] = "normal"
        return env

    def _effective_balatro_paths(self) -> tuple[Path, Path]:
        love_path = Path(self.love_path.get())
        lovely_path = Path(self.lovely_path.get())
        if os.name == "nt" and self.headless.get() and self.tiny_startup.get():
            copy_dir = Path(".balatro-headless")
            self._log_message(f"Preparing tiny-startup Balatro copy in {copy_dir}...")
            love_path, copied_lovely = ensure_tiny_startup_copy(love_path, copy_dir)
            if copied_lovely is not None:
                lovely_path = copied_lovely
            self._log_message(f"Using tiny-startup Balatro exe: {love_path}")
        return love_path, lovely_path

    def _bridge_args(self, *, port: int, log_dir: Path, love_path: Path, lovely_path: Path) -> list[str]:
        args = [
            self.uvx_path.get(),
            "balatrobot",
            "serve",
            "--host",
            self.host.get(),
            "--port",
            str(port),
            "--fps-cap",
            str(_int_value(self.fps_cap, "FPS cap")),
            "--gamespeed",
            str(_int_value(self.gamespeed, "Game speed")),
            "--animation-fps",
            str(_int_value(self.animation_fps, "Anim FPS")),
            "--logs-path",
            str(log_dir),
            "--love-path",
            str(love_path),
            "--lovely-path",
            str(lovely_path),
        ]
        args.append("--headless" if self.headless.get() else "--no-headless")
        args.append("--fast" if self.fast.get() else "--no-fast")
        args.append("--no-debug")
        if self.no_shaders.get():
            args.append("--no-shaders")
        if self.render_on_api.get():
            args.append("--render-on-api")
        args.append("--audio" if self.audio.get() else "--no-audio")
        return args

    def _wait_for_health(self, port: int) -> None:
        deadline = time.time() + 45
        while time.time() < deadline:
            if self.stop_event.is_set():
                raise RuntimeError(f"Bridge launch on port {port} was stopped.")
            if _health_ok(self.host.get(), port):
                return
            time.sleep(1)
        raise RuntimeError(f"Bridge on port {port} did not become healthy.")

    def _health_check(self) -> None:
        try:
            worker_count = _int_value(self.workers, "Workers")
            base_port = _int_value(self.base_port, "Base port")
            ports = [base_port + index for index in range(worker_count)]
        except ValueError:
            messagebox.showerror("Invalid Parameters", "Workers and base port must be integers.")
            return

        for port in ports:
            status = "OK" if _health_ok(self.host.get(), port) else "MISSING"
            self._log_message(f"{status} health http://{self.host.get()}:{port}")

    def _stop_bridges(self) -> None:
        clean_log_dirs: list[Path] = []
        discard_log_dirs: list[Path] = []
        for bridge in self.bridge_processes:
            if bridge.process.poll() is None:
                self._log_message(f"Stopping owned bridge on port {bridge.port}...")
                _kill_process_tree(bridge.process.pid)
                try:
                    bridge.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            if bridge.clean_logs:
                clean_log_dirs.append(bridge.log_dir)
            if bridge.discard_logs:
                discard_log_dirs.append(bridge.log_dir)
        if clean_log_dirs:
            self._clean_bridge_log_dirs(clean_log_dirs)
        if discard_log_dirs:
            self._discard_bridge_log_dirs(discard_log_dirs)
        self.bridge_processes.clear()

    def _clean_bridge_log_dirs(self, log_dirs: list[Path]) -> None:
        try:
            results = clean_logs(log_dirs, replace=True)
        except OSError as exc:
            self._log_message(f"WARNING: Could not clean bridge logs: {exc}")
            return
        if results:
            self._log_message(summarize_results(results))

    def _discard_bridge_log_dirs(self, log_dirs: list[Path]) -> None:
        removed = 0
        for log_dir in log_dirs:
            try:
                shutil.rmtree(log_dir, ignore_errors=True)
                removed += 1
            except OSError as exc:
                self._log_message(f"WARNING: Could not discard bridge logs at {log_dir}: {exc}")
        if removed:
            self._log_message(f"Discarded bridge logs for {removed} worker(s).")

    def _stop_owned_workers_clicked(self) -> None:
        self._request_stop()
        self._stop_bridges()

    def _stop_all_bridges_clicked(self) -> None:
        self._request_stop()
        self._stop_existing_bridge_processes()

    def _stop_existing_bridge_processes(self) -> None:
        self._log_message("Stopping existing Balatro/BalatroBot processes...")
        for image_name in ("Balatro.exe", "balatrobot.exe", "uvx.exe", "uv.exe"):
            subprocess.run(
                ["taskkill", "/F", "/T", "/IM", image_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=_creation_flags(),
                check=False,
            )
        time.sleep(2)

    def _clear_log(self) -> None:
        with self.log_lock:
            self.log_lines.clear()
        self.log.delete("1.0", "end")

    def _log_message(self, message: str) -> None:
        with self.log_lock:
            self.log_lines.append(message)
        self.messages.put(message)

    def _request_stop(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive() and not self.stop_event.is_set():
            self.stop_event.set()
            self._log_message("Stop requested; no new seeds will be scheduled.")

    def _poll_messages(self) -> None:
        while True:
            try:
                message = self.messages.get_nowait()
            except Empty:
                break
            if message == "__ENABLE_START__":
                self.start_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
            else:
                self.log.insert("end", message + "\n")
                self.log.see("end")
        self.root.after(100, self._poll_messages)

    def _on_close(self) -> None:
        self.stop_event.set()
        self._stop_bridges()
        self.root.destroy()


def _combo(parent: ttk.Frame, label: str, variable: StringVar, values: tuple[str, ...], row: int, column: int) -> None:
    ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 4), pady=3)
    ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=18).grid(
        row=row, column=column + 1, sticky="ew", pady=3
    )


def _entry(parent: ttk.Frame, label: str, variable: StringVar | IntVar, row: int, column: int) -> None:
    ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 4), pady=3)
    ttk.Entry(parent, textvariable=variable, width=14).grid(row=row, column=column + 1, sticky="ew", pady=3)


def _wide_entry(
    parent: ttk.Frame,
    label: str,
    variable: StringVar,
    row: int,
    column: int,
    column_span: int = 3,
) -> None:
    ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 4), pady=3)
    ttk.Entry(parent, textvariable=variable).grid(
        row=row, column=column + 1, columnspan=column_span, sticky="ew", pady=3
    )


def _path_entry(
    parent: ttk.Frame,
    label: str,
    variable: StringVar,
    row: int,
    column: int,
    command,
    column_span: int = 3,
) -> None:
    ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 4), pady=3)
    ttk.Entry(parent, textvariable=variable).grid(
        row=row, column=column + 1, columnspan=column_span, sticky="ew", pady=3
    )
    ttk.Button(parent, text="Browse", command=command).grid(row=row, column=column + column_span + 1, sticky="e", pady=3)


def _int_value(variable: IntVar, label: str) -> int:
    try:
        return int(variable.get())
    except (TclError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc


def _float_value(variable: StringVar, label: str) -> float:
    try:
        return float(variable.get())
    except (TclError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number.") from exc


def _health_ok(host: str, port: int) -> bool:
    endpoint = f"http://{host}:{port}"
    payload = {"jsonrpc": "2.0", "method": "health", "id": 1}
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
    return data.get("result", {}).get("status") == "ok"


def _default_uvx_path() -> str:
    return str(Path(os.environ.get("APPDATA", "")) / "Python" / "Python312" / "Scripts" / "uvx.exe")


def _creation_flags() -> int:
    return 0x08000000 if os.name == "nt" else 0


def _kill_process_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=_creation_flags(),
            check=False,
        )
    else:
        try:
            os.kill(pid, 15)
        except OSError:
            pass


def main() -> int:
    root = Tk()
    BenchmarkApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
