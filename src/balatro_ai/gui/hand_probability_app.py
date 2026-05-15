"""Tkinter GUI for estimating Balatro hand-type odds."""

from __future__ import annotations

import threading
from queue import Empty, Queue
from tkinter import IntVar, StringVar, TclError, Tk, messagebox
from tkinter import ttk

from balatro_ai.probability.hand_type_odds import (
    CHECKERED_DECK_PRESET,
    DECK_PRESETS,
    RANKS,
    SUITS,
    SimulationConfig,
    SimulationResult,
    deck_preset,
    deck_size,
    estimate_hand_type_probabilities,
)


class HandProbabilityApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Botlatro Hand Odds")
        self.root.geometry("1120x720")

        self.messages: Queue[tuple[str, object]] = Queue()
        self.stop_event = threading.Event()
        self.worker_thread: threading.Thread | None = None

        self.hand_size = IntVar(value=8)
        self.play_size = IntVar(value=5)
        self.discard_size = IntVar(value=5)
        self.discards = IntVar(value=4)
        self.hands = IntVar(value=4)
        self.trials = IntVar(value=2_000)
        self.seed = IntVar(value=1)
        self.preset = StringVar(value=DECK_PRESETS[0])
        self.status = StringVar(value="Ready")
        self.deck_total = StringVar(value="52 cards")

        self.deck_vars = [
            [IntVar(value=count) for count in row]
            for row in deck_preset(self.preset.get())
        ]

        self._build_ui()
        self._poll_messages()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)

        controls = ttk.LabelFrame(outer, text="Run", padding=10)
        controls.grid(row=0, column=0, sticky="ew")
        for column in range(12):
            controls.columnconfigure(column, weight=1)

        _spin(controls, "Hand size", self.hand_size, 1, 30, 0, 0)
        _spin(controls, "Play size", self.play_size, 1, 5, 0, 2)
        _spin(controls, "Discard size", self.discard_size, 1, 5, 0, 4)
        _spin(controls, "Discards", self.discards, 0, 20, 0, 6)
        _spin(controls, "Hands", self.hands, 1, 20, 0, 8)
        _spin(controls, "Trials", self.trials, 100, 1_000_000, 1, 0, increment=1000)
        _spin(controls, "Seed", self.seed, 0, 1_000_000_000, 1, 2)

        ttk.Label(controls, text="Preset").grid(row=1, column=4, sticky="w", padx=(8, 4), pady=3)
        preset = ttk.Combobox(controls, textvariable=self.preset, values=DECK_PRESETS, state="readonly", width=18)
        preset.grid(row=1, column=5, sticky="ew", pady=3)
        preset.bind("<<ComboboxSelected>>", lambda _event: self._apply_preset())

        self.run_button = ttk.Button(controls, text="Run", command=self._start)
        self.run_button.grid(row=1, column=6, sticky="ew", padx=(8, 0), pady=3)
        self.stop_button = ttk.Button(controls, text="Stop", command=self._stop, state="disabled")
        self.stop_button.grid(row=1, column=7, sticky="ew", padx=(8, 0), pady=3)
        ttk.Button(controls, text="Standard", command=lambda: self._set_preset(DECK_PRESETS[0])).grid(
            row=1, column=8, sticky="ew", padx=(8, 0), pady=3
        )
        ttk.Button(controls, text="Abandoned", command=lambda: self._set_preset(DECK_PRESETS[1])).grid(
            row=1, column=9, sticky="ew", padx=(8, 0), pady=3
        )
        ttk.Button(controls, text="Checkered", command=lambda: self._set_preset(CHECKERED_DECK_PRESET)).grid(
            row=1, column=10, sticky="ew", padx=(8, 0), pady=3
        )

        deck_frame = ttk.LabelFrame(outer, text="Deck", padding=10)
        deck_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        deck_frame.columnconfigure(0, weight=0)
        for column in range(1, len(RANKS) + 1):
            deck_frame.columnconfigure(column, weight=1)
        for column, rank in enumerate(RANKS, start=1):
            ttk.Label(deck_frame, text=rank, anchor="center").grid(row=0, column=column, sticky="ew", padx=2)
        for row, suit in enumerate(SUITS, start=1):
            ttk.Label(deck_frame, text=suit).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
            for column, _rank in enumerate(RANKS, start=1):
                spinbox = ttk.Spinbox(
                    deck_frame,
                    from_=0,
                    to=20,
                    textvariable=self.deck_vars[row - 1][column - 1],
                    width=4,
                    command=self._refresh_deck_total,
                )
                spinbox.grid(row=row, column=column, sticky="ew", padx=2, pady=2)
                spinbox.bind("<KeyRelease>", lambda _event: self._refresh_deck_total())
        ttk.Label(deck_frame, textvariable=self.deck_total).grid(
            row=len(SUITS) + 1,
            column=0,
            columnspan=len(RANKS) + 1,
            sticky="w",
            pady=(6, 0),
        )

        result_frame = ttk.LabelFrame(outer, text="Probabilities", padding=8)
        result_frame.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)

        self.table = ttk.Treeview(
            result_frame,
            columns=("hand", "opening", "after_discards", "full_blind"),
            show="headings",
            height=14,
        )
        self.table.heading("hand", text="Hand Type")
        self.table.heading("opening", text="Opening")
        self.table.heading("after_discards", text="After Discards")
        self.table.heading("full_blind", text="After Hands")
        self.table.column("hand", width=210, anchor="w")
        self.table.column("opening", width=190, anchor="e")
        self.table.column("after_discards", width=190, anchor="e")
        self.table.column("full_blind", width=190, anchor="e")
        self.table.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns")

        footer = ttk.Frame(outer)
        footer.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status).grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(footer, mode="determinate", maximum=100)
        self.progress.grid(row=0, column=1, sticky="ew", padx=(12, 0))
        footer.columnconfigure(1, weight=1)

    def _set_preset(self, name: str) -> None:
        self.preset.set(name)
        self._apply_preset()

    def _apply_preset(self) -> None:
        counts = deck_preset(self.preset.get())
        for suit, row in enumerate(counts):
            for rank, count in enumerate(row):
                self.deck_vars[suit][rank].set(count)
        self._refresh_deck_total()

    def _refresh_deck_total(self) -> None:
        try:
            total = deck_size(self._deck_counts())
        except ValueError:
            total = 0
        self.deck_total.set(f"{total} cards")

    def _start(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            return
        try:
            config = self._read_config()
        except ValueError as exc:
            messagebox.showerror("Invalid Parameters", str(exc))
            return

        self.stop_event.clear()
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.progress.configure(value=0)
        self.status.set("Running")
        self.worker_thread = threading.Thread(target=self._run_background, args=(config,), daemon=True)
        self.worker_thread.start()

    def _stop(self) -> None:
        self.stop_event.set()
        self.status.set("Stopping")

    def _read_config(self) -> SimulationConfig:
        return SimulationConfig(
            deck_counts=self._deck_counts(),
            hand_size=_int_value(self.hand_size, "Hand size"),
            play_size=_int_value(self.play_size, "Play size"),
            discard_size=_int_value(self.discard_size, "Discard size"),
            discards=_int_value(self.discards, "Discards"),
            hands=_int_value(self.hands, "Hands"),
            trials=_int_value(self.trials, "Trials"),
            seed=_int_value(self.seed, "Seed"),
        )

    def _deck_counts(self):
        rows: list[tuple[int, ...]] = []
        for suit_vars in self.deck_vars:
            row: list[int] = []
            for variable in suit_vars:
                value = _int_value(variable, "Deck card count")
                if value < 0:
                    raise ValueError("Deck card counts cannot be negative.")
                row.append(value)
            rows.append(tuple(row))
        return tuple(rows)

    def _run_background(self, config: SimulationConfig) -> None:
        try:
            result = estimate_hand_type_probabilities(
                config,
                progress=lambda done, total: self.messages.put(("progress", (done, total))),
                should_stop=self.stop_event.is_set,
            )
        except Exception as exc:  # noqa: BLE001 - GUI should surface all failures.
            self.messages.put(("error", str(exc)))
            return
        self.messages.put(("result", result))

    def _poll_messages(self) -> None:
        while True:
            try:
                kind, payload = self.messages.get_nowait()
            except Empty:
                break
            if kind == "progress":
                done, total = payload
                self.progress.configure(value=(done / max(1, total)) * 100)
                self.status.set(f"Running {done:,}/{total:,}")
            elif kind == "result":
                self._show_result(payload)
                self.run_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
            elif kind == "error":
                self.run_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                self.status.set("Error")
                messagebox.showerror("Simulation Failed", str(payload))
        self.root.after(100, self._poll_messages)

    def _show_result(self, result: SimulationResult) -> None:
        self.table.delete(*self.table.get_children())
        for row in result.rows:
            self.table.insert(
                "",
                "end",
                values=(
                    row.hand_type.value,
                    _format_probability(row.opening),
                    _format_probability(row.after_discards),
                    _format_probability(row.after_discards_and_hands),
                ),
            )
        self.progress.configure(value=100)
        self.status.set(
            f"Done: {result.trials:,} trials, {result.deck_size} cards, "
            f"hand {result.hand_size}, play {result.play_size}"
        )


def _spin(
    parent: ttk.Frame,
    label: str,
    variable: IntVar,
    from_: int,
    to: int,
    row: int,
    column: int,
    *,
    increment: int = 1,
) -> None:
    ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(8, 4), pady=3)
    ttk.Spinbox(parent, from_=from_, to=to, increment=increment, textvariable=variable, width=10).grid(
        row=row,
        column=column + 1,
        sticky="ew",
        pady=3,
    )


def _int_value(variable: IntVar, label: str) -> int:
    try:
        return int(variable.get())
    except (TclError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc


def _format_probability(probability: float) -> str:
    if probability <= 0:
        return "0.00%"
    if probability >= 1:
        return "100.00%"
    return f"{probability * 100:.2f}%  1:{1 / probability:.1f}"


def main() -> int:
    root = Tk()
    HandProbabilityApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
