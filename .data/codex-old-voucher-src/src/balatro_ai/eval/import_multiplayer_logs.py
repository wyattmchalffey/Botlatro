"""Extract weak training signals from Lovely multiplayer logs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any, Iterable


INFO_LINE_PATTERN = re.compile(
    r"^INFO - \[[^\]]+\] (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) :: "
    r"(?P<level>\w+)\s+:: (?P<logger>[^:]+?) :: (?P<message>.*)$"
)
CLIENT_SENT_PATTERN = re.compile(r"^Client sent message: (?P<payload>.*)$")
CLIENT_GOT_PATTERN = re.compile(r"^Client got (?P<name>[A-Za-z0-9_]+) message:\s*(?P<pairs>.*)$")
PAREN_PAIR_PATTERN = re.compile(r"\(([^:()]+):\s*(.*?)\)\s*")
SELECTED_CARD_PATTERN = re.compile(
    r"^\((?P<selector>Idol|Mail)\) Selected card (?P<card>.+?) "
    r"with weight (?P<weight>\d+) of total (?P<total>\d+)$"
)

PRIVATE_FIELD_KEYS = frozenset(
    {
        "code",
        "guest",
        "guestHash",
        "host",
        "hostHash",
        "modHash",
        "password",
        "reconnectToken",
        "reconnect_token",
        "token",
        "username",
    }
)
LARGE_ENCODED_KEYS = frozenset({"keys"})
CONNECTION_ONLY_ACTIONS = frozenset(
    {
        "connect",
        "connected",
        "handyMPExtensionLobbyEnabled",
        "joinLobby",
        "joinedLobby",
        "lobbyInfo",
        "syncClient",
        "username",
        "version",
    }
)
SHOP_SIGNAL_ACTIONS = frozenset(
    {
        "boughtCardFromShop",
        "rerollShop",
        "soldCard",
        "soldJoker",
        "spentLastShop",
        "usedCard",
    }
)
LOCAL_SIGNAL_ACTIONS = SHOP_SIGNAL_ACTIONS | frozenset(
    {
        "moneyMoved",
        "newRound",
        "playHand",
        "readyBlind",
        "setAnte",
        "setFurthestBlind",
        "setLocation",
    }
)
TAROT_AND_PLANET_NAMES = frozenset(
    {
        "Fool",
        "The Fool",
        "Magician",
        "The Magician",
        "High Priestess",
        "The High Priestess",
        "Empress",
        "The Empress",
        "Emperor",
        "The Emperor",
        "Hierophant",
        "The Hierophant",
        "Lovers",
        "The Lovers",
        "Chariot",
        "The Chariot",
        "Justice",
        "The Justice",
        "Hermit",
        "The Hermit",
        "Wheel of Fortune",
        "The Wheel of Fortune",
        "Strength",
        "The Strength",
        "Hanged Man",
        "The Hanged Man",
        "Death",
        "The Death",
        "Temperance",
        "The Temperance",
        "Devil",
        "The Devil",
        "Tower",
        "The Tower",
        "Star",
        "The Star",
        "Moon",
        "The Moon",
        "Sun",
        "The Sun",
        "Judgement",
        "The Judgement",
        "World",
        "The World",
        "Mercury",
        "Venus",
        "Earth",
        "Mars",
        "Jupiter",
        "Saturn",
        "Uranus",
        "Neptune",
        "Pluto",
        "Planet X",
        "Ceres",
        "Eris",
    }
)


@dataclass(frozen=True, slots=True)
class MultiplayerLogImportSummary:
    files_scanned: int
    logs_imported: int
    events_written: int
    run_rows_written: int
    replay_summary_rows_written: int
    shop_rows_written: int
    build_rows_written: int
    malformed_payloads: int
    output_dir: Path
    output_files: dict[str, Path]
    observed_wins: int
    observed_losses: int

    def to_text(self) -> str:
        lines = [
            "Multiplayer log import",
            f"Files scanned: {self.files_scanned}",
            f"Logs imported: {self.logs_imported}",
            f"Events written: {self.events_written}",
            f"Run rows written: {self.run_rows_written}",
            f"Replay-compatible summaries: {self.replay_summary_rows_written}",
            f"Shop signal rows: {self.shop_rows_written}",
            f"Endgame build rows: {self.build_rows_written}",
            f"Known local wins/losses: {self.observed_wins}/{self.observed_losses}",
            f"Malformed payloads skipped: {self.malformed_payloads}",
            f"Output directory: {self.output_dir}",
            "Output files:",
        ]
        lines.extend(f"- {name}: {path}" for name, path in sorted(self.output_files.items()))
        return "\n".join(lines)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "files_scanned": self.files_scanned,
            "logs_imported": self.logs_imported,
            "events_written": self.events_written,
            "run_rows_written": self.run_rows_written,
            "replay_summary_rows_written": self.replay_summary_rows_written,
            "shop_rows_written": self.shop_rows_written,
            "build_rows_written": self.build_rows_written,
            "malformed_payloads": self.malformed_payloads,
            "observed_wins": self.observed_wins,
            "observed_losses": self.observed_losses,
            "output_dir": str(self.output_dir),
            "output_files": {name: str(path) for name, path in sorted(self.output_files.items())},
        }


@dataclass(frozen=True, slots=True)
class _ParsedLog:
    events: tuple[dict[str, object], ...]
    malformed_payloads: int
    redacted_fields: Counter[str]
    mod_version_sets: tuple[tuple[str, ...], ...]


def import_multiplayer_logs(
    sources: Iterable[Path],
    *,
    dest: Path,
    player_id: str = "multiplayer",
    overwrite: bool = False,
    min_support: int = 3,
    include_events: bool = True,
) -> MultiplayerLogImportSummary:
    """Convert Lovely multiplayer logs into sanitized weak-signal datasets."""

    source_files = _expand_sources(sources)
    output_dir = _resolve_output_dir(dest, overwrite=overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_events: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    replay_summary_rows: list[dict[str, object]] = []
    shop_rows: list[dict[str, object]] = []
    build_rows: list[dict[str, object]] = []
    malformed_payloads = 0
    redacted_fields: Counter[str] = Counter()
    mod_version_sets: set[tuple[str, ...]] = set()

    for index, source in enumerate(source_files, start=1):
        run_id = _run_id(index, source)
        parsed = _parse_log(source)
        if not parsed.events:
            malformed_payloads += parsed.malformed_payloads
            redacted_fields.update(parsed.redacted_fields)
            mod_version_sets.update(parsed.mod_version_sets)
            continue

        events = _with_source_metadata(parsed.events, run_id=run_id, source=source, player_id=player_id)
        run_row = _build_run_row(events, run_id=run_id, source=source, player_id=player_id)
        run_rows.append(run_row)
        if include_events:
            all_events.extend(events)
        shop_rows.extend(_build_shop_rows(events, run_row))
        build_rows.extend(_build_build_rows(events, run_row))
        replay_row = _build_replay_summary_row(run_row)
        if replay_row is not None:
            replay_summary_rows.append(replay_row)

        malformed_payloads += parsed.malformed_payloads
        redacted_fields.update(parsed.redacted_fields)
        mod_version_sets.update(parsed.mod_version_sets)

    card_stats = _aggregate_card_stats(run_rows, shop_rows, build_rows, min_support=min_support)
    import_summary = _build_import_summary(
        source_files=source_files,
        run_rows=run_rows,
        events=all_events,
        shop_rows=shop_rows,
        build_rows=build_rows,
        replay_summary_rows=replay_summary_rows,
        malformed_payloads=malformed_payloads,
        redacted_fields=redacted_fields,
        mod_version_sets=mod_version_sets,
        card_stats=card_stats,
    )

    output_files = {
        "events": output_dir / "events.jsonl",
        "runs": output_dir / "runs.jsonl",
        "replay_summaries": output_dir / "replay_summaries.jsonl",
        "shop_actions": output_dir / "shop_actions.jsonl",
        "endgame_builds": output_dir / "endgame_builds.jsonl",
        "card_stats": output_dir / "card_stats.json",
        "summary": output_dir / "summary.json",
    }
    _write_jsonl(output_files["events"], all_events)
    _write_jsonl(output_files["runs"], run_rows)
    _write_jsonl(output_files["replay_summaries"], replay_summary_rows)
    _write_jsonl(output_files["shop_actions"], shop_rows)
    _write_jsonl(output_files["endgame_builds"], build_rows)
    _write_json(output_files["card_stats"], card_stats)
    _write_json(output_files["summary"], import_summary)

    return MultiplayerLogImportSummary(
        files_scanned=len(source_files),
        logs_imported=len(run_rows),
        events_written=len(all_events),
        run_rows_written=len(run_rows),
        replay_summary_rows_written=len(replay_summary_rows),
        shop_rows_written=len(shop_rows),
        build_rows_written=len(build_rows),
        malformed_payloads=malformed_payloads,
        output_dir=output_dir,
        output_files=output_files,
        observed_wins=sum(1 for row in run_rows if row.get("won") is True),
        observed_losses=sum(1 for row in run_rows if row.get("won") is False),
    )


def _parse_log(path: Path) -> _ParsedLog:
    events: list[dict[str, object]] = []
    malformed_payloads = 0
    redacted_fields: Counter[str] = Counter()
    mod_version_sets: set[tuple[str, ...]] = set()

    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            match = INFO_LINE_PATTERN.match(line.rstrip("\n"))
            if not match:
                continue
            timestamp = match.group("ts").replace(" ", "T")
            logger = match.group("logger").strip()
            message = match.group("message").strip()

            if logger == "DefaultLogger":
                selected = SELECTED_CARD_PATTERN.match(message)
                if selected:
                    events.append(
                        _event(
                            "stochasticSelection",
                            timestamp,
                            line_number,
                            "stochastic_selection",
                            fields={
                                "selector": selected.group("selector"),
                                "selected_card": selected.group("card"),
                                "weight": int(selected.group("weight")),
                                "total_weight": int(selected.group("total")),
                            },
                        )
                    )
                continue

            if logger != "MULTIPLAYER":
                continue

            sent = CLIENT_SENT_PATTERN.match(message)
            if sent:
                raw_fields, malformed = _parse_sent_payload(sent.group("payload").strip())
                malformed_payloads += int(malformed)
                clean, redactions, versions = _sanitize_fields(raw_fields)
                redacted_fields.update(redactions.keys())
                mod_version_sets.update(versions)
                action = str(raw_fields.get("action") or "sentMessage")
                if _is_connection_only(action, clean):
                    continue
                events.append(_event(action, timestamp, line_number, "multiplayer_message", "sent", clean, redactions))
                continue

            got = CLIENT_GOT_PATTERN.match(message)
            if got:
                raw_fields = _parse_parenthesized_pairs(got.group("pairs"))
                clean, redactions, versions = _sanitize_fields(raw_fields)
                redacted_fields.update(redactions.keys())
                mod_version_sets.update(versions)
                action = str(raw_fields.get("action") or got.group("name"))
                if _is_connection_only(action, clean):
                    continue
                events.append(_event(action, timestamp, line_number, "multiplayer_message", "got", clean, redactions))
                continue

            if message.startswith("Sending end game jokers:"):
                raw = message.split(":", 1)[1].strip()
                events.append(
                    _event(
                        "sendEndGameJokers",
                        timestamp,
                        line_number,
                        "endgame_jokers",
                        "sent",
                        {"jokers_raw": raw, "jokers": _parse_joker_list(raw)},
                    )
                )
                continue

            if message.startswith("Received end game jokers:"):
                raw = message.split(":", 1)[1].strip()
                events.append(
                    _event(
                        "receivedEndGameJokers",
                        timestamp,
                        line_number,
                        "endgame_jokers",
                        "got",
                        {"jokers_raw": raw, "jokers": _parse_joker_list(raw)},
                    )
                )
                continue

            if message == "Toggling Ready":
                events.append(_event("note", timestamp, line_number, "multiplayer_note", fields={"message": message}))

    enriched = _enrich_events(events)
    return _ParsedLog(
        events=tuple(enriched),
        malformed_payloads=malformed_payloads,
        redacted_fields=redacted_fields,
        mod_version_sets=tuple(sorted(mod_version_sets)),
    )


def _parse_sent_payload(payload: str) -> tuple[dict[str, object], bool]:
    if payload.startswith("{"):
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            return {"raw_payload": payload}, True
        if isinstance(decoded, dict):
            return dict(decoded), False
        return {"raw_payload": payload}, True
    return _parse_loose_fields(payload), False


def _parse_loose_fields(payload: str) -> dict[str, object]:
    fields: dict[str, object] = {}
    for part in payload.split(","):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        fields[key.strip()] = _parse_scalar(value)
    return fields


def _parse_parenthesized_pairs(text: str) -> dict[str, object]:
    return {key.strip(): _parse_scalar(value) for key, value in PAREN_PAIR_PATTERN.findall(text)}


def _parse_scalar(value: object) -> object:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return ""
    lower = text.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"nil", "null"}:
        return None
    if re.fullmatch(r"-?\d+", text):
        try:
            return int(text)
        except ValueError:
            return text
    if re.fullmatch(r"-?\d+\.\d+", text):
        try:
            return float(text)
        except ValueError:
            return text
    return text


def _sanitize_fields(
    fields: dict[str, object],
) -> tuple[dict[str, object], dict[str, str], set[tuple[str, ...]]]:
    clean: dict[str, object] = {}
    redactions: dict[str, str] = {}
    mod_version_sets: set[tuple[str, ...]] = set()
    for key, value in fields.items():
        if key in {"modHash", "hostHash", "guestHash"}:
            versions = _mod_versions_from_hash(value)
            if versions:
                mod_version_sets.add(tuple(versions))
        if key in PRIVATE_FIELD_KEYS:
            redactions[key] = "removed"
            continue
        if key in LARGE_ENCODED_KEYS:
            clean[f"{key}_present"] = True
            clean[f"{key}_length"] = len(str(value))
            redactions[key] = "omitted encoded blob"
            continue
        clean[key] = value
    return clean, redactions, mod_version_sets


def _mod_versions_from_hash(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    versions: list[str] = []
    for token in value.split(";"):
        token = token.strip()
        if not token or "=" in token:
            continue
        versions.append(token)
    return versions


def _is_connection_only(action: str, fields: dict[str, object]) -> bool:
    if action not in CONNECTION_ONLY_ACTIONS:
        return False
    useful_keys = set(fields) - {"action", "hostCached", "guestCached", "guestReady", "isCached", "isHost", "version"}
    return not useful_keys


def _event(
    action: str,
    timestamp: str,
    source_line: int,
    category: str,
    direction: str | None = None,
    fields: dict[str, object] | None = None,
    redactions: dict[str, str] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "timestamp": timestamp,
        "source_line": source_line,
        "category": category,
        "action": action,
    }
    if direction is not None:
        row["direction"] = direction
    if fields:
        row["fields"] = fields
    if redactions:
        row["redactions"] = redactions
    return row


def _enrich_events(events: list[dict[str, object]]) -> list[dict[str, object]]:
    location: object = None
    ante: object = None
    furthest_blind: object = None
    enriched: list[dict[str, object]] = []
    for row in events:
        fields = row.get("fields")
        if not isinstance(fields, dict):
            fields = {}
        if row.get("direction") == "sent" and row.get("action") == "setLocation":
            location = fields.get("location")
        if row.get("direction") == "sent" and row.get("action") == "setAnte":
            ante = fields.get("ante")
        if row.get("direction") == "sent" and row.get("action") == "setFurthestBlind":
            furthest_blind = fields.get("furthestBlind")
        context = {
            key: value
            for key, value in (
                ("local_location", location),
                ("local_ante", ante),
                ("local_furthest_blind", furthest_blind),
            )
            if value is not None
        }
        if context:
            row = dict(row)
            row["context"] = context
        if row.get("action") == "receiveNemesisDeck" and isinstance(fields.get("cards"), str):
            fields = dict(fields)
            fields["cards_parsed"] = _parse_card_list(str(fields["cards"]))
            fields["card_count"] = len(fields["cards_parsed"])
            row = dict(row)
            row["fields"] = fields
        enriched.append(row)
    return enriched


def _with_source_metadata(
    events: tuple[dict[str, object], ...],
    *,
    run_id: str,
    source: Path,
    player_id: str,
) -> list[dict[str, object]]:
    dataset_source = _dataset_source(player_id=player_id, source=source)
    rows: list[dict[str, object]] = []
    for event in events:
        row = dict(event)
        row["run_id"] = run_id
        row["source_file"] = source.name
        row["dataset_source"] = dict(dataset_source)
        rows.append(row)
    return rows


def _build_run_row(
    events: list[dict[str, object]],
    *,
    run_id: str,
    source: Path,
    player_id: str,
) -> dict[str, object]:
    action_counts = Counter(str(row.get("action")) for row in events)
    timestamps = [str(row.get("timestamp")) for row in events if row.get("timestamp")]
    start_timestamp = min(timestamps) if timestamps else None
    end_timestamp = max(timestamps) if timestamps else None
    duration_seconds = _duration_seconds(start_timestamp, end_timestamp)
    latest_lobby = _latest_fields(events, "lobbyOptions")
    start_game = _latest_fields(events, "startGame")
    result = _last_result(events)
    won = True if result == "winGame" else False if result == "loseGame" else None
    source_seed = _source_seed(events) or _optional_string(latest_lobby.get("custom_seed")) or ""
    numeric_seed = _stable_numeric_seed(source_seed or source.name, source.name)
    local_play_scores = [_int(row.get("fields", {}).get("score")) for row in events if _is_local_action(row, "playHand")]
    money_moves = [_number(row.get("fields", {}).get("amount")) for row in events if _is_local_action(row, "moneyMoved")]
    money_moves = [value for value in money_moves if value is not None]
    reroll_costs = [_number(row.get("fields", {}).get("cost")) for row in events if _is_local_action(row, "rerollShop")]
    reroll_costs = [value for value in reroll_costs if value is not None]
    tarot_usage = _used_tarots(events)

    row: dict[str, object] = {
        "record_type": "multiplayer_log_summary",
        "run_id": run_id,
        "source_file": source.name,
        "dataset_source": _dataset_source(player_id=player_id, source=source),
        "imported_by": "balatro_ai.eval.import_multiplayer_logs",
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "runtime_seconds": duration_seconds,
        "result": result,
        "won": won,
        "seed": numeric_seed,
        "source_seed": source_seed,
        "max_ante": _max_field(events, "setAnte", "ante", direction="sent"),
        "max_furthest_blind": _max_field(events, "setFurthestBlind", "furthestBlind", direction="sent"),
        "final_score": local_play_scores[-1] if local_play_scores else 0,
        "max_score": max(local_play_scores) if local_play_scores else 0,
        "stake": _stake_label(latest_lobby.get("stake")),
        "deck": _optional_string(latest_lobby.get("back")) or _optional_string(start_game.get("deck")) or "unknown",
        "match_setup": {
            "latest_lobby_options": latest_lobby,
            "start_game": start_game,
            "mod_version_sets": _mod_version_sets(events),
        },
        "economy": {
            "money_income_total": sum(value for value in money_moves if value > 0),
            "money_spent_total": -sum(value for value in money_moves if value < 0),
            "money_observed_net": sum(money_moves),
            "money_spent_rerolls": sum(reroll_costs),
            "reroll_count": sum(1 for row_event in events if _is_local_action(row_event, "rerollShop")),
        },
        "tarot_usage": tarot_usage,
        "action_counts": dict(sorted(action_counts.items())),
        "local_final_jokers": _local_final_jokers(events),
        "opponent_final_jokers": _opponent_final_jokers(events),
        "local_final_deck_card_count": _final_deck_count(events, direction="sent"),
        "opponent_final_deck_card_count": _final_deck_count(events, direction="got"),
    }
    return row


def _build_replay_summary_row(run_row: dict[str, object]) -> dict[str, object] | None:
    won = run_row.get("won")
    if won not in {True, False}:
        return None
    economy = run_row.get("economy") if isinstance(run_row.get("economy"), dict) else {}
    return {
        "record_type": "run_summary",
        "bot_version": "external_multiplayer_log",
        "seed": run_row.get("seed", 0),
        "source_seed": run_row.get("source_seed", ""),
        "stake": run_row.get("stake", "unknown"),
        "deck": run_row.get("deck", "unknown"),
        "won": won,
        "outcome": "win" if won is True else "loss",
        "ante": run_row.get("max_ante", 0),
        "final_score": run_row.get("final_score", 0),
        "final_money": _int(economy.get("money_observed_net")),
        "runtime_seconds": run_row.get("runtime_seconds", 0),
        "death_reason": None if won is True else "multiplayer_loss",
        "economy": economy,
        "tarot_usage": run_row.get("tarot_usage", {}),
        "dataset_source": run_row.get("dataset_source", {}),
        "source_record_type": "multiplayer_log_summary",
        "source_file": run_row.get("source_file", ""),
        "run_id": run_row.get("run_id", ""),
    }


def _build_shop_rows(events: list[dict[str, object]], run_row: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in events:
        action = str(event.get("action"))
        if action not in SHOP_SIGNAL_ACTIONS or event.get("direction") != "sent":
            continue
        fields = event.get("fields") if isinstance(event.get("fields"), dict) else {}
        context = event.get("context") if isinstance(event.get("context"), dict) else {}
        item = fields.get("card") if action in {"boughtCardFromShop", "soldCard", "usedCard"} else None
        row = {
            "record_type": "multiplayer_shop_signal",
            "run_id": run_row.get("run_id"),
            "source_file": run_row.get("source_file"),
            "source_line": event.get("source_line"),
            "timestamp": event.get("timestamp"),
            "action": action,
            "item": item,
            "cost": fields.get("cost"),
            "amount": fields.get("amount"),
            "local_location": context.get("local_location"),
            "local_ante": context.get("local_ante"),
            "known_result": run_row.get("won") in {True, False},
            "won": run_row.get("won"),
            "max_ante": run_row.get("max_ante"),
            "max_furthest_blind": run_row.get("max_furthest_blind"),
            "source_seed": run_row.get("source_seed"),
            "dataset_source": run_row.get("dataset_source"),
        }
        rows.append({key: value for key, value in row.items() if value is not None})
    return rows


def _build_build_rows(events: list[dict[str, object]], run_row: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    local_won = run_row.get("won")
    for perspective, direction in (("local", "sent"), ("opponent", "got")):
        jokers = _build_jokers_for_perspective(events, perspective)
        deck = _build_deck_for_direction(events, direction)
        if not jokers and not deck:
            continue
        won = local_won if perspective == "local" else (not local_won if local_won in {True, False} else None)
        row = {
            "record_type": "multiplayer_endgame_build",
            "run_id": run_row.get("run_id"),
            "source_file": run_row.get("source_file"),
            "perspective": perspective,
            "won": won,
            "source_seed": run_row.get("source_seed"),
            "max_ante": run_row.get("max_ante"),
            "max_furthest_blind": run_row.get("max_furthest_blind"),
            "jokers": jokers,
            "deck_cards": deck,
            "deck_card_count": len(deck),
            "dataset_source": run_row.get("dataset_source"),
        }
        rows.append(row)
    return rows


def _aggregate_card_stats(
    run_rows: list[dict[str, object]],
    shop_rows: list[dict[str, object]],
    build_rows: list[dict[str, object]],
    *,
    min_support: int,
) -> dict[str, object]:
    run_by_id = {str(row.get("run_id")): row for row in run_rows}
    buckets: dict[tuple[str, str], dict[str, object]] = defaultdict(
        lambda: {"occurrences": 0, "run_ids": set(), "wins": set(), "losses": set(), "unknown": set(), "antes": []}
    )

    for row in shop_rows:
        item = row.get("item")
        if not item:
            continue
        _add_item_observation(buckets, str(row.get("action")), str(item), str(row.get("run_id")), run_by_id)

    for row in build_rows:
        if row.get("perspective") != "local":
            continue
        run_id = str(row.get("run_id"))
        for joker in row.get("jokers", ()):
            if isinstance(joker, dict):
                key = str(joker.get("key") or joker.get("raw") or "")
                if key:
                    _add_item_observation(buckets, "final_joker", key, run_id, run_by_id)

    known_runs = [row for row in run_rows if row.get("won") in {True, False}]
    baseline = sum(1 for row in known_runs if row.get("won") is True) / len(known_runs) if known_runs else 0.0
    by_action: dict[str, dict[str, object]] = defaultdict(dict)
    signal_rows: list[dict[str, object]] = []
    for (action, item), bucket in sorted(buckets.items()):
        run_ids = bucket["run_ids"]
        wins = bucket["wins"]
        losses = bucket["losses"]
        unknown = bucket["unknown"]
        observed = len(wins) + len(losses)
        win_rate = len(wins) / observed if observed else None
        antes = bucket["antes"]
        payload = {
            "occurrences": bucket["occurrences"],
            "run_count": len(run_ids),
            "known_result_runs": observed,
            "wins": len(wins),
            "losses": len(losses),
            "unknown_result_runs": len(unknown),
            "win_rate": win_rate,
            "lift_vs_baseline": (win_rate - baseline) if win_rate is not None else None,
            "avg_max_ante": mean(antes) if antes else 0.0,
        }
        by_action[action][item] = payload
        if observed >= min_support and win_rate is not None:
            signal = {"action": action, "item": item, **payload}
            signal_rows.append(signal)

    signal_rows.sort(key=lambda row: (float(row.get("lift_vs_baseline") or 0.0), int(row["known_result_runs"])), reverse=True)
    negative_rows = sorted(
        signal_rows,
        key=lambda row: (float(row.get("lift_vs_baseline") or 0.0), -int(row["known_result_runs"])),
    )
    return {
        "baseline": {
            "known_result_runs": len(known_runs),
            "wins": sum(1 for row in known_runs if row.get("won") is True),
            "losses": sum(1 for row in known_runs if row.get("won") is False),
            "win_rate": baseline,
            "min_support": min_support,
        },
        "by_action": {action: dict(items) for action, items in sorted(by_action.items())},
        "top_positive_item_signals": signal_rows[:25],
        "top_negative_item_signals": negative_rows[:25],
    }


def _add_item_observation(
    buckets: dict[tuple[str, str], dict[str, object]],
    action: str,
    item: str,
    run_id: str,
    run_by_id: dict[str, dict[str, object]],
) -> None:
    bucket = buckets[(action, item)]
    bucket["occurrences"] = int(bucket["occurrences"]) + 1
    run_ids = bucket["run_ids"]
    if not isinstance(run_ids, set):
        return
    if run_id in run_ids:
        return
    run_ids.add(run_id)
    run = run_by_id.get(run_id, {})
    won = run.get("won")
    if won is True:
        cast_set(bucket["wins"]).add(run_id)
    elif won is False:
        cast_set(bucket["losses"]).add(run_id)
    else:
        cast_set(bucket["unknown"]).add(run_id)
    cast_list(bucket["antes"]).append(_int(run.get("max_ante")))


def cast_set(value: object) -> set[str]:
    return value if isinstance(value, set) else set()


def cast_list(value: object) -> list[int]:
    return value if isinstance(value, list) else []


def _build_import_summary(
    *,
    source_files: tuple[Path, ...],
    run_rows: list[dict[str, object]],
    events: list[dict[str, object]],
    shop_rows: list[dict[str, object]],
    build_rows: list[dict[str, object]],
    replay_summary_rows: list[dict[str, object]],
    malformed_payloads: int,
    redacted_fields: Counter[str],
    mod_version_sets: set[tuple[str, ...]],
    card_stats: dict[str, object],
) -> dict[str, object]:
    known_rows = [row for row in run_rows if row.get("won") in {True, False}]
    wins = sum(1 for row in known_rows if row.get("won") is True)
    losses = sum(1 for row in known_rows if row.get("won") is False)
    antes = [_int(row.get("max_ante")) for row in run_rows if _int(row.get("max_ante")) > 0]
    return {
        "record_type": "multiplayer_log_import_summary",
        "files_scanned": len(source_files),
        "logs_imported": len(run_rows),
        "events_written": len(events),
        "run_rows_written": len(run_rows),
        "replay_summary_rows_written": len(replay_summary_rows),
        "shop_rows_written": len(shop_rows),
        "build_rows_written": len(build_rows),
        "malformed_payloads": malformed_payloads,
        "known_result_runs": len(known_rows),
        "observed_wins": wins,
        "observed_losses": losses,
        "observed_win_rate": wins / len(known_rows) if known_rows else None,
        "average_max_ante": mean(antes) if antes else 0.0,
        "redacted_fields": dict(sorted(redacted_fields.items())),
        "sanitized_mod_version_sets": [list(version_set) for version_set in sorted(mod_version_sets)],
        "top_positive_item_signals": card_stats.get("top_positive_item_signals", []),
        "top_negative_item_signals": card_stats.get("top_negative_item_signals", []),
        "privacy_notes": [
            "Only source filenames are stored; full local source paths are omitted.",
            "Usernames, lobby codes, reconnect tokens, mod hashes, and encoded endgame blobs are removed.",
            "This is weak multiplayer telemetry, not full state-action replay data.",
        ],
    }


def _expand_sources(sources: Iterable[Path]) -> tuple[Path, ...]:
    files: list[Path] = []
    for source in sources:
        if source.is_dir():
            files.extend(sorted(path for path in source.rglob("*.log") if path.is_file()))
        elif source.is_file():
            files.append(source)
    return tuple(files)


def _resolve_output_dir(dest: Path, *, overwrite: bool) -> Path:
    output_names = {
        "events.jsonl",
        "runs.jsonl",
        "replay_summaries.jsonl",
        "shop_actions.jsonl",
        "endgame_builds.jsonl",
        "card_stats.json",
        "summary.json",
    }
    if overwrite:
        return dest
    if not dest.exists() or not any((dest / name).exists() for name in output_names):
        return dest
    for suffix in range(2, 10000):
        candidate = dest.with_name(f"{dest.name}_{suffix}")
        if not candidate.exists() or not any((candidate / name).exists() for name in output_names):
            return candidate
    raise FileExistsError(f"Could not find a free output directory for {dest}")


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write("\n")


def _dataset_source(*, player_id: str, source: Path) -> dict[str, object]:
    return {
        "player_id": player_id,
        "source_file": source.name,
        "imported_by": "balatro_ai.eval.import_multiplayer_logs",
    }


def _run_id(index: int, source: Path) -> str:
    return f"{index:05d}_{_safe_name(source.stem)}"


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "log"


def _stable_numeric_seed(seed: object, source_name: str) -> int:
    digest = hashlib.sha256(f"{seed}|{source_name}".encode("utf-8", errors="replace")).hexdigest()
    return int(digest[:12], 16) % 2_000_000_000


def _duration_seconds(start: str | None, end: str | None) -> float:
    if not start or not end:
        return 0.0
    try:
        return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    except ValueError:
        return 0.0


def _latest_fields(events: list[dict[str, object]], action: str) -> dict[str, object]:
    for row in reversed(events):
        if row.get("action") != action:
            continue
        fields = row.get("fields")
        return dict(fields) if isinstance(fields, dict) else {}
    return {}


def _last_result(events: list[dict[str, object]]) -> str | None:
    result = None
    for row in events:
        if row.get("action") in {"winGame", "loseGame"}:
            result = str(row.get("action"))
    return result


def _source_seed(events: list[dict[str, object]]) -> str | None:
    for row in reversed(events):
        fields = row.get("fields")
        if isinstance(fields, dict) and fields.get("seed"):
            return str(fields["seed"])
    return None


def _max_field(events: list[dict[str, object]], action: str, key: str, *, direction: str | None = None) -> int:
    values: list[int] = []
    for row in events:
        if row.get("action") != action:
            continue
        if direction is not None and row.get("direction") != direction:
            continue
        fields = row.get("fields")
        if isinstance(fields, dict):
            values.append(_int(fields.get(key)))
    return max(values) if values else 0


def _is_local_action(row: dict[str, object], action: str) -> bool:
    return row.get("action") == action and row.get("direction") == "sent"


def _number(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _stake_label(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, int):
        return f"stake_{value}"
    text = str(value)
    return text if text else "unknown"


def _used_tarots(events: list[dict[str, object]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in events:
        if not _is_local_action(row, "usedCard"):
            continue
        fields = row.get("fields")
        card = fields.get("card") if isinstance(fields, dict) else None
        if isinstance(card, str) and (card in TAROT_AND_PLANET_NAMES or card.startswith("The ")):
            counter[card] += 1
    return dict(sorted(counter.items()))


def _mod_version_sets(events: list[dict[str, object]]) -> list[list[str]]:
    sets: set[tuple[str, ...]] = set()
    for row in events:
        fields = row.get("fields")
        if isinstance(fields, dict):
            versions = fields.get("mod_versions")
            if isinstance(versions, list):
                sets.add(tuple(str(version) for version in versions))
    return [list(version_set) for version_set in sorted(sets)]


def _local_final_jokers(events: list[dict[str, object]]) -> list[dict[str, object]]:
    for row in reversed(events):
        if row.get("action") == "sendEndGameJokers":
            fields = row.get("fields")
            jokers = fields.get("jokers") if isinstance(fields, dict) else None
            return list(jokers) if isinstance(jokers, list) else []
    return []


def _opponent_final_jokers(events: list[dict[str, object]]) -> list[dict[str, object]]:
    for row in events:
        if row.get("action") == "receivedEndGameJokers":
            fields = row.get("fields")
            jokers = fields.get("jokers") if isinstance(fields, dict) else None
            return list(jokers) if isinstance(jokers, list) else []
    return []


def _final_deck_count(events: list[dict[str, object]], *, direction: str) -> int:
    deck = _build_deck_for_direction(events, direction)
    return len(deck)


def _build_jokers_for_perspective(events: list[dict[str, object]], perspective: str) -> list[dict[str, object]]:
    if perspective == "local":
        return _local_final_jokers(events)
    return _opponent_final_jokers(events)


def _build_deck_for_direction(events: list[dict[str, object]], direction: str) -> list[dict[str, object]]:
    for row in reversed(events):
        if row.get("action") == "receiveNemesisDeck" and row.get("direction") == direction:
            fields = row.get("fields")
            if isinstance(fields, dict) and isinstance(fields.get("cards_parsed"), list):
                return list(fields["cards_parsed"])
    return []


def _parse_joker_list(raw: str) -> list[dict[str, object]]:
    jokers: list[dict[str, object]] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split("-")
        row: dict[str, object] = {"raw": item, "key": parts[0] if parts else item}
        if len(parts) > 1:
            row["edition"] = _none_if(parts[1])
        if len(parts) > 2:
            row["enhancement"] = _none_if(parts[2])
        if len(parts) > 3:
            row["seal"] = _none_if(parts[3])
        jokers.append(row)
    return jokers


def _parse_card_list(raw: str) -> list[dict[str, object]]:
    suits = {"S": "Spades", "H": "Hearts", "C": "Clubs", "D": "Diamonds"}
    ranks = {"A": "Ace", "K": "King", "Q": "Queen", "J": "Jack", "T": "10"}
    cards: list[dict[str, object]] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split("-")
        row: dict[str, object] = {"raw": item}
        if len(parts) >= 2:
            suit = suits.get(parts[0], parts[0])
            rank = ranks.get(parts[1], parts[1])
            row.update({"suit": suit, "rank": rank, "name": f"{rank} of {suit}"})
        if len(parts) > 2:
            row["enhancement"] = _none_if(parts[2])
        if len(parts) > 3:
            row["edition"] = _none_if(parts[3])
        if len(parts) > 4:
            row["seal"] = _none_if(parts[4])
        cards.append(row)
    return cards


def _none_if(value: str) -> str | None:
    return None if value in {"", "nil", "none", "null"} else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract useful weak-signal data from Lovely multiplayer logs.")
    parser.add_argument("--source", nargs="+", type=Path, required=True, help="Source .log file(s) or directories.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(".data") / "multiplayer-log-dataset",
        help="Output directory for extracted JSONL/JSON files.",
    )
    parser.add_argument("--player-id", default="multiplayer", help="Dataset player/source identifier.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output files in --dest.")
    parser.add_argument("--min-support", type=int, default=3, help="Minimum known-result runs for item signal lists.")
    parser.add_argument("--no-events", action="store_true", help="Write summaries/stats without the large events.jsonl stream.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable import summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = import_multiplayer_logs(
        args.source,
        dest=args.dest,
        player_id=args.player_id,
        overwrite=args.overwrite,
        min_support=max(1, args.min_support),
        include_events=not args.no_events,
    )
    if args.json:
        print(json.dumps(summary.to_json_dict(), indent=2, sort_keys=True))
    else:
        print(summary.to_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
