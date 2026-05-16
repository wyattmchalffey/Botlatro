"""General small utility helpers for the basic strategy bot."""

from __future__ import annotations


def _int_or_default(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
