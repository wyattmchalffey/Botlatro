"""Bot lookup helpers for CLI commands."""

from __future__ import annotations

from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot
from balatro_ai.bots.base import Bot
from balatro_ai.bots.greedy_bot import GreedyBot
from balatro_ai.bots.random_bot import RandomBot
from balatro_ai.bots.search_bot import SearchBot
from balatro_ai.search.shop_search import ShopSearchConfig


def create_bot(name: str, seed: int | None = None) -> Bot:
    normalized = name.lower().replace("-", "_")
    if normalized == "random_bot":
        return RandomBot(seed=seed)
    if normalized == "greedy_bot":
        return GreedyBot(seed=seed)
    if normalized in {"basic_strategy_bot", "basic_bot", "rule_bot"}:
        return BasicStrategyBot(seed=seed)
    if normalized in {"search_bot", "search_bot_v0"}:
        return SearchBot(seed=seed)
    if normalized in {"search_bot_v1", "shop_search_bot"}:
        return SearchBot(seed=seed, enable_shop_search=True, name="search_bot_v1")
    if normalized in {"search_bot_v1_trace", "shop_search_bot_trace"}:
        return SearchBot(
            seed=seed,
            enable_shop_search=True,
            shop_config=ShopSearchConfig(seed=seed or 0, reroll_samples=8, trace_top_paths=8),
            name="search_bot_v1_trace",
        )
    raise ValueError(f"Unknown bot: {name}")
