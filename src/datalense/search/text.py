"""Text-based semantic search."""

from __future__ import annotations

from typing import Any, Callable

import lancedb


def search(
    table: lancedb.table.Table,
    query: str,
    embed_fn: Callable[[str], list[float]],
    limit: int = 10,
    filter: str | None = None,
) -> list[dict[str, Any]]:
    """Search using natural language text query.

    Args:
        table: LanceDB table to search.
        query: Natural language search query.
        embed_fn: Function to convert text to embedding vector.
        limit: Maximum number of results to return.
        filter: Optional SQL filter expression.

    Returns:
        List of matching records with distances.
    """
    query_vector = embed_fn(query)

    search_query = table.search(query_vector).limit(limit)

    if filter:
        search_query = search_query.where(filter)

    return search_query.to_list()
