"""Nearest neighbor search using vector embeddings."""

from __future__ import annotations

from typing import Any

import lancedb


def search(
    table: lancedb.table.Table,
    query_vector: list[float],
    limit: int = 10,
    filter: str | None = None,
) -> list[dict[str, Any]]:
    """Find nearest neighbors by vector similarity.

    Args:
        table: LanceDB table to search.
        query_vector: Query embedding vector.
        limit: Maximum number of results to return.
        filter: Optional SQL filter expression.

    Returns:
        List of matching records with distances.
    """
    query = table.search(query_vector).limit(limit)

    if filter:
        query = query.where(filter)

    return query.to_list()


def search_by_id(
    table: lancedb.table.Table,
    id: str,
    limit: int = 10,
    filter: str | None = None,
) -> list[dict[str, Any]]:
    """Find nearest neighbors to an existing record.

    Args:
        table: LanceDB table to search.
        id: ID of the record to find neighbors for.
        limit: Maximum number of results to return.
        filter: Optional SQL filter expression.

    Returns:
        List of matching records with distances.
    """
    # Get the vector for the given ID
    results = table.search().where(f"id = '{id}'").limit(1).to_list()

    if not results:
        raise ValueError(f"No record found with id: {id}")

    query_vector = results[0]["vector"]
    return search(table, query_vector, limit=limit, filter=filter)
