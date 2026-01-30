"""LanceDB client for managing multimodal data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lancedb


class LenseDB:
    """Main database interface for storing and querying multimodal data."""

    def __init__(self, path: str | Path = ".lensedb") -> None:
        """Initialize LenseDB connection.

        Args:
            path: Path to the database directory.
        """
        self.path = Path(path)
        self._db = lancedb.connect(self.path)

    def create_table(
        self,
        name: str,
        data: list[dict[str, Any]] | None = None,
        schema: Any | None = None,
    ) -> lancedb.table.Table:
        """Create a new table for storing embeddings and metadata.

        Args:
            name: Table name.
            data: Initial data to populate the table.
            schema: PyArrow schema for the table.

        Returns:
            The created table.
        """
        if data is not None:
            return self._db.create_table(name, data=data)
        elif schema is not None:
            return self._db.create_table(name, schema=schema)
        else:
            raise ValueError("Either data or schema must be provided")

    def open_table(self, name: str) -> lancedb.table.Table:
        """Open an existing table.

        Args:
            name: Table name.

        Returns:
            The opened table.
        """
        return self._db.open_table(name)

    def list_tables(self) -> list[str]:
        """List all tables in the database.

        Returns:
            List of table names.
        """
        return self._db.table_names()

    def drop_table(self, name: str) -> None:
        """Drop a table from the database.

        Args:
            name: Table name to drop.
        """
        self._db.drop_table(name)
