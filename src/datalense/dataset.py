"""Core Dataset class for DataLense."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Sequence

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}


class DatasetItem:
    """A single item in a Dataset."""

    def __init__(
        self,
        id: str,
        path: str | Path | None = None,
        image: Image.Image | None = None,
        embedding: NDArray[np.float32] | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.id = id
        self.path = Path(path) if path else None
        self._image = image
        self.embedding = embedding
        self.metadata = metadata or {}

    @property
    def image(self) -> Image.Image:
        """Lazy-load image from path if not already loaded."""
        if self._image is None:
            if self.path is None:
                raise ValueError(f"Item {self.id} has no path or image")
            self._image = Image.open(self.path)
        return self._image

    def show(self) -> Image.Image:
        """Display the image."""
        img = self.image
        img.show()
        return img

    def __repr__(self) -> str:
        return f"DatasetItem(id={self.id!r}, path={self.path})"


class SearchResults:
    """Results from a search query."""

    def __init__(self, items: list[DatasetItem], scores: list[float] | None = None) -> None:
        self.items = items
        self.scores = scores or []

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> DatasetItem:
        return self.items[idx]

    def __iter__(self) -> Iterator[DatasetItem]:
        return iter(self.items)

    def show(self, cols: int = 4, figsize: tuple[int, int] | None = None) -> Image.Image:
        """Display results as a grid."""
        from datalense.viz.grids import create_image_grid

        images = [item.image for item in self.items]
        labels = [f"{s:.3f}" for s in self.scores] if self.scores else None
        return create_image_grid(images, cols=cols, labels=labels, figsize=figsize)

    def __repr__(self) -> str:
        return f"SearchResults({len(self)} items)"


class Dataset:
    """
    A collection of multimodal data that can be indexed, searched, and visualized.

    Examples
    --------
    >>> import datalense as dl
    >>> dataset = dl.Dataset.from_folder("./images/")
    >>> dataset.index()
    >>> results = dataset.search("a cat", k=10)
    >>> results.show()
    """

    def __init__(self, items: list[DatasetItem] | None = None) -> None:
        self._items: list[DatasetItem] = items or []
        self._indexed = False
        self._db = None  # LanceDB table, set after indexing

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> DatasetItem:
        return self._items[idx]

    def __iter__(self) -> Iterator[DatasetItem]:
        return iter(self._items)

    @classmethod
    def from_folder(
        cls,
        path: str | Path,
        recursive: bool = True,
        extensions: set[str] | None = None,
    ) -> Dataset:
        """
        Create a Dataset from a folder of images.

        Parameters
        ----------
        path : str | Path
            Path to folder containing images.
        recursive : bool
            If True, search subdirectories recursively.
        extensions : set[str] | None
            File extensions to include. Defaults to common image formats.

        Returns
        -------
        Dataset
            A new Dataset containing the images.

        Examples
        --------
        >>> dataset = Dataset.from_folder("./my_images/")
        >>> print(f"Loaded {len(dataset)} images")
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Folder not found: {path}")
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        extensions = extensions or IMAGE_EXTENSIONS
        extensions = {ext.lower() for ext in extensions}

        items: list[DatasetItem] = []
        pattern = "**/*" if recursive else "*"

        for file_path in sorted(path.glob(pattern)):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                item_id = str(file_path.relative_to(path))
                items.append(
                    DatasetItem(
                        id=item_id,
                        path=file_path,
                        metadata={"filename": file_path.name},
                    )
                )

        return cls(items)

    @classmethod
    def from_dataframe(
        cls,
        df,
        image_col: str = "image",
        embedding_col: str | None = None,
        id_col: str | None = None,
    ) -> Dataset:
        """
        Create a Dataset from a pandas DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing image paths or data.
        image_col : str
            Column containing image paths or PIL Images.
        embedding_col : str | None
            Column containing pre-computed embeddings.
        id_col : str | None
            Column to use as item IDs. Defaults to DataFrame index.

        Returns
        -------
        Dataset
            A new Dataset.
        """
        items: list[DatasetItem] = []

        for idx, row in df.iterrows():
            item_id = str(row[id_col]) if id_col else str(idx)
            image_val = row[image_col]

            # Handle different image column types
            if isinstance(image_val, (str, Path)):
                path = Path(image_val)
                image = None
            elif isinstance(image_val, Image.Image):
                path = None
                image = image_val
            else:
                raise ValueError(f"Unsupported image type: {type(image_val)}")

            embedding = None
            if embedding_col and embedding_col in row:
                embedding = np.array(row[embedding_col], dtype=np.float32)

            # Collect other columns as metadata
            metadata = {
                k: v
                for k, v in row.items()
                if k not in {image_col, embedding_col, id_col}
            }

            items.append(
                DatasetItem(
                    id=item_id,
                    path=path,
                    image=image,
                    embedding=embedding,
                    metadata=metadata,
                )
            )

        return cls(items)

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        image_col: str = "image",
        embedding_col: str | None = None,
        id_col: str | None = None,
    ) -> Dataset:
        """
        Create a Dataset from a Parquet file.

        Parameters
        ----------
        path : str | Path
            Path to Parquet file.
        image_col : str
            Column containing image paths.
        embedding_col : str | None
            Column containing pre-computed embeddings.
        id_col : str | None
            Column to use as item IDs.

        Returns
        -------
        Dataset
            A new Dataset.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for from_parquet()")

        df = pd.read_parquet(path)
        return cls.from_dataframe(
            df,
            image_col=image_col,
            embedding_col=embedding_col,
            id_col=id_col,
        )

    def index(
        self,
        db_path: str | Path | None = None,
        embedding_fn=None,
    ) -> Dataset:
        """
        Index the dataset for fast search.

        This computes embeddings (if not already present) and stores them
        in a LanceDB database for fast nearest neighbor search.

        Parameters
        ----------
        db_path : str | Path | None
            Path to store the LanceDB database. Defaults to a temp directory.
        embedding_fn : callable | None
            Function to compute embeddings. Signature: (images) -> embeddings.
            If None, uses a default CLIP model.

        Returns
        -------
        Dataset
            Self, for method chaining.
        """
        try:
            import lancedb
        except ImportError:
            raise ImportError(
                "lancedb is required for indexing. "
                "Install with: pip install datalense[db]"
            )

        from datalense.db.client import create_table, connect

        # Compute embeddings for items that don't have them
        items_needing_embeddings = [
            item for item in self._items if item.embedding is None
        ]

        if items_needing_embeddings:
            if embedding_fn is None:
                raise ValueError(
                    "Some items don't have embeddings. "
                    "Provide an embedding_fn or pre-compute embeddings."
                )
            images = [item.image for item in items_needing_embeddings]
            embeddings = embedding_fn(images)
            for item, emb in zip(items_needing_embeddings, embeddings):
                item.embedding = np.array(emb, dtype=np.float32)

        # Create LanceDB table
        db = connect(db_path)
        self._db = create_table(db, "dataset", self._items)
        self._indexed = True

        return self

    def search(self, query: str, k: int = 10) -> SearchResults:
        """
        Search for items using a text query.

        Parameters
        ----------
        query : str
            Text query to search for.
        k : int
            Number of results to return.

        Returns
        -------
        SearchResults
            Matching items with similarity scores.
        """
        if not self._indexed:
            raise RuntimeError("Dataset not indexed. Call .index() first.")

        from datalense.search.text import text_search

        return text_search(self._db, self._items, query, k=k)

    def find_similar(
        self,
        query: DatasetItem | NDArray[np.float32] | Image.Image,
        k: int = 10,
    ) -> SearchResults:
        """
        Find items similar to a query image or embedding.

        Parameters
        ----------
        query : DatasetItem | NDArray | Image
            Query item, embedding vector, or image.
        k : int
            Number of results to return.

        Returns
        -------
        SearchResults
            Similar items with similarity scores.
        """
        if not self._indexed:
            raise RuntimeError("Dataset not indexed. Call .index() first.")

        from datalense.search.nearest import nearest_neighbor_search

        return nearest_neighbor_search(self._db, self._items, query, k=k)

    def sample(self, n: int = 10, seed: int | None = None) -> SearchResults:
        """
        Return a random sample of items.

        Parameters
        ----------
        n : int
            Number of items to sample.
        seed : int | None
            Random seed for reproducibility.

        Returns
        -------
        SearchResults
            Sampled items.
        """
        rng = np.random.default_rng(seed)
        n = min(n, len(self._items))
        indices = rng.choice(len(self._items), size=n, replace=False)
        items = [self._items[i] for i in indices]
        return SearchResults(items)

    def show(self, n: int = 16, cols: int = 4) -> Image.Image:
        """
        Display a sample of images from the dataset.

        Parameters
        ----------
        n : int
            Number of images to show.
        cols : int
            Number of columns in the grid.

        Returns
        -------
        Image
            Grid image.
        """
        return self.sample(n).show(cols=cols)

    def __repr__(self) -> str:
        status = "indexed" if self._indexed else "not indexed"
        return f"Dataset({len(self)} items, {status})"
