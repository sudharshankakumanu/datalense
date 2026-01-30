"""Multimodal data loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from PIL import Image


def load_images(paths: list[str | Path]) -> list[Image.Image]:
    """Load multiple images from paths.

    Args:
        paths: List of image file paths.

    Returns:
        List of PIL Images.
    """
    images = []
    for path in paths:
        img = Image.open(path)
        img.load()  # Force load into memory
        images.append(img)
    return images


def load_from_directory(
    directory: str | Path,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
    recursive: bool = False,
) -> Iterator[tuple[Path, Image.Image]]:
    """Load images from a directory.

    Args:
        directory: Directory path.
        extensions: File extensions to include.
        recursive: Whether to search subdirectories.

    Yields:
        Tuples of (path, image).
    """
    directory = Path(directory)

    pattern = "**/*" if recursive else "*"

    for path in directory.glob(pattern):
        if path.suffix.lower() in extensions:
            img = Image.open(path)
            img.load()
            yield path, img
