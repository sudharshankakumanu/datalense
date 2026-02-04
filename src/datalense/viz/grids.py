"""Grid displays for browsing multiple samples."""

from __future__ import annotations

from typing import Sequence

from PIL import Image, ImageDraw, ImageFont


def create_image_grid(
    images: Sequence[Image.Image],
    cols: int = 4,
    labels: Sequence[str] | None = None,
    cell_size: tuple[int, int] = (224, 224),
    padding: int = 4,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    figsize: tuple[int, int] | None = None,
) -> Image.Image:
    """Create a grid of images with optional labels.

    Args:
        images: List of PIL Images.
        cols: Number of columns in the grid.
        labels: Optional labels to display under each image.
        cell_size: Size to resize each image (width, height).
        padding: Padding between images in pixels.
        bg_color: Background color RGB.
        figsize: Optional overall figure size (width, height).

    Returns:
        Combined grid image.
    """
    if not images:
        raise ValueError("No images provided")

    n = len(images)
    rows = (n + cols - 1) // cols

    cell_w, cell_h = cell_size
    label_height = 20 if labels else 0
    grid_w = cols * cell_w + (cols + 1) * padding
    grid_h = rows * (cell_h + label_height) + (rows + 1) * padding

    grid = Image.new("RGB", (grid_w, grid_h), bg_color)
    draw = ImageDraw.Draw(grid)

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols

        # Resize image to fit cell
        img_resized = img.copy()
        if img_resized.mode != "RGB":
            img_resized = img_resized.convert("RGB")
        img_resized.thumbnail(cell_size, Image.Resampling.LANCZOS)

        # Center in cell
        x = padding + col * (cell_w + padding) + (cell_w - img_resized.width) // 2
        y = padding + row * (cell_h + label_height + padding) + (cell_h - img_resized.height) // 2

        grid.paste(img_resized, (x, y))

        # Draw label if provided
        if labels and idx < len(labels):
            label_x = padding + col * (cell_w + padding) + cell_w // 2
            label_y = padding + row * (cell_h + label_height + padding) + cell_h + 2
            draw.text((label_x, label_y), labels[idx], fill=(0, 0, 0), anchor="mt")

    # Resize to figsize if specified
    if figsize:
        grid = grid.resize(figsize, Image.Resampling.LANCZOS)

    return grid


def create_grid(
    images: Sequence[Image.Image],
    cols: int = 4,
    cell_size: tuple[int, int] = (224, 224),
    padding: int = 4,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Create a grid of images for visual browsing.

    Args:
        images: List of PIL Images.
        cols: Number of columns in the grid.
        cell_size: Size to resize each image (width, height).
        padding: Padding between images in pixels.
        bg_color: Background color RGB.

    Returns:
        Combined grid image.
    """
    if not images:
        raise ValueError("No images provided")

    n = len(images)
    rows = (n + cols - 1) // cols

    cell_w, cell_h = cell_size
    grid_w = cols * cell_w + (cols + 1) * padding
    grid_h = rows * cell_h + (rows + 1) * padding

    grid = Image.new("RGB", (grid_w, grid_h), bg_color)

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols

        # Resize image to fit cell
        img_resized = img.copy()
        img_resized.thumbnail(cell_size, Image.Resampling.LANCZOS)

        # Center in cell
        x = padding + col * (cell_w + padding) + (cell_w - img_resized.width) // 2
        y = padding + row * (cell_h + padding) + (cell_h - img_resized.height) // 2

        grid.paste(img_resized, (x, y))

    return grid
