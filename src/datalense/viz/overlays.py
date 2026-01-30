"""Visual overlays for images - bounding boxes, masks, keypoints."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw


def draw_boxes(
    image: Image.Image,
    boxes: Sequence[tuple[float, float, float, float]],
    labels: Sequence[str] | None = None,
    scores: Sequence[float] | None = None,
    color: str = "red",
    width: int = 2,
) -> Image.Image:
    """Draw bounding boxes on an image.

    Args:
        image: PIL Image to draw on.
        boxes: List of boxes as (x1, y1, x2, y2).
        labels: Optional labels for each box.
        scores: Optional confidence scores for each box.
        color: Box color.
        width: Line width.

    Returns:
        Image with boxes drawn.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for i, box in enumerate(boxes):
        draw.rectangle(box, outline=color, width=width)

        if labels or scores:
            text_parts = []
            if labels and i < len(labels):
                text_parts.append(labels[i])
            if scores and i < len(scores):
                text_parts.append(f"{scores[i]:.2f}")

            if text_parts:
                text = " ".join(text_parts)
                draw.text((box[0], box[1] - 12), text, fill=color)

    return img


def draw_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
) -> Image.Image:
    """Overlay a segmentation mask on an image.

    Args:
        image: PIL Image to draw on.
        mask: Binary mask array (H, W).
        color: RGB color for the mask.
        alpha: Transparency (0-1).

    Returns:
        Image with mask overlay.
    """
    img = image.copy().convert("RGBA")

    mask_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
    mask_array = np.array(mask_img)

    mask_bool = mask.astype(bool)
    mask_array[mask_bool] = (*color, int(255 * alpha))

    mask_img = Image.fromarray(mask_array)
    img = Image.alpha_composite(img, mask_img)

    return img.convert("RGB")


def draw_keypoints(
    image: Image.Image,
    keypoints: Sequence[tuple[float, float]],
    connections: Sequence[tuple[int, int]] | None = None,
    color: str = "lime",
    radius: int = 4,
) -> Image.Image:
    """Draw keypoints and skeleton connections on an image.

    Args:
        image: PIL Image to draw on.
        keypoints: List of (x, y) coordinates.
        connections: Optional list of (idx1, idx2) pairs for skeleton.
        color: Point and line color.
        radius: Keypoint circle radius.

    Returns:
        Image with keypoints drawn.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Draw connections first (behind points)
    if connections:
        for i, j in connections:
            if i < len(keypoints) and j < len(keypoints):
                x1, y1 = keypoints[i]
                x2, y2 = keypoints[j]
                draw.line([(x1, y1), (x2, y2)], fill=color, width=2)

    # Draw keypoints
    for x, y in keypoints:
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color,
            outline=color,
        )

    return img
