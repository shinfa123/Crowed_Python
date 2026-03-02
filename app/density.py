"""
Crowd density calculation module.

Density is calculated as the ratio of frame area occupied by detected
people to the total frame area, clamped to [0.0, 1.0].
"""

from __future__ import annotations

import numpy as np
from app.utils import clamp


def calculate_density(
    frame: np.ndarray,
    boxes: list[list[int]],
    max_expected_density: float = 0.60,
) -> float:
    """
    Compute a normalised crowd density score.

    Strategy
    --------
    We sum the pixel area of every detected person bounding box, divide
    by the total frame area, then normalise against a configurable
    *max_expected_density* cap so that the score reaches 1.0 at realistic
    indoor/event-level crowd densities rather than only when every pixel
    is covered.

    Parameters
    ----------
    frame:
        Original BGR image used to determine frame dimensions.
    boxes:
        Bounding boxes from :func:`app.detection.detect_persons`.
    max_expected_density:
        The raw pixel-coverage ratio that maps to score 1.0.
        Increase this value for outdoor/sparse scenes.

    Returns
    -------
    float
        Density score in [0.0, 1.0].
    """
    frame_height, frame_width = frame.shape[:2]
    frame_area: int = frame_height * frame_width

    if frame_area == 0 or not boxes:
        return 0.0

    covered_area: int = 0
    for x1, y1, x2, y2 in boxes:
        covered_area += max(0, x2 - x1) * max(0, y2 - y1)

    raw_ratio: float = covered_area / frame_area
    normalised: float = raw_ratio / max_expected_density

    return clamp(normalised)
