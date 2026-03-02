"""
Movement analysis module.

For a single-image endpoint we cannot compute true optical flow between
consecutive frames.  Instead we use the *Laplacian variance* (a.k.a.
blur detection) localised to each detected person's bounding box as a
proxy for motion blur / movement intensity.

When the caller provides a *previous frame* (e.g., via a stateful
wrapper or video pipeline), Farneback dense optical flow is used instead
for a more accurate score.
"""

from __future__ import annotations

import cv2
import numpy as np
from app.utils import clamp


# Tuning parameters
_BLUR_MAX_VARIANCE = 500.0   # Laplacian variance that maps to score 1.0
_FLOW_MAX_MAGNITUDE = 15.0   # Mean optical-flow magnitude → score 1.0


def _motion_score_from_blur(
    frame: np.ndarray,
    boxes: list[list[int]],
) -> float:
    """
    Estimate motion via Laplacian variance within person bounding boxes.

    High blur (low variance) → frame is sharp / people are still.
    Low blur (high variance) → motion blur → people are moving fast.

    Note: This is an approximation for single-frame analysis.
    """
    if not boxes:
        return 0.0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variances: list[float] = []

    for x1, y1, x2, y2 in boxes:
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        variance = float(cv2.Laplacian(roi, cv2.CV_64F).var())
        variances.append(variance)

    if not variances:
        return 0.0

    mean_variance = float(np.mean(variances))
    # Higher variance = sharper = less motion blur → lower score
    # We invert: score ∝ 1 - (variance / max)
    score = 1.0 - clamp(mean_variance / _BLUR_MAX_VARIANCE)
    return clamp(score)


def _motion_score_from_optical_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
) -> float:
    """
    Compute a movement score using Farneback dense optical flow.

    Parameters
    ----------
    prev_gray, curr_gray:
        Consecutive greyscale frames.

    Returns
    -------
    float
        Movement score in [0.0, 1.0].
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_magnitude = float(np.mean(magnitude))
    return clamp(mean_magnitude / _FLOW_MAX_MAGNITUDE)


def calculate_movement(
    frame: np.ndarray,
    boxes: list[list[int]],
    prev_frame: np.ndarray | None = None,
) -> float:
    """
    Return a normalised movement score in [0.0, 1.0].

    If *prev_frame* is supplied, Farneback optical flow is used.
    Otherwise, per-person Laplacian variance (motion-blur proxy) is used.

    Parameters
    ----------
    frame:
        Current BGR frame.
    boxes:
        Bounding boxes for detected persons.
    prev_frame:
        Optional previous BGR frame for optical-flow computation.

    Returns
    -------
    float
        Movement score in [0.0, 1.0].
    """
    if prev_frame is not None:
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        return _motion_score_from_optical_flow(prev_gray, curr_gray)

    return _motion_score_from_blur(frame, boxes)
