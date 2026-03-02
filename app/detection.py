"""
Person detection module using YOLOv8.

The detector is loaded once at module level (singleton pattern) so the
heavy model weights are not reloaded on every request.
"""

from __future__ import annotations

import os
import numpy as np
from ultralytics import YOLO

# Fix for PyTorch 2.6 breaking change: disable strict weights_only loading 
# since YOLO needs to deserialize complex model architectures.
# This env var affects both PyTorch and Ultralytics internals.
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"



# YOLO class indices for "person" in the COCO dataset
_PERSON_CLASS_IDS = [0]

# Model is loaded once and reused across all requests
_model: YOLO | None = None


def _get_model(model_name: str = "yolov8n.pt") -> YOLO:
    """Lazy-load the YOLOv8 model (downloads weights on first call)."""
    global _model
    if _model is None:
        _model = YOLO(model_name)
    return _model


def detect_persons(
    frame: np.ndarray,
    confidence_threshold: float = 0.25,
    model_name: str = "yolov8n.pt",
) -> list[list[int]]:
    """
    Run YOLOv8 inference on a single BGR frame and return bounding boxes
    for every detected person.

    Parameters
    ----------
    frame:
        OpenCV BGR image (H × W × 3).
    confidence_threshold:
        Minimum confidence to retain a detection.
    model_name:
        YOLOv8 model variant.

    Returns
    -------
    list[list[int]]
        Each inner list is [x1, y1, x2, y2] in pixel coordinates.
    """
    model = _get_model(model_name)

    # Using default imgsz (640) for fast CPU processing
    results = model.predict(
        source=frame,
        classes=_PERSON_CLASS_IDS,
        conf=confidence_threshold,
        verbose=False,
    )

    boxes: list[list[int]] = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append([int(x1), int(y1), int(x2), int(y2)])

    return boxes
