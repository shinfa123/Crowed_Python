"""
Shared utility helpers used across multiple modules.
"""

import io
import base64
import numpy as np
import cv2
import cv2


def bytes_to_frame(image_bytes: bytes) -> np.ndarray:
    """
    Decode raw image bytes (JPEG, PNG, …) into an OpenCV BGR ndarray.

    Parameters
    ----------
    image_bytes:
        Raw bytes of the uploaded image file.

    Returns
    -------
    np.ndarray
        BGR image array of shape (H, W, 3).

    Raises
    ------
    ValueError
        If the bytes cannot be decoded as a valid image.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(
            "Could not decode the uploaded file as an image. "
            "Supported formats: JPEG, PNG, BMP, WEBP."
        )
    return frame


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp *value* to [low, high]."""
    return max(low, min(high, value))


def draw_boxes_and_encode(frame: np.ndarray, boxes: list[list[int]]) -> str:
    """
    Draw bounding boxes on a frame and return it as a base64-encoded JPEG string.

    Parameters
    ----------
    frame:
        Original BGR image array.
    boxes:
        List of [x1, y1, x2, y2] bounding boxes.

    Returns
    -------
    str
        Base64 string of the processed image (suitable for <img src="data:image/jpeg;base64,...">)
    """
    annotated = frame.copy()
    
    # Draw red rectangles around each detected person
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
    # Encode to JPEG
    success, buffer = cv2.imencode(".jpg", annotated)
    if not success:
        return ""
        
    # Convert to base64 string
    base64_str = base64.b64encode(buffer).decode("utf-8")
    return base64_str

