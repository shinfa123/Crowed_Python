"""
AI-Based Crowd Panic Prediction System
=======================================
FastAPI microservice entry point.

Run locally:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Swagger UI:
    http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
import tempfile
import cv2

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from app.detection import detect_persons
from app.density import calculate_density
from app.movement import calculate_movement
from app.risk import classify_risk
from app.schemas import AnalysisResponse, VideoAnalysisResponse, ErrorResponse
from app.utils import bytes_to_frame, draw_boxes_and_encode

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Crowd Panic Prediction API",
    description=(
        "Upload a crowd image to receive a structured risk assessment "
        "including people count, density score, movement score, and "
        "a traffic-light risk level (Green / Yellow / Red)."
    ),
    version="1.0.0",
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Returns service health status."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------
@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Analyse a crowd image for panic risk",
    tags=["Analysis"],
)
async def analyze(
    file: UploadFile = File(
        ...,
        description="Crowd image file (JPEG, PNG, BMP, or WebP).",
    ),
) -> AnalysisResponse:
    """
    **POST /analyze**

    Upload a single crowd image to receive:

    | Field           | Type  | Description                                     |
    |-----------------|-------|-------------------------------------------------|
    | `people_count`  | int   | Number of detected persons                      |
    | `density_score` | float | Normalised crowd density (0.0 – 1.0)            |
    | `movement_score`| float | Normalised movement intensity (0.0 – 1.0)       |
    | `risk_level`    | str   | `Green` / `Yellow` / `Red`                     |
    """
    # ---- Validate content type -------------------------------------------
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '{file.content_type}'. Upload an image.",
        )

    # ---- Read & decode image ---------------------------------------------
    try:
        raw_bytes = await file.read()
        frame = bytes_to_frame(raw_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error reading uploaded file.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read the uploaded file.",
        ) from exc

    # ---- Run analysis pipeline ------------------------------------------
    try:
        logger.info("Running person detection on frame %s …", frame.shape)
        boxes = detect_persons(frame)
        people_count = len(boxes)
        logger.info("Detected %d person(s).", people_count)

        density_score = calculate_density(frame, boxes)
        movement_score = calculate_movement(frame, boxes)
        risk_level = classify_risk(density_score, movement_score)
        
        # Generate visual output for frontend
        annotated_b64 = draw_boxes_and_encode(frame, boxes)

        logger.info(
            "density=%.2f  movement=%.2f  risk=%s",
            density_score,
            movement_score,
            risk_level,
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("Analysis pipeline error.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {exc}",
        ) from exc

    return AnalysisResponse(
        people_count=people_count,
        density_score=round(density_score, 4),
        movement_score=round(movement_score, 4),
        risk_level=risk_level,
        annotated_image_base64=annotated_b64,
    )


@app.post(
    "/analyze-video",
    response_model=VideoAnalysisResponse,
    summary="Analyse a crowd video sequence for panic risk",
    tags=["Analysis"],
)
async def analyze_video(
    file: UploadFile = File(
        ...,
        description="Crowd video file (MP4, AVI, MOV).",
    ),
) -> VideoAnalysisResponse:
    """
    **POST /analyze-video**

    Upload a video file to receive an aggregated risk assessment using true optical flow:

    | Field                   | Type  | Description                                     |
    |-------------------------|-------|-------------------------------------------------|
    | `average_people_count`  | int   | Average people detected per frame               |
    | `average_density_score` | float | Average crowd density (0.0 – 1.0)               |
    | `movement_score`        | float | Optical flow motion intensity (0.0 – 1.0)       |
    | `risk_level`            | str   | `Green` / `Yellow` / `Red`                     |
    """
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '{file.content_type}'. Upload a video.",
        )

    # Save uploaded video to a temp file for OpenCV to read
    try:
        raw_bytes = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(raw_bytes)
            tmp_path = tmp_file.name
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to buffer video file.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read the uploaded video file.",
        ) from exc

    # Process video
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("OpenCV could not open the video file.")

        # Determine video FPS to sample roughly 1 frame per second
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_skip = max(int(fps), 10) # process 1 frame per second max
        max_duration_secs = 60 # only process up to 1 minute of video to prevent timeout
        max_frames_to_process = max_duration_secs

        people_counts = []
        density_scores = []
        movement_scores = []
        
        prev_frame = None
        frame_idx = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            processed_count += 1
            if processed_count > max_frames_to_process:
                logger.warning("Reached maximum frames limit for video analysis. Stopping early.")
                break

            # 1. Resize frame for faster processing (downscale to max 1280px width)
            h, w = frame.shape[:2]
            if w > 1280:
                scale = 1280 / w
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            # 2. Run detection & density on current frame (detect_persons handles resizing internally via YOLO)
            boxes = detect_persons(frame)
            people_counts.append(len(boxes))
            density_scores.append(calculate_density(frame, boxes))

            # 3. Run optical flow on a smaller 640px version of the frame for speed
            if prev_frame is not None:
                # Downscale again for optical flow - it doesn't need high res
                flow_scale = 640 / frame.shape[1]
                small_frame = cv2.resize(frame, (int(frame.shape[1] * flow_scale), int(frame.shape[0] * flow_scale)))
                small_prev = cv2.resize(prev_frame, (int(prev_frame.shape[1] * flow_scale), int(prev_frame.shape[0] * flow_scale)))
                
                mov = calculate_movement(small_frame, boxes, prev_frame=small_prev)
                movement_scores.append(mov)

            # Store current frame for the next iteration's optical flow calculation
            prev_frame = frame.copy()

        cap.release()
        
        if not people_counts:
            raise ValueError("Video was too short or contained no readable frames.")

        avg_people = int(sum(people_counts) / len(people_counts))
        avg_density = sum(density_scores) / len(density_scores)
        # Average the movement scores, or 0.0 if not enough frames
        avg_movement = sum(movement_scores) / len(movement_scores) if movement_scores else 0.0
        
        final_risk = classify_risk(avg_density, avg_movement)
        
        # Encode the very last processed frame for frontend visualization
        final_annotated_b64 = None
        if prev_frame is not None:
            # Re-run detection on the last frame just to get boxes for drawing
            final_boxes = detect_persons(prev_frame)
            final_annotated_b64 = draw_boxes_and_encode(prev_frame, final_boxes)
        
        logger.info(
            "Video processed: avg_density=%.2f avg_movement=%.2f risk=%s",
            avg_density, avg_movement, final_risk
        )

    except Exception as exc:  # pragma: no cover
        logger.exception("Video analysis pipeline error.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video analysis failed: {exc}",
        ) from exc
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return VideoAnalysisResponse(
        average_people_count=avg_people,
        average_density_score=round(avg_density, 4),
        movement_score=round(avg_movement, 4),
        risk_level=final_risk,
        annotated_image_base64=final_annotated_b64,
    )
