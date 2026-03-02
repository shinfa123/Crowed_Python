# AI-Based Crowd Panic Prediction System

A **FastAPI** microservice that analyses crowd images using **YOLOv8** (person detection) and **OpenCV** to predict crowd panic risk in real time.

---

## Project Structure

```
Crowd Panic Phython/
├── main.py                  # FastAPI entry point
├── requirements.txt         # Python dependencies
└── app/
    ├── __init__.py
    ├── schemas.py           # Pydantic request/response models
    ├── utils.py             # Shared helpers (image decoding, clamp)
    ├── detection.py         # YOLOv8 person detection
    ├── density.py           # Crowd density calculation
    ├── movement.py          # Movement / optical-flow analysis
    └── risk.py              # Risk classification (Green / Yellow / Red)
```

---

## Installation

```bash
pip install -r requirements.txt
```

> On the first run, YOLOv8 will automatically download the `yolov8n.pt` model weights (~6 MB).

---

## Running the Service

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open the interactive API docs at **http://localhost:8000/docs**.

---

## API Reference

### `GET /health`
Returns a simple health-check response.

```json
{ "status": "ok" }
```

---

### `POST /analyze`

Upload a crowd image (JPEG / PNG / BMP / WebP) to receive a risk assessment.

**Request** — `multipart/form-data`

| Field  | Type | Description               |
|--------|------|---------------------------|
| `file` | File | Crowd image to analyse    |

**Response** — `application/json`

| Field            | Type   | Description                                  |
|------------------|--------|----------------------------------------------|
| `people_count`   | int    | Number of people detected by YOLOv8          |
| `density_score`  | float  | Normalised density score (0.0 – 1.0)         |
| `movement_score` | float  | Normalised movement score (0.0 – 1.0)         |
| `risk_level`     | string | `"Green"` / `"Yellow"` / `"Red"`            |

**Example response:**

```json
{
  "people_count": 120,
  "density_score": 0.72,
  "movement_score": 0.55,
  "risk_level": "Yellow",
  "annotated_image_base64": "/9j/4AAQSkZJRgABAQAAAQ..."
}
```

**Quick test with curl:**

```bash
curl -X POST http://localhost:8000/analyze \
     -F "file=@crowd.jpg"
```

---

### `POST /analyze-video`

Upload a crowd video (MP4 / AVI / MOV) to receive an aggregated risk assessment. The system processes the video frame-by-frame and uses dense optical flow for high-accuracy movement calculation.

**Request** — `multipart/form-data`

| Field  | Type | Description               |
|--------|------|---------------------------|
| `file` | File | Crowd video to analyse    |

**Response** — `application/json`

| Field                   | Type   | Description                                  |
|-------------------------|--------|----------------------------------------------|
| `average_people_count`  | int    | Average people detected per frame            |
| `average_density_score` | float  | Average density score (0.0 – 1.0)            |
| `movement_score`        | float  | Overall optical flow motion (0.0 – 1.0)      |
| `risk_level`            | string | `"Green"` / `"Yellow"` / `"Red"`             |

**Example response:**

```json
{
  "average_people_count": 85,
  "average_density_score": 0.61,
  "movement_score": 0.75,
  "risk_level": "Red",
  "annotated_image_base64": "/9j/4AAQSkZJRgABAQAAAQ..."
}
```

---

## Risk Level Logic

| Condition                             | Risk Level |
|---------------------------------------|------------|
| Density < 0.35 AND Movement < 0.35   | 🟢 Green   |
| Density ≥ 0.35 OR  Movement ≥ 0.35   | 🟡 Yellow  |
| Density ≥ 0.65 OR  Movement ≥ 0.65   | 🔴 Red     |

---

## Module Overview

| Module          | Responsibility                                                |
|-----------------|---------------------------------------------------------------|
| `detection.py`  | Runs YOLOv8 on a frame; returns bounding boxes for persons    |
| `density.py`    | Computes normalised density from bounding-box pixel coverage  |
| `movement.py`   | Single-frame: Laplacian variance proxy; multi-frame: Farneback optical flow |
| `risk.py`       | Maps density + movement → Green / Yellow / Red                |
| `schemas.py`    | Pydantic models for request validation and response shaping   |
| `utils.py`      | `bytes_to_frame()` image decoder, `clamp()` helper            |

---

## Tech Stack

| Technology       | Purpose                          |
|------------------|----------------------------------|
| FastAPI          | REST API framework               |
| Uvicorn          | ASGI server                      |
| Ultralytics YOLOv8 | Real-time person detection     |
| OpenCV           | Video / image processing         |
| NumPy            | Array operations                 |
| Pydantic v2      | Data validation & serialisation  |
