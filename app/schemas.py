"""
Pydantic models for request and response validation.
"""

from typing import Literal
from pydantic import BaseModel, Field


class AnalysisResponse(BaseModel):
    """Response schema returned by the /analyze endpoint."""

    people_count: int = Field(
        ...,
        ge=0,
        description="Total number of people detected in the frame.",
    )
    density_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalised crowd density score (0.0 = empty, 1.0 = fully packed).",
    )
    movement_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalised movement / optical-flow score (0.0 = still, 1.0 = chaotic).",
    )
    risk_level: Literal["Green", "Yellow", "Red"] = Field(
        ...,
        description="Overall risk classification derived from density and movement scores.",
    )
    annotated_image_base64: str | None = Field(
        default=None,
        description="Base64-encoded JPEG image with detected bounding boxes drawn.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "people_count": 120,
                "density_score": 0.72,
                "movement_score": 0.55,
                "risk_level": "Yellow",
                "annotated_image_base64": "/9j/4AAQSkZJRgABAQEASABIAAD...",
            }
        }
    }


class VideoAnalysisResponse(BaseModel):
    """Response schema returned by the /analyze-video endpoint."""

    average_people_count: int = Field(
        ...,
        ge=0,
        description="Average number of people detected across analyzed frames.",
    )
    average_density_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average normalized crowd density score across frames.",
    )
    movement_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall movement score based on dense optical flow analysis.",
    )
    risk_level: Literal["Green", "Yellow", "Red"] = Field(
        ...,
        description="Overall risk classification derived from density and movement scores.",
    )
    annotated_image_base64: str | None = Field(
        default=None,
        description="Base64-encoded JPEG of the most recent processed frame with bounding boxes.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "average_people_count": 85,
                "average_density_score": 0.61,
                "movement_score": 0.75,
                "risk_level": "Red",
                "annotated_image_base64": "/9j/4AAQSkZJRgABAQEASABIAAD...",
            }
        }
    }


class ErrorResponse(BaseModel):
    """Generic error response schema."""

    detail: str
