"""
Risk classification module.

Combines density and movement scores into a three-level risk label:
  Green  – low crowd density AND low movement
  Yellow – moderate density OR notable movement
  Red    – high density OR chaotic movement suggesting panic
"""

from __future__ import annotations

from typing import Literal

RiskLevel = Literal["Green", "Yellow", "Red"]

# Thresholds — tune these for your deployment environment
_DENSITY_GREEN_MAX = 0.35
_DENSITY_YELLOW_MAX = 0.65
_MOVEMENT_GREEN_MAX = 0.35
_MOVEMENT_YELLOW_MAX = 0.65


def classify_risk(
    density_score: float,
    movement_score: float,
) -> RiskLevel:
    """
    Classify crowd risk into Green / Yellow / Red.

    Decision logic
    --------------
    Red:
        Density ≥ 0.65  OR  Movement ≥ 0.65
        (high density or highly chaotic motion = panic risk)

    Yellow:
        Density ≥ 0.35  OR  Movement ≥ 0.35
        (noticeable crowd or above-average motion = caution needed)

    Green:
        Everything else (low density and calm movement)

    Parameters
    ----------
    density_score:
        Normalised density in [0.0, 1.0].
    movement_score:
        Normalised movement in [0.0, 1.0].

    Returns
    -------
    str
        One of ``"Green"``, ``"Yellow"``, or ``"Red"``.
    """
    if density_score >= _DENSITY_YELLOW_MAX or movement_score >= _MOVEMENT_YELLOW_MAX:
        return "Red"

    if density_score >= _DENSITY_GREEN_MAX or movement_score >= _MOVEMENT_GREEN_MAX:
        return "Yellow"

    return "Green"
