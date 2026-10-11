"""Immutable records shared between perception and cognition.

The CNN only consumes current, valid eye crops.  ``None`` deliberately means
missing evidence; it must never be replaced with a numeric prediction.
"""

from dataclasses import dataclass
import math
from typing import Any, Optional, Tuple


OPEN, CLOSED, UNKNOWN = "OPEN", "CLOSED", "UNKNOWN"


@dataclass(frozen=True)
class Frame:
    sample_id: str
    t: float
    image: Any  # OpenCV BGR image


@dataclass(frozen=True)
class Eye:
    sample_id: str
    t: float
    track: str  # Continuity epoch, never a biometric identity.
    side: str   # The driver's anatomical left or right eye.
    points: tuple
    crop: Any
    geometry_ok: bool
    crop_ok: bool
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Estimate:
    sample_id: str
    t: float
    track: str
    side: str
    method: str
    value: Optional[float]
    valid: bool
    reason: str = ""

    def __post_init__(self) -> None:
        if self.side not in ("left", "right") or not math.isfinite(self.t):
            raise ValueError("Invalid estimate identity or time")
        if self.valid and (self.value is None or not math.isfinite(self.value)):
            raise ValueError("A valid estimate needs a finite value")
        if not self.valid and self.value is not None:
            raise ValueError("Missing evidence must be None")


def bilateral(left: str, right: str) -> str:
    """Return a state only when both current eyes agree."""
    if left == right and left in (OPEN, CLOSED):
        return left
    return UNKNOWN
