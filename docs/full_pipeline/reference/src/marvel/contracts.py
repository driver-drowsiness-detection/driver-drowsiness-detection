from dataclasses import dataclass
from typing import Any, Optional, Tuple
import math

OPEN, CLOSED, UNKNOWN = "OPEN", "CLOSED", "UNKNOWN"


@dataclass(frozen=True)
class Frame:
    sample_id: str
    t: float                 # Observation time, seconds in ONE clock domain.
    image: Any               # OpenCV BGR array. Never send this to the LLM.


@dataclass(frozen=True)
class Eye:
    sample_id: str
    t: float
    track: str               # Continuity epoch, not biometric identity.
    side: str                # Subject's anatomical left or right.
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
    value: Optional[float]   # Raw EAR or probability-shaped closure score.
    valid: bool
    reason: str = ""

    def __post_init__(self):
        if self.side not in ("left", "right") or not math.isfinite(self.t):
            raise ValueError("Invalid estimate identity or time")
        if self.valid and (self.value is None or not math.isfinite(self.value)):
            raise ValueError("A valid estimate needs a finite value")
        if not self.valid and self.value is not None:
            raise ValueError("Missing evidence must be None")


def bilateral(left: str, right: str) -> str:
    if left == right and left in (OPEN, CLOSED):
        return left
    return UNKNOWN
