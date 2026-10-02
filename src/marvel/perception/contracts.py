import math
from dataclasses import dataclass

import numpy as np

@dataclass (frozen=True)
class FramePacket:
    source_id: str
    frame_id: int
    timestamp_s: float   # event time, seconds
    received_s: float    # host receipt time, seconds
    image: np.ndarray    # (height, width, 3)
    synthetic: bool = False


def packet_problem(packet, previous=None):
    """Return a reason string if the packet is unusable, else None."""
    image = packet.image
    if image is None or image.size == 0:
        return "empty_frame"
    if image.ndim != 3 or image.shape[2] != 3:
        return "unexpected_image_shape"
    if packet.frame_id < 0:
        return "invalid_frame_id"
    if not (math.isfinite(packet.timestamp_s) and math.isfinite(packet.received_s)):
        return "nonfinite_time"
    if previous is not None:
        if packet.source_id != previous.source_id:
            return "source_changed"
        if packet.frame_id <= previous.frame_id:
            return "non_increasing_frame_id"
        if packet.timestamp_s <= previous.timestamp_s:
            return "non_increasing_time"
    return None


def make_synthetic_packet(frame_id, fps=30.0, height=480, width=640,
                          source_id="synthetic"):
    if fps <= 0:
        raise ValueError("fps must be positive")
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = frame_id % 256
    t = frame_id / fps
    return FramePacket(source_id, frame_id, t, t, image, synthetic=True)