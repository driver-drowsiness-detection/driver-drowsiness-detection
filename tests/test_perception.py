import numpy as np

from marvel.perception.contracts import FramePacket, make_synthetic_packet, packet_problem


def test_synthetic_packet_is_labelled_and_carries_its_own_pixels():
    p = make_synthetic_packet(7)
    assert p.synthetic and p.frame_id == 7
    assert p.image[0, 0, 0] == 7
    assert packet_problem(p) is None


def test_non_increasing_ids_and_times_are_rejected():
    a, b = make_synthetic_packet(1), make_synthetic_packet(1)
    assert packet_problem(b, previous=a) == "non_increasing_frame_id"
    late = FramePacket("synthetic", 2, 0.0, 0.0, a.image)
    assert packet_problem(late, previous=a) == "non_increasing_time"


def test_empty_and_nonfinite_rejected():
    empty = FramePacket("s", 0, 0.0, 0.0, np.zeros((0, 0, 3), np.uint8))
    assert packet_problem(empty) == "empty_frame"
    nan = FramePacket("s", 0, float("nan"), 0.0, np.zeros((4, 4, 3), np.uint8))
    assert packet_problem(nan) == "nonfinite_time"