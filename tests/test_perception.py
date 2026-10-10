import numpy as np
import pytest

from marvel.perception.contracts import FramePacket, make_synthetic_packet, packet_problem
from marvel.perception.crops import crop_box

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

def make_image():
    return np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3)


def test_known_box_gives_expected_shape():
    crop, reason = crop_box(make_image(), (2, 3, 7, 8))
    assert reason is None and crop.shape == (5, 5, 3)


@pytest.mark.parametrize("box", [(-1, 3, 7, 8), (2, 3, 2, 8), (2, 3, 13, 8),
                                 (float("nan"), 3, 7, 8)])
def test_bad_boxes_are_invalid(box):
    crop, reason = crop_box(make_image(), box)
    assert crop is None and reason is not None


def test_crop_is_independent_of_original():
    image = make_image()
    crop, _ = crop_box(image, (2, 3, 7, 8))
    before = crop.copy()
    image[:] = 0
    assert np.array_equal(crop, before)