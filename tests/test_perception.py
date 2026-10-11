import numpy as np
import pytest

from marvel.perception.contracts import FramePacket, make_synthetic_packet, packet_problem
from marvel.perception.crops import crop_box
from marvel.perception.quality import crop_validity
from marvel.perception.quality import (EyeObservation, Validity, combine_eye_validity, crop_validity)

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

def test_valid_crop_has_no_reasons():
    crop = np.zeros((20, 30, 3), np.uint8)
    result = crop_validity(crop, None, 8)
    assert result.valid and result.reasons == ()


def test_missing_crop_keeps_its_reason():
    result = crop_validity(None, "box_outside_or_empty", 8)
    assert not result.valid
    assert result.reasons == ("box_outside_or_empty",)


def test_missing_crop_without_reason_uses_no_crop():
    result = crop_validity(None, None, 8)
    assert result.reasons == ("no_crop",)


def test_small_crop_is_rejected():
    crop = np.zeros((5, 30, 3), np.uint8)
    result = crop_validity(crop, None, 8)
    assert not result.valid and result.reasons == ("crop_too_small",)

def test_fractional_box_is_expanded_outward():
    image = make_image()
    crop, reason = crop_box(image, (2.5, 3.5, 7.2, 8.1))
    assert reason is None and crop.shape == (6, 6, 3)
    assert np.array_equal(crop, image[3:9, 2:8])


def test_box_covering_whole_image_is_valid():
    image = make_image()
    crop, reason = crop_box(image, (0, 0, 12, 10))
    assert reason is None and np.array_equal(crop, image)


def test_grayscale_image_is_rejected():
    crop, reason = crop_box(np.zeros((10, 12), np.uint8), (1, 1, 5, 5))
    assert crop is None and reason == "unexpected_image_shape"


def test_box_with_wrong_length_is_rejected():
    crop, reason = crop_box(make_image(), (1, 2, 3))
    assert crop is None and reason == "invalid_box"


def test_rows_come_before_columns():
    image = make_image()
    crop, reason = crop_box(image, (4, 1, 9, 3))   # left, top, right, bottom
    assert reason is None and crop.shape == (2, 5, 3)
    assert np.array_equal(crop[0, 0], image[1, 4])  # point x=4, y=1

def test_no_face_makes_both_invalid():
    good = Validity(True)
    obs = combine_eye_validity(False, good, good)
    assert obs.geometry.reasons == ("no_face",)
    assert obs.crop.reasons == ("no_face",)


def test_geometry_can_fail_while_crop_is_valid():
    bad_geometry = Validity(False, ("zero_eye_width",))
    obs = combine_eye_validity(True, bad_geometry, Validity(True))
    assert not obs.geometry.valid and obs.crop.valid


def test_crop_can_fail_while_geometry_is_valid():
    bad_crop = Validity(False, ("crop_too_small",))
    obs = combine_eye_validity(True, Validity(True), bad_crop)
    assert obs.geometry.valid and not obs.crop.valid