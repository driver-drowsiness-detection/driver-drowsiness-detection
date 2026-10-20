from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from marvel.contracts import Frame
from marvel.perception.crops import eye_crop
from marvel.perception.detection import Detector, EYES
from marvel.perception.pipeline import process_frame
from marvel.perception.quality import image_quality, points_ok
from marvel.observability.overlay import draw_preview, eye_panels

QUALITY = dict(min_crop_pixels=8, brightness_min=10, brightness_max=245, laplacian_min=0)


def face(closed=False):
    points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
    for side, cx in (("left", 0.65), ("right", 0.35)):
        gap = 0 if closed else 0.02
        coords = [(cx-.05,.5),(cx-.02,.5-gap),(cx+.02,.5-gap),
                  (cx+.05,.5),(cx+.02,.5+gap),(cx-.02,.5+gap)]
        for index, (x, y) in zip(EYES[side], coords):
            points[index] = SimpleNamespace(x=x, y=y)
    return points


class Backend:
    def __init__(self, faces):
        self.faces = faces
        self.calls = 0
        self.closed = False

    def detect_bgr(self, image):
        self.calls += 1
        return self.faces

    def close(self):
        self.closed = True


def detector(faces=None, **kwargs):
    backend = Backend([face()] if faces is None else faces)
    return Detector(None, kwargs.get("roi", [0, 0, 1, 1]), QUALITY, backend=backend)


def frame(index=1, t=1.0):
    return Frame(str(index), t, np.full((240, 640, 3), 100, np.uint8))


def test_pixel_coordinates_identity_anatomical_sides_and_detached_crops():
    source = frame()
    d = detector()
    eyes, status, metrics = d.process(source)
    assert status == "valid"
    assert [e.side for e in eyes] == ["left", "right"]
    assert eyes[0].points[0] == pytest.approx((384, 120))
    assert {e.sample_id for e in eyes} == {source.sample_id}
    assert {e.t for e in eyes} == {source.t}
    assert len({e.track for e in eyes}) == 1
    assert all(e.geometry_ok and e.crop_ok for e in eyes)
    assert not np.shares_memory(eyes[0].crop, source.image)
    assert metrics["left"]["brightness"] == 100
    d.close()
    assert d.backend.closed


def test_closed_eyes_retain_same_crop_area():
    opened = detector().process(frame())[0]
    closed = detector([face(True)]).process(frame())[0]
    assert all(e.crop_ok for e in closed)
    assert [e.crop.shape for e in opened] == [e.crop.shape for e in closed]


@pytest.mark.parametrize("faces,reason", [([], "no_face"), ([face(), face()], "multiple_faces")])
def test_selection_rejects_missing_or_ambiguous_faces(faces, reason):
    d = detector(faces)
    assert d.process(frame()) == ([], reason, {})


def test_one_face_outside_roi_is_rejected():
    assert detector(roi=[0, 0, .1, .1]).process(frame())[1] == "outside_driver_roi"


def test_track_changes_after_loss_and_repeated_samples_do_not_pass():
    d = detector()
    first = d.process(frame())[0][0].track
    assert d.process(frame())[1] == "repeated_frame"
    d.backend.faces = []
    assert d.process(frame(2, 2))[1] == "no_face"
    d.backend.faces = [face()]
    assert d.process(frame(3, 3))[0][0].track != first


def test_nonfinite_landmarks_reject_entire_observation():
    f = face()
    f[33].x = float("nan")
    assert detector([f]).process(frame())[1] == "invalid_landmarks"


def test_bad_eye_points_and_bounds_remain_missing():
    f = face()
    f[33].x = -0.1
    eyes = detector([f]).process(frame())[0]
    bad = next(e for e in eyes if e.side == "right")
    assert not bad.geometry_ok and not bad.crop_ok and bad.crop is None
    assert "invalid_points" in bad.reasons
    assert not points_ok(((0, 0),) * 5, 640, 480)
    assert eye_crop(frame().image, ((0, 0),) * 6) is None


def test_exposure_and_blur_are_measured_and_gate_evidence():
    black = np.zeros((32, 64, 3), np.uint8)
    valid, reasons, metrics = image_quality(black, QUALITY)
    assert not valid and "exposure" in reasons and metrics["brightness"] == 0
    flat = np.full_like(black, 100)
    assert image_quality(flat, dict(QUALITY, laplacian_min=1))[1] == ("blur",)
    assert image_quality(flat, QUALITY)[0]  # Explicit development setting disables blur gate.
    eyes = detector().process(replace(frame(), image=np.zeros((240, 640, 3), np.uint8)))[0]
    assert all(e.crop is None and not e.crop_ok and not e.geometry_ok for e in eyes)


def test_freshness_checked_before_and_after_inference():
    d = detector()
    assert process_frame(frame(), d, 1, clock=lambda: 3)[1] == "stale_frame"
    assert d.backend.calls == 0
    ticks = iter([1.1, 3.0])
    assert process_frame(frame(), d, 1, clock=lambda: next(ticks))[:3] == ([], "stale_frame", {})
    assert d.previous is None


def test_display_does_not_modify_input_and_invalid_panels_are_cleared():
    source = frame()
    eyes, status, metrics = detector().process(source)
    before = source.image.copy()
    canvas = draw_preview(source, eyes, status, 10, .05, metrics)
    assert np.array_equal(before, source.image)
    assert not np.array_equal(canvas, source.image)
    valid_panel = eye_panels(eyes)["left"]
    invalid_panel = eye_panels([replace(eyes[0], crop=None, crop_ok=False)])["left"]
    assert not np.array_equal(valid_panel, invalid_panel)
    assert np.array_equal(invalid_panel, eye_panels([])["left"])


def test_float_images_rejected_before_model_call():
    d = detector()
    assert d.process(replace(frame(), image=frame().image.astype(float)))[1] == "unexpected_image_format"
    assert d.backend.calls == 0
