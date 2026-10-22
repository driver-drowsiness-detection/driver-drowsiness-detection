"""CEW inspection must preserve unknown identities and require source metadata."""

from pathlib import Path
import runpy

from PIL import Image
import pytest

API = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "inspect_cew_release.py"))


def make_release(tmp_path):
    for folder, name, color in (
        ("ClosedFace", "closed_eye_0001.jpg_face_1.jpg", 40),
        ("OpenFace", "Example_Person_0001.jpg", 180),
    ):
        (tmp_path / folder).mkdir()
        Image.new("RGB", (100, 100), (color, color, color)).save(tmp_path / folder / name)
        (tmp_path / f"EyeCoordinatesInfo_{folder}.txt").write_text(name + " 30 40 70 40\n")
    return tmp_path


def test_closed_filename_is_not_a_subject_id():
    identity, group = API["source_identity"]("closed_eye_0033.jpg_face_3.jpg", "closed")
    assert identity == ""
    assert group == "closed_eye_0033.jpg"


def test_inventory_stays_out_of_training(tmp_path):
    rows, summary = API["inspect_release"](make_release(tmp_path))
    assert summary["classes"] == {"closed": 1, "open": 1}
    assert summary["status"] == "inspection_only_not_ready_for_training"
    assert summary["blockers"]
    assert all("split" not in row and "label" not in row for row in rows)


def test_missing_coordinate_record_rejected(tmp_path):
    make_release(tmp_path)
    (tmp_path / "EyeCoordinatesInfo_OpenFace.txt").write_text("")
    with pytest.raises(ValueError, match="disagree"):
        API["inspect_release"](tmp_path)


def test_identical_pixels_across_classes_are_reported(tmp_path):
    make_release(tmp_path)
    closed = tmp_path / "ClosedFace" / "closed_eye_0001.jpg_face_1.jpg"
    opened = tmp_path / "OpenFace" / "Example_Person_0001.jpg"
    opened.write_bytes(closed.read_bytes())
    _, summary = API["inspect_release"](tmp_path)
    assert len(summary["decoded_pixel_duplicate_groups"]) == 1
