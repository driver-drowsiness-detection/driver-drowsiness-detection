from pathlib import Path

import pytest
import yaml

from marvel.perception.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["runtime.yaml", "jetson-camera.yaml"])
def test_shipped_configuration_resolves_model_against_repository(name):
    config = load_config("configs/" + name, ROOT)
    assert config["landmarker"] == str(ROOT / "models/face_landmarker.task")


@pytest.mark.parametrize("key,value", [("max_frame_age_s", float("nan")),
                                    ("capture_timeout_s", 0),
                                    ("driver_roi", [0, 0, 2, 1])])
def test_bad_settings_are_rejected_before_opening_camera(tmp_path, key, value):
    config = yaml.safe_load((ROOT / "configs/runtime.yaml").read_text())
    config[key] = value
    target = tmp_path / "invalid.yaml"
    target.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_config(target, ROOT)
