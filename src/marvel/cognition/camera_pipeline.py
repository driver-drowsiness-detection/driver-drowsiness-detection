"""Connect current camera eyes to optional CNN and comparison histories."""
from dataclasses import asdict
from time import perf_counter

from marvel.contracts import UNKNOWN, bilateral
from .pipeline import EARCognition
from .comparison import Comparison
from .fusion import score_state
from .perclos import ClosureWindow
from .trigger import Trigger


class CameraCognition:
    def __init__(self, *, policy, run_id, model=None, cnn_threshold=None,
                 comparison=None, alert_mode="ear", pose_calibration=None):
        self.ear = EARCognition(**policy, run_id=run_id, pose_calibration=pose_calibration)
        self.model, self.cnn_threshold = model, cnn_threshold
        self.comparison = None
        self.alert_mode = alert_mode
        self.cnn_trigger = self.cnn_window = None
        if alert_mode not in ("ear", "cnn", "fusion"):
            raise ValueError("Unknown alert mode")
        if model is not None:
            from .fusion import _unit
            _unit(cnn_threshold, "CNN threshold")
            timing = {key: policy[key] for key in ("close_s", "recover_s", "max_gap_s", "unavailable_s")}
            self.cnn_trigger = Trigger(**timing, run_id=run_id, mode="cnn")
            self.cnn_window = ClosureWindow(policy["window_s"], policy["max_gap_s"], policy["min_coverage"])
        if comparison is not None:
            if model is None:
                raise ValueError("Comparison requires a CNN artifact")
            if set(comparison) != {"ear_calibration", "fusion_threshold", "alpha"}:
                raise ValueError("Comparison JSON needs ear_calibration, fusion_threshold and alpha only")
            self.comparison = Comparison(**policy, **comparison, cnn_threshold=cnn_threshold, run_id=run_id)
        if alert_mode == "cnn" and model is None:
            raise ValueError("CNN alerts require a CNN artifact")
        if alert_mode == "fusion" and self.comparison is None:
            raise ValueError("Fusion alerts require explicit comparison settings")

    @property
    def trigger(self):
        if self.comparison:
            return self.comparison.triggers[self.alert_mode]
        return self.cnn_trigger if self.alert_mode == "cnn" else self.ear.trigger

    def active_episodes(self):
        if self.comparison:
            return {mode: trigger.episode_id for mode, trigger in self.comparison.triggers.items()}
        result = {"ear": self.ear.trigger.episode_id}
        if self.cnn_trigger:
            result["cnn"] = self.cnn_trigger.episode_id
        return result

    def update(self, sample_id, t, eyes, face_status, metrics=None, image_shape=None,
               *, evidence_age_s=None, max_age_s=None):
        # Check observation clock before running inference or mutating histories.
        import math
        if not math.isfinite(t) or (self.trigger.last is not None and t <= self.trigger.last):
            raise ValueError("Time must be finite and strictly increasing")
        started = perf_counter()
        aligned = (face_status in ("valid", "invalid_eye") and len(eyes) == 2
                   and {e.side for e in eyes} == {"left", "right"}
                   and all((e.sample_id, e.t, e.track) == (sample_id, t, eyes[0].track) for e in eyes)
                   and isinstance(eyes[0].track, str) and bool(eyes[0].track.strip()))
        track = eyes[0].track if aligned else None
        cnn = [self.model.estimate(eye) for eye in eyes] if aligned and self.model else []
        # An inference adapter may not substitute a cached crop's identity.
        for eye, estimate in zip(eyes, cnn):
            if (estimate.sample_id, estimate.t, estimate.track, estimate.side) != (eye.sample_id, eye.t, eye.track, eye.side):
                raise ValueError("CNN returned mismatched observation identity")
            from .fusion import validate_score
            validate_score(estimate, "cnn")
        if evidence_age_s is not None and max_age_s is not None and evidence_age_s + perf_counter() - started > max_age_s:
            eyes, cnn, metrics, track, face_status = [], [], {}, None, "stale_after_cognition"
        result = self.ear.update(sample_id, t, eyes, face_status, metrics, image_shape)
        modes = {"ear": {"decision": result["decision"], "rolling": result["rolling"],
                         "estimates": result["eye_estimates"]}}
        if self.comparison:
            from marvel.contracts import Estimate
            compared = self.comparison.update(sample_id=sample_id, t=t, track=track,
                ear=[Estimate(**e) for e in result["eye_estimates"]], cnn=cnn)
            modes = compared["modes"]
            result["common_observable"] = compared["common_observable"]
        elif self.cnn_trigger:
            observation = bilateral(*(score_state(e, self.cnn_threshold) for e in cnn)) if len(cnn) == 2 else UNKNOWN
            modes["cnn"] = {"estimates": [asdict(e) for e in cnn],
                "decision": asdict(self.cnn_trigger.update(t, observation, track, sample_id=sample_id)),
                "rolling": self.cnn_window.update(t, observation, track)}
        result.update(modes=modes, alert_mode=self.alert_mode,
                      decision=modes[self.alert_mode]["decision"], rolling=modes[self.alert_mode]["rolling"])
        return result
