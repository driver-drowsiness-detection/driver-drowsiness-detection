from dataclasses import asdict
from marvel.contracts import UNKNOWN, bilateral
from marvel.cognition.ear_classifier import measure, probability, state as ear_state
from marvel.cognition.fusion import combine, state as score_state
from marvel.cognition.trigger import Trigger
from marvel.cognition.perclos import ClosureWindow
from marvel.cognition.state_machine import FatigueState


class Pipeline:
    def __init__(self, config, bundle=None, cnn=None):
        self.config, self.bundle, self.cnn = config, bundle, cnn
        self.modes = ("ear", "cnn", "fusion") if config["mode"] == "compare" else (config["mode"],)
        if any(m != "ear" for m in self.modes) and (bundle is None or cnn is None):
            raise ValueError("CNN and fusion need a frozen bundle")
        self.triggers = {m: Trigger(**config["temporal"]) for m in self.modes}
        self.windows = {m: ClosureWindow(**config["window"]) for m in self.modes}
        self.tiers = {m: FatigueState(config["fatigue_policy"]) for m in self.modes}
        self.track = "not_acquired"

    def update(self, sample_id, t, eyes):
        observations = {m: UNKNOWN for m in self.modes}
        evidence = []
        if len(eyes) == 2:
            if len({e.track for e in eyes}) != 1 or {e.side for e in eyes} != {"left", "right"}:
                raise ValueError("Invalid bilateral identity")
            if any(e.sample_id != sample_id or e.t != t for e in eyes):
                raise ValueError("Frame and eye identity mismatch")
            self.track = eyes[0].track
            states = {m: [] for m in self.modes}
            for eye in eyes:
                ear = measure(eye)
                evidence.append(asdict(ear))
                if "ear" in states:
                    threshold = self.bundle["ear_threshold"] if self.bundle else self.config["ear_threshold"]
                    states["ear"].append(ear_state(ear, threshold))
                if self.cnn:
                    cnn = self.cnn.estimate(eye)
                    evidence.append(asdict(cnn))
                    if "cnn" in states:
                        states["cnn"].append(score_state(cnn, self.bundle["cnn_threshold"]))
                    if "fusion" in states:
                        fused = combine(probability(ear, self.bundle["ear_calibration"]), cnn, self.bundle["alpha"])
                        evidence.append(asdict(fused))
                        states["fusion"].append(score_state(fused, self.bundle["fusion_threshold"]))
            observations = {m: bilateral(*values) for m, values in states.items()}
        results = {}
        for mode, observation in observations.items():
            result = self.triggers[mode].update(t, observation, self.track)
            window = self.windows[mode].update(t, observation, self.track)
            tier = self.tiers[mode].update(t, window["closure_fraction"], result["alarm_active"],
                                         observation != UNKNOWN, self.track)
            results[mode] = dict(result, observation=observation, window=window, tier=tier)
        return {"type": "observation", "sample_id": sample_id, "t": t,
                "track": self.track, "evidence": evidence, "modes": results}
