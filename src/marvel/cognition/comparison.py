"""Replay aligned estimates through three independent experimental histories."""
from dataclasses import asdict
import math

from marvel.contracts import Estimate, UNKNOWN, bilateral
from .ear_classifier import probability, state
from .fusion import combine, score_state, validate_score, _unit
from .perclos import ClosureWindow
from .trigger import Trigger


class Comparison:
    def __init__(self, *, run_id, ear_threshold, ear_calibration, cnn_threshold,
                 fusion_threshold, alpha, close_s, recover_s, max_gap_s,
                 unavailable_s, window_s, min_coverage):
        if not math.isfinite(ear_threshold) or ear_threshold <= 0:
            raise ValueError("EAR threshold must be finite and positive")
        for name, value in (("CNN threshold", cnn_threshold),
                            ("Fusion threshold", fusion_threshold), ("Alpha", alpha)):
            _unit(value, name)
        self.calibration = dict(ear_calibration)
        probability(Estimate("check", 0, "check", "left", "ear", None, False), self.calibration)
        self.ear_threshold, self.cnn_threshold = ear_threshold, cnn_threshold
        self.fusion_threshold, self.alpha = fusion_threshold, alpha
        self.triggers = {mode: Trigger(close_s, recover_s, max_gap_s, unavailable_s,
                                      mode=mode, run_id=run_id)
                         for mode in ("ear", "cnn", "fusion")}
        self.windows = {mode: ClosureWindow(window_s, max_gap_s, min_coverage)
                        for mode in self.triggers}
        self.last = None

    @staticmethod
    def _align(estimates, method, sample_id, t, track):
        # Reject the whole branch when extra, duplicate or stale eyes occur.
        estimates = list(estimates)
        for estimate in estimates:
            if estimate.method != method:
                raise ValueError(f"Expected {method} evidence")
            if method == "cnn":
                validate_score(estimate, method)
            elif estimate.valid and estimate.value < 0:
                raise ValueError("Raw EAR cannot be negative")
        aligned = (track is not None and len(estimates) == 2
                   and {e.side for e in estimates} == {"left", "right"}
                   and all((e.sample_id, e.t, e.track) == (sample_id, t, track)
                           for e in estimates))
        if aligned:
            return sorted(estimates, key=lambda e: e.side)
        return [Estimate(sample_id, t, track or "", side, method, None, False,
                         "missing_or_unaligned_branch") for side in ("left", "right")]

    def update(self, *, sample_id, t, track, ear, cnn):
        # Validate all evidence before advancing any mode's clock.
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("A sample identifier is required")
        if not math.isfinite(t) or (self.last is not None and t <= self.last):
            raise ValueError("Time must be finite and strictly increasing")
        if track is not None and (not isinstance(track, str) or not track.strip()):
            raise ValueError("Track must be nonempty or None")
        ears = self._align(ear, "ear", sample_id, t, track)
        cnns = self._align(cnn, "cnn", sample_id, t, track)
        mapped = [probability(e, self.calibration) for e in ears]
        fused = ([combine(e, c, alpha=self.alpha) for e, c in zip(mapped, cnns)]
                 if track is not None else
                 [Estimate(sample_id, t, "", side, "fusion", None, False, "no_track")
                  for side in ("left", "right")])
        observations = {
            "ear": bilateral(*(state(e, self.ear_threshold) for e in ears)),
            "cnn": bilateral(*(score_state(c, self.cnn_threshold) for c in cnns)),
            "fusion": bilateral(*(score_state(f, self.fusion_threshold) for f in fused)),
        }
        evidence = {"ear": ears, "cnn": cnns, "fusion": fused}
        modes = {}
        for mode, observation in observations.items():
            decision = self.triggers[mode].update(t, observation, track, sample_id=sample_id)
            modes[mode] = {
                "estimates": [asdict(e) for e in evidence[mode]],
                "decision": asdict(decision),
                "rolling": self.windows[mode].update(t, observation, track),
            }
        self.last = t
        return {"sample_id": sample_id, "t": t, "track": track,
                "experimental": True, "modes": modes,
                "common_observable": all(s != UNKNOWN for s in observations.values())}
