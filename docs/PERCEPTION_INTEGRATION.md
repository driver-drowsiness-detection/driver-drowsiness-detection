# Comparison, decisions and verification

Specification: `MARVEL_PERCEPTION_LAYER_REFERENCE.md`, supplied on 11 October 2026.
Reference file SHA256: `d22eb9e3877cc79d65d399b412eafda9b4be3d1be57a9e5a71c299c607443a02`.
Its current endpoint is camera and eye perception. Mouth/pose, EAR/CNN inference,
temporal decisions, alarms and speech remain separate work.

## Inspected branch snapshots

| Branch | Inspected commit | Contribution | Gap relative to reference |
| --- | --- | --- | --- |
| main | 0dadd584332836343951e944ac8c7e503a1d364d | Scaffold and team guides | No executable perception preview |
| perception | d1399d6e863385e497d1b432f7a7a56c9a00975d | Camera adapters, landmark indices, preliminary orchestration | Missing imports/types, blocking capture, largest-face selection, no preview entry point |
| perception01 | 154aa8591b460f0cdcda411ca0d96007cb55901d | Bounds-safe crop copy, validity helpers and tests | Different FramePacket, no landmark-to-crop geometry, no integrated capture/detection |

The remote also has `cnn` at 730d436f9cdd6864a87e630457ffaafb11bd8405.
The integration does not merge its training code. The canonical contracts file is
identical to that branch's file so later CNN integration uses the same records.

## Resolved differences

| Area | Final behavior |
| --- | --- |
| Shared types | `marvel.contracts.Frame` and `Eye`; old perception contracts module only reexports these |
| Package imports | Lightweight __init__; camera mode never loads MediaPipe |
| Capture | Spawned worker, one queued frame, separate error pipe, bounded read and shutdown |
| Windows camera | Preserve contributor's DirectShow backend; other camera platforms use CAP_ANY |
| Jetson | CAP_GSTREAMER with explicit build-support check; source mode negotiation left automatic |
| Timestamps | Monotonic read-completion seconds; session-prefixed advancing sample IDs |
| Landmarks | One MediaPipe Tasks IMAGE model loaded once; BGR to RGB, original pixel coordinates |
| Driver rule | Exactly one detected face globally and inside the ROI; multiple faces reject as ambiguous |
| Crops | Corner distance determines 1.6x width / 0.8x height, strict bounds, independent image copy |
| Quality | Coordinates, crop dimensions, exposure and measured sharpness, explicit invalid reasons |
| Freshness | Checks before AND after detection; stale output carries no eye evidence |
| Display | Separate canvas, eye points and rectangles, metrics and validity, fresh UNAVAILABLE panels |
| Entry point | `preview_perception.py camera` or `perception`; optional saved-image/headless checks |
| Reproduction | Windows dependency lock, versioned model URL and required SHA256, automated checks |

Keep useful tests from perception01, migrating its packet tests onto Frame. The
old incomplete Perception orchestrator was replaced by Detector plus the freshness
boundary. Video replay, automatic reopening, pose gates and EAR calculations from
the initial branch are outside this reference endpoint and remain recoverable in
the original commit history. There is one runnable implementation in src/marvel.

The reference's default `laplacian_min: 0.0` is retained and explicitly announced
at runtime: it measures sharpness but does not reject blur. A measured nonzero
threshold can be placed in `configs/local-*.yaml` before blur acceptance. No
threshold is presented as validated for driving.

This version retains the reference's conservative policy: crop quality also gates
geometry validity. This does not change the contract's ability to represent
independent validity flags. Rejected crops are set to None to prevent accidental
inference on invalid evidence. Geometry points remain available for debugging.

## Stable handoff

```python
from marvel.contracts import Frame
from marvel.perception.detection import Detector
from marvel.perception.pipeline import process_frame

# Construct Detector once using the validated configuration.
# LiveCapture.read() returns a NEW Frame or None; never reuse a previous Frame.
eyes, status, metrics, age = process_frame(frame, detector, max_age=1.0)
for eye in eyes:
    # Preserve sample_id, t, track, side.
    # EAR reads eye.points only if eye.geometry_ok.
    # CNN reads eye.crop only if eye.crop_ok.
    pass
```

`Detector.process(frame)` returns exactly `(eyes, status, metrics)`, matching the
new reference. Live consumers call process_frame to enforce freshness. Missing
face/ambiguous/stale input produces `eyes=[]`; selected but rejected eyes have
false flags, a missing crop and reasons. Later state logic must map either case
to UNKNOWN. The CNN branch's older offline preparation script expects a different
four-value detector result and an EAR measure function; adapt that script during
the later CNN/geometry merge. It is not part of this live perception endpoint.

## Verification performed locally

Windows x86-64; isolated Python 3.11.4 environment. Exact packages are recorded in
`environments/windows-perception-lock.txt`. `pip check` reported no broken requirements.

- Final run: 46 automated checks passed on 12 October 2026, including geometry,
  contracts, quality, freshness, config, real spawned workers, CLI operation from
  another directory, q during unavailable input, and Python 3.6 probe syntax.
- Real Face Landmarker asset loaded, processed a blank image as no_face, and closed.
- Webcam delivered 640x480 BGR frames; camera preview opened and closed automatically.
- Camera reopened for a second run. Initial integrated use of CAP_ANY timed out.
  Restoring the contributor's Windows DirectShow backend resolved the observed
  startup failure in the subsequent run; this is one-machine evidence, not a
  universal claim about camera drivers.
- Integrated headless webcam run completed 10 frames; both eyes were reported
  valid with crops (20x40 and 21x41 pixels in the logged observation). That
  observation's read age was approximately 0.047 seconds. No images were saved.

The controlled human checklist (visual alignment, blink, leave/return, second
person, occlusion, dim lighting, q/restart) remains for each laptop. No Jetson was
connected to this session. Jetson native preview, Python capture and landmark
compatibility must be recorded separately. Synthetic tests verify code behavior;
they do not establish real-world accuracy or occlusion detection.

## References

The supplied reference is the project specification. MediaPipe API usage was also
checked against Google's [Face Landmarker Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python).
The chosen IMAGE mode has synchronous inference; a stuck detector is not covered
by the camera-worker timeout. See the separate acceptance record and Jetson guide.
