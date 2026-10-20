# Perception hardware acceptance record

Copy this file for each laptop and for the Jetson. Record actual observations;
do not change unchecked boxes to passed because an automated test succeeded.
Keep local reports under ignored `reports/local/` and share a cleaned summary.

Date / tester:
Machine / OS / architecture:
Git branch and full commit (`git rev-parse HEAD`):
Uncommitted changes (`git status --short`):
Python version / interpreter:
Dependency lock used / differences:
Model SHA256:
Command / configuration:
Camera source / image shape:
Observed processed FPS / read age:

| Test | Expected | Actual / pass / fail |
| --- | --- | --- |
| Camera startup | Moving image, advancing IDs and timestamps | |
| q and restart | Windows close; camera reopens | |
| Single face | Correct anatomical eyes and six points each | |
| Closed eyes | Crops keep area and remain valid with adequate quality | |
| Leave / return | no_face, empty panels; new track on return | |
| Second detected face | multiple_faces, empty panels | |
| Outside driver ROI | outside_driver_roi, empty panels | |
| Frame edge | Out-of-bounds crops are unavailable | |
| Dim / bright light | Scores reported; configured exposure gate applied | |
| Blur | Set a measured nonzero threshold and record behavior | |
| One eye covered | Record actual behavior and limitation | |
| Camera unavailable | Explicit open/read failure or bounded timeout | |
| Camera stops producing | Waiting panel, eye panels clear, q remains usable | |
| Slow inference | stale_frame after processing; no usable eye output | |

Known limitations / failures:
Next action / owner:

Run for about five minutes for the controlled review. This is eye perception,
not a drowsiness diagnosis. Default `laplacian_min: 0.0` reports sharpness but
does not reject blur. A predicted landmark can remain on a covered eye.

Jetson additionally records board model, JetPack/L4T, OS, Python, OpenCV location
and GStreamer support. Separate native preview, Python camera, saved-image
landmarks, and live perception results. Never mark all four from one successful
camera window.
