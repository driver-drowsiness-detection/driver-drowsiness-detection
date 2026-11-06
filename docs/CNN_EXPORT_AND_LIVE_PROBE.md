# CNN export, parity and laptop live-path check

**Run date:** 6 November 2026

The experimental MRL checkpoint was exported to ONNX opset 11 on Arushi's
Windows laptop. The export is a software handoff candidate, not a validated
Jetson model or a live driver-warning policy.

## What ran

| Check | Observed result |
| --- | --- |
| Export dependency | ONNX 1.19.0 with existing protobuf 4.25.9; `pip check` clean |
| Exported artifact | `models/exports/mrl_full_3epoch_opset11.onnx` (local and ignored by Git) |
| ONNX SHA256 | `da843bb78db4696989b855abbce26597a16ff39c6d8153a461820a15b354d756` |
| Source checkpoint SHA256 | `f33202048a72f14e0aeb2aa4d5dae622af036d56619c52e815dca534d8bb75bd` |
| ONNX checker | Passed; graph reports opset 11 |
| Laptop OpenCV DNN parser | Loaded the graph successfully |
| Parity sample | 100 **validation** eyes spread across four subjects and both classes |
| Maximum absolute logit error | `0.0000042915` |
| Decisions at raw logit zero | 0 disagreements on the 100 sampled eyes |
| MRL test images | Not used |

The parity script checks both absolute and relative tolerance of `1e-4`,
plus the raw 0.5-score decision. This small laptop sample does not prove
Jetson parser support or parity on all images. The local JSON report is
`reports/runs/mrl_onnx_parity_v2.json`.

The existing preview was extended with optional `--cnn-checkpoint` and
`--cnn-onnx`. Both show **experimental raw closure scores**, preserve the
shared eye identity and return a missing score for an invalid current crop.
Neither option makes an alert decision. The `--cnn-onnx` path loads only
OpenCV DNN for the CNN, although perception still needs its own compatible
face-landmark runtime.

Two no-display laptop webcam runs completed, each receiving three frames and
reopening the camera on the second run. In each run the printed observation
had a valid face, two valid eye crops and two CNN scores. No images were
recorded. The saved local JSON lines are
`reports/runs/cnn_live_probe_v1.jsonl` and `v2.jsonl`. This is an integration
smoke check, **not** eye-state accuracy evidence: those webcam eyes have no
independent labels. Three frames cannot measure dependable FPS, error rate,
lighting performance or camera reliability.

## Reproduce on the laptop

From the `cnn` branch and repository root, after installing the existing
Windows training environment and acquiring the local checkpoint:

```powershell
.\.venv-cognition\Scripts\python.exe -m pip install -r environments/windows-export.txt
.\.venv-cognition\Scripts\python.exe -m pip check
.\.venv-cognition\Scripts\python.exe scripts/export_onnx.py models/checkpoints/mrl_full_3epoch.pt models/exports/mrl_full_3epoch_opset11_v2.onnx
.\.venv-cognition\Scripts\python.exe scripts/check_onnx.py data/processed/mrl_manifest_v1.csv models/checkpoints/mrl_full_3epoch.pt models/exports/mrl_full_3epoch_opset11_v2.onnx reports/runs/mrl_onnx_parity_v3.json
.\.venv-cognition\Scripts\python.exe scripts/preview_perception.py perception --cnn-onnx models/exports/mrl_full_3epoch_opset11_v2.onnx --no-display --max-frames 3 --max-seconds 20
```

Use new output paths: export and parity scripts refuse overwrite. For a
visible preview, omit `--no-display` and press `q` to close. The exported
ONNX and adjacent `.json` metadata must travel together; the runtime checks
their hashes and preprocessing policy. The code and documentation are in Git,
but the licensed source images, checkpoint, ONNX file and local reports are
ignored. A teammate's clone does **not** contain the model artifact.

## Remaining acceptance gates

1. **Representative labels:** The user confirmed that no independently
   labelled RGB webcam/Jetson eyes or NTHU-DDD recordings are available yet.
   Without them, camera accuracy, closed-eye recall, false-eye-closure rate,
   subject generalization and the chosen operating point cannot be verified.
   `scripts/extract_frames.py` can make a blank human-review queue from an
   actual video, but cannot supply the missing labels or real subject IDs.
2. **Decision policy:** The user prioritizes missed closures. A provisional
   95%-recall calibration rule was evaluated, but it called 52.61% of open
   validation eyes closed. This is not safe to freeze as the live rule.
3. **Jetson:** The board's installed Python, OpenCV, camera, landmarker and
   ONNX parser need inspection. Run a saved-image parser/parity check there,
   then camera integration and measured latency/memory/FPS. No result from
   this laptop establishes board compatibility.
4. **Project outcome:** CNN alone classifies one eye crop. Paired EAR/CNN
   comparison, bilateral state and time-based alert evaluation need labelled
   recordings and integration with Vishruth's EAR branch.

Do not run the held-out MRL test split until a model and eye-state decision
rule are selected and frozen. This export does not change the checkpoint or
make its temperature fit reliable on validation data.
