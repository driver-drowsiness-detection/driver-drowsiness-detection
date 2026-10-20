# Jetson camera and perception handoff

Do not install the Windows lock or copy a Windows environment onto the Nano.
The same source contracts are the target, but actual Python/OpenCV/MediaPipe
compatibility determines whether the full program can run on this board.

## 1. Pull the reviewed integration commit

```bash
git status --short
git fetch origin
git switch --track origin/codex/perception-integration
git rev-parse HEAD
```

Save existing work before switching. If the branch already exists, switch to it
and `git pull --ff-only origin codex/perception-integration`. For older Git without
`switch`, use `git checkout -b codex/perception-integration origin/codex/perception-integration`
once. Compare the commit with the laptops' acceptance records.

## 2. Native preview and board audit

In the graphical desktop, rerun the team's previously working command:

```bash
gst-launch-1.0 -e nvarguscamerasrc sensor-id=0 ! nvvidconv ! 'video/x-raw,format=I420' ! xvimagesink sync=false
```

Stop with Ctrl+C before Python opens the camera. Then record:

```bash
cat /proc/device-tree/model
cat /etc/nv_tegra_release
cat /etc/os-release
uname -m
python3 -V
python3 -c "import cv2; print(cv2.__version__); print(cv2.__file__); print(cv2.getBuildInformation())"
```

GStreamer must show YES in OpenCV build information. Preserve the working camera
stack. Changing a ribbon requires shutdown and disconnected power first.

## 3. Python camera with an older board interpreter

For Python 3.6 or later with working system OpenCV and NumPy:

```bash
python3 scripts/jetson_camera_probe.py
```

This standalone probe imports neither the MARVEL package, YAML nor MediaPipe. It
uses a camera worker so its parent can report a stalled read and handle q. It
prints shape, frame ID and read age. Press q with its window focused and restart.
For diagnostics without a desktop window:

```bash
python3 scripts/jetson_camera_probe.py --no-display --max-frames 30
```

The default pipeline is a candidate with BGR appsink. No sensor resolution or FPS
is forced. Pass `--pipeline 'your validated pipeline'` if the board requires
different negotiated caps. Capture failure is a camera-stack issue to inspect;
do not replace system OpenCV with a pip wheel as a blind repair.

## 4. Full package compatibility gate

The full laptop environment is tested on Python 3.11 Windows x86-64, not Nano
ARM64. Before any installation, establish an interpreter with a compatible
OpenCV binding, NumPy, PyYAML and MediaPipe Tasks package on that exact JetPack.
If such a combination is unavailable, report the audit and choose a supported
landmark deployment; camera success does not remove this blocker.

Once a compatible full environment is demonstrated, from the repository root:

```bash
python3 scripts/preview_perception.py camera configs/jetson-camera.yaml
python3 scripts/download_landmarker.py
python3 scripts/preview_perception.py perception configs/jetson-camera.yaml --image /path/to/reviewed-face.png
python3 scripts/preview_perception.py perception configs/jetson-camera.yaml
```

Stop after the first failing command and preserve the full error. Do not proceed
from native preview straight to live landmarks without testing Python capture and
saved-image inference. Record measured FPS and read age before considering any
threshold or model optimization. Use PERCEPTION_ACCEPTANCE.md for all hardware
checks. Model hash must match the laptops.
