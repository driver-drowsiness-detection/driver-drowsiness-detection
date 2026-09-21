# MARVEL team environment setup

## Purpose

This guide starts from the first repository check and ends when all four laptops have reproducible local environments. It also defines the separate Jetson Nano audit. Follow the steps in order.

The source code is shared through Git. A virtual environment is local to one computer. Model weights and configuration move between training and deployment, but Windows and Jetson do not share one Python environment.

```text
GitHub main
    |
    +--> Laptop clone --> local .venv --> development and tests
    |
    +--> Training laptop --> local .venv --> trained model artifact
    |
    +--> Jetson clone --> Jetson compatible environment --> inference
```

## Current repository facts

The updated structure is on the GitHub `main` branch at commit `32aef5b`.

The correct clone on Vishruth's laptop is:

```text
C:\Users\alwan\Documents\driver-drowsiness-detection
```

The updated root contains:

```text
configs
data
docs
environments
legacy
models
reports
scripts
src
tests
.gitignore
pyproject.toml
README.md
```

If File Explorer shows `benchmarks` and a root `requirements` file but no `environments`, it is showing the older layout or an unupdated clone. Do not install that root requirements file. The earlier implementation is now preserved under `legacy/phase1_initial`.

## Team rules before setup

1. Each person uses their own clone.
2. Each person creates their own `.venv`.
3. Nobody commits `.venv`.
4. Nobody copies `.venv` between laptops.
5. Nobody installs project packages globally.
6. Windows dependency files are not installed on Jetson.
7. Only tested dependency versions are committed.
8. Python 3.14 may remain installed, but this project will use Python 3.11.
9. If a laptop is not Windows x86 64, report its operating system and architecture before installing the Windows dependency files.
10. Do not begin feature work until the basic environment verification passes.

## Why Python 3.11 is selected

Python 3.14 is newer than the package set previously considered for this project. The archived MediaPipe `0.10.14` candidate provides Windows wheels through Python 3.12, including Python 3.11, but not Python 3.14. The CNN stack also needs a tested combination of Python, PyTorch and related libraries.

Three choices were considered:

| Choice | Benefit | Problem |
| :--- | :--- | :--- |
| Use Python 3.14 | Already installed on some laptops | Native computer vision package support is not established |
| Remove Python 3.14 and install 3.11 | Leaves one version | Unnecessary and may break unrelated projects |
| Keep 3.14 and install 3.11 beside it | Safe separation and broad package compatibility | Team must explicitly select 3.11 when creating `.venv` |

Recommendation: keep Python 3.14 for other work and install 64 bit Python 3.11 beside it. The MARVEL `.venv` must be created with Python 3.11.

Decision record: Used Python 3.11 for the Windows project environment because the selected computer vision candidates have known Python 3.11 wheels, while Python 3.14 compatibility has not been established.

# Step 1: Locate or clone the updated repository

## Vishruth

Paste this into the File Explorer address bar:

```text
C:\Users\alwan\Documents\driver-drowsiness-detection
```

Press Enter and refresh File Explorer. Confirm that `environments` is visible.

Open PowerShell in that folder and run:

```powershell
git rev-parse --show-toplevel
git branch
git status
```

Expected Git root:

```text
C:/Users/alwan/Documents/driver-drowsiness-detection
```

Expected branch:

```text
* main
```

Do not use the older Desktop copy for development.

## Other team members cloning for the first time

Open PowerShell in a local folder such as `Documents`:

```powershell
cd Documents
git clone https://github.com/driver-drowsiness-detection/driver-drowsiness-detection.git
cd driver-drowsiness-detection
git branch
git status
Get-ChildItem environments
```

They should see `main`, a clean working tree and the three environment files.

## Team members with an older clone

First run:

```powershell
git status
```

If the working tree is clean:

```powershell
git switch main
git pull origin main
Get-ChildItem environments
```

If `git status` shows modified or untracked work, do not pull yet. Send Vishruth the complete `git status` output so the work can be preserved before updating.

## Step 1 checkpoint

Each member sends:

```text
Git root:
Current branch:
Git status:
Environments folder visible: yes or no
```

Do not proceed if the Git root points to a Desktop folder containing unrelated files.

# Step 2: Audit every laptop

Different laptops may have different GPUs, drivers, RAM and Python installations. Record facts before installing packages.

Run on Windows:

```powershell
python -V
python -c "import sys; print(sys.executable)"
where.exe python
py -0p
nvidia-smi
```

Some commands may fail. Preserve the exact failure instead of guessing.

If Python can run, also execute:

```powershell
python scripts\inspect_environment.py
```

Each member records:

| Field | Value |
| :--- | :--- |
| GitHub username | Fill this |
| Operating system | Fill this |
| Processor architecture | Fill this |
| Python version | Fill this |
| Python path | Fill this |
| RAM | Fill this |
| GPU | Fill this |
| GPU driver | Fill this |
| Webcam available | Fill this |
| Free storage | Fill this |
| Git root correct | Fill this |

Do not commit full reports containing private usernames or personal filesystem paths. Share a cleaned summary with the team.

# Step 3: Handle an existing Python 3.14 installation

Do not uninstall Python 3.14.

First identify it in the same terminal where `python -V` reports 3.14:

```powershell
python -V
python -c "import sys; print(sys.executable)"
where.exe python
py -0p
```

Possible situations:

1. `py -0p` lists Python 3.14 only. Install Python 3.11 beside it.
2. `py -0p` lists both 3.14 and 3.11. Use `py -3.11` for this project.
3. `python` works but `py` does not. Use the full Python 3.11 executable path after installation.
4. `python` opens Microsoft Store. Disable the Store alias or use the full Python 3.11 path.
5. Python works in an editor but not normal PowerShell. The editor may be injecting its own path. Record `sys.executable` and install or expose the intended project Python.

Install 64 bit Python 3.11 from the official Python distribution. During installation, enable the Python launcher if offered. Adding Python to the user path is helpful, but the project should still select Python 3.11 explicitly.

After installation:

```powershell
py -0p
py -3.11 -V
```

Expected result:

```text
Python 3.11.x
```

If `py` is unavailable, find the installed executable:

```powershell
where.exe python
```

Then use its full path in the next step.

# Step 4: Create the local virtual environment

Run from the correct repository root.

Preferred command when the Python launcher works:

```powershell
py -3.11 -m venv .venv
```

If the launcher is unavailable, use the full Python 3.11 executable path:

```powershell
& "C:\path\to\Python311\python.exe" -m venv .venv
```

Verify the environment:

```powershell
.\.venv\Scripts\python.exe -V
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
.\.venv\Scripts\python.exe -m pip -V
git status
```

Expected Python version:

```text
Python 3.11.x
```

Expected Python path contains:

```text
driver-drowsiness-detection\.venv
```

Expected Git status does not contain `.venv` because `.gitignore` excludes it.

Activation is optional:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, do not change security settings immediately. Use `.\.venv\Scripts\python.exe` directly.

If `.venv` was accidentally created with Python 3.14, confirm that the directory is the project `.venv`, remove only that disposable directory and recreate it explicitly with Python 3.11.

## Step 4 checkpoint

Each member sends:

```text
.venv created: yes or no
.venv Python version:
.venv Python path:
pip version:
.venv absent from git status: yes or no
```

# Step 5: Do not install project requirements yet

At the time this guide was created, these files contain comments only:

```text
environments/windows-runtime.txt
environments/windows-training.txt
```

Installing them now would install nothing.

The archived file below is historical and must not be used as the revised environment source:

```text
legacy/phase1_initial/requirements.txt
```

Vishruth and Arushuuu validate the dependency files first. The juniors wait for the environment pull requests to be reviewed and merged.

# Step 6: Vishruth validates the Windows runtime environment

Vishruth owns the first environment increment.

Update `main` and create a setup branch:

```powershell
git switch main
git pull origin main
git switch -c setup/windows_runtime
```

Create a clean Python 3.11 environment using Step 4.

The first runtime candidate set contains these direct dependency categories:

1. NumPy for arrays and geometry.
2. OpenCV for camera frames and image operations.
3. MediaPipe as the first Windows landmark candidate.
4. PyYAML for configuration files.
5. pytest for automated tests.

Do not install `dlib` during this increment. Do not install PyTorch yet.

The validation target is:

```text
Import dependencies
    |
    v
Open webcam
    |
    v
Read a changing frame
    |
    v
Print height, width and channels
    |
    v
Close cleanly and release the camera
```

After the smoke test succeeds, write the tested versions into:

```text
environments/windows-runtime.txt
```

Record the Python version, laptop details, commands and results in the pull request. A successful import alone is not enough. The camera path must also be checked.

Commit only relevant files:

```powershell
git status
git diff
git add environments\windows-runtime.txt
git add path\to\the\runtime\verification\file
git diff HEAD
git commit -m "Validate Windows runtime environment"
git push origin setup/windows_runtime
```

Open a pull request to `main`. Arushuuu reviews it.

# Step 7: Arushuuu validates the Windows training environment

Arushuuu first completes Steps 1 through 4 and records their laptop GPU and driver.

After the runtime environment pull request is merged:

```powershell
git switch main
git pull origin main
git switch -c setup/windows_training
```

Create a fresh Python 3.11 `.venv`, then install the validated runtime file:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r environments\windows-runtime.txt
```

Validate these training dependency categories:

1. PyTorch.
2. torchvision.
3. scikit learn.
4. pandas.
5. Matplotlib.

The PyTorch build must be chosen using the actual GPU and driver. Do not install a random CUDA toolkit to repair a failed PyTorch check.

Verify PyTorch:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__)"
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

`True` means the selected PyTorch build sees a compatible CUDA device. `False` must be recorded and investigated. CPU execution can still support small tests.

After validation, update:

```text
environments/windows-training.txt
```

Open a pull request to `main`. Vishruth reviews it.

# Step 8: Juniors reproduce the validated environment

The juniors do not choose or change package versions during initial setup.

After the runtime environment pull request is merged, both run:

```powershell
git switch main
git pull origin main
.\.venv\Scripts\python.exe -m pip install -r environments\windows-runtime.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

Verify the package import:

```powershell
.\.venv\Scripts\python.exe -c "import marvel; print(marvel.__file__)"
```

The printed path should point into `src\marvel` in their clone.

Each junior then runs the agreed import and camera checks. If installation fails, send:

```text
Laptop operating system:
Python version:
Python executable:
Command executed:
Complete error:
Package being installed:
What was already attempted:
```

Do not fix dependency failures by globally installing packages.

# Step 9: Begin implementation branches

After environment reproduction succeeds:

## shamaa007

```powershell
git switch main
git pull origin main
git switch -c feature/camera_capture
```

First deliverable:

1. Open the webcam.
2. Read fresh frames.
3. attach a monotonic timestamp.
4. Display frame dimensions.
5. Handle failed camera reads.
6. Exit cleanly.
7. Release the camera.

## niharika526

```powershell
git switch main
git pull origin main
git switch -c feature/event_logger
```

First deliverable:

1. Receive a synthetic observation record.
2. Validate required fields.
3. Write a structured log record.
4. Read the record back.
5. Verify run ID, frame ID, timestamp and validity.
6. Label all synthetic records clearly.

## Vishruth

After the environment branch is merged:

```powershell
git switch main
git pull origin main
git switch -c feature/ear_geometry
```

First ML deliverable: EAR using known pixel coordinates, point count validation and invalid eye width handling.

## Arushuuu

After the environment branch is merged:

```powershell
git switch main
git pull origin main
git switch -c feature/dataset_audit
```

First ML deliverable: dataset source, licence, labels, subject identifiers, split suitability and support for both landmark and CNN evaluation.

# Step 10: Handle different laptop types

| Laptop | Allowed work | Environment rule |
| :--- | :--- | :--- |
| Windows x86 64 with NVIDIA GPU | All runtime work and CUDA training | Python 3.11 plus validated Windows files |
| Windows x86 64 without NVIDIA GPU | Runtime, CPU inference and small CPU tests | Same runtime file; record CPU training limitations |
| macOS | Source work and compatible local testing | Do not install the Windows file until a macOS specification is reviewed |
| Linux x86 64 | Source work and compatible local testing | Do not assume Windows binary parity |
| Jetson Nano ARM64 | Target capture and inference | Use a separate audited Jetson procedure |

Different laptop performance is acceptable. Silent package version drift is not.

# Step 11: Audit Jetson Nano separately

Do not install either Windows dependency file on the Jetson.

Before changing the board, run:

```bash
cat /etc/nv_tegra_release
lsb_release -a
uname -m
python3 -V
which python3
free -h
```

Inspect OpenCV:

```bash
python3 -c "import cv2; print(cv2.__version__); print(cv2.getBuildInformation())"
```

Inspect the CSI camera plugin:

```bash
gst-inspect-1.0 nvarguscamerasrc
```

Save the reviewed facts in:

```text
environments/jetson-runtime.md
```

Do not upgrade JetPack, replace system Python, replace OpenCV or install MediaPipe until this audit is reviewed.

A Jetson environment created with `--system-site-packages` may see JetPack's system OpenCV. It does not solve incompatible Python, CUDA or native binary versions. The decision is made only after checking the installed Python and OpenCV binding.

# Step 12: Daily branch workflow after setup

Before every task:

```powershell
git status
git switch main
git pull origin main
git switch -c category/task_name
```

During work:

```powershell
git status
git diff
```

Before committing:

```powershell
git add exact\file\one.py
git add exact\file\two.py
git diff HEAD
```

After testing:

```powershell
git commit -m "Describe the completed behavior"
git push origin category/task_name
```

Open a pull request. Do not commit feature work directly to `main`.

After merge:

```powershell
git switch main
git pull origin main
```

Create a new branch for the next task. Do not reuse the completed branch.

# Completion checklist

Every member must reach all applicable checks:

1. Correct Git root.
2. Updated `main` branch.
3. `environments` folder visible.
4. Operating system and architecture recorded.
5. Python 3.11 available.
6. `.venv` created using Python 3.11.
7. `.venv` excluded from Git.
8. Validated dependency file installed after merge.
9. `marvel` imports from `src\marvel`.
10. Assigned smoke test passes.
11. First feature branch created from the latest `main`.

# Common mistakes

1. Opening the old clone shown in the screenshot.
2. Creating `.venv` with Python 3.14 accidentally.
3. Installing the archived requirements file.
4. Installing packages globally.
5. Running `pip freeze` in an environment containing unrelated experiments.
6. Letting every member choose different package versions.
7. Installing Windows packages on Jetson.
8. Assuming a successful import proves camera support.
9. Installing CUDA separately before diagnosing PyTorch.
10. Starting feature work from an outdated `main`.

# Interview explanation

We separated Windows runtime, Windows training and Jetson deployment environments because they use different hardware, processor architectures and native libraries. Each contributor recreates a local Python 3.11 environment from reviewed dependency files. The Jetson preserves its JetPack controlled libraries and receives a separate compatibility audit before installation. This makes package choices reproducible without pretending that a Windows environment can be copied to an ARM64 embedded device.

# Retention questions

1. Why can Python 3.14 remain installed while this project uses Python 3.11?
2. Why must each teammate create their own `.venv`?
3. Why is the archived requirements file not the current source of truth?
4. Why can two laptops share package versions but have different training speeds?
5. Why does `--system-site-packages` not solve every Jetson compatibility problem?
6. Why must the camera be tested separately from landmark and CNN inference?
