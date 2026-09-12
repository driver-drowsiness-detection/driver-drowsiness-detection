# Team setup and first deliverables

Updated 12 September 2026. This guide supplements the existing Phase 1 design and records Vishruth's requested senior ML pair and independent junior runtime pair. This guide describes the migration branch and ownership plan. Organization permissions and project-board setup are separate from this scaffold migration.

## 1. What exists now

The revised target tree from section 7 of PHASE_1_SYSTEM_DESIGN.md has been scaffolded. Source modules contain responsibility notes. Application entry points report that they are unimplemented. Tests contain pending scenarios, not passing tests. Configurations and dependency lists intentionally contain no invented settings or tested versions.

The operational utility is scripts/inspect_environment.py. It reports the interpreter, operating system, architecture, virtual environment status and actual Git root. It does not access the camera, install software or certify ML compatibility.

## 2. Dedicated Git clone and scaffold migration

The working clone is now C:/Users/alwan/Documents/driver-drowsiness-detection. This location has its own .git directory and origin points to https://github.com/driver-drowsiness-detection/driver-drowsiness-detection.git.

The previous Desktop folder inherited a repository rooted at the entire Desktop. It remains a local reference copy. Open the Documents clone for ongoing development and do not commit project work from the old folder.

The remote main branch contained initial baseline code and tests at commit 1188bc7663e423364e92e0c4cb4756bbaf31a05d. The migration preserves every tracked file from that revision byte for byte under legacy/phase1_initial, including the old requirements, README and tests. The revised src/marvel scaffold is the current development target. Legacy code is retained for inspection and selective reuse; it has not been ported into the new contracts.

The migration is prepared on codex/revised-phase1-scaffold for review into main. Arushuuu should review the pull request before merging. Other members should clone main after merge. To inspect the proposal before merge, fetch and switch to the migration branch explicitly.

Confirm the checkout from your editor terminal:

```powershell
git rev-parse --show-toplevel
git remote -v
git status
```

On Vishruth's machine, the first command should show C:/Users/alwan/Documents/driver-drowsiness-detection. Each teammate uses their own clone path. The remote should identify the shared organization repository. Each person creates their own virtual environment in their own clone.

## 3. Team division

Vishruth requested that the two experienced members work together on most ML work, while the two juniors own independent engineering deliverables. This supersedes the earlier mixed-experience pairing. Vishruth supervises junior work through issue definitions and checkpoint reviews. Seniors do not implement the juniors' assigned modules for them.

Pair A, ML and experiments: alwandi-vishruth-0407 and Arushuuu.

Pair B, runtime and tooling: shamaa007 and niharika526.

| Person | Primary implementation ownership | Reviewer | First deliverable |
| :--- | :--- | :--- | :--- |
| alwandi-vishruth-0407 | Landmark adapter, eye crops and quality, EAR/MAR, EAR classifier, fusion, temporal controller, calibration and fusion tuning | Arushuuu | Define observation contracts; implement and explain geometric baseline with known-coordinate tests |
| Arushuuu | Dataset audit and splits, training dataset and transforms, CNN architecture, training, inference, frozen evaluation and model provenance | Vishruth | Dataset suitability decision and label policy; manifest validation; model compatibility check before extensive training |
| shamaa007 | Webcam/video capture, preview entry point, overlays, frame timestamps and capture/overlay tests | niharika526 first; Vishruth for acceptance | Independent webcam preview with dimensions, monotonic timestamps, clear failure handling and clean shutdown |
| niharika526 | Structured logger, fixed audio adapter, environment inspection improvements, profiling and logging/audio tests | shamaa007 first; Vishruth for acceptance | Log synthetic observations and play one bounded sound from a synthetic event, each independently testable |

Both seniors jointly design and cross-review the complete ML pipeline. Primary ownership identifies who implements a change and avoids simultaneous conflicting edits; it does not isolate either senior from the other's ML work. Each senior must reproduce and explain the other's major experiment. Alternate the driver during shared debugging and retain individual commits and experiment notes.

All module paths below are relative to src/marvel unless shown otherwise.

| Area | Implementation owner | Boundary |
| :--- | :--- | :--- |
| perception/detection.py, crops.py, quality.py | Vishruth | Produce named points, valid eye crops and explicit invalid reasons |
| cognition/features.py, ear_classifier.py, fusion.py, trigger.py | Vishruth | Geometry, compatible score fusion and timestamp-based decisions |
| training/ and cognition/cnn_architecture.py, cnn_inference.py | Arushuuu | Same preprocessing and class mapping in training and inference |
| scripts/prepare_dataset.py, train_cnn.py, evaluate.py | Arushuuu | Subject split integrity, training evidence and frozen evaluation |
| scripts/calibrate_scores.py, tune_fusion.py | Vishruth | Calibration and validation partitions have different jobs |
| data/, models/, training.yaml, windows-training.txt | Arushuuu, with Vishruth review | Dataset and model choices remain senior-owned ML work |
| experiments.yaml and runtime decision thresholds | Both seniors; Vishruth coordinates | Juniors must not guess model thresholds or experiment policy |
| perception/capture.py, observability/overlay.py | shamaa007 | Work with raw frames and explicit display records |
| observability/logger.py, actuation/audio.py | niharika526 | Consume records/events; never infer drowsiness inside logging or audio |
| scripts/inspect_environment.py, profile_runtime.py | niharika526 | Report observed values and distinguish unavailable measurements |
| main.py | shamaa007 owns preview and lifecycle; Vishruth owns later ML wiring | Separate issues and handoff before editing the same file |
| contracts.py, pyproject.toml | Both seniors; Vishruth coordinates | Publish a small agreed interface before parallel implementation |
| windows-runtime.txt and runtime source/display/audio settings | Juniors verify their setup; Vishruth reviews | Record tested versions and evidence, not arbitrary package pins |
| environments/jetson-runtime.md | niharika526 records runtime inventory; both seniors assess model compatibility | Inspection can begin before trained weights exist |
| tests/ | Each implementation owner | Existing five test placeholders belong to seniors; juniors add capture, overlay, logger and audio tests as those behaviours are built |

The juniors own real software modules, their tests and their setup instructions. They work together to diagnose routine issues, then bring Vishruth a concise blocker report with the command, expected result, actual result and attempted fixes. Independent work still needs a reviewed interface and bounded tasks.

## 4. How the pairs connect

```text
JUNIORS: shamaa007 + niharika526
Camera/video -> FramePacket (image + source + ID + observation time)
                            |
                            v
SENIORS: Vishruth + Arushuuu
Landmarks -> eye crops and quality -> EAR / CNN / calibrated fusion
                                                |
                                                v
                               Duration controller -> observation + event
                                                            |
                                                            v
JUNIORS:                                      overlay + logger + fixed audio

SENIORS OFFLINE:
Dataset decisions -> subject splits -> training -> calibration -> validation
                                         |                           |
                                         v                           v
                                   saved weights             frozen settings
                                         +--------------+------------+
                                                        v
                                          runtime and final evaluation
```

Before a real detector exists, juniors pass synthetic OPEN, CLOSED and UNKNOWN records and synthetic alert events into their display, logger and audio modules. Mark these records as synthetic, use separate test run IDs and keep them out of accuracy reports. Synthetic inputs test engineering behaviour; they provide no evidence of model performance.

This removes the trained-model dependency from the juniors' first deliverables. Their camera module can already produce real timestamped frames. Neither logger nor audio should decide whether someone is drowsy: they consume the seniors' records and events.

## 5. What everyone initializes individually

1. Accept access to the same organization repository using your own GitHub account. Configure your own Git identity. Do not share accounts or credentials.
2. Install or verify Git and an editor. Open the dedicated clone as the editor's project root. Juniors should learn which folder their terminal currently points at.
3. Verify that an actual Python interpreter is installed. Run python -V and python -c "import sys; print(sys.executable)". If Python is missing or opens a Store prompt, record that blocker. The leads must choose the laptop Python version after checking candidate library support; do not independently install four different latest versions.
4. Run the environment inspection script with an available Python interpreter. Record its output and your RAM, GPU and webcam status in the environment checklist below. Review local usernames in paths before sharing output publicly.
5. After both leads select and record a supported laptop interpreter, create your own .venv in your own clone. Never copy another person's .venv. Never commit it.
6. Select .venv/Scripts/python.exe in your editor. Use that interpreter for the terminal and editor run button.
7. Install the dependency file for your pair only after its pins have been tested and added. Both files currently contain comments only. A successful installation of an empty file proves nothing.
8. Make one small branch, commit, push and pull request. A personal setup record with real evidence is enough for the first PR.

Use these commands from your dedicated clone, once python resolves to the agreed interpreter:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
.\.venv\Scripts\python.exe scripts/inspect_environment.py
```

Command 1 creates an isolated local environment. Command 2 shows exactly which Python will run your code. Command 3 prints the setup inventory. Calling that interpreter directly avoids needing to change PowerShell execution policy to activate it.

After dependency validation, the junior runtime pair uses:

```powershell
.\.venv\Scripts\python.exe -m pip install -r environments/windows-runtime.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

The senior ML pair uses the reviewed windows-training.txt and also needs the common runtime dependencies for integration. The seniors should define and test how those dependency files share common requirements. The -e option installs this project's package in editable mode: edits in src/marvel remain visible to imports.

The first dependency increment is a tested camera setup. The training environment comes after the Python, framework and optional GPU compatibility check. Jetson has its own environment record and installation procedure.

## 6. One-time organization setup by the supervisor

1. Verify all four members have the repository access they need. Membership alone is not proof of write access to this repository.
2. Use the existing shared repository. Optionally create runtime and learning-data GitHub teams for ownership; four separate repositories are unnecessary.
3. Agree main as the integration branch, matching the currently observed remote branch.
4. Require pull request review for main if the repository's settings and plan support enforcement. Otherwise explicitly follow the same review rule as a team convention.
5. Create small issues with an owner, reviewer, affected files and evidence of completion. Use Backlog, In progress, Review and Done columns if a board helps.
6. Merge the scaffold before the other three begin changing modules. Agree file ownership and contracts at the first team meeting.

The dedicated clone and migration branch are prepared. Organization membership, branch rules and project-board instructions remain a supervisor checklist unless separately verified.

## 7. First working session

All four complete the setup inventory and explain clone, branch, commit, push and pull request. The two seniors publish the smallest useful FramePacket, display record and AlertEvent contracts, including timestamp units, source ID, valid flags and synthetic-data markers. Use section 8 of the system design as the starting point. Do not wait for every future field to be designed before starting the preview.

Senior ML pair:

1. Vishruth uses the dedicated Documents clone. Arushuuu reviews the scaffold migration PR before the team starts module work from main.
2. Together select suitable data based on licence, eye labels, subject identity and support for both geometry and CNN evaluation. Arushuuu records the decision and initial label policy.
3. Vishruth begins EAR on known pixel coordinates and invalid-geometry cases. This can be tested before a detector or dataset is ready.
4. Arushuuu defines a grouped manifest and deterministic preprocessing, then investigates a compatible CNN and training environment. Training waits for valid data and a checked input policy.
5. Cross-review geometry, splitting and preprocessing. Both must explain leakage prevention and how a score becomes a temporal observation.

Junior runtime pair:

1. shamaa007 independently implements a minimal webcam preview, then dimensions, timestamps and clean shutdown. First explain an image as height by width by channels. Produce a clear error if camera opening or reading fails.
2. niharika526 first runs and explains the existing environment script. Then implement a logger that writes a small set of synthetic observations with run ID, frame ID, timestamp and validity. Read the file back and verify those fields.
3. niharika526 adds a separately callable fixed audio adapter. Verify that one synthetic event requests one sound and that sound failure is reported. Do not embed alert classification in it.
4. Together connect capture to overlay and logger. Use the agreed synthetic records where ML outputs are needed. Verify output is labelled synthetic.
5. Each junior reviews and explains the other's module. Submit small PRs to Vishruth with a demonstration and at least one relevant failure case.

Vishruth reviews completed increments rather than sitting beside every coding step. If a task is too broad, reduce it to one observable behaviour. Do not replace a junior's implementation with a complete senior-written solution. A short hint or interface example can unblock them while preserving ownership.

Book the first Jetson inspection early. Juniors can inventory runtime and capture behaviour; seniors must assess detector and CNN compatibility. Do not wait until the final demo to test that assumption.

## 8. Progression and completion evidence

| Gate | Senior ML pair | Junior runtime pair | Evidence |
| :--- | :--- | :--- | :--- |
| First | Contracts, data audit, synthetic-coordinate EAR tests | Camera preview, synthetic-record logger, isolated audio | Real setup records; independently reproducible demos |
| Next | Landmarks, valid crops, grouped dataset and preprocessing | Video replay, overlays and runtime failure tests | Timestamp and frame ID agreement |
| Then | EAR controller, CNN training and inference | Integrated display/log/audio adapters using actual outputs | Same input records arrive at all outputs |
| Then | Calibration, fusion, frozen evaluation | Profiling and repeatable scenario runner | Measured results separated from synthetic test data |
| Finally | Model export compatibility and interpretation of results | Jetson capture/output checks and reproduction guide | Reproducible target demonstration |

Junior learning progression, after core deliverables:

1. shamaa007 moves from image arrays and capture to video timing, bounding-box overlays, invalid observations and capture tests.
2. niharika526 moves from Python dictionaries and JSON records to event handling, audio errors, profiling and automated verification.
3. After the seniors establish the baseline, both juniors run inference with frozen weights and explain crop size, colour order and class mapping. They can reproduce a bounded experiment on development data with senior review. They do not tune against the final test set.

LLM speech, a mouth CNN, a polished dashboard and extensive optimization remain later work. Both seniors keep their own experiment notes: question, implementation, data split, configuration, result, failure analysis and contribution. Resume statements must describe completed personal work and measured evidence; being assigned a module is not yet a contribution.

## 9. Beginner Git workflow

A repository stores project history. A clone is your local copy. A branch holds a focused change. A commit is a local snapshot. A push sends commits to GitHub. A pull request asks others to review before merging. A virtual environment isolates Python packages and is unrelated to a Git branch.

For a new small task, first check status and finish or preserve any existing work. Then:

```powershell
git switch main
git pull --ff-only
git switch -c codex/camera-preview
```

Use your own task name for each branch. Do not have both pair members push unrelated work onto one shared branch. After implementing and checking a change, stage only the relevant files in your editor, inspect the staged diff, commit and push the branch. Open a pull request and name the reviewer.

Include: what changed, why, how to run it, evidence, and remaining limitations. If you see a conflict, identify the other file owner and resolve it together. Do not overwrite their version blindly. After merge, everyone updates main before starting the next task.

## 10. Environment checklist

Fill real values; unknown is acceptable.

| Member | OS | Python version and path | RAM | GPU and driver | Webcam tested | Dedicated Git root confirmed | Blocker |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| alwandi-vishruth-0407 | Windows 11 observed | Personal interpreter setup pending | Pending | Pending | Pending | Yes: dedicated Documents clone | Select project Python environment |
| Arushuuu | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| shamaa007 | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| niharika526 | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

## 11. Explain the working script

scripts/inspect_environment.py uses only standard library modules:

1. json formats the output as a readable record.
2. platform reads Python version, OS and machine architecture.
3. shutil.which checks whether Git is available on PATH.
4. subprocess runs a fixed read-only Git command and captures success or the exact error. The timeout prevents an indefinite wait.
5. sys identifies the current interpreter and whether it belongs to a virtual environment.
6. Path locates the project from the script's location, so the report does not depend on accidentally opening a terminal elsewhere.
7. git_root compares Git's reported root with the project directory. That comparison exposes the Desktop repository issue.
8. main assembles the measured values and explicitly lists checks it has not performed.
9. The __name__ guard runs the report when the file is executed, while allowing its functions to be imported without running it.

No prediction algorithm is implemented at this stage. The first vision algorithm to explain and test later is EAR: average vertical eye opening divided by eye width.

Interview note for the planned project: Vishruth and Arushuuu develop the geometric baseline, learned eye classifier, temporal decision logic and calibrated comparison. The junior pair builds camera input, runtime outputs and tooling against shared contracts. The same preprocessing and timing policies support meaningful comparisons. Say planned until the modules and evidence actually exist.

Retention check: each junior should explain why an eye crop must keep its source frame ID, why neighbouring frames should not be randomly split across training and test, and why a .venv must be recreated on another laptop.

## References

Python environment instructions: https://docs.python.org/3/library/venv.html

Repository cloning: https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository

Organization teams: https://docs.github.com/en/organizations/organizing-members-into-teams/about-teams
