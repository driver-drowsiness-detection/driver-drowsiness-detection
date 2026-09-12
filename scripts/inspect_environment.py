"""Print a read-only setup inventory using only the Python standard library."""

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def git_root(project):
    """Return Git's actual root, or its error, without changing the repository."""
    if shutil.which("git") is None:
        return {"error": "Git was not found on PATH"}
    try:
        result = subprocess.run(
            ["git", "-C", str(project), "rev-parse", "--show-toplevel"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}
    if result.returncode != 0:
        return {"error": result.stderr.strip()}
    root = Path(result.stdout.strip()).resolve()
    return {"path": str(root), "matches_project": root == project}


def main():
    project = Path(__file__).resolve().parents[1]
    report = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "os": platform.platform(),
        "architecture": platform.machine(),
        "virtual_environment": sys.prefix != sys.base_prefix,
        "project_directory": str(project),
        "git": git_root(project),
        "manual_checks_pending": [
            "RAM capacity",
            "GPU model and driver, if present",
            "Webcam permission and actual capture",
            "Python and package compatibility for the selected backend",
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
