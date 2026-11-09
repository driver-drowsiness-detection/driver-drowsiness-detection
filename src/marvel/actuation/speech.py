"""Offline Windows speech synthesis to a file; text is passed as data."""
import json
from pathlib import Path
import subprocess
import sys
import wave

SCRIPT = """
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$data = ConvertFrom-Json -InputObject ([Console]::In.ReadToEnd())
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $speaker.SetOutputToWaveFile($data.path)
    $speaker.Speak($data.text)
} finally { $speaker.Dispose() }
"""


def render_speech(text, path):
    if sys.platform != "win32":
        raise RuntimeError("Windows SAPI is the current TTS backend; Jetson adapter is pending")
    if not isinstance(text, str) or not text.strip() or len(text) > 220:
        raise ValueError("Speech must be a nonempty short sentence")
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    result = subprocess.run(
        ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", SCRIPT],
        input=json.dumps({"path": str(path), "text": text}), capture_output=True,
        text=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if result.returncode:
        raise RuntimeError("Speech synthesis failed: " + result.stderr[-500:])
    with wave.open(str(path), "rb") as stream:
        if stream.getnframes() == 0:
            raise RuntimeError("Speech synthesis produced empty audio")
        seconds = stream.getnframes() / stream.getframerate()
    return {"audio_path": str(path), "audio_seconds": seconds, "tts_backend": "windows_sapi"}
