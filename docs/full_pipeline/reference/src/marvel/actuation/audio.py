import math
import os
import shutil
import struct
import subprocess
import time
import wave


class Audio:
    def __init__(self, folder, enabled, tts):
        from pathlib import Path
        self.enabled, self.tts = enabled, tts
        self.speech = self.tone = None
        self.speech_started = 0
        self.tone_until = 0
        self.path = Path(folder)/"fixed_alert.wav"
        if enabled:
            with wave.open(str(self.path), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(16000)
                f.writeframes(b"".join(struct.pack("<h", int(7000*math.sin(2*math.pi*900*i/16000)))
                                       for i in range(6400)))

    def stop_speech(self):
        if self.speech and self.speech.poll() is None:
            self.speech.terminate()
            try:
                self.speech.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                self.speech.kill()
        self.speech = None

    def alert(self):
        self.stop_speech()
        if not self.enabled:
            return "sound_disabled"
        self.tone_until = time.monotonic()+0.5
        if os.name == "nt":
            import winsound
            winsound.PlaySound(str(self.path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            return "sound_requested"
        if not shutil.which("aplay"):
            return "sound_failed_aplay_missing"
        if self.tone and self.tone.poll() is None:
            self.tone.terminate()
        self.tone = subprocess.Popen(["aplay", "-q", str(self.path)],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "sound_requested"

    def speak(self, message):
        if not self.enabled or self.tts == "disabled":
            return "speech_disabled"
        self.stop_speech()
        kwargs = {"stdin": subprocess.PIPE, "stdout": subprocess.DEVNULL,
                  "stderr": subprocess.DEVNULL, "text": True}
        if self.tts == "windows_sapi" and os.name == "nt":
            # Message is input DATA. It is never interpolated into shell code.
            script = ("Add-Type -AssemblyName System.Speech; "
                      "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                      "$sentence = [Console]::In.ReadToEnd(); $speaker.Speak($sentence)")
            self.speech = subprocess.Popen(["powershell.exe", "-NoProfile", "-NonInteractive",
                                            "-Command", script], creationflags=0x08000000, **kwargs)
        elif self.tts == "espeak" and shutil.which("espeak-ng"):
            self.speech = subprocess.Popen(["espeak-ng", "--stdin"], **kwargs)
        else:
            return "speech_failed_backend_unavailable"
        self.speech.stdin.write(message)
        self.speech.stdin.close()
        self.speech_started = time.monotonic()
        return "speech_requested"

    def poll(self):
        if self.speech and self.speech.poll() is not None:
            result = "speech_completed" if self.speech.returncode == 0 else "speech_failed"
            self.speech = None
            return result
        if self.speech and time.monotonic()-self.speech_started > 10:
            self.stop_speech()
            return "speech_timeout"
        if self.tone and self.tone.poll() is not None:
            result = "sound_completed" if self.tone.returncode == 0 else "sound_failed"
            self.tone = None
            return result
        return None

    def close(self):
        self.stop_speech()
        if self.tone and self.tone.poll() is None:
            self.tone.terminate()
        if self.enabled and os.name == "nt":
            import winsound
            winsound.PlaySound(None, 0)
