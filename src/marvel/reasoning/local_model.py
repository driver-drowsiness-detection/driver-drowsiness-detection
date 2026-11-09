"""Pinned offline text generation. This module cannot decide whether to alert."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"
FALLBACK = "Your eyes stayed closed; please stop in a safe place and take a break."


def validate_event(event):
    if (event.get("kind") != "PROLONGED_CLOSURE"
            or not isinstance(event.get("event_id"), str) or not event["event_id"]
            or not isinstance(event.get("episode_id"), str) or not event["episode_id"]
            or not isinstance(event.get("closure_s"), (int, float))
            or not math.isfinite(event["closure_s"]) or event["closure_s"] <= 0):
        raise ValueError("Voice requires a confirmed closure event with identifiers and duration")


def checked_wording(text):
    """Conservative format/content filter, not a proof of semantic safety."""
    text = text.strip().strip('"')
    lower = text.lower()
    if (not 5 <= len(text.split()) <= 30 or len(text) > 220
            or not re.fullmatch(r"[A-Za-z][A-Za-z ,;'\-]*[.!]?", text)
            or "safe" not in lower or not any(x in lower for x in ("stop", "pull over", "break"))
            or any(x in lower for x in ("asleep", "microsleep", "unconscious", "continue driving",
                                       "speed up", "close your eyes", "diagnos", "ignore"))):
        return FALLBACK, "fixed_fallback"
    return text, "local_llm"


class LocalGenerator:
    def __init__(self, folder, device=None):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        folder = Path(folder)
        manifest = json.loads((folder / "model-manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("repo") != MODEL_ID or manifest.get("revision") != REVISION
                or manifest.get("status") != "download_complete"):
            raise ValueError("Expected the pinned, completely downloaded voice model")
        for name, expected in manifest["files"].items():
            if Path(name).name != name:
                raise ValueError("Invalid model manifest filename")
            with (folder / name).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError("Model asset hash mismatch: " + name)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(2)
        self.tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            folder, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()

    def generate(self, event):
        validate_event(event)
        messages = [
            {"role": "system", "content": "Write exactly one short English warning for a driver. "
             "Use only the reported event. Do not diagnose sleep or fatigue. "
             "Ask the driver to stop in a safe place and take a break. "
             "No numbers, explanations, lists or quotation marks."},
            {"role": "user", "content": "The monitoring system detected prolonged eye closure. "
             "Write the spoken warning in one sentence."},
        ]
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                    return_tensors="pt", return_dict=True).to(self.device)
        start = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=48, do_sample=False,
                                         repetition_penalty=1.1,
                                         pad_token_id=self.tokenizer.eos_token_id)
        raw = self.tokenizer.decode(output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        elapsed = time.perf_counter() - start
        text, source = checked_wording(raw)
        return dict(event_id=event["event_id"], episode_id=event["episode_id"], text=text,
                    source=source, raw_text=raw, generation_s=elapsed, model=MODEL_ID,
                    revision=REVISION, device=self.device)
