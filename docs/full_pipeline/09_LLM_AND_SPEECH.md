# 09. LLM contract, endpoint options and speech

## Requirement and current decision

The user confirmed on 4 October 2026 that the LLM endpoint has not been selected. Keep model and endpoint **undecided**. Proceed with asynchronous context-aware speech after an immediate fixed alert, and supply options rather than pretending one is chosen.

The original proposal prefers on-device language generation, with a paired local device as a fallback. This guide preserves that preference.

## Understand the components

| Component | Purpose | Example |
| --- | --- | --- |
| Model weights | Learned language parameters | A separately chosen small instruction model |
| Quantization | Reduce numeric precision/storage | A reviewed 4-bit GGUF artifact |
| Runtime | Executes the model | A compatible llama.cpp build |
| Endpoint | A process exposes a request interface | Local chat-completions URL |
| Prompt | Instructions and measured context | Event kind and closure duration |
| Validation | Check output belongs to the event and obeys policy | Exact event ID and approved sentence |
| TTS | Produce speech from text | SAPI, eSpeak-NG or selected Piper voice |
| TTL | Maximum lifetime of an event's speech request | Discard old results after the deadline |

Model, runtime and endpoint are separate decisions. A GGUF file does not automatically include a running server. An API-compatible endpoint does not prove model quality or Nano compatibility.

## Deployment options from the project direction

| Option | Data flow | Benefit | What must be measured |
| --- | --- | --- | --- |
| On Nano | Telemetry stays on the board | Closest to the on-device aim | Build support, total RAM, CV slowdown, first-token/completion latency, thermals |
| Paired laptop on local network | Only small event telemetry leaves Nano | More compute headroom; proposal fallback | Network loss, reconnection, response delay and disclosure of hybrid deployment |
| Fixed sentences only | No language model inference | Reliable reference fallback | Whether it satisfies the final demonstration requirement; it does not prove LLM integration |

Develop the interface with a mock first. Evaluate candidate local models later. The proposal names Gemma, Qwen and SmolLM families; these are candidate families only. Verify the exact model card, licence, parameter size, quantization artifact and runtime support before selecting a release. Do not copy the proposal's blanket licence or throughput claims.

For model memory, start with `parameters * bits / 8` as **weights-only lower-order accounting**, then add runtime buffers, tokenizer, context/KV cache, vision memory, operating system and speech engine. Measure peak resident memory and swap. A model that loads alone may fail when the camera and CNN are active.

## Why speech is asynchronous

At event creation the runtime requests a fixed sound immediately, then optionally starts language generation. Camera and cognition continue while the worker operates. A slow or malformed response cannot veto the warning.

The reference worker has capacity one. A new event cancels the old request before replacing it. Its dedicated process can be terminated at a hard deadline; a thread timeout alone would not necessarily stop blocked inference work.

The worker returns text to the main process. It never operates the speaker itself. The main process checks whether the same event is still current before requesting speech.

## Initial output policy: constrained language

The supplied first policy asks the LLM to select one approved sentence for the measured event and return:

```json
{"event_id":"run:method:sequence","message":"Prolonged eye closure detected. Please stop somewhere safe and rest."}
```

`event_id` must match exactly. Unexpected fields or a message outside the allowed set are rejected. Failure uses a fixed fallback. This is deliberately constrained LLM output selection, not unrestricted original sentence generation. It establishes integration and failure handling but offers limited language-model value compared with a deterministic template.

If the final project requires freely phrased context-aware sentences, treat that as a later policy change. Keep the same asynchronous contract, add factual consistency tests and a reviewed output evaluator, compare against templates, and report unsupported claims and fallback rate. A length check or prohibited-word list alone cannot prove factual correctness.

The prompt contains only event ID, kind, method and measured closure duration. It does not include raw frames, names, biometric identity, ground-truth labels or invented road conditions. The LLM does not classify fatigue and cannot advise that it is safe to continue driving.

## Follow the worker lifecycle

1. `submit` refuses a second in-flight item.
2. It creates a child process and records submission monotonic time.
3. `_run` invokes the selected backend and catches errors into a bounded fallback result.
4. `poll` checks current event identity and TTL first.
5. If a response exists, return it.
6. If the hard timeout expires, terminate the process and return fallback.
7. On recovery or a changed track, discard pending work.
8. On shutdown, terminate/join the worker and close its queue.

Socket timeout and worker timeout serve different purposes. The parent process deadline still limits a server that trickles data slowly or an unexpected library hang.

## Test without a real endpoint

Edit the full runtime configuration's LLM section:

```yaml
llm:
  enabled: false
  backend: mock
  endpoint: null
  model: null
  timeout_s: 3.0
  ttl_s: 5.0
```

Run:

```powershell
python scripts\check_llm.py configs\runtime.yaml
```

Expected: `source` is `mock`, and the result contains a valid approved message. This checks the adapter path; it does not call an LLM.

The test-only `test_slow` backend deliberately sleeps so the hard deadline can be verified. Never call its output real model performance.

## Connect a selected endpoint later

The supplied `local_chat` adapter supports the local chat-completions contract documented by llama.cpp. The chosen server must accept `messages`, `model`, `temperature`, `max_tokens`, `stream` and the requested JSON response format, and return `choices[0].message.content`.

Set `backend: local_chat`, fill `model` with the actual configured model name, and fill `endpoint` with the actual local URL ending in the chat-completions route. The code rejects missing values, external public hosts, embedded credentials and redirects. It disables proxy inheritance for the local request.

An Ollama runtime is another proposal option, but API compatibility must be checked against its selected version. This package does not claim every OpenAI-style server supports every field. If the endpoint differs, write and test a separate adapter rather than silently ignoring response-format failures.

There is no real server launch command here because the model, runtime build and endpoint are undecided. The complete client and test harness are supplied. After selection, record the exact server launch command, model hash, licence and measured limits in the decision register.

## TTS options

| Engine | Reference status | Decision criterion |
| --- | --- | --- |
| Windows SAPI | Adapter supplied for Windows development | Audible smoke test and preemption |
| eSpeak-NG | Adapter supplied for Linux if executable exists | Small runtime, intelligibility and board package support |
| Piper | Proposal's neural TTS candidate; target adapter not selected | Current engine/voice licence, compatible build, memory and audio latency |

For Windows, use `tts: windows_sapi` after a listening test. For a compatible Linux environment, use `tts: espeak`. Leave `tts: disabled` until the engine is ready. LLM and TTS are separate: a good generated message can still fail to play.

The current maintained Piper repository is linked in Guide 12. Inspect voice licences separately from the engine. The guide does not install an unverified neural voice on Nano.

## Required failure tests

Test: no endpoint, connection refused, slow server, oversized response, malformed JSON, mismatched event ID, unsupported text, full worker, recovery before completion, driver change, TTL expiry, TTS failure, new fixed alert during speech and shutdown during generation.

Measure CV timing with speech disabled, mock speech, real LLM generation, and LLM plus TTS. Independent processes can still compete for RAM, CPU and memory bandwidth. Asynchronous design reduces blocking but does not eliminate resource contention.

Used an optional worker because the warning must not depend on a text model. Used strict initial sentence validation because a demonstration needs predictable event wording before experimenting with freer language.

Interview explanation: “The language model receives measured telemetry after the alert decision. We bound work, validate event identity, discard stale results and retain fixed output when generation fails.”

Retention exercise: explain why a 3-second HTTP timeout does not by itself prove that the worker process can never exceed 3 seconds.

<!-- CODE -->

## Complete source files

### File: `src/marvel/reasoning/prompt_builder.py`

[Open the companion file](reference/src/marvel/reasoning/prompt_builder.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `build`: lines 12 to 22.
* `validate`: lines 25 to 31.
* `fallback`: lines 34 to 35.

```python
import json

MESSAGES = {
    "PROLONGED_CLOSURE": (
        "Prolonged eye closure detected. Please stop somewhere safe and rest.",
        "Your eyes have stayed closed. Please take a break somewhere safe."),
    "MONITORING_UNAVAILABLE": (
        "Driver monitoring is unavailable. Please check the camera when safely stopped.",)
}


def build(event):
    kind = event["kind"]
    if kind not in MESSAGES:
        raise ValueError("This event has no speech policy")
    telemetry = {k: event[k] for k in ("event_id", "kind", "method", "closure_s")}
    system = ("You verbalize a measured monitoring event. Return JSON containing only "
              "event_id and message. Copy event_id exactly. Choose one of the supplied "
              "approved messages verbatim. Do not infer diagnosis, road conditions, "
              "driver identity, recovery or permission to continue driving.")
    user = json.dumps({"telemetry": telemetry, "approved_messages": MESSAGES[kind]}, allow_nan=False)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate(text, event):
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"event_id", "message"}:
        raise ValueError("Unexpected response fields")
    if value["event_id"] != event["event_id"] or value["message"] not in MESSAGES[event["kind"]]:
        raise ValueError("Unsupported message or mismatched event")
    return value["message"]


def fallback(event):
    return MESSAGES[event["kind"]][0]
```

### File: `src/marvel/reasoning/llm_inference.py`

[Open the companion file](reference/src/marvel/reasoning/llm_inference.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `generate`: lines 8 to 45.

```python
import json
import ipaddress
import urllib.request
from urllib.parse import urlparse
from .prompt_builder import build, validate, fallback


def generate(config, event):
    if config["backend"] == "mock":
        return {"message": fallback(event), "source": "mock"}
    if config["backend"] == "test_slow":
        import time
        time.sleep(10)
        return {"message": fallback(event), "source": "test_slow"}
    if config["backend"] != "local_chat":
        raise ValueError("Unknown LLM backend")
    endpoint, model = config.get("endpoint"), config.get("model")
    if not endpoint or not model:
        raise ValueError("LLM endpoint and model are undecided")
    url = urlparse(endpoint)
    host = url.hostname
    local = host == "localhost"
    if not local:
        try:
            address = ipaddress.ip_address(host)
            local = address.is_private or address.is_loopback
        except ValueError:
            local = False
    if url.scheme not in ("http", "https") or not local or url.username or url.password:
        raise ValueError("Configure an explicit local endpoint without embedded credentials")
    body = json.dumps({"model": model, "messages": build(event), "temperature": 0,
                       "max_tokens": 100, "stream": False,
                       "response_format": {"type": "json_object"}}).encode()
    request = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json"})
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=config["timeout_s"]) as response:
        raw = response.read(32769)
    if len(raw) > 32768:
        raise ValueError("LLM response too large")
    data = json.loads(raw)
    message = validate(data["choices"][0]["message"]["content"], event)
    return {"message": message, "source": "local_chat"}
```

### File: `src/marvel/reasoning/worker.py`

[Open the companion file](reference/src/marvel/reasoning/worker.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `_run`: lines 7 to 13.
* `SpeechWorker`: lines 16 to 58. Capacity one. Parent owns deadlines, event freshness and all sound output.

```python
import multiprocessing as mp
import queue
import time
from .prompt_builder import fallback


def _run(config, event, output):
    from .llm_inference import generate
    try:
        output.put(generate(config, event))
    except Exception as exc:
        output.put({"message": fallback(event), "source": "fallback",
                    "error": type(exc).__name__ + ": " + str(exc)[:200]})


class SpeechWorker:
    """Capacity one. Parent owns deadlines, event freshness and all sound output."""
    def __init__(self, config):
        self.config, self.process, self.output = config, None, None
        self.event = self.started = None

    def submit(self, event):
        if self.process is not None:
            return False
        context = mp.get_context("spawn")
        self.output = context.Queue(maxsize=1)
        self.event, self.started = dict(event), time.monotonic()
        self.process = context.Process(target=_run, args=(self.config, self.event, self.output))
        self.process.start()
        return True

    def poll(self, current_event_id):
        if self.process is None:
            return None
        age = time.monotonic()-self.started
        if current_event_id != self.event["event_id"] or age > self.config["ttl_s"]:
            self.close()
            return {"source": "discarded", "message": None}
        try:
            result = self.output.get_nowait()
        except queue.Empty:
            if age <= self.config["timeout_s"]:
                return None
            result = {"message": fallback(self.event), "source": "timeout_fallback"}
        result["event_id"] = self.event["event_id"]
        result["worker_seconds"] = age
        self.close()
        return result

    def close(self):
        if self.process is not None:
            self.process.join(0.02)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(0.2)
            self.output.cancel_join_thread()
            self.output.close()
        self.process = self.output = None
```

### File: `scripts/check_llm.py`

[Open the companion file](reference/scripts/check_llm.py). Copy the complete block to this path in your reviewed implementation.

Reading map, using line numbers in the companion file:

* `main`: lines 9 to 23.

```python
"""python scripts/check_llm.py configs/runtime.yaml"""
import json
import sys
import time
from pathlib import Path
from marvel.reasoning.worker import SpeechWorker


def main():
    import yaml
    config = yaml.safe_load(Path(sys.argv[1]).read_text())["llm"]
    event = {"event_id": "contract-test:1", "kind": "PROLONGED_CLOSURE", "method": "fusion", "closure_s": 2.0}
    worker = SpeechWorker(config)
    try:
        worker.submit(event)
        while True:
            result = worker.poll(event["event_id"])
            if result:
                print(json.dumps(result, indent=2))
                break
            time.sleep(0.02)
    finally:
        worker.close()


if __name__ == "__main__":
    main()
```

