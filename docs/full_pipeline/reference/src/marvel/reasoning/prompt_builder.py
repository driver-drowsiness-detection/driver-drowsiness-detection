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
