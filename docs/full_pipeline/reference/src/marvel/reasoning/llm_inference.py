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
