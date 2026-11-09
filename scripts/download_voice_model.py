"""Download a pinned public instruction model for local, offline inference."""
import hashlib
import json
import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

REPO = "HuggingFaceTB/SmolLM2-360M-Instruct"
REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"
FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
         "tokenizer_config.json", "special_tokens_map.json", "merges.txt", "vocab.json", "README.md"]


def main():
    from huggingface_hub import snapshot_download
    target = Path(__file__).resolve().parents[1] / "models/llm/SmolLM2-360M-Instruct"
    snapshot_download(REPO, revision=REVISION, local_dir=target, allow_patterns=FILES, token=False)
    hashes = {}
    for name in FILES:
        with (target / name).open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = dict(repo=REPO, revision=REVISION, license="apache-2.0", files=hashes,
                    purpose="experimental_local_alert_wording", status="download_complete")
    (target / "model-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(dict(path=str(target), revision=REVISION, files=len(hashes))))


if __name__ == "__main__":
    main()
