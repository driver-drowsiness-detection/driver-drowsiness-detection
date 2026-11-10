"""Fetch checksum-verified official RT-BENE files for a bounded public pilot."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import subprocess

FILES = {
    "rt_bene_subjects.csv": "a07810daaf6d2053a9ed1f782e5f360c",
    "s003_blink_labels.csv": "4523566e1784c65fea7d57c6465f0926",
    "s003_noglasses_eyes.tar": "e5455af8d9acd1be8eb11118529b9a4e",
    "s007_blink_labels.csv": "1b62a3a2c45efbd0af3ef36900151ead",
    "s007_noglasses_eyes.tar": "7ca6f32635c4d8d70a075fd1bcd195a8",
    "s012_blink_labels.csv": "7722524624f3dcf53116e7081a7de5c7",
    "s012_noglasses_eyes.tar": "e925025262025b3c6145c42b08255ba1",
    "s015_blink_labels.csv": "825af2e4609c3f1534438777fbf7c45e",
    "s015_noglasses_eyes.tar": "cc2547496bb28ed5298f11c587b67fc6",
    "s016_blink_labels.csv": "c449b5da2fee9ecc1c8cbb0f15d179f3",
    "s016_noglasses_eyes.tar": "d64f7e91f234ab3ac3ddbf19bbef427a",
}


def fetch(item):
    name, expected = item
    root = Path("data/raw/rt_bene")
    root.mkdir(parents=True, exist_ok=True)
    path = root/name
    def checksum(file):
        with file.open("rb") as handle:
            return hashlib.file_digest(handle, "md5").hexdigest()
    if path.exists():
        if checksum(path) != expected:
            raise ValueError("Existing file checksum mismatch: " + name)
        return name + " verified"
    partial = path.with_suffix(path.suffix + ".part")
    url = "https://zenodo.org/records/3685316/files/" + name + "?download=1"
    # A total deadline prevents a server that trickles data from hanging forever.
    subprocess.run(["curl.exe", "--fail", "--location", "--retry", "1", "--max-time", "600",
                    "--silent", "--show-error", "--output", str(partial), url], check=True)
    if checksum(partial) != expected:
        raise ValueError("Download checksum mismatch: " + name)
    partial.replace(path)
    return name + " downloaded and verified"


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(fetch, FILES.items()):
            print(result, flush=True)
