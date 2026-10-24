"""Download benchmark files into data/raw/ and verify their SHA-256.

    python scripts/download_data.py musique_dev          # 30.4 MB
    python scripts/download_data.py musique_train        # 241 MB (router labels / tuning)

Sources are pinned to a Hugging Face revision. The two mirrors listed in
docs/novelty_review.md (dgslibisey/MuSiQue, bdsaglam/musique) serve byte-identical
copies of the official MuSiQue v1.0 files; the official release is a Google Drive zip
(github.com/StonyBrookNLP/musique). MuSiQue is licensed CC BY 4.0. The data is not
redistributed in this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_HF = "https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/c8f4f8c9465fb69d31a8eae894c3fd509c4ca321"

FILES = {
    "musique_dev": {
        "url": f"{_HF}/musique_ans_v1.0_dev.jsonl",
        "path": "data/raw/musique/musique_ans_v1.0_dev.jsonl",
        "sha256": "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b",
        "bytes": 30_439_728,
    },
    "musique_train": {
        "url": f"{_HF}/musique_ans_v1.0_train.jsonl",
        "path": "data/raw/musique/musique_ans_v1.0_train.jsonl",
        "sha256": "83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a",
        "bytes": 241_046_755,
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str) -> Path:
    spec = FILES[name]
    dest = ROOT / spec["path"]
    if dest.exists() and sha256(dest) == spec["sha256"]:
        print(f"{name}: already present and verified ({dest})")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"{name}: downloading {spec['bytes'] / 1e6:.1f} MB from {spec['url']}")
    urllib.request.urlretrieve(spec["url"], tmp)
    digest = sha256(tmp)
    if digest != spec["sha256"]:
        tmp.unlink()
        sys.exit(f"{name}: SHA-256 mismatch ({digest}); file discarded")
    tmp.replace(dest)
    print(f"{name}: verified -> {dest}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", choices=sorted(FILES))
    for name in ap.parse_args().names:
        fetch(name)


if __name__ == "__main__":
    main()
