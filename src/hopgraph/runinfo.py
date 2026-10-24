"""Run metadata: logging, Git commit, environment and run IDs."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_PACKAGES = (
    "numpy", "scipy", "pandas", "networkx", "scikit-learn", "torch", "sentence-transformers",
    "transformers", "faiss-cpu", "bm25s", "PyStemmer", "spacy", "en-core-web-sm", "pyyaml", "psutil",
)


def setup_logging(run_id: str = "-", level: int = logging.INFO, log_file: Path | None = None) -> None:
    fmt = f"%(asctime)s [{run_id}] %(levelname)s %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    for noisy in ("httpx", "urllib3", "huggingface_hub", "sentence_transformers", "filelock",
                  "bm25s", "faiss.loader"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _git() -> str | None:
    return os.environ.get("GIT_EXECUTABLE") or shutil.which("git")


def git_commit() -> dict[str, str | bool | None]:
    exe = _git()
    if exe is None:
        return {"commit": None, "dirty": None, "note": "git not found (set GIT_EXECUTABLE)"}
    try:
        commit = subprocess.run([exe, "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        status = subprocess.run([exe, "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": commit, "dirty": bool(status)}
    except (subprocess.CalledProcessError, OSError) as e:
        return {"commit": None, "dirty": None, "note": f"git failed: {e}"}


def environment() -> dict:
    versions = {}
    for pkg in _PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    try:
        import psutil

        mem_gb = round(psutil.virtual_memory().total / 1024**3, 2)
        cores = psutil.cpu_count(logical=False)
    except ImportError:  # pragma: no cover
        mem_gb, cores = None, None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "physical_cores": cores,
        "logical_cores": os.cpu_count(),
        "ram_gb": mem_gb,
        "packages": versions,
    }


def run_id(config_hash: str, commit: str | None, data_hash: str) -> str:
    blob = json.dumps([config_hash, commit, data_hash]).encode()
    return hashlib.sha256(blob).hexdigest()[:10]
