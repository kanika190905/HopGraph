"""Cross-platform task runner, the Windows equivalent of the Makefile targets.

    python scripts/tasks.py data      --config configs/musique_200.yaml
    python scripts/tasks.py index     --config configs/musique_200.yaml
    python scripts/tasks.py eval      --config configs/musique_200.yaml [--tag day3]
    python scripts/tasks.py reproduce --config configs/musique_200.yaml
    python scripts/tasks.py test
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run([PY, *args], cwd=ROOT, check=True)


def data(cfg: str) -> None:
    run("scripts/download_data.py", "musique_dev")
    run("scripts/build_corpus.py", "--config", cfg)


def index(cfg: str) -> None:
    run("scripts/build_bm25.py", "--config", cfg)
    run("scripts/embed.py", "--config", cfg)
    run("scripts/build_graph.py", "--config", cfg)


def evaluate(cfg: str, tag: str) -> None:
    run("scripts/run_eval.py", "--config", cfg, *(["--tag", tag] if tag else []))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["data", "index", "eval", "reproduce", "test"])
    ap.add_argument("--config", default="configs/musique_200.yaml")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.task == "data":
        data(a.config)
    elif a.task == "index":
        index(a.config)
    elif a.task == "eval":
        evaluate(a.config, a.tag)
    elif a.task == "reproduce":
        data(a.config)
        index(a.config)
        evaluate(a.config, a.tag)
    else:
        run("-m", "pytest", "-q")


if __name__ == "__main__":
    main()
