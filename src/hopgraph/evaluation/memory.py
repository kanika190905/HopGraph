"""Peak-RSS sampling with a hard safety limit."""

from __future__ import annotations

import _thread
import logging
import threading
import time
from types import TracebackType

import psutil

log = logging.getLogger(__name__)


class RSSMonitor:
    """Samples this process's RSS every ``interval`` seconds in a daemon thread.

    If RSS exceeds ``limit_gb`` the main thread is interrupted (KeyboardInterrupt)
    so the stage aborts instead of pushing the machine into swap.
    """

    def __init__(self, interval: float = 0.1, limit_gb: float | None = 12.0):
        self.interval = interval
        self.limit_bytes = int(limit_gb * 1024**3) if limit_gb else None
        self.peak_bytes = 0
        self.start_bytes = 0
        self.exceeded = False
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss
            self.peak_bytes = max(self.peak_bytes, rss)
            if self.limit_bytes and rss > self.limit_bytes and not self.exceeded:
                self.exceeded = True
                log.error("RSS %.2f GB exceeded limit %.2f GB; aborting stage",
                          rss / 1024**3, self.limit_bytes / 1024**3)
                _thread.interrupt_main()
            self._stop.wait(self.interval)

    def __enter__(self) -> RSSMonitor:
        self.start_bytes = self._proc.memory_info().rss
        self.peak_bytes = self.start_bytes
        self._thread = threading.Thread(target=self._run, daemon=True, name="rss-monitor")
        self._thread.start()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.interval + 1)
        self.peak_bytes = max(self.peak_bytes, self._proc.memory_info().rss)

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / 1024**2

    @property
    def start_mb(self) -> float:
        return self.start_bytes / 1024**2


def current_rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024**2


def wait_for_sample(seconds: float = 0.25) -> None:
    """Let the monitor take at least one sample after a short stage."""
    time.sleep(seconds)
