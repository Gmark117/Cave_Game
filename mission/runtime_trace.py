"""Thread-safe JSONL runtime tracing for diagnosing live missions."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from config.simulation_config import TraceConfig


class RuntimeTraceLogger:
    """Write compact structured mission events to a JSONL file."""

    def __init__(
        self,
        project_root: Path,
        config: TraceConfig,
    ) -> None:
        """Create a trace file when tracing is enabled."""
        self.config = config
        self.enabled = bool(config.enabled)
        self.path: Path | None = None
        self._lock = threading.RLock()
        self._file = None
        self._last_interval: dict[str, float] = {}

        if not self.enabled:
            return

        directory = Path(config.directory)
        if not directory.is_absolute():
            directory = Path(project_root) / directory
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = directory / f"mission_trace_{timestamp}.jsonl"
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self.record("trace_started", path=str(self.path))

    def record(self, event: str, **fields: Any) -> None:
        """Append one event payload to the trace file."""
        if not self.enabled or self._file is None:
            return

        payload = {
            "event": str(event),
            "wall_time": time.time(),
            "perf_time": time.perf_counter(),
        }
        payload.update(
            {
                str(key): self._normalize(value)
                for key, value in fields.items()
            }
        )
        line = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self._lock:
            if self._file is not None:
                self._file.write(line + "\n")

    def should_record_interval(
        self,
        key: str,
        now: float,
        interval: float,
    ) -> bool:
        """Return True when an interval-gated event is due."""
        if not self.enabled:
            return False
        if interval <= 0.0:
            return True
        previous = self._last_interval.get(key)
        if previous is not None and (now - previous) < interval:
            return False
        self._last_interval[key] = now
        return True

    def close(self) -> None:
        """Flush and close the trace file."""
        with self._lock:
            if self._file is None:
                return
            self.record("trace_closed")
            self._file.close()
            self._file = None

    @classmethod
    def disabled(cls) -> RuntimeTraceLogger:
        """Return a disabled trace logger for tests and null defaults."""
        return cls(Path.cwd(), TraceConfig(enabled=False))

    @staticmethod
    def _normalize(value: Any) -> Any:
        """Convert common runtime values into JSON-safe structures."""
        if is_dataclass(value):
            return RuntimeTraceLogger._normalize(asdict(value))
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [RuntimeTraceLogger._normalize(item) for item in value]
        if isinstance(value, list):
            return [RuntimeTraceLogger._normalize(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): RuntimeTraceLogger._normalize(item)
                for key, item in value.items()
            }
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)
