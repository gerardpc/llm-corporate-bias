"""Optional tracing helpers used by experiment runners."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass
class NoOpTrace:
    """Trace object with the same method shape used by experiment code."""

    def update_current_trace(self, **_: Any) -> None:
        """Accept trace updates without sending them anywhere."""


@contextmanager
def trace_context(*_: Any, **__: Any) -> Iterator[NoOpTrace]:
    """Provide a local no-op tracing context."""
    yield NoOpTrace()
