"""Tracing hooks.

Provider-neutral tracing. Spans are written to the supplied state's ``trace`` list so
that benchmarks and reports can consume them. A LangSmith provider can wrap this
later without changing call sites.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.state import ResearchState


@contextmanager
def trace_span(
    name: str,
    state: ResearchState | None = None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Open a span. If ``state`` is provided, the span is appended to ``state.trace``."""

    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": dict(attributes or {}),
        "duration_seconds": None,
        "status": "ok",
    }
    try:
        yield span
    except Exception as exc:
        span["status"] = "error"
        span["error"] = repr(exc)
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
        if state is not None:
            state.add_trace_event(name, span)


def export_trace(state: ResearchState) -> list[dict[str, Any]]:
    """Return a JSON-friendly snapshot of the trace timeline.

    Every row has a stable shape: ``name``, ``duration_seconds``, ``status`` and an
    ``attributes`` bag. Other payload keys are surfaced under ``attributes`` so the
    table consumer never has to special-case event types.
    """

    rows: list[dict[str, Any]] = []
    for event in state.trace:
        payload = dict(event.get("payload", {}))
        duration = payload.pop("duration_seconds", 0.0)
        status = payload.pop("status", "ok")
        attributes = payload.pop("attributes", {})
        payload.pop("name", None)
        attributes.update(payload)
        rows.append(
            {
                "name": event["name"],
                "duration_seconds": duration,
                "status": status,
                "attributes": attributes,
            }
        )
    return rows
