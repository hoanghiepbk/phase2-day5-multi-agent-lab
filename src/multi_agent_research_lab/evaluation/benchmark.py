"""Benchmark: single-agent vs multi-agent.

Measures, per (run_name, query):
  - latency (wall-clock seconds)
  - estimated cost (USD, from token usage)
  - quality_score (0-10, optional override; mock heuristic by default)
  - citation_coverage (0-1, from CriticAgent if available, else regex)
  - failure (bool, True if state.failed or no final_answer)
"""

from __future__ import annotations

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]
CITATION_RE = re.compile(r"\[\d+\]")


def _citation_coverage(state: ResearchState) -> float:
    for result in reversed(state.agent_results):
        if result.agent.value == "critic":
            return float(result.metadata.get("citation_coverage", 0.0))
    answer = state.final_answer or ""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer) if len(s) > 30]
    if not sentences:
        return 0.0
    cited = [s for s in sentences if CITATION_RE.search(s)]
    return len(cited) / len(sentences)


def _heuristic_quality(state: ResearchState) -> float:
    """Cheap proxy 0-10 used when no human/LLM rater is available."""

    if state.failed or not state.final_answer:
        return 0.0
    score = 4.0
    if state.research_notes:
        score += 1.5
    if state.analysis_notes:
        score += 1.5
    if state.sources:
        score += 1.0
    score += 2.0 * _citation_coverage(state)
    return min(round(score, 2), 10.0)


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
    quality_override: float | None = None,
) -> tuple[ResearchState, BenchmarkMetrics]:
    started = perf_counter()
    failure = False
    try:
        state = runner(query)
    except Exception as exc:
        failure = True
        from multi_agent_research_lab.core.schemas import ResearchQuery

        request = ResearchQuery.model_construct(query=query, max_sources=5, audience="n/a")
        state = ResearchState(request=request)
        state.errors.append(f"runner_exception:{exc!r}")
        state.failed = True
    latency = perf_counter() - started

    coverage = _citation_coverage(state)
    quality = quality_override if quality_override is not None else _heuristic_quality(state)
    backend = "n/a"
    for r in reversed(state.agent_results):
        if "backend" in r.metadata:
            backend = str(r.metadata["backend"])
            break
    notes = (
        f"iters={state.iteration}, sources={len(state.sources)}, "
        f"errors={len(state.errors)}, backend={backend}"
    )

    metrics = BenchmarkMetrics(
        run_name=run_name,
        query=query,
        latency_seconds=latency,
        estimated_cost_usd=round(state.total_cost_usd, 6),
        quality_score=quality,
        citation_coverage=round(coverage, 3),
        failure=failure or state.failed,
        iterations=state.iteration,
        notes=notes,
    )
    return state, metrics
