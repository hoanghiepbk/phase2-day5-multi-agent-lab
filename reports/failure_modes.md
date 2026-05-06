# Failure modes & fixes

- **Author:** Phạm Hữu Hoàng Hiệp
- **Student ID:** 2A202600415

This file is deliverable #4: a concrete walk-through of how the system breaks and
how the implementation responds.

## Mode 1 — Researcher returns no sources (search backend dead / rate-limited)

**Symptom.** `SearchClient.search` raises, or returns an empty list. Without sources,
the Analyst has nothing to compare and the Writer would otherwise hallucinate.

**Where it surfaces in the trace.** A `researcher` span with `status="error"` (full
exception captured), or a `researcher` span where `state.errors` contains
`researcher.no_sources`.

**Fix wired into the code.**
1. `ResearcherAgent.run` wraps the search call in `try/except`, records
   `researcher.search_failed` in `state.errors`, then raises `AgentExecutionError`.
2. `MultiAgentWorkflow.run` catches `AgentExecutionError`, marks `state.failed=True`
   and breaks the loop instead of cascading the failure into Analyst/Writer.
3. The benchmark records `failure=True` in `BenchmarkMetrics`, so the failure shows
   up as a row in section 4 of `benchmark_report.md` rather than silently producing
   a low-quality answer.

**Why not just retry forever.** Retrying inside the agent would extend latency
without fixing the cause. The retry budget belongs to the LLM client (3 attempts,
exponential backoff via `tenacity`), not to the search call where transient failures
usually mean the upstream is dead.

## Mode 2 — Routing oscillation / runaway iterations

**Symptom.** A buggy supervisor routes back to a worker that already produced its
artifact (e.g. asks the analyst to run again because of a stale state read).

**Where it surfaces in the trace.** `supervisor.route` events with the same `next`
field repeated, while `iteration` keeps growing.

**Fix wired into the code.**
1. `SupervisorAgent.decide` is deterministic and reads only completion booleans
   (`has_research`, `has_analysis`, `has_final`). It cannot oscillate by design.
2. Two independent caps:
   - `Settings.max_iterations` (default 6) checked inside the supervisor.
   - `MultiAgentWorkflow` enforces a hard wall at `2 * max_iterations` to catch the
     case where a future LLM-based supervisor loses its grip.
3. A separate wall-clock check (`Settings.timeout_seconds`) guarantees the run
   terminates even if every iteration is fast but the supervisor keeps choosing
   work to do.

## Mode 3 — Final answer without citations (hallucination)

**Symptom.** Writer produces fluent prose with no `[n]` markers, so claims cannot be
audited.

**Where it surfaces in the trace.** `critic` span with `metadata.citation_coverage`
below 0.5 and `state.errors` containing `critic.low_citation_coverage=...`.

**Fix wired into the code.**
1. `WriterAgent` system prompt instructs every non-trivial sentence to end with a
   citation marker.
2. `CriticAgent` runs after the Writer (when enabled) and computes
   `citation_coverage = cited_sentences / substantive_sentences`.
3. Coverage is recorded in `BenchmarkMetrics.citation_coverage` and appears in the
   per-run table in `benchmark_report.md`, so a regression is visible without
   reading every answer.

## Mode 4 — LLM provider transient error / timeout

**Symptom.** The provider returns a 5xx, a connection reset, or stalls.

**Where it surfaces in the trace.** Three retry attempts with exponential backoff
(`tenacity`), all recorded under the same agent span; if all fail, the agent raises
`AgentExecutionError`.

**Fix wired into the code.**
1. Retry/timeout policy lives once, in `LLMClient.complete`, not duplicated in each
   agent.
2. The OpenAI call passes `timeout=settings.timeout_seconds` so a hung request fails
   fast instead of blocking the whole workflow.
3. Mock fallback (`backend="mock"`) keeps the lab fully runnable when the provider
   is offline or no API key is configured, so the rest of the pipeline (and these
   failure modes!) can still be exercised.
