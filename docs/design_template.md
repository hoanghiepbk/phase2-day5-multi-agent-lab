# Design

- **Author:** Phạm Hữu Hoàng Hiệp
- **Student ID:** 2A202600415

## Problem

Take an open-ended research question (e.g. "Research GraphRAG state-of-the-art and
write a 500-word summary"), retrieve evidence, analyse it, and produce a faithful,
cited answer for technical learners. The system must run end-to-end with deterministic
guardrails and emit metrics that allow comparison against a single-agent baseline.

## Why multi-agent?

A single agent has to plan, search, analyse, and write in one prompt. As soon as the
question requires multiple sources and a structured answer, that single prompt
becomes a bottleneck: it either truncates evidence, blends roles, or skips citations.
Splitting the work into Supervisor / Researcher / Analyst / Writer lets each role
have a focused prompt and isolated failure mode, which makes both quality and
debugging better. The trade-off is added latency (~1.5–4x in our benchmark) and
token cost, so multi-agent is only justified when the task is genuinely
decomposable.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Decide the next route based on which artifacts are missing; enforce iteration cap | `state` | next route in {researcher, analyst, writer, done} | Oscillation — bounded by deterministic policy + 2x max_iterations cap |
| Researcher | Call the search backend and summarise the snippets into compact notes with [n] citations | `state.request.query` | `state.sources`, `state.research_notes` | Search backend dead → `AgentExecutionError`, workflow marks failure |
| Analyst | Extract 3–5 key claims, compare viewpoints, flag weak evidence | `state.research_notes` | `state.analysis_notes` | No notes → raises (caught by workflow) |
| Writer | Produce final, audience-aware answer with inline `[n]` citations | research + analysis | `state.final_answer` | Hallucinates without citations → caught by Critic |
| Critic (bonus) | Compute citation coverage, append issues to `state.errors` | `state.final_answer` | `state.critic_notes`, coverage metric | Conservative regex; relies on `[n]` convention |

## Shared state

`ResearchState` (see [state.py](../src/multi_agent_research_lab/core/state.py)) holds:

- `request` — the typed query (Pydantic, min length, max sources, audience).
- `iteration`, `route_history` — for the supervisor cap and trace inspection.
- `sources` — list of `SourceDocument`; the Writer cites by index into this list.
- `research_notes`, `analysis_notes`, `final_answer`, `critic_notes` — one field per
  artifact so handoffs are explicit and the supervisor can detect "missing" with a
  truthiness check.
- `agent_results` — append-only history with per-step token / cost / backend
  metadata (used by the benchmark).
- `trace` — list of span dicts (used by the trace JSON export).
- `errors` — append-only diagnostics; presence is what flips `failed=True`.
- `total_input_tokens`, `total_output_tokens`, `total_cost_usd` — aggregate usage so
  the benchmark does not have to walk `agent_results`.

Each agent reads only the fields it needs and writes only the fields it owns. No
agent inspects another agent's metadata.

## Routing policy

```
            +----------------+
            |  supervisor    |  <-----------------+
            +--------+-------+                    |
                     |                            |
   route=researcher  |  route=analyst             |
                     v        |  route=writer     |
            +-----------+ +---------+ +---------+ |
            | researcher| | analyst | | writer  | |
            +-----+-----+ +----+----+ +----+----+ |
                  |            |           |      |
                  +------------+-----------+------+
                                     |
                                     v
                              route=done -> END (then critic if enabled)
```

Implementation: `SupervisorAgent.decide` returns the first incomplete artifact, in
order; `done` when all three are present or `iteration >= max_iterations`.

## Guardrails

- **Max iterations:** `Settings.max_iterations` (default 6), checked in the
  supervisor. Workflow enforces a hard cap at `2 * max_iterations` as a backstop.
- **Timeout:** `Settings.timeout_seconds` (default 60); workflow checks wall-clock
  before each step, LLM client passes timeout to the provider call.
- **Retry:** `LLMClient.complete` is wrapped in `tenacity.retry` (3 attempts,
  exponential backoff). Retry budget belongs to the LLM call, not the agents.
- **Fallback:** if the provider is unavailable or no API key is set, `LLMClient`
  switches to a deterministic mock backend so the pipeline still produces a valid
  state. `SearchClient` does the same with a curated mock corpus.
- **Validation:** all I/O between layers is Pydantic models (`ResearchQuery`,
  `SourceDocument`, `AgentResult`, `BenchmarkMetrics`). Invalid input raises before
  any agent runs.

## Benchmark plan

Three queries from `configs/lab_default.yaml`, run twice (baseline vs multi-agent),
six measurements total. Metrics per measurement: latency (s), estimated cost (USD,
from token usage and a static price table), heuristic quality (0–10), citation
coverage (0–1, from Critic), iteration count, failure flag.

Expected outcome: multi-agent produces higher quality and citation coverage at the
cost of higher latency (~3x in mock mode, larger with a real LLM) and roughly 3x
the token spend. Both runs should report `failures=0/3` on the canonical query set.

The full report is regenerated by `python -m multi_agent_research_lab.cli benchmark`
into [reports/benchmark_report.md](../reports/benchmark_report.md).
