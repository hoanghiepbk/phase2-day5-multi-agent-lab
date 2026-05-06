"""Command-line entrypoint for the lab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import export_trace, trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


BASELINE_SYSTEM_PROMPT = (
    "You are a single answering agent for technical learners. Given a question, produce "
    "a concise, well-cited answer in Markdown. If you have no external sources available, "
    "state that explicitly. Use [n] markers for any citation you do include."
)


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _baseline_run(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    llm = LLMClient()
    with trace_span("baseline", state, {"query": query}):
        response = llm.complete(BASELINE_SYSTEM_PROMPT, query)
        state.final_answer = response.content
        state.add_usage(response.input_tokens, response.output_tokens, response.cost_usd)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                    "backend": response.backend,
                },
            )
        )
    return state


def _multi_run(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline (one LLM call, no search)."""

    _init()
    state = _baseline_run(query)
    console.print(Panel.fit(state.final_answer or "(empty)", title="Single-Agent Baseline"))
    console.print(
        f"[dim]tokens={state.total_input_tokens}+{state.total_output_tokens}, "
        f"cost=${state.total_cost_usd:.6f}[/dim]"
    )


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    save_trace: Annotated[
        bool, typer.Option("--save-trace", help="Save trace to reports/")
    ] = False,
) -> None:
    """Run the multi-agent workflow."""

    _init()
    state = _multi_run(query)
    console.print(Panel.fit(state.final_answer or "(empty)", title="Multi-Agent Final Answer"))

    table = Table(title="Trace timeline")
    table.add_column("#", justify="right")
    table.add_column("Step")
    table.add_column("Duration (s)", justify="right")
    table.add_column("Status")
    for i, ev in enumerate(state.trace, start=1):
        payload = ev["payload"]
        table.add_row(
            str(i),
            ev["name"],
            f"{(payload.get('duration_seconds') or 0):.3f}",
            payload.get("status", "ok"),
        )
    console.print(table)
    console.print(
        f"[dim]iterations={state.iteration}, errors={len(state.errors)}, "
        f"cost=${state.total_cost_usd:.6f}[/dim]"
    )

    if save_trace:
        store = LocalArtifactStore()
        path = store.write_text(
            "trace_last_run.json",
            json.dumps(export_trace(state), indent=2, default=str),
        )
        console.print(f"[green]Trace saved to {path}[/green]")


@app.command()
def benchmark(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/lab_default.yaml"),
    out_dir: Annotated[Path, typer.Option("--out", "-o")] = Path("reports"),
) -> None:
    """Run baseline + multi-agent across the configured queries and write artifacts."""

    _init()
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    queries: list[str] = cfg["benchmark"]["queries"]

    metrics = []
    last_multi_state: ResearchState | None = None

    for q in queries:
        console.print(f"[bold]baseline[/bold] :: {q}")
        _, m_b = run_benchmark("baseline", q, _baseline_run)
        metrics.append(m_b)

        console.print(f"[bold]multi-agent[/bold] :: {q}")
        state, m_m = run_benchmark("multi-agent", q, _multi_run)
        metrics.append(m_m)
        last_multi_state = state

    out_dir.mkdir(parents=True, exist_ok=True)
    report_md = render_markdown_report(metrics)
    (out_dir / "benchmark_report.md").write_text(report_md, encoding="utf-8")
    console.print(f"[green]Report -> {out_dir / 'benchmark_report.md'}[/green]")

    if last_multi_state is not None:
        trace_path = out_dir / "trace_last_run.json"
        trace_path.write_text(
            json.dumps(export_trace(last_multi_state), indent=2, default=str),
            encoding="utf-8",
        )
        console.print(f"[green]Trace -> {trace_path}[/green]")


if __name__ == "__main__":
    app()
