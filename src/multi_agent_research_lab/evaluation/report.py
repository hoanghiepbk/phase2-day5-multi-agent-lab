"""Benchmark report rendering."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from statistics import mean

from multi_agent_research_lab.core.schemas import BenchmarkMetrics

REPORT_AUTHOR = "Phạm Hữu Hoàng Hiệp"
REPORT_STUDENT_ID = "2A202600415"


def _fmt(v: float | None, prec: int = 2) -> str:
    return "-" if v is None else f"{v:.{prec}f}"


def render_markdown_report(
    metrics: list[BenchmarkMetrics],
    author: str = REPORT_AUTHOR,
    student_id: str = REPORT_STUDENT_ID,
) -> str:
    """Render benchmark metrics to a richer markdown report.

    Sections:
      1. Per-run table (every (run_name, query) pair)
      2. Aggregate summary by run_name
      3. Comparison block: baseline vs multi-agent
      4. Failure modes spotted in ``notes``
    """

    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = [
        "# Benchmark Report",
        "",
        f"- **Author:** {author}",
        f"- **Student ID:** {student_id}",
        f"- **Generated:** {generated_at}",
        "",
    ]

    header_run = (
        "| Run | Query | Latency (s) | Cost (USD) | Quality | "
        "Citation cov. | Iters | Failure | Notes |"
    )
    lines += [
        "## 1. Per-run results",
        "",
        header_run,
        "|---|---|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for m in metrics:
        q_short = (m.query[:48] + "…") if len(m.query) > 50 else m.query
        lines.append(
            f"| {m.run_name} | {q_short} | {m.latency_seconds:.2f} | "
            f"{_fmt(m.estimated_cost_usd, 4)} | {_fmt(m.quality_score, 1)} | "
            f"{_fmt(m.citation_coverage, 2)} | {m.iterations} | "
            f"{'YES' if m.failure else 'no'} | {m.notes} |"
        )
    lines.append("")

    grouped: dict[str, list[BenchmarkMetrics]] = defaultdict(list)
    for m in metrics:
        grouped[m.run_name].append(m)

    header_agg = (
        "| Run | Mean latency (s) | Mean cost (USD) | Mean quality | "
        "Mean citation cov. | Failures |"
    )
    lines += [
        "## 2. Aggregate by run",
        "",
        header_agg,
        "|---|---:|---:|---:|---:|---:|",
    ]
    for run_name, items in grouped.items():
        latency = mean(i.latency_seconds for i in items)
        costs = [i.estimated_cost_usd for i in items if i.estimated_cost_usd is not None]
        quality = [i.quality_score for i in items if i.quality_score is not None]
        cov = [i.citation_coverage for i in items if i.citation_coverage is not None]
        fails = sum(1 for i in items if i.failure)
        lines.append(
            f"| {run_name} | {latency:.2f} | "
            f"{_fmt(mean(costs), 4) if costs else '-'} | "
            f"{_fmt(mean(quality), 2) if quality else '-'} | "
            f"{_fmt(mean(cov), 2) if cov else '-'} | "
            f"{fails}/{len(items)} |"
        )
    lines.append("")

    if "baseline" in grouped and "multi-agent" in grouped:
        b = grouped["baseline"]
        m = grouped["multi-agent"]
        lat_ratio = mean(i.latency_seconds for i in m) / max(
            mean(i.latency_seconds for i in b), 1e-6
        )
        cost_b = [i.estimated_cost_usd or 0 for i in b]
        cost_m = [i.estimated_cost_usd or 0 for i in m]
        qual_b = [i.quality_score for i in b if i.quality_score is not None]
        qual_m = [i.quality_score for i in m if i.quality_score is not None]
        lines += [
            "## 3. Baseline vs Multi-agent",
            "",
            f"- Latency multiplier (multi / baseline): **{lat_ratio:.2f}x**",
            f"- Cost delta: baseline ${mean(cost_b):.4f} -> multi ${mean(cost_m):.4f}",
            (
                f"- Quality delta: baseline {mean(qual_b):.2f} -> multi {mean(qual_m):.2f}"
                if qual_b and qual_m
                else "- Quality: insufficient samples"
            ),
            "",
        ]

    failure_lines = [m for m in metrics if m.failure]
    lines += ["## 4. Failures observed", ""]
    if not failure_lines:
        lines.append("No failures recorded across the benchmark set.")
    else:
        for m in failure_lines:
            lines.append(f"- **{m.run_name}** on `{m.query[:60]}` -> {m.notes}")
    lines.append("")

    return "\n".join(lines) + "\n"
