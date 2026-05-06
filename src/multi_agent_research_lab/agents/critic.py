"""Critic agent.

Lightweight post-write check: verifies that the final answer cites at least one
source per main claim and computes a coarse citation-coverage score that the
benchmark can record.
"""

from __future__ import annotations

import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

CITATION_RE = re.compile(r"\[\d+\]")


class CriticAgent(BaseAgent):
    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("critic", state, {"has_final": bool(state.final_answer)}):
            answer = state.final_answer or ""
            sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer) if len(s) > 30]
            cited = [s for s in sentences if CITATION_RE.search(s)]
            coverage = (len(cited) / len(sentences)) if sentences else 0.0

            issues: list[str] = []
            if coverage < 0.5:
                issues.append(f"low_citation_coverage={coverage:.2f}")
            if not state.sources:
                issues.append("no_sources_recorded")

            critique = (
                f"Citation coverage: {coverage:.2%} "
                f"({len(cited)}/{len(sentences)} substantive sentences cited).\n"
                f"Issues: {', '.join(issues) if issues else 'none'}."
            )
            state.critic_notes = critique
            for issue in issues:
                state.errors.append(f"critic.{issue}")

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.CRITIC,
                    content=critique,
                    metadata={"citation_coverage": coverage, "issues": issues},
                )
            )
        return state
