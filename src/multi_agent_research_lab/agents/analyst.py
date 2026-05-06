"""Analyst agent.

Reads research notes, extracts key claims, compares viewpoints, and flags weak
evidence. Output is structured Markdown so the Writer can lift it directly.
"""

from __future__ import annotations

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

SYSTEM_PROMPT = (
    "You are an analytical agent. Given research notes, extract 3-5 key claims, "
    "compare differing viewpoints, and flag any claim with weak or single-source "
    "evidence. Cite sources by [n] index. Output valid Markdown."
)


class AnalystAgent(BaseAgent):
    name = "analyst"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("analyst", state, {"has_notes": bool(state.research_notes)}):
            if not state.research_notes:
                state.errors.append("analyst.no_research_notes")
                raise AgentExecutionError("Analyst requires research_notes")

            user_prompt = (
                f"Question: {state.request.query}\n\n"
                f"Research notes:\n{state.research_notes}\n\n"
                "Produce the analysis now."
            )
            response = self.llm.complete(SYSTEM_PROMPT, user_prompt)
            state.analysis_notes = response.content
            state.add_usage(response.input_tokens, response.output_tokens, response.cost_usd)
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
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
