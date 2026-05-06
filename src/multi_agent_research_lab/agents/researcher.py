"""Researcher agent.

Calls the search backend, then asks the LLM to summarise the retrieved snippets
into compact notes. Keeps source list intact so downstream agents can cite by index.
"""

from __future__ import annotations

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

SYSTEM_PROMPT = (
    "You are a careful research agent. Read the provided sources and produce concise "
    "notes (max 8 bullets). Reference each bullet with a [n] index matching the source "
    "list. Do not invent facts beyond the snippets."
)


class ResearcherAgent(BaseAgent):
    name = "researcher"

    def __init__(
        self,
        llm: LLMClient | None = None,
        search: SearchClient | None = None,
    ) -> None:
        self.llm = llm or LLMClient()
        self.search = search or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("researcher", state, {"query": state.request.query}):
            try:
                docs = self.search.search(
                    state.request.query, max_results=state.request.max_sources
                )
            except Exception as exc:
                state.errors.append(f"researcher.search_failed: {exc!r}")
                raise AgentExecutionError("Researcher search failed") from exc

            if not docs:
                state.errors.append("researcher.no_sources")
                state.research_notes = "No sources retrieved."
                return state

            state.sources.extend(docs)
            user_prompt = self._format_prompt(state.request.query, docs)
            response = self.llm.complete(SYSTEM_PROMPT, user_prompt)
            state.research_notes = response.content
            state.add_usage(response.input_tokens, response.output_tokens, response.cost_usd)
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                        "backend": response.backend,
                        "n_sources": len(docs),
                    },
                )
            )
        return state

    @staticmethod
    def _format_prompt(query: str, docs: list[SourceDocument]) -> str:
        lines = [f"Question: {query}", "", "Sources:"]
        for i, doc in enumerate(docs, start=1):
            lines.append(f"[{i}] {doc.title} ({doc.url or 'no-url'}): {doc.snippet}")
        lines.append("")
        lines.append("Write the research notes now.")
        return "\n".join(lines)
