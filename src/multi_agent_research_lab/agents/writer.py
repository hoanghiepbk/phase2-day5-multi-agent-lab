"""Writer agent.

Synthesises research_notes + analysis_notes into the final answer, audience-aware,
with inline citations referencing ``state.sources`` by index.
"""

from __future__ import annotations

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

SYSTEM_PROMPT_TMPL = (
    "You are a writer agent producing the final answer for {audience}. "
    "Use the research notes and analysis to write a clear, well-structured response. "
    "Every non-trivial claim must end with a citation marker [n] referring to the "
    "numbered source list. Keep the answer faithful to the notes."
)


class WriterAgent(BaseAgent):
    name = "writer"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("writer", state, {"audience": state.request.audience}):
            if not state.research_notes or not state.analysis_notes:
                state.errors.append("writer.missing_inputs")
                raise AgentExecutionError("Writer requires research and analysis notes")

            system_prompt = SYSTEM_PROMPT_TMPL.format(audience=state.request.audience)
            sources_block = "\n".join(
                f"[{i}] {doc.title} ({doc.url or 'no-url'})"
                for i, doc in enumerate(state.sources, start=1)
            )
            user_prompt = (
                f"Question: {state.request.query}\n\n"
                f"Research notes:\n{state.research_notes}\n\n"
                f"Analysis:\n{state.analysis_notes}\n\n"
                f"Source list:\n{sources_block}\n\n"
                "Write the final answer now."
            )
            response = self.llm.complete(system_prompt, user_prompt)
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
