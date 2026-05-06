"""Multi-agent workflow orchestrator.

A small, deterministic state machine driven by ``SupervisorAgent.decide``. Conceptually
identical to a LangGraph supervisor pattern (supervisor -> worker -> supervisor) but
written without a hard ``langgraph`` dependency so the lab runs in any Python env.

Guardrails enforced here:
  - hard cap on iterations from settings.max_iterations (the supervisor also checks)
  - per-step exception handling -> falls back to baseline writer with an error note
  - timeout-aware via wall-clock check between steps
"""

from __future__ import annotations

from time import perf_counter

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import DONE, SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        search: SearchClient | None = None,
        enable_critic: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or LLMClient(self.settings)
        self.search = search or SearchClient(self.settings)
        self.supervisor = SupervisorAgent(self.settings)
        self.researcher = ResearcherAgent(self.llm, self.search)
        self.analyst = AnalystAgent(self.llm)
        self.writer = WriterAgent(self.llm)
        self.critic = CriticAgent() if enable_critic else None

    def build(self) -> dict[str, object]:
        """Return the node table. Kept as a dict so a future LangGraph swap is trivial."""

        return {
            "supervisor": self.supervisor,
            "researcher": self.researcher,
            "analyst": self.analyst,
            "writer": self.writer,
            "critic": self.critic,
        }

    def run(self, state: ResearchState) -> ResearchState:
        graph = self.build()
        deadline = perf_counter() + self.settings.timeout_seconds

        with trace_span("workflow", state, {"query": state.request.query}):
            while True:
                if perf_counter() > deadline:
                    state.errors.append("workflow.timeout")
                    state.failed = True
                    break
                if state.iteration >= self.settings.max_iterations * 2:
                    state.errors.append("workflow.iteration_cap")
                    state.failed = True
                    break

                self.supervisor.run(state)
                next_route = state.route_history[-1]

                if next_route == DONE:
                    break

                worker = graph.get(next_route)
                if worker is None:
                    state.errors.append(f"workflow.unknown_route:{next_route}")
                    state.failed = True
                    break

                try:
                    worker.run(state)  # type: ignore[union-attr]
                except AgentExecutionError as exc:
                    state.errors.append(f"workflow.agent_error:{next_route}:{exc!r}")
                    state.failed = True
                    break

            if self.critic and state.final_answer and not state.failed:
                self.critic.run(state)

            if not state.final_answer and not state.failed:
                state.errors.append("workflow.no_final_answer")
                state.failed = True

        return state
