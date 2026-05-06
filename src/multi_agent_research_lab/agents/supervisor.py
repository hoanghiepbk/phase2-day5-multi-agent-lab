"""Supervisor / router.

Deterministic policy:
  - if iteration >= max_iterations  -> done (failure if final_answer missing)
  - if not research_notes           -> researcher
  - if not analysis_notes           -> analyst
  - if not final_answer             -> writer
  - else                            -> done

Keeping the policy deterministic makes the trace easy to read and the failure mode
analysis easy to reason about. An LLM-routed supervisor can replace this without
changing the rest of the graph.
"""

from __future__ import annotations

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState

DONE = "done"


class SupervisorAgent(BaseAgent):
    name = "supervisor"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def decide(self, state: ResearchState) -> str:
        if state.iteration >= self.settings.max_iterations:
            return DONE
        if not state.research_notes:
            return "researcher"
        if not state.analysis_notes:
            return "analyst"
        if not state.final_answer:
            return "writer"
        return DONE

    def run(self, state: ResearchState) -> ResearchState:
        decision_at = state.iteration
        route = self.decide(state)
        state.record_route(route)
        state.add_trace_event(
            "supervisor.route",
            {
                "decision_at_iteration": decision_at,
                "iteration_after": state.iteration,
                "next": route,
                "has_research": bool(state.research_notes),
                "has_analysis": bool(state.analysis_notes),
                "has_final": bool(state.final_answer),
                "duration_seconds": 0.0,
                "status": "ok",
            },
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.SUPERVISOR,
                content=f"route={route}",
                metadata={"decision_at_iteration": decision_at},
            )
        )
        return state
