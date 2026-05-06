from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.agents.supervisor import DONE
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_to_researcher_first() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    SupervisorAgent().run(state)
    assert state.route_history == ["researcher"]


def test_supervisor_finishes_when_final_answer_exists() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    state.final_answer = "answer"
    SupervisorAgent().run(state)
    assert state.route_history == [DONE]


def test_supervisor_caps_iterations() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.iteration = 10**3
    SupervisorAgent().run(state)
    assert state.route_history == [DONE]
