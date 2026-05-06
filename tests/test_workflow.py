from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def _runner(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    return MultiAgentWorkflow().run(state)


def test_multi_agent_run_produces_final_answer() -> None:
    state = _runner("Research GraphRAG state-of-the-art and write a 500-word summary")
    assert state.final_answer
    assert state.research_notes
    assert state.analysis_notes
    assert state.sources
    assert not state.failed
    routes = state.route_history
    assert "researcher" in routes
    assert "analyst" in routes
    assert "writer" in routes
    assert routes[-1] == "done"


def test_benchmark_returns_full_metrics() -> None:
    _, metrics = run_benchmark(
        "multi-agent",
        "Summarise production guardrails for LLM agents",
        _runner,
    )
    assert metrics.run_name == "multi-agent"
    assert metrics.latency_seconds >= 0
    assert metrics.estimated_cost_usd is not None
    assert metrics.quality_score is not None
    assert metrics.citation_coverage is not None
    assert metrics.iterations > 0
