from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


def _settings_no_keys() -> Settings:
    return Settings(
        OPENAI_API_KEY=None,
        TAVILY_API_KEY=None,
    )  # type: ignore[call-arg]


def test_llm_client_falls_back_to_mock_without_key() -> None:
    client = LLMClient(_settings_no_keys())
    assert client.backend == "mock"
    response = client.complete("You are a writer agent.", "topic")
    assert response.content
    assert response.backend == "mock"
    assert response.input_tokens > 0
    assert response.output_tokens > 0


def test_search_client_falls_back_to_mock_without_key() -> None:
    client = SearchClient(_settings_no_keys())
    assert client.backend == "mock"
    docs = client.search("multi agent systems", max_results=3)
    assert len(docs) == 3
    assert all(d.url for d in docs)
