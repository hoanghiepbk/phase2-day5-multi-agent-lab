"""Search client abstraction for ResearcherAgent.

Backends:
- ``tavily`` if ``TAVILY_API_KEY`` is set and the ``tavily`` package is installed.
- ``mock`` deterministic fallback so the lab works offline.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
from typing import Any

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.backend = self._select_backend()
        self._tavily: Any = None
        if self.backend == "tavily":
            try:
                from tavily import TavilyClient  # type: ignore[import-not-found]

                self._tavily = TavilyClient(api_key=self.settings.tavily_api_key)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning("Falling back to mock search (tavily import failed: %s)", exc)
                self.backend = "mock"

    def _select_backend(self) -> str:
        if self.settings.tavily_api_key and importlib.util.find_spec("tavily") is not None:
            return "tavily"
        if self.settings.tavily_api_key:
            logger.warning("tavily package not installed; using mock search")
        return "mock"

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        if self.backend == "tavily":
            return self._search_tavily(query, max_results)
        return self._search_mock(query, max_results)

    def _search_tavily(self, query: str, max_results: int) -> list[SourceDocument]:
        assert self._tavily is not None
        result = self._tavily.search(query=query, max_results=max_results)
        docs: list[SourceDocument] = []
        for item in result.get("results", []):
            docs.append(
                SourceDocument(
                    title=item.get("title", "Untitled"),
                    url=item.get("url"),
                    snippet=item.get("content", "")[:600],
                    metadata={"score": item.get("score")},
                )
            )
        return docs

    def _search_mock(self, query: str, max_results: int) -> list[SourceDocument]:
        digest = hashlib.sha1(query.encode()).hexdigest()[:6]
        templates = [
            (
                "Anthropic - Building effective agents",
                "https://www.anthropic.com/engineering/building-effective-agents",
                "Production agents prioritise simple control loops, explicit tool boundaries, "
                "and step-level evaluation over emergent behaviour.",
            ),
            (
                "LangGraph concepts",
                "https://langchain-ai.github.io/langgraph/concepts/",
                "LangGraph models multi-agent workflows as a state graph with conditional edges, "
                "enabling supervisor / worker patterns with explicit stop conditions.",
            ),
            (
                "OpenAI Agents - orchestration & handoffs",
                "https://developers.openai.com/api/docs/guides/agents/orchestration",
                "Handoffs let a supervisor route to specialised workers; guardrails such as "
                "max iterations and timeouts prevent runaway loops.",
            ),
            (
                "LangSmith tracing docs",
                "https://docs.smith.langchain.com/",
                "LangSmith captures spans, token usage, and latency per agent step, enabling "
                "comparative benchmarks between single-agent and multi-agent runs.",
            ),
            (
                "Survey: Multi-agent LLM systems",
                "https://arxiv.org/abs/2402.01680",
                "Multi-agent systems improve quality on decomposable tasks but add 1.5-3x "
                "latency vs strong single-agent baselines; cost grows with iteration count.",
            ),
        ]
        results = []
        for i, (title, url, snippet) in enumerate(templates[:max_results]):
            results.append(
                SourceDocument(
                    title=title,
                    url=url,
                    snippet=snippet,
                    metadata={"rank": i + 1, "mock_digest": digest, "query": query},
                )
            )
        return results
