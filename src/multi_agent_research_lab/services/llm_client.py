"""LLM client abstraction.

Agents must depend on this interface, not on a vendor SDK directly. The client
supports two backends:

- ``openai`` if ``OPENAI_API_KEY`` is set and the ``openai`` package is installed.
- ``mock`` deterministic fallback so the lab is fully runnable without keys.

Retry, timeout, and token accounting live here, not inside agents.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
from dataclasses import dataclass
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


PRICING_USD_PER_1K = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "gpt-4.1-mini": (0.0004, 0.0016),
}


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    backend: str = "mock"


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = PRICING_USD_PER_1K.get(model, (0.0, 0.0))
    return (input_tokens / 1000.0) * in_rate + (output_tokens / 1000.0) * out_rate


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class LLMClient:
    """Provider-agnostic LLM client."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.backend = self._select_backend()
        self._openai: Any = None
        if self.backend == "openai":
            try:
                from openai import OpenAI  # type: ignore[import-not-found]

                self._openai = OpenAI(api_key=self.settings.openai_api_key)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning("Falling back to mock LLM (openai import failed: %s)", exc)
                self.backend = "mock"

    def _select_backend(self) -> str:
        if self.settings.openai_api_key and importlib.util.find_spec("openai") is not None:
            return "openai"
        if self.settings.openai_api_key:
            logger.warning("openai package not installed; using mock LLM")
        return "mock"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with retry, timeout, and usage tracking."""

        if self.backend == "openai":
            return self._complete_openai(system_prompt, user_prompt)
        return self._complete_mock(system_prompt, user_prompt)

    def _complete_openai(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        assert self._openai is not None
        result = self._openai.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            timeout=self.settings.timeout_seconds,
        )
        content = result.choices[0].message.content or ""
        usage = getattr(result, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0
        cost = _estimate_cost(self.settings.openai_model, in_tok, out_tok)
        return LLMResponse(
            content=content,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            backend="openai",
        )

    def _complete_mock(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Deterministic stand-in. Role is inferred from the system prompt; order of the
        checks matters because some agents share keywords (e.g. the baseline prompt may
        mention "research" while still acting as a writer)."""

        digest = hashlib.sha1((system_prompt + "||" + user_prompt).encode()).hexdigest()[:8]
        role = "agent"
        sp_lower = system_prompt.lower()
        if "answering agent" in sp_lower or "single answering" in sp_lower:
            role = "baseline"
        elif "writer agent" in sp_lower:
            role = "writer"
        elif "critic" in sp_lower or "fact" in sp_lower:
            role = "critic"
        elif "analytical agent" in sp_lower or "analy" in sp_lower:
            role = "analyst"
        elif "research agent" in sp_lower:
            role = "researcher"

        content = self._mock_template(role, user_prompt, digest)
        in_tok = _approx_tokens(system_prompt) + _approx_tokens(user_prompt)
        out_tok = _approx_tokens(content)
        cost = _estimate_cost(self.settings.openai_model, in_tok, out_tok)
        return LLMResponse(
            content=content,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            backend="mock",
        )

    @staticmethod
    def _mock_template(role: str, user_prompt: str, digest: str) -> str:
        topic = user_prompt.split("\n")[0][:80].strip() or "the topic"
        if role == "baseline":
            return (
                f"Answer for '{topic}':\n\n"
                "A single agent attempting this task without retrieval can only state the "
                "most general definition and tradeoffs. Without sources I cannot cite "
                "specific benchmarks or production case studies. For a richer answer, "
                "use the multi-agent workflow which performs retrieval before writing.\n"
                f"(mock-digest:{digest})"
            )
        if role == "researcher":
            return (
                f"Research summary for '{topic}':\n"
                f"- Source [1] highlights core definition and recent benchmarks.\n"
                f"- Source [2] presents tradeoffs around latency and cost.\n"
                f"- Source [3] documents production-grade guardrails.\n"
                f"(mock-digest:{digest})"
            )
        if role == "analyst":
            return (
                f"Key claims for '{topic}':\n"
                f"1. Multi-agent helps when tasks are decomposable [1].\n"
                f"2. Latency overhead is real, ~1.5-3x baseline [2].\n"
                f"3. Guardrails (max iterations, retries) prevent runaway loops [3].\n"
                f"Weak evidence: vendor benchmarks self-reported.\n"
                f"(mock-digest:{digest})"
            )
        if role == "writer":
            return (
                f"# {topic}\n\n"
                f"Multi-agent systems decompose complex tasks across specialised agents [1]. "
                f"Latency increases compared with a single-agent baseline, typically 1.5-3x [2], "
                f"so they are best reserved for multi-step research, planning, or review tasks. "
                f"Production deployments add guardrails such as max iterations and retries to "
                f"avoid runaway loops [3].\n\n"
                f"Sources: [1], [2], [3]. (mock-digest:{digest})"
            )
        if role == "critic":
            return (
                f"Critic review for '{topic}':\n"
                f"- Citation coverage looks adequate (3/3 main claims have a source).\n"
                f"- Watch for over-claiming on latency numbers; cite specific benchmark.\n"
                f"(mock-digest:{digest})"
            )
        return f"Mock response for '{topic}' (digest:{digest})"
