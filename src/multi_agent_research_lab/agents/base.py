"""Base agent contract.

Every concrete agent implements ``run(state) -> state``. Agents read the fields they
depend on and write only the fields they own; coordination lives in the workflow,
not in the agents.
"""

from abc import ABC, abstractmethod

from multi_agent_research_lab.core.state import ResearchState


class BaseAgent(ABC):
    """Minimal interface every agent must implement."""

    name: str

    @abstractmethod
    def run(self, state: ResearchState) -> ResearchState:
        """Read and update shared state, then return it."""
