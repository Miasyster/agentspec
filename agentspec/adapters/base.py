"""Base adapter protocols for agent testing."""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class AgentAdapter(Protocol):
    """Minimal protocol for a testable agent.

    Any class with a ``run(prompt, call_tool, **kwargs) -> str`` method
    qualifies — no subclassing required.
    """

    def run(self, prompt: str, call_tool: Callable[..., Any], **kwargs: Any) -> str: ...


@runtime_checkable
class AsyncAgentAdapter(Protocol):
    """Async variant of AgentAdapter."""

    async def run(self, prompt: str, call_tool: Callable[..., Any], **kwargs: Any) -> str: ...
