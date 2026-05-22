"""Agent test harness — the main entry point for testing any agent.

The missing piece: give agentspec an agent, and it tests it.

    from agentspec import test_agent, TraceAssertions

    def my_agent(prompt, call_tool):
        result = call_tool("search", query=prompt)
        return f"Found: {result}"

    trace = test_agent(my_agent, prompt="python testing", mock_tools={"search": ["doc1"]})  # noqa: F811
    TraceAssertions(trace).assert_tool_called("search")
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable

from agentspec.adapters.base import AgentAdapter, AsyncAgentAdapter
from agentspec.mock import MockTool
from agentspec.recorder import AgentRecorder
from agentspec.trace import ExecutionTrace


class AgentHarness:
    """Instrument, run, and record any agent.

    Usage::

        harness = AgentHarness()
        harness.mock("get_weather", returns={"temp": 28})

        def my_agent(prompt, call_tool):
            weather = call_tool("get_weather", city="Shanghai")
            return f"Weather: {weather['temp']}°C"

        trace = harness.run(my_agent, prompt="What's the weather?")
    """

    def __init__(
        self,
        *,
        max_steps: int = 50,
        max_cost: float = 1.0,
        max_consecutive_repeats: int = 5,
    ) -> None:
        self._recorder = AgentRecorder(
            max_steps=max_steps,
            max_cost=max_cost,
            max_consecutive_repeats=max_consecutive_repeats,
        )
        self._used = False

    def mock(
        self,
        name: str,
        *,
        returns: Any = None,
        raises: Exception | None = None,
        side_effect: Callable[..., Any] | None = None,
    ) -> MockTool:
        """Register a mock tool."""
        self._recorder.mock_tool(name, returns=returns, raises=raises, side_effect=side_effect)
        return self._recorder.mocks.get(name)  # type: ignore[return-value]

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a real tool function."""
        self._recorder.register_tool(name, fn)

    @property
    def tool_names(self) -> list[str]:
        """Names of all registered tools (mocks + real)."""
        names: list[str] = []
        names.extend(self._recorder.mocks._mocks.keys())
        names.extend(self._recorder._real_tools.keys())
        return sorted(set(names))

    @property
    def trace(self) -> ExecutionTrace:
        """Access the recorded trace (available after run)."""
        return self._recorder.trace

    def _ensure_fresh(self) -> None:
        if self._used:
            old_mocks = self._recorder.mocks
            old_tools = self._recorder._real_tools
            self._recorder = AgentRecorder(
                max_steps=self._recorder.budget.max_steps,
                max_cost=self._recorder.budget.max_cost,
                max_consecutive_repeats=self._recorder.loop_detector.max_consecutive,
            )
            self._recorder.mocks = old_mocks
            self._recorder._real_tools = old_tools
            self._used = False

    def run(
        self,
        agent: Callable[..., str] | AgentAdapter,
        *,
        prompt: str = "",
        **kwargs: Any,
    ) -> ExecutionTrace:
        """Run a sync agent and return its execution trace.

        The ``agent`` can be:
        - A callable ``(prompt: str, call_tool: Callable) -> str``
        - An ``AgentAdapter`` with a ``.run(prompt, call_tool)`` method
        """
        self._ensure_fresh()
        self._used = True
        self._recorder.trace.prompt = prompt

        call_tool = self._recorder.call_tool

        try:
            if isinstance(agent, AgentAdapter):
                output = agent.run(prompt, call_tool, **kwargs)
            else:
                sig = inspect.signature(agent)
                params = list(sig.parameters.keys())
                if len(params) >= 2:
                    output = agent(prompt, call_tool, **kwargs)
                else:
                    output = agent(prompt, **kwargs)
        except Exception as exc:
            self._recorder.finish(f"[error: {exc}]")
            raise

        output_str = str(output) if output is not None else ""
        self._recorder.finish(output_str)
        return self._recorder.trace

    async def arun(
        self,
        agent: Callable[..., Any] | AsyncAgentAdapter,
        *,
        prompt: str = "",
        **kwargs: Any,
    ) -> ExecutionTrace:
        """Run an async agent and return its execution trace."""
        self._ensure_fresh()
        self._used = True
        self._recorder.trace.prompt = prompt

        call_tool = self._recorder.call_tool

        try:
            if isinstance(agent, AsyncAgentAdapter):
                output = await agent.run(prompt, call_tool, **kwargs)
            else:
                output = await agent(prompt, call_tool, **kwargs)
        except Exception as exc:
            self._recorder.finish(f"[error: {exc}]")
            raise

        output_str = str(output) if output is not None else ""
        self._recorder.finish(output_str)
        return self._recorder.trace

    def run_openai(
        self,
        client: Any,
        *,
        model: str = "deepseek-chat",
        prompt: str,
        tools: dict[str, dict[str, Any]] | None = None,
        system_prompt: str = "You are a helpful assistant.",
        max_steps: int | None = None,
        max_cost: float | None = None,
    ) -> ExecutionTrace:
        """Run an OpenAI-compatible tool-use agent loop.

        ``tools`` maps tool name to ``{"fn": callable, "schema": {...}}``.
        Tools already mocked on this harness are used as-is (no fn needed).

        Example::

            trace = harness.run_openai(
                client,
                model="deepseek-chat",
                prompt="What's the weather?",
                tools={
                    "get_weather": {
                        "fn": lambda city: {"temp": 28},
                        "schema": {
                            "description": "Get weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                },
            )
        """
        from agentspec.integrations.agent_runner import AgentRunner

        runner = AgentRunner(
            client=client,
            model=model,
            max_steps=max_steps or self._recorder.budget.max_steps,
            max_cost=max_cost or self._recorder.budget.max_cost,
            system_prompt=system_prompt,
        )

        for name, mock in self._recorder.mocks._mocks.items():
            schema = {}
            if tools and name in tools:
                schema = tools[name].get("schema", {})
            runner.register_tool(name, mock, schema=schema)

        if tools:
            for name, spec in tools.items():
                if self._recorder.mocks.has(name):
                    continue
                fn = spec.get("fn")
                schema = spec.get("schema", {})
                if fn is None:
                    raise ValueError(f"Tool '{name}' has no fn and is not mocked")
                runner.register_tool(name, fn, schema=schema)

        return runner.run(prompt)


def _setup_harness(
    harness: AgentHarness,
    mock_tools: dict[str, Any] | None,
    real_tools: dict[str, Callable[..., Any]] | None,
) -> None:
    if mock_tools:
        for name, returns in mock_tools.items():
            if isinstance(returns, Exception):
                harness.mock(name, raises=returns)
            elif callable(returns) and not isinstance(returns, (dict, list, str, int, float, bool)):
                harness.mock(name, side_effect=returns)
            else:
                harness.mock(name, returns=returns)
    if real_tools:
        for name, fn in real_tools.items():
            harness.register(name, fn)


def run_agent_test(
    agent: Callable[..., str],
    *,
    prompt: str,
    mock_tools: dict[str, Any] | None = None,
    real_tools: dict[str, Callable[..., Any]] | None = None,
    max_steps: int = 50,
    max_cost: float = 1.0,
) -> ExecutionTrace:
    """Test a callable agent in one line.

    The agent must accept ``(prompt: str, call_tool: Callable) -> str``.

    ``mock_tools`` maps tool name to return value.
    ``real_tools`` maps tool name to real function.

    Example::

        def my_agent(prompt, call_tool):
            result = call_tool("search", query=prompt)
            return f"Found: {result}"

        trace = test_agent(my_agent, prompt="test", mock_tools={"search": ["doc1"]})
    """
    harness = AgentHarness(max_steps=max_steps, max_cost=max_cost)
    _setup_harness(harness, mock_tools, real_tools)
    return harness.run(agent, prompt=prompt)




async def arun_agent_test(
    agent: Callable[..., Any],
    *,
    prompt: str,
    mock_tools: dict[str, Any] | None = None,
    real_tools: dict[str, Callable[..., Any]] | None = None,
    max_steps: int = 50,
    max_cost: float = 1.0,
) -> ExecutionTrace:
    """Async version of test_agent."""
    harness = AgentHarness(max_steps=max_steps, max_cost=max_cost)
    _setup_harness(harness, mock_tools, real_tools)
    return await harness.arun(agent, prompt=prompt)




def run_openai_agent_test(
    client: Any,
    *,
    model: str = "deepseek-chat",
    prompt: str,
    tools: dict[str, dict[str, Any]],
    max_steps: int = 20,
    max_cost: float = 0.50,
    system_prompt: str = "You are a helpful assistant.",
) -> ExecutionTrace:
    """Test an OpenAI-compatible tool-use agent in one line.

    ``tools`` maps tool name to ``{"fn": callable, "schema": {...}}``.

    Example::

        from openai import OpenAI
        trace = test_openai_agent(
            OpenAI(api_key="..."),
            model="deepseek-chat",
            prompt="What's the weather?",
            tools={
                "get_weather": {
                    "fn": lambda city: {"temp": 28},
                    "schema": {
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            },
        )
    """
    harness = AgentHarness(max_steps=max_steps, max_cost=max_cost)
    return harness.run_openai(
        client,
        model=model,
        prompt=prompt,
        tools=tools,
        system_prompt=system_prompt,
    )


