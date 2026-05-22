"""Record agent execution into traces."""

from __future__ import annotations

import time
from typing import Any, Callable

from agentspec.trace import ExecutionTrace, ToolCall, LLMCall
from agentspec.mock import MockToolRegistry
from agentspec.budget import BudgetGuard
from agentspec.loop_detector import LoopDetector


class AgentRecorder:
    """Records tool calls and LLM calls into an ExecutionTrace.

    Use as a middleware between your agent and its tools:
    - Register mock tools for testing
    - Record all interactions
    - Enforce budgets and detect loops
    """

    def __init__(
        self,
        *,
        max_steps: int = 50,
        max_cost: float = 1.0,
        max_consecutive_repeats: int = 5,
    ) -> None:
        self.trace = ExecutionTrace()
        self.mocks = MockToolRegistry()
        self.budget = BudgetGuard(max_steps=max_steps, max_cost=max_cost)
        self.loop_detector = LoopDetector(max_consecutive=max_consecutive_repeats)
        self._real_tools: dict[str, Callable[..., Any]] = {}

    def register_tool(self, name: str, fn: Callable[..., Any]) -> None:
        self._real_tools[name] = fn

    def mock_tool(
        self,
        name: str,
        *,
        returns: Any = None,
        raises: Exception | None = None,
        side_effect: Callable[..., Any] | None = None,
    ) -> None:
        self.mocks.register(name, returns=returns, raises=raises, side_effect=side_effect)

    def call_tool(self, name: str, **kwargs: Any) -> Any:
        self.budget.check_step(self.trace)
        self.loop_detector.check(name)

        start = time.time()
        error = None
        result = None

        try:
            if self.mocks.has(name):
                result = self.mocks.resolve(name, **kwargs)
            elif name in self._real_tools:
                result = self._real_tools[name](**kwargs)
            else:
                raise KeyError(f"Tool '{name}' not registered and no mock available")
        except (BudgetExceeded, LoopDetected):
            raise
        except Exception as e:
            error = str(e)
            raise
        finally:
            elapsed = (time.time() - start) * 1000
            call = ToolCall(
                name=name,
                arguments=kwargs,
                result=result,
                error=error,
                duration_ms=elapsed,
            )
            self.trace.add_tool_call(call)

        return result

    def record_llm_call(
        self,
        *,
        model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost: float = 0.0,
    ) -> None:
        call = LLMCall(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost,
        )
        self.trace.add_llm_call(call)
        self.budget.check_cost(self.trace)

    def finish(self, output: str = "") -> ExecutionTrace:
        self.trace.finish(output)
        return self.trace


class BudgetExceeded(RuntimeError):
    pass


class LoopDetected(RuntimeError):
    pass
