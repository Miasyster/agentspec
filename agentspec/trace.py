"""Execution trace data models."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0


@dataclass
class LLMCall:
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.total_tokens == 0 and (self.prompt_tokens or self.completion_tokens):
            self.total_tokens = self.prompt_tokens + self.completion_tokens


@dataclass
class ExecutionTrace:
    prompt: str = ""
    steps: list[ToolCall] = field(default_factory=list)
    llm_calls: list[LLMCall] = field(default_factory=list)
    final_output: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.llm_calls)

    @property
    def total_cost(self) -> float:
        return sum(c.cost for c in self.llm_calls)

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def duration_ms(self) -> float:
        if self.finished_at > 0:
            return (self.finished_at - self.started_at) * 1000
        return (time.time() - self.started_at) * 1000

    @property
    def tool_names(self) -> list[str]:
        return [s.name for s in self.steps]

    @property
    def unique_tools(self) -> set[str]:
        return set(self.tool_names)

    def tool_call_count(self, name: str) -> int:
        return sum(1 for s in self.steps if s.name == name)

    def get_tool_calls(self, name: str) -> list[ToolCall]:
        return [s for s in self.steps if s.name == name]

    def add_tool_call(self, call: ToolCall) -> None:
        self.steps.append(call)

    def add_llm_call(self, call: LLMCall) -> None:
        self.llm_calls.append(call)

    def finish(self, output: str = "") -> None:
        self.finished_at = time.time()
        if output:
            self.final_output = output
