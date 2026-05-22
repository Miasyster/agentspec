"""Monkey-patch OpenAI SDK to automatically record LLM calls and tool use.

Works with any OpenAI-compatible API (DeepSeek, Together, Groq, etc.).
"""

from __future__ import annotations

import time
import functools
from typing import Any

from agentspec.trace import LLMCall, ToolCall

COST_PER_1K = {
    "deepseek-chat": {"input": 0.0001, "output": 0.0002},
    "deepseek-reasoner": {"input": 0.0004, "output": 0.0016},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4.1": {"input": 0.002, "output": 0.008},
    "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
    "gpt-4.1-nano": {"input": 0.0001, "output": 0.0004},
    "claude-sonnet-4-6": {"input": 0.003, "output": 0.015},
    "claude-haiku-4-5-20251001": {"input": 0.0008, "output": 0.004},
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = COST_PER_1K.get(model)
    if rates is None:
        for key, val in COST_PER_1K.items():
            if key in model:
                rates = val
                break
    if rates is None:
        return 0.0
    return (prompt_tokens / 1000) * rates["input"] + (completion_tokens / 1000) * rates["output"]


def _extract_tool_calls(response: Any) -> list[dict[str, Any]]:
    tool_calls = []
    for choice in getattr(response, "choices", []):
        msg = getattr(choice, "message", None)
        if msg is None:
            continue
        for tc in getattr(msg, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            import json

            try:
                args = json.loads(fn.arguments) if isinstance(fn.arguments, str) else fn.arguments
            except (json.JSONDecodeError, TypeError):
                args = {"raw": fn.arguments}
            tool_calls.append({"name": fn.name, "arguments": args})
    return tool_calls


class OpenAIPatch:
    """Patches an OpenAI client to record all calls into an AgentRecorder."""

    def __init__(self, client: Any, recorder: Any) -> None:
        self._client = client
        self._recorder = recorder
        self._original_create = None
        self._patched = False

    def activate(self) -> None:
        if self._patched:
            return
        self._original_create = self._client.chat.completions.create
        self._client.chat.completions.create = self._wrapped_create
        self._patched = True

    def deactivate(self) -> None:
        if not self._patched:
            return
        self._client.chat.completions.create = self._original_create
        self._patched = False
        self._original_create = None

    def __enter__(self) -> OpenAIPatch:
        self.activate()
        return self

    def __exit__(self, *args: Any) -> None:
        self.deactivate()

    @functools.wraps(lambda: None)
    def _wrapped_create(self, *args: Any, **kwargs: Any) -> Any:
        start = time.time()
        response = self._original_create(*args, **kwargs)
        elapsed = (time.time() - start) * 1000

        usage = getattr(response, "usage", None)
        model = getattr(response, "model", kwargs.get("model", "unknown"))
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        self._recorder.trace.add_llm_call(
            LLMCall(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cost=_estimate_cost(model, prompt_tokens, completion_tokens),
                duration_ms=elapsed,
            )
        )

        for tc in _extract_tool_calls(response):
            self._recorder.trace.add_tool_call(
                ToolCall(name=tc["name"], arguments=tc["arguments"])
            )

        return response


def patch_openai(client: Any, recorder: Any) -> OpenAIPatch:
    """Convenience function to create and activate an OpenAI patch."""
    p = OpenAIPatch(client, recorder)
    p.activate()
    return p
