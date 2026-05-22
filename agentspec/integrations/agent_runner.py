"""Run a tool-using agent loop with automatic recording.

Supports any OpenAI-compatible API (DeepSeek, OpenAI, Together, etc.).
"""

from __future__ import annotations

import json
from typing import Any, Callable

from agentspec.recorder import AgentRecorder
from agentspec.integrations.openai_patch import OpenAIPatch


class AgentRunner:
    """Runs an OpenAI-compatible tool-use agent loop and records everything.

    Usage:
        from openai import OpenAI
        client = OpenAI(base_url="https://api.deepseek.com", api_key="...")

        runner = AgentRunner(client=client, model="deepseek-chat")
        runner.register_tool("get_weather", get_weather_fn, schema={...})

        trace = runner.run("What's the weather in Shanghai?")
        # trace contains all LLM calls, tool calls, costs, etc.
    """

    def __init__(
        self,
        client: Any,
        model: str = "deepseek-chat",
        *,
        max_steps: int = 20,
        max_cost: float = 0.50,
        system_prompt: str = "You are a helpful assistant.",
    ) -> None:
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.recorder = AgentRecorder(max_steps=max_steps, max_cost=max_cost)
        self._tool_fns: dict[str, Callable[..., Any]] = {}
        self._tool_schemas: list[dict[str, Any]] = []

    def register_tool(
        self,
        name: str,
        fn: Callable[..., Any],
        schema: dict[str, Any],
    ) -> None:
        self._tool_fns[name] = fn
        self.recorder.register_tool(name, fn)
        tool_def = {"type": "function", "function": {"name": name, **schema}}
        self._tool_schemas.append(tool_def)

    def run(self, prompt: str) -> Any:
        from agentspec.trace import ExecutionTrace

        self.recorder.trace.prompt = prompt
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            for _ in range(self.recorder.budget.max_steps):
                kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
                if self._tool_schemas:
                    kwargs["tools"] = self._tool_schemas

                import time as _time
                from agentspec.trace import LLMCall
                from agentspec.integrations.openai_patch import _estimate_cost

                start = _time.time()
                response = self.client.chat.completions.create(**kwargs)
                elapsed = (_time.time() - start) * 1000

                usage = getattr(response, "usage", None)
                model = getattr(response, "model", self.model)
                pt = getattr(usage, "prompt_tokens", 0) if usage else 0
                ct = getattr(usage, "completion_tokens", 0) if usage else 0
                self.recorder.trace.add_llm_call(LLMCall(
                    model=model,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    total_tokens=pt + ct,
                    cost=_estimate_cost(model, pt, ct),
                    duration_ms=elapsed,
                ))

                choice = response.choices[0]
                msg = choice.message

                if not getattr(msg, "tool_calls", None):
                    self.recorder.finish(msg.content or "")
                    break

                messages.append(msg.model_dump())

                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        args = {}

                    try:
                        result = self.recorder.call_tool(fn_name, **args)
                        result_str = json.dumps(result) if not isinstance(result, str) else result
                    except Exception as e:
                        result_str = f"Error: {e}"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    })
            else:
                self.recorder.finish("[max steps reached]")
        except Exception:
            self.recorder.finish("[error]")
            raise

        return self.recorder.trace
