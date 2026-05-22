"""Live integration test with DeepSeek API.

Run with: DEEPSEEK_API_KEY=sk-xxx pytest examples/test_deepseek_live.py -v -s
"""

import os
import pytest
from agentspec import TraceAssertions
from agentspec.integrations.agent_runner import AgentRunner


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
skip_no_key = pytest.mark.skipif(not DEEPSEEK_API_KEY, reason="DEEPSEEK_API_KEY not set")


def _make_runner() -> AgentRunner:
    from openai import OpenAI

    client = OpenAI(base_url="https://api.deepseek.com", api_key=DEEPSEEK_API_KEY)
    runner = AgentRunner(client=client, model="deepseek-chat", max_steps=10, max_cost=0.05)

    runner.register_tool(
        "get_weather",
        lambda city: {"city": city, "temp": 28, "condition": "sunny", "humidity": 60},
        schema={
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    )

    runner.register_tool(
        "get_exchange_rate",
        lambda base, target: {"base": base, "target": target, "rate": 7.24},
        schema={
            "description": "Get exchange rate between two currencies",
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {"type": "string", "description": "Base currency code"},
                    "target": {"type": "string", "description": "Target currency code"},
                },
                "required": ["base", "target"],
            },
        },
    )

    return runner


@skip_no_key
def test_deepseek_single_tool_call():
    """DeepSeek should call get_weather and return a formatted response."""
    runner = _make_runner()
    trace = runner.run("What's the weather in Shanghai?")

    print(f"\n--- Trace ---")
    print(f"Steps: {trace.total_steps}")
    print(f"Tools: {trace.tool_names}")
    print(f"LLM calls: {len(trace.llm_calls)}")
    print(f"Total tokens: {trace.total_tokens}")
    print(f"Total cost: ${trace.total_cost:.4f}")
    print(f"Output: {trace.final_output[:200]}")

    a = TraceAssertions(trace)
    a.assert_tool_called("get_weather")
    a.assert_tool_not_called("get_exchange_rate")
    a.assert_steps_within(5)
    a.assert_cost_within(0.05)
    a.assert_no_errors()


@skip_no_key
def test_deepseek_multi_tool_call():
    """DeepSeek should call multiple tools for a complex query."""
    runner = _make_runner()
    trace = runner.run(
        "I'm traveling to Shanghai. What's the weather there? "
        "Also, what's the USD to CNY exchange rate?"
    )

    print(f"\n--- Trace ---")
    print(f"Steps: {trace.total_steps}")
    print(f"Tools: {trace.tool_names}")
    print(f"LLM calls: {len(trace.llm_calls)}")
    print(f"Total tokens: {trace.total_tokens}")
    print(f"Total cost: ${trace.total_cost:.4f}")
    print(f"Output: {trace.final_output[:300]}")

    a = TraceAssertions(trace)
    a.assert_tool_called("get_weather")
    a.assert_tool_called("get_exchange_rate")
    a.assert_steps_within(10)
    a.assert_cost_within(0.05)
    a.assert_no_errors()


@skip_no_key
def test_deepseek_no_tool_needed():
    """DeepSeek should not call tools for a simple question."""
    runner = _make_runner()
    trace = runner.run("What is 2 + 2?")

    print(f"\n--- Trace ---")
    print(f"Steps: {trace.total_steps}")
    print(f"Tools: {trace.tool_names}")
    print(f"Output: {trace.final_output[:200]}")

    a = TraceAssertions(trace)
    a.assert_steps_within(0)
    a.assert_output_contains("4")
