"""Tests for AgentHarness — the core 'give it an agent, test it' interface."""

import asyncio

import pytest

from agentspec import TraceAssertions
from agentspec.budget import BudgetExceeded
from agentspec.harness import AgentHarness, run_agent_test, arun_agent_test
from agentspec.loop_detector import LoopDetected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def simple_agent(prompt, call_tool):
    """A trivial agent: calls search, returns result."""
    result = call_tool("search", query=prompt)
    return f"Found: {result}"


def multi_tool_agent(prompt, call_tool):
    """Agent that uses multiple tools in sequence."""
    weather = call_tool("get_weather", city="Shanghai")
    rate = call_tool("get_exchange_rate", base="USD", target="CNY")
    return f"Weather: {weather['temp']}°C, Rate: {rate['rate']}"


def no_tool_agent(prompt, call_tool):
    """Agent that ignores tools entirely."""
    return f"I don't need tools. The answer is: {prompt}"


def error_agent(prompt, call_tool):
    """Agent that raises an exception."""
    call_tool("search", query=prompt)
    raise RuntimeError("Something went wrong")


def looping_agent(prompt, call_tool):
    """Agent that calls the same tool in a loop."""
    for _ in range(10):
        call_tool("retry", attempt=True)
    return "done"


# ---------------------------------------------------------------------------
# AgentHarness basic lifecycle
# ---------------------------------------------------------------------------

class TestAgentHarnessBasic:
    def test_mock_and_run(self):
        harness = AgentHarness()
        harness.mock("search", returns=["doc1", "doc2"])

        trace = harness.run(simple_agent, prompt="python testing")

        assert trace.total_steps == 1
        assert trace.tool_names == ["search"]
        assert "Found:" in trace.final_output
        assert trace.prompt == "python testing"

    def test_real_tool(self):
        harness = AgentHarness()
        harness.register("search", lambda query: [f"result for {query}"])

        trace = harness.run(simple_agent, prompt="testing")

        assert trace.total_steps == 1
        assert "result for testing" in trace.final_output

    def test_budget_exceeded(self):
        harness = AgentHarness(max_steps=2)
        harness.mock("retry", returns="fail")

        def greedy_agent(prompt, call_tool):
            for _ in range(5):
                call_tool("retry")
            return "done"

        with pytest.raises(BudgetExceeded):
            harness.run(greedy_agent, prompt="go")

    def test_loop_detected(self):
        harness = AgentHarness(max_consecutive_repeats=3)
        harness.mock("retry", returns="fail")

        with pytest.raises(LoopDetected):
            harness.run(looping_agent, prompt="go")

    def test_agent_error_captured_in_trace(self):
        harness = AgentHarness()
        harness.mock("search", returns=["doc1"])

        with pytest.raises(RuntimeError, match="Something went wrong"):
            harness.run(error_agent, prompt="test")

        assert harness.trace.total_steps == 1
        assert "[error:" in harness.trace.final_output

    def test_tool_names_property(self):
        harness = AgentHarness()
        harness.mock("b_tool", returns=1)
        harness.mock("a_tool", returns=2)
        harness.register("c_tool", lambda: 3)

        assert harness.tool_names == ["a_tool", "b_tool", "c_tool"]


# ---------------------------------------------------------------------------
# Agent protocol variations
# ---------------------------------------------------------------------------

class TestAgentProtocol:
    def test_callable_agent(self):
        harness = AgentHarness()
        harness.mock("search", returns=["doc1"])

        trace = harness.run(simple_agent, prompt="test")
        assert trace.total_steps == 1
        assert trace.final_output == "Found: ['doc1']"

    def test_adapter_protocol(self):
        class MyAdapter:
            def run(self, prompt, call_tool, **kwargs):
                result = call_tool("fetch", url=prompt)
                return f"Fetched: {result}"

        harness = AgentHarness()
        harness.mock("fetch", returns="<html>hi</html>")

        trace = harness.run(MyAdapter(), prompt="http://example.com")
        assert trace.final_output == "Fetched: <html>hi</html>"

    def test_agent_returns_none(self):
        def none_agent(prompt, call_tool):
            call_tool("noop")
            return None

        harness = AgentHarness()
        harness.mock("noop", returns="ok")

        trace = harness.run(none_agent, prompt="test")
        assert trace.final_output == ""

    def test_agent_ignores_tools(self):
        harness = AgentHarness()
        harness.mock("search", returns=["doc1"])

        trace = harness.run(no_tool_agent, prompt="42")
        assert trace.total_steps == 0
        assert "42" in trace.final_output

    def test_multi_tool_agent(self):
        harness = AgentHarness()
        harness.mock("get_weather", returns={"temp": 28, "condition": "sunny"})
        harness.mock("get_exchange_rate", returns={"base": "USD", "target": "CNY", "rate": 7.24})

        trace = harness.run(multi_tool_agent, prompt="travel info")

        assert trace.total_steps == 2
        assert trace.tool_names == ["get_weather", "get_exchange_rate"]
        assert "28" in trace.final_output
        assert "7.24" in trace.final_output

    def test_single_arg_agent(self):
        """Agent that only takes prompt (no call_tool)."""
        def simple(prompt):
            return f"Echo: {prompt}"

        harness = AgentHarness()
        trace = harness.run(simple, prompt="hello")
        assert trace.final_output == "Echo: hello"


# ---------------------------------------------------------------------------
# Convenience function: run_agent_test()
# ---------------------------------------------------------------------------

class TestTestAgent:
    def test_basic(self):
        trace = run_agent_test(
            simple_agent,
            prompt="python testing",
            mock_tools={"search": ["doc1", "doc2"]},
        )
        assert trace.total_steps == 1
        assert trace.tool_names == ["search"]

    def test_real_tools(self):
        trace = run_agent_test(
            simple_agent,
            prompt="test",
            real_tools={"search": lambda query: [f"real_{query}"]},
        )
        assert "real_test" in trace.final_output

    def test_custom_budget(self):
        def greedy(prompt, call_tool):
            for _ in range(5):
                call_tool("t")
            return "done"

        with pytest.raises(BudgetExceeded):
            run_agent_test(greedy, prompt="go", mock_tools={"t": "ok"}, max_steps=3)

    def test_mock_exception(self):
        def agent_that_catches(prompt, call_tool):
            try:
                call_tool("fail_tool")
            except ValueError:
                return "caught error"
            return "no error"

        trace = run_agent_test(
            agent_that_catches,
            prompt="test",
            mock_tools={"fail_tool": ValueError("bad input")},
        )
        assert trace.final_output == "caught error"

    def test_mock_side_effect(self):
        call_count = {"n": 0}

        def counter(**kwargs):
            call_count["n"] += 1
            return call_count["n"]

        def counting_agent(prompt, call_tool):
            a = call_tool("counter")
            b = call_tool("counter")
            return f"{a},{b}"

        trace = run_agent_test(
            counting_agent,
            prompt="count",
            mock_tools={"counter": counter},
        )
        assert trace.final_output == "1,2"
        assert trace.total_steps == 2


# ---------------------------------------------------------------------------
# Async support
# ---------------------------------------------------------------------------

class TestAsync:
    def test_arun_async_agent(self):
        async def async_agent(prompt, call_tool):
            result = call_tool("search", query=prompt)
            return f"Async found: {result}"

        harness = AgentHarness()
        harness.mock("search", returns=["doc1"])

        trace = asyncio.get_event_loop().run_until_complete(
            harness.arun(async_agent, prompt="test")
        )
        assert "Async found" in trace.final_output
        assert trace.total_steps == 1

    def test_arun_async_adapter(self):
        class AsyncAdapter:
            async def run(self, prompt, call_tool, **kwargs):
                result = call_tool("fetch", url=prompt)
                return f"Fetched: {result}"

        harness = AgentHarness()
        harness.mock("fetch", returns="data")

        trace = asyncio.get_event_loop().run_until_complete(
            harness.arun(AsyncAdapter(), prompt="http://example.com")
        )
        assert trace.final_output == "Fetched: data"

    def test_atest_agent_convenience(self):
        async def async_agent(prompt, call_tool):
            r = call_tool("search", query=prompt)
            return f"Got: {r}"

        trace = asyncio.get_event_loop().run_until_complete(
            arun_agent_test(async_agent, prompt="test", mock_tools={"search": "result"})
        )
        assert "Got: result" in trace.final_output


# ---------------------------------------------------------------------------
# Integration with TraceAssertions
# ---------------------------------------------------------------------------

class TestHarnessWithAssertions:
    def test_full_flow(self):
        trace = run_agent_test(
            multi_tool_agent,
            prompt="travel info",
            mock_tools={
                "get_weather": {"temp": 28, "condition": "sunny"},
                "get_exchange_rate": {"base": "USD", "target": "CNY", "rate": 7.24},
            },
        )

        a = TraceAssertions(trace)
        a.assert_tool_called("get_weather", times=1)
        a.assert_tool_called("get_exchange_rate", times=1)
        a.assert_tool_order("get_weather", "get_exchange_rate")
        a.assert_steps_within(5)
        a.assert_output_contains("28")
        a.assert_no_errors()
        a.assert_no_repeated_tool()

    def test_tool_not_called(self):
        trace = run_agent_test(
            no_tool_agent,
            prompt="42",
            mock_tools={"search": "unused"},
        )

        a = TraceAssertions(trace)
        a.assert_tool_not_called("search")
        a.assert_steps_within(0)

    def test_output_assertions(self):
        trace = run_agent_test(
            simple_agent,
            prompt="hello",
            mock_tools={"search": ["world"]},
        )

        a = TraceAssertions(trace)
        a.assert_output_contains("Found")
        a.assert_output_not_contains("Error")


# ---------------------------------------------------------------------------
# Harness reuse
# ---------------------------------------------------------------------------

class TestHarnessReuse:
    def test_run_twice_creates_fresh_trace(self):
        harness = AgentHarness()
        harness.mock("search", returns="first")

        trace1 = harness.run(simple_agent, prompt="one")
        assert trace1.total_steps == 1

        harness.mock("search", returns="second")
        trace2 = harness.run(simple_agent, prompt="two")
        assert trace2.total_steps == 1
        assert trace2.prompt == "two"

    def test_traces_are_independent(self):
        harness = AgentHarness()
        harness.mock("search", returns="result")

        trace1 = harness.run(simple_agent, prompt="a")
        trace2 = harness.run(simple_agent, prompt="b")

        assert trace1.prompt == "a"
        assert trace2.prompt == "b"
        assert trace1 is not trace2
