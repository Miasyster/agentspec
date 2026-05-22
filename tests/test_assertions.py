"""Tests for trace assertions."""

import pytest
from agentspec.trace import ExecutionTrace, ToolCall, LLMCall
from agentspec.assertions import TraceAssertions


def _make_trace(*tool_names: str, cost: float = 0.0, output: str = "") -> ExecutionTrace:
    trace = ExecutionTrace()
    for name in tool_names:
        trace.add_tool_call(ToolCall(name=name))
    if cost > 0:
        trace.add_llm_call(LLMCall(cost=cost, prompt_tokens=100, completion_tokens=50))
    trace.finish(output)
    return trace


class TestToolAssertions:
    def test_assert_tool_called(self):
        a = TraceAssertions(_make_trace("search", "read"))
        a.assert_tool_called("search")
        a.assert_tool_called("read")

    def test_assert_tool_called_fail(self):
        a = TraceAssertions(_make_trace("search"))
        with pytest.raises(AssertionError, match="'write' was never called"):
            a.assert_tool_called("write")

    def test_assert_tool_called_times(self):
        a = TraceAssertions(_make_trace("search", "read", "search"))
        a.assert_tool_called("search", times=2)

    def test_assert_tool_called_times_fail(self):
        a = TraceAssertions(_make_trace("search"))
        with pytest.raises(AssertionError, match="called 1 times, expected 3"):
            a.assert_tool_called("search", times=3)

    def test_assert_tool_not_called(self):
        a = TraceAssertions(_make_trace("search"))
        a.assert_tool_not_called("write")

    def test_assert_tool_not_called_fail(self):
        a = TraceAssertions(_make_trace("search"))
        with pytest.raises(AssertionError):
            a.assert_tool_not_called("search")


class TestOrderAssertions:
    def test_assert_tool_order(self):
        a = TraceAssertions(_make_trace("search", "analyze", "write"))
        a.assert_tool_order("search", "write")
        a.assert_tool_order("search", "analyze", "write")

    def test_assert_tool_order_fail(self):
        a = TraceAssertions(_make_trace("search", "analyze", "write"))
        with pytest.raises(AssertionError, match="Expected tool order"):
            a.assert_tool_order("write", "search")


class TestBudgetAssertions:
    def test_assert_steps_within(self):
        a = TraceAssertions(_make_trace("a", "b", "c"))
        a.assert_steps_within(5)
        a.assert_steps_within(3)

    def test_assert_steps_within_fail(self):
        a = TraceAssertions(_make_trace("a", "b", "c"))
        with pytest.raises(AssertionError, match="3 steps"):
            a.assert_steps_within(2)

    def test_assert_cost_within(self):
        a = TraceAssertions(_make_trace("a", cost=0.05))
        a.assert_cost_within(0.10)

    def test_assert_cost_within_fail(self):
        a = TraceAssertions(_make_trace("a", cost=0.05))
        with pytest.raises(AssertionError, match="cost"):
            a.assert_cost_within(0.01)


class TestOutputAssertions:
    def test_assert_output_contains(self):
        a = TraceAssertions(_make_trace(output="Meeting booked for 2pm"))
        a.assert_output_contains("2pm")

    def test_assert_output_contains_fail(self):
        a = TraceAssertions(_make_trace(output="Meeting booked for 2pm"))
        with pytest.raises(AssertionError):
            a.assert_output_contains("3pm")

    def test_assert_output_not_contains(self):
        a = TraceAssertions(_make_trace(output="success"))
        a.assert_output_not_contains("error")


class TestErrorAssertions:
    def test_assert_no_errors(self):
        a = TraceAssertions(_make_trace("a", "b"))
        a.assert_no_errors()

    def test_assert_no_errors_fail(self):
        trace = ExecutionTrace()
        trace.add_tool_call(ToolCall(name="bad", error="timeout"))
        a = TraceAssertions(trace)
        with pytest.raises(AssertionError, match="1 tool errors"):
            a.assert_no_errors()


class TestLoopAssertions:
    def test_assert_no_repeated_tool(self):
        a = TraceAssertions(_make_trace("a", "b", "a", "b"))
        a.assert_no_repeated_tool(max_consecutive=3)

    def test_assert_no_repeated_tool_fail(self):
        a = TraceAssertions(_make_trace("a", "a", "a"))
        with pytest.raises(AssertionError, match="possible loop"):
            a.assert_no_repeated_tool(max_consecutive=3)
