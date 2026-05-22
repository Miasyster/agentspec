"""Tests for AgentRecorder."""

import pytest
from agentspec.recorder import AgentRecorder
from agentspec.budget import BudgetExceeded
from agentspec.loop_detector import LoopDetected


class TestAgentRecorder:
    def test_mock_tool_call(self):
        rec = AgentRecorder()
        rec.mock_tool("search", returns={"results": ["a", "b"]})

        result = rec.call_tool("search", q="test")
        assert result == {"results": ["a", "b"]}

        trace = rec.finish("done")
        assert trace.total_steps == 1
        assert trace.steps[0].name == "search"
        assert trace.steps[0].arguments == {"q": "test"}

    def test_real_tool_call(self):
        rec = AgentRecorder()
        rec.register_tool("add", lambda x, y: x + y)

        result = rec.call_tool("add", x=3, y=4)
        assert result == 7

    def test_missing_tool(self):
        rec = AgentRecorder()
        with pytest.raises(KeyError, match="not registered"):
            rec.call_tool("unknown")

    def test_llm_recording(self):
        rec = AgentRecorder()
        rec.record_llm_call(model="gpt-4", prompt_tokens=100, completion_tokens=50, cost=0.01)
        rec.record_llm_call(model="gpt-4", prompt_tokens=200, completion_tokens=80, cost=0.02)

        trace = rec.finish()
        assert len(trace.llm_calls) == 2
        assert trace.total_tokens == 430
        assert trace.total_cost == 0.03

    def test_budget_step_exceeded(self):
        rec = AgentRecorder(max_steps=2)
        rec.mock_tool("t", returns="ok")

        rec.call_tool("t")
        rec.call_tool("t")
        with pytest.raises(BudgetExceeded, match="Step budget"):
            rec.call_tool("t")

    def test_budget_cost_exceeded(self):
        rec = AgentRecorder(max_cost=0.05)
        rec.record_llm_call(cost=0.03)
        with pytest.raises(BudgetExceeded, match="Cost budget"):
            rec.record_llm_call(cost=0.03)

    def test_loop_detection(self):
        rec = AgentRecorder(max_consecutive_repeats=3)
        rec.mock_tool("retry", returns="fail")

        rec.call_tool("retry")
        rec.call_tool("retry")
        with pytest.raises(LoopDetected):
            rec.call_tool("retry")

    def test_full_workflow(self):
        rec = AgentRecorder()
        rec.mock_tool("search", returns={"items": ["doc1"]})
        rec.mock_tool("read", returns="content of doc1")
        rec.mock_tool("summarize", returns="summary")

        rec.record_llm_call(model="claude", prompt_tokens=50, completion_tokens=20, cost=0.005)
        rec.call_tool("search", query="python testing")
        rec.record_llm_call(model="claude", prompt_tokens=100, completion_tokens=30, cost=0.008)
        rec.call_tool("read", path="doc1")
        rec.record_llm_call(model="claude", prompt_tokens=200, completion_tokens=50, cost=0.015)
        rec.call_tool("summarize", text="content of doc1")

        trace = rec.finish("Here is the summary of doc1...")

        assert trace.total_steps == 3
        assert trace.tool_names == ["search", "read", "summarize"]
        assert trace.total_cost == pytest.approx(0.028, abs=1e-6)
        assert "summary" in trace.final_output
