"""Tests for execution trace."""

from agentspec.trace import ExecutionTrace, ToolCall, LLMCall


def test_empty_trace():
    trace = ExecutionTrace(prompt="hello")
    assert trace.total_steps == 0
    assert trace.total_tokens == 0
    assert trace.total_cost == 0.0
    assert trace.tool_names == []
    assert trace.unique_tools == set()


def test_add_tool_calls():
    trace = ExecutionTrace()
    trace.add_tool_call(ToolCall(name="search", arguments={"q": "test"}, result="found"))
    trace.add_tool_call(ToolCall(name="read", arguments={"path": "/a"}, result="content"))
    trace.add_tool_call(ToolCall(name="search", arguments={"q": "other"}, result="found2"))

    assert trace.total_steps == 3
    assert trace.tool_names == ["search", "read", "search"]
    assert trace.unique_tools == {"search", "read"}
    assert trace.tool_call_count("search") == 2
    assert trace.tool_call_count("read") == 1
    assert trace.tool_call_count("write") == 0


def test_add_llm_calls():
    trace = ExecutionTrace()
    trace.add_llm_call(LLMCall(model="gpt-4", prompt_tokens=100, completion_tokens=50, cost=0.01))
    trace.add_llm_call(LLMCall(model="gpt-4", prompt_tokens=200, completion_tokens=80, cost=0.02))

    assert trace.total_tokens == 430
    assert trace.total_cost == 0.03


def test_get_tool_calls():
    trace = ExecutionTrace()
    trace.add_tool_call(ToolCall(name="a", result=1))
    trace.add_tool_call(ToolCall(name="b", result=2))
    trace.add_tool_call(ToolCall(name="a", result=3))

    a_calls = trace.get_tool_calls("a")
    assert len(a_calls) == 2
    assert a_calls[0].result == 1
    assert a_calls[1].result == 3


def test_finish():
    trace = ExecutionTrace()
    trace.finish("done")
    assert trace.final_output == "done"
    assert trace.finished_at > 0
    assert trace.duration_ms >= 0
