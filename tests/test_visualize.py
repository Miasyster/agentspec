"""Tests for trace visualization."""

from __future__ import annotations

import os
import tempfile
import time

from agentspec.trace import ExecutionTrace, ToolCall, LLMCall
from agentspec.visualize import (
    render_trace_html,
    render_multi_trace_html,
    save_trace_report,
    print_trace_summary,
)


# ---------------------------------------------------------------------------
# Helpers: build sample traces
# ---------------------------------------------------------------------------

def _make_trace(
    *,
    prompt: str = "What is the weather?",
    num_tools: int = 2,
    num_llm: int = 2,
    include_error: bool = False,
    final_output: str = "The weather is sunny.",
) -> ExecutionTrace:
    """Construct a realistic ExecutionTrace for testing."""
    base_ts = time.time()
    trace = ExecutionTrace(prompt=prompt, started_at=base_ts)

    for i in range(num_llm):
        trace.add_llm_call(
            LLMCall(
                model="deepseek-chat",
                prompt_tokens=100 + i * 50,
                completion_tokens=40 + i * 20,
                cost=0.001 * (i + 1),
                timestamp=base_ts + i * 0.5,
                duration_ms=300 + i * 100,
            )
        )

    for i in range(num_tools):
        error = "connection timeout" if (include_error and i == num_tools - 1) else None
        trace.add_tool_call(
            ToolCall(
                name=f"tool_{i}",
                arguments={"query": f"arg_{i}", "verbose": True},
                result=f"result_{i}" if not error else None,
                error=error,
                timestamp=base_ts + 0.25 + i * 0.5,
                duration_ms=150 + i * 50,
            )
        )

    trace.finish(final_output)
    return trace


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_render_trace_html_valid_structure():
    """HTML output contains DOCTYPE, html, head, and body tags."""
    trace = _make_trace()
    html = render_trace_html(trace)
    assert "<!DOCTYPE html>" in html
    assert "<html" in html
    assert "<head>" in html
    assert "<body>" in html
    assert "</html>" in html


def test_summary_stats_appear_in_html():
    """Quick-stats section contains steps, LLM calls, tokens, cost."""
    trace = _make_trace(num_tools=3, num_llm=4)
    html = render_trace_html(trace)
    # Steps count
    assert ">3<" in html  # 3 tool steps
    # LLM calls count
    assert ">4<" in html  # 4 LLM calls
    # Cost string
    assert "$" in html


def test_tool_calls_appear_in_html():
    """Each tool call name shows up in the rendered HTML."""
    trace = _make_trace(num_tools=3)
    html = render_trace_html(trace)
    assert "tool_0" in html
    assert "tool_1" in html
    assert "tool_2" in html


def test_llm_calls_appear_in_html():
    """LLM model name and token info appear in rendered HTML."""
    trace = _make_trace(num_llm=2)
    html = render_trace_html(trace)
    assert "deepseek-chat" in html
    # Token counts exist somewhere
    assert "tokens" in html.lower() or "tok" in html.lower()


def test_errors_highlighted_in_html():
    """Error tool calls are highlighted and the error section appears."""
    trace = _make_trace(include_error=True)
    html = render_trace_html(trace)
    assert "connection timeout" in html
    assert "Errors" in html
    assert "error" in html.lower()


def test_empty_trace_renders():
    """An empty trace (no steps, no LLM calls) renders without error."""
    trace = ExecutionTrace(prompt="empty")
    trace.finish("")
    html = render_trace_html(trace)
    assert "<!DOCTYPE html>" in html
    assert "empty" in html


def test_multi_trace_renders():
    """render_multi_trace_html with multiple traces includes overview table."""
    t1 = _make_trace(prompt="Trace A", num_tools=1, num_llm=1)
    t2 = _make_trace(prompt="Trace B", num_tools=2, num_llm=3)
    html = render_multi_trace_html([t1, t2])
    assert "<!DOCTYPE html>" in html
    assert "Trace A" in html
    assert "Trace B" in html
    assert "Overview" in html


def test_multi_trace_single_delegates():
    """When given a single trace, render_multi_trace_html defers to single-trace."""
    trace = _make_trace(prompt="Only one")
    html = render_multi_trace_html([trace])
    # Should look like a normal single-trace report, not a tabbed view
    assert "Only one" in html
    assert "Overview" not in html


def test_multi_trace_empty():
    """Empty list of traces renders gracefully."""
    html = render_multi_trace_html([])
    assert "No traces" in html


def test_print_trace_summary_runs(capsys):
    """print_trace_summary executes without raising and prints output."""
    trace = _make_trace()
    print_trace_summary(trace)
    captured = capsys.readouterr()
    assert "weather" in captured.out.lower() or "Steps" in captured.out


def test_save_trace_report_creates_file():
    """save_trace_report writes an HTML file and returns its path."""
    trace = _make_trace()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "report.html")
        result = save_trace_report(trace, path)
        assert os.path.isfile(result)
        with open(result, encoding="utf-8") as f:
            content = f.read()
        assert "<!DOCTYPE html>" in content


def test_save_trace_report_creates_subdirectory():
    """save_trace_report creates intermediate directories if needed."""
    trace = _make_trace()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sub", "dir", "report.html")
        result = save_trace_report(trace, path)
        assert os.path.isfile(result)


def test_cost_breakdown_section_present():
    """Cost Breakdown section appears when there are LLM calls with cost."""
    trace = _make_trace(num_llm=3)
    html = render_trace_html(trace)
    assert "Cost Breakdown" in html


def test_cost_breakdown_absent_without_llm():
    """Cost Breakdown section is absent when there are no LLM calls."""
    trace = _make_trace(num_llm=0)
    html = render_trace_html(trace)
    assert "Cost Breakdown" not in html


def test_tool_usage_section_present():
    """Tool Usage section appears when there are tool calls."""
    trace = _make_trace(num_tools=2)
    html = render_trace_html(trace)
    assert "Tool Usage" in html


def test_tool_usage_absent_without_tools():
    """Tool Usage section is absent when there are no tool calls."""
    trace = _make_trace(num_tools=0)
    html = render_trace_html(trace)
    assert "Tool Usage" not in html


def test_final_output_in_html():
    """Final output section appears with the output text."""
    trace = _make_trace(final_output="The answer is 42.")
    html = render_trace_html(trace)
    assert "The answer is 42." in html
    assert "Final Output" in html


def test_custom_title():
    """Custom title appears in the HTML."""
    trace = _make_trace()
    html = render_trace_html(trace, title="My Custom Report")
    assert "My Custom Report" in html
    assert "<title>My Custom Report</title>" in html


def test_theme_toggle_present():
    """The dark/light theme toggle button is present in the HTML."""
    trace = _make_trace()
    html = render_trace_html(trace)
    assert "theme-toggle" in html


def test_arguments_truncated():
    """Long arguments are truncated in the HTML output."""
    trace = ExecutionTrace(prompt="test")
    trace.add_tool_call(
        ToolCall(
            name="big_tool",
            arguments={"data": "x" * 500},
            result="ok",
        )
    )
    trace.finish("done")
    html = render_trace_html(trace)
    # The raw 500-char string should not appear in full
    assert "x" * 500 not in html
    # But a truncated version should
    assert "..." in html
