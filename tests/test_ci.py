"""Tests for CI/CD integration helpers."""

from agentspec.ci import CIReporter
from agentspec.trace import ExecutionTrace, ToolCall, LLMCall


def _make_trace(
    prompt: str = "test prompt",
    *,
    steps: int = 0,
    cost: float = 0.0,
    tokens: int = 0,
) -> ExecutionTrace:
    trace = ExecutionTrace(prompt=prompt)
    for i in range(steps):
        trace.add_tool_call(ToolCall(name=f"tool_{i}"))
    if cost > 0 or tokens > 0:
        pt = tokens // 2 if tokens else 100
        ct = tokens - pt if tokens else 50
        trace.add_llm_call(LLMCall(
            model="test-model",
            prompt_tokens=pt,
            completion_tokens=ct,
            cost=cost,
        ))
    trace.finish("done")
    return trace


class TestDetectCI:
    def test_returns_none_when_not_in_ci(self, monkeypatch):
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.delenv("GITLAB_CI", raising=False)
        monkeypatch.delenv("JENKINS_URL", raising=False)
        assert CIReporter.detect_ci() is None

    def test_returns_github_when_github_actions(self, monkeypatch):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        assert CIReporter.detect_ci() == "github"

    def test_returns_gitlab_when_gitlab_ci(self, monkeypatch):
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.setenv("GITLAB_CI", "true")
        assert CIReporter.detect_ci() == "gitlab"

    def test_returns_jenkins_when_jenkins_url(self, monkeypatch):
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.delenv("GITLAB_CI", raising=False)
        monkeypatch.setenv("JENKINS_URL", "http://jenkins.example.com")
        assert CIReporter.detect_ci() == "jenkins"


class TestGithubSummary:
    def test_generates_valid_markdown(self):
        traces = [_make_trace("What is the weather?", steps=2, cost=0.01, tokens=500)]
        md = CIReporter.github_summary(traces)
        assert "## Agent Test Results" in md
        assert "| # | Prompt" in md
        assert "What is the weather?" in md
        assert "$0.0100" in md

    def test_empty_traces_no_crash(self):
        md = CIReporter.github_summary([])
        assert "No traces recorded" in md

    def test_truncates_long_prompt(self):
        long_prompt = "A" * 100
        traces = [_make_trace(long_prompt, steps=1, cost=0.005)]
        md = CIReporter.github_summary(traces)
        assert "..." in md
        assert "A" * 100 not in md


class TestFormatCostReport:
    def test_shows_costs(self):
        traces = [
            _make_trace("prompt one", steps=3, cost=0.05, tokens=1000),
            _make_trace("prompt two", steps=1, cost=0.02, tokens=400),
        ]
        report = CIReporter.format_cost_report(traces)
        assert "Agent Test Cost Report" in report
        assert "$0.0700" in report
        assert "Traces:     2" in report
        assert "Tool calls: 4" in report
        assert "1,400" in report

    def test_empty_traces_no_crash(self):
        report = CIReporter.format_cost_report([])
        assert "No traces recorded" in report


class TestCheckBudgetGate:
    def test_passes_when_under_budget(self):
        traces = [_make_trace(cost=0.10), _make_trace(cost=0.20)]
        assert CIReporter.check_budget_gate(traces, max_total_cost=1.0) is True

    def test_fails_when_over_budget(self):
        traces = [_make_trace(cost=0.60), _make_trace(cost=0.50)]
        assert CIReporter.check_budget_gate(traces, max_total_cost=1.0) is False

    def test_passes_at_exact_budget(self):
        traces = [_make_trace(cost=0.50), _make_trace(cost=0.50)]
        assert CIReporter.check_budget_gate(traces, max_total_cost=1.0) is True

    def test_empty_traces_passes(self):
        assert CIReporter.check_budget_gate([], max_total_cost=1.0) is True


class TestMultipleTraceAggregation:
    def test_aggregates_costs_across_traces(self):
        traces = [
            _make_trace("a", steps=2, cost=0.10, tokens=500),
            _make_trace("b", steps=3, cost=0.20, tokens=1000),
            _make_trace("c", steps=1, cost=0.05, tokens=200),
        ]
        report = CIReporter.format_cost_report(traces)
        assert "Traces:     3" in report
        assert "Tool calls: 6" in report
        assert "$0.3500" in report
        assert "1,700" in report

    def test_github_summary_aggregates(self):
        traces = [
            _make_trace("first", steps=2, cost=0.10, tokens=500),
            _make_trace("second", steps=3, cost=0.20, tokens=800),
        ]
        md = CIReporter.github_summary(traces)
        assert "2 traces" in md.lower() or "**2 traces**" in md
        assert "5 tool calls" in md.lower() or "**5 tool calls**" in md
        assert "$0.3000" in md
