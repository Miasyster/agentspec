"""Pytest plugin for agentspec."""

from __future__ import annotations

from typing import Any

import pytest

from agentspec.recorder import AgentRecorder
from agentspec.mock import MockToolRegistry
from agentspec.trace import ExecutionTrace
from agentspec.assertions import TraceAssertions


# ---------------------------------------------------------------------------
# CLI option
# ---------------------------------------------------------------------------

def pytest_addoption(parser: Any) -> None:
    parser.addoption(
        "--agentspec-report",
        default=None,
        metavar="PATH",
        help="Path for HTML trace report generated from agent_recorder fixtures.",
    )
    group = parser.getgroup("agentspec", "Agent testing options")
    group.addoption(
        "--agentspec-ci",
        action="store_true",
        default=False,
        help="Enable CI mode output (cost summary, GitHub Actions integration)",
    )
    group.addoption(
        "--agentspec-budget",
        type=float,
        default=None,
        help="Max total cost budget for CI gate (fails session if exceeded)",
    )


# ---------------------------------------------------------------------------
# Collector: gather traces across all tests
# ---------------------------------------------------------------------------

# Module-level list shared between the fixture and the session-finish hook.
_collected_traces: list[ExecutionTrace] = []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent_recorder():
    """Provides an AgentRecorder for capturing agent execution.

    When ``--agentspec-report`` is passed, finished traces are automatically
    collected and written to the HTML report at the end of the session.
    """
    recorder = AgentRecorder()
    yield recorder
    # After the test finishes, snapshot the trace if it has any activity.
    trace = recorder.trace
    if trace.steps or trace.llm_calls:
        _collected_traces.append(trace)


@pytest.fixture
def mock_tools():
    """Provides a MockToolRegistry for mocking agent tools."""
    registry = MockToolRegistry()
    yield registry
    registry.clear()


@pytest.fixture
def assert_trace():
    """Returns a factory that creates TraceAssertions for a given trace."""

    def _factory(trace: ExecutionTrace) -> TraceAssertions:
        return TraceAssertions(trace)

    return _factory


@pytest.fixture
def mcp_schema_validator():
    """Provides the MCPSchemaValidator class for testing MCP tool schemas.

    Usage in tests::

        def test_my_server_schema(mcp_schema_validator):
            v = mcp_schema_validator(my_tool_schemas)
            v.assert_tool_exists("my_tool")
            v.assert_parameter_required("my_tool", "query")
    """
    from agentspec.mcp import MCPSchemaValidator

    return MCPSchemaValidator


# ---------------------------------------------------------------------------
# Session-finish hook: write combined HTML report
# ---------------------------------------------------------------------------

def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    report_path = session.config.getoption("--agentspec-report")
    if not report_path:
        return
    if not _collected_traces:
        return

    from agentspec.visualize import render_multi_trace_html, save_trace_report

    if len(_collected_traces) == 1:
        save_trace_report(_collected_traces[0], report_path, title="Agent Trace Report")
    else:
        import os

        html_content = render_multi_trace_html(
            _collected_traces, title="Agent Trace Report"
        )
        resolved = os.path.abspath(report_path)
        os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(html_content)

    # Reset for potential re-use (e.g. in-process test runners).
    _collected_traces.clear()


# ---------------------------------------------------------------------------
# CI mode: cost summary + budget gate
# ---------------------------------------------------------------------------

class _AgentSpecCIPlugin:
    """Collects traces during test session and reports in CI mode."""

    def __init__(self) -> None:
        self.traces: list[ExecutionTrace] = []
        self._budget_failed = False

    @pytest.hookimpl(trylast=True)
    def pytest_runtest_makereport(self, item: Any, call: Any) -> None:
        if call.when != "call":
            return
        # Collect traces from recorder fixtures used in this test
        funcargs = getattr(item, "funcargs", {})
        for val in funcargs.values():
            if isinstance(val, AgentRecorder) and val.trace.llm_calls:
                self.traces.append(val.trace)

    def pytest_terminal_summary(
        self, terminalreporter: Any, config: Any
    ) -> None:
        if not self.traces:
            return

        from agentspec.ci import CIReporter

        report = CIReporter.format_cost_report(self.traces)
        terminalreporter.write_line("")
        for line in report.split("\n"):
            terminalreporter.write_line(line)
        terminalreporter.write_line("")

        budget = config.getoption("--agentspec-budget")
        if budget is not None:
            total = sum(t.total_cost for t in self.traces)
            if CIReporter.check_budget_gate(self.traces, max_total_cost=budget):
                terminalreporter.write_line(
                    f"Budget gate PASSED: ${total:.4f} within ${budget:.4f}"
                )
            else:
                self._budget_failed = True
                terminalreporter.write_line(
                    f"Budget gate FAILED: ${total:.4f} exceeds ${budget:.4f}"
                )

    def pytest_sessionfinish(self, session: Any, exitstatus: int) -> None:
        if not self.traces:
            return

        from agentspec.ci import CIReporter

        # Write GitHub Actions summary if available
        CIReporter.write_github_summary(self.traces)


def pytest_configure(config: Any) -> None:
    if config.getoption("--agentspec-ci", default=False):
        plugin = _AgentSpecCIPlugin()
        config.pluginmanager.register(plugin, "agentspec-ci")


def pytest_unconfigure(config: Any) -> None:
    plugin = config.pluginmanager.get_plugin("agentspec-ci")
    if plugin is None:
        return
    if plugin._budget_failed:
        session = getattr(config, "_session", None)
        if session is not None:
            session.exitstatus = 1
