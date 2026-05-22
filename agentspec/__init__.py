"""agentspec — Testing framework for AI agent workflows."""

from agentspec.trace import ExecutionTrace, ToolCall, LLMCall
from agentspec.mock import MockTool, MockToolRegistry
from agentspec.assertions import TraceAssertions
from agentspec.recorder import AgentRecorder
from agentspec.budget import BudgetGuard
from agentspec.loop_detector import LoopDetector
from agentspec.harness import AgentHarness, run_agent_test, run_openai_agent_test

# Convenience aliases — use these in test files.
# Named without ``test_`` prefix at module level to prevent pytest
# from collecting them as test items when the plugin is loaded.
spec_agent = run_agent_test
spec_openai_agent = run_openai_agent_test
from agentspec.mcp import MCPServerUnderTest, MCPSchemaValidator, MCPTestHarness
from agentspec.visualize import render_trace_html, save_trace_report, print_trace_summary
from agentspec.ci import CIReporter

__version__ = "0.1.0"

__all__ = [
    "ExecutionTrace",
    "ToolCall",
    "LLMCall",
    "MockTool",
    "MockToolRegistry",
    "TraceAssertions",
    "AgentRecorder",
    "BudgetGuard",
    "LoopDetector",
    "AgentHarness",
    "run_agent_test",
    "run_openai_agent_test",
    "spec_agent",
    "spec_openai_agent",
    "MCPServerUnderTest",
    "MCPSchemaValidator",
    "MCPTestHarness",
    "render_trace_html",
    "save_trace_report",
    "print_trace_summary",
    "CIReporter",
]
