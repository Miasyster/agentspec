"""agentspec — Testing framework for AI agent workflows."""

from agentspec.trace import ExecutionTrace, ToolCall, LLMCall
from agentspec.mock import MockTool, MockToolRegistry
from agentspec.assertions import TraceAssertions
from agentspec.recorder import AgentRecorder
from agentspec.budget import BudgetGuard
from agentspec.loop_detector import LoopDetector
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
    "MCPServerUnderTest",
    "MCPSchemaValidator",
    "MCPTestHarness",
    "render_trace_html",
    "save_trace_report",
    "print_trace_summary",
    "CIReporter",
]
