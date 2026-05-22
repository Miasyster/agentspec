# agentspec

**Testing framework for AI agent workflows. pytest for agents.**

[![CI](https://github.com/Miasyster/agentspec/actions/workflows/ci.yml/badge.svg)](https://github.com/Miasyster/agentspec/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/agentspec)](https://pypi.org/project/agentspec/)
[![Python](https://img.shields.io/pypi/pyversions/agentspec)](https://pypi.org/project/agentspec/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Agents are the new software — but testing them is still the wild west. `agentspec` brings the rigor of `pytest` to AI agent workflows: give it your agent, mock its tools, and assert on its behavior.

```python
from agentspec import AgentHarness, TraceAssertions

def my_booking_agent(prompt, call_tool):
    slots = call_tool("calendar.list_slots", date="2026-05-22")
    call_tool("calendar.book", slot=slots["slots"][1])
    return "Meeting booked for 2pm"

def test_booking_agent():
    harness = AgentHarness()
    harness.mock("calendar.list_slots", returns={"slots": ["9am", "2pm"]})
    harness.mock("calendar.book", returns={"status": "confirmed"})

    trace = harness.run(my_booking_agent, prompt="Book a meeting")

    a = TraceAssertions(trace)
    a.assert_tool_order("calendar.list_slots", "calendar.book")
    a.assert_steps_within(5)
    a.assert_no_errors()
```

## Why agentspec?

| Problem | agentspec Solution |
|---------|-------------------|
| "How do I test my agent?" | `AgentHarness` — give it your agent, get a trace |
| Agent calls wrong tools or in wrong order | `assert_tool_called`, `assert_tool_order` |
| Agent gets stuck in loops | `LoopDetector` catches consecutive & pattern loops |
| Agent costs spiral out of control | `BudgetGuard` + `assert_cost_within` |
| No visibility into agent execution | `ExecutionTrace` records every tool call, LLM call, cost, timing |
| MCP servers lack test coverage | `MCPSchemaValidator` + `MCPTestHarness` |
| CI has no agent-aware reporting | `--agentspec-ci` flag + GitHub Actions summary |

## Install

```bash
pip install agentspec
```

With optional integrations:

```bash
pip install agentspec[openai]     # OpenAI/DeepSeek/Together integration
pip install agentspec[mcp]        # MCP server testing
pip install agentspec[dev]        # Development (pytest + ruff)
```

## Quick Start

### 1. Test Any Agent (AgentHarness)

Your agent is just a function that takes `(prompt, call_tool)` and returns a string. `AgentHarness` provides the instrumented `call_tool`, runs your agent, and records everything into a trace.

```python
from agentspec import AgentHarness, TraceAssertions

def my_agent(prompt, call_tool):
    results = call_tool("web_search", query=prompt)
    summary = call_tool("summarize", text=str(results))
    return f"Summary: {summary}"

def test_search_agent():
    harness = AgentHarness(max_steps=10, max_cost=0.10)
    harness.mock("web_search", returns={"results": ["doc1", "doc2"]})
    harness.mock("summarize", returns="Here's a summary...")

    trace = harness.run(my_agent, prompt="python testing")

    a = TraceAssertions(trace)
    a.assert_tool_called("web_search", times=1)
    a.assert_tool_order("web_search", "summarize")
    a.assert_steps_within(5)
    a.assert_output_contains("Summary")
    a.assert_no_errors()
```

Or as a one-liner:

```python
from agentspec import run_agent_test

trace = run_agent_test(
    my_agent,
    prompt="python testing",
    mock_tools={"web_search": ["doc1"], "summarize": "summary"},
)
```

### 2. Test with Adapter Protocol

Any class with a `.run(prompt, call_tool)` method works:

```python
class TravelAgent:
    def run(self, prompt, call_tool, **kwargs):
        weather = call_tool("get_weather", city="Shanghai")
        if weather["temp"] > 30:
            return "Pack light clothes!"
        return "Bring a jacket."

def test_travel_agent():
    harness = AgentHarness()
    harness.mock("get_weather", returns={"temp": 35})
    trace = harness.run(TravelAgent(), prompt="Shanghai trip")
    TraceAssertions(trace).assert_output_contains("light clothes")
```

### 3. Live LLM Integration (DeepSeek/OpenAI)

```python
from openai import OpenAI
from agentspec import TraceAssertions
from agentspec.integrations.agent_runner import AgentRunner

def test_weather_agent_live():
    client = OpenAI(base_url="https://api.deepseek.com", api_key="sk-...")
    runner = AgentRunner(client=client, model="deepseek-chat", max_steps=10, max_cost=0.05)

    runner.register_tool(
        "get_weather",
        lambda city: {"city": city, "temp": 28, "condition": "sunny"},
        schema={
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    )

    trace = runner.run("What's the weather in Shanghai?")

    a = TraceAssertions(trace)
    a.assert_tool_called("get_weather")
    a.assert_steps_within(5)
    a.assert_cost_within(0.05)
    a.assert_no_errors()
```

### 3. MCP Server Testing

```python
from agentspec.mcp import MCPSchemaValidator

def test_mcp_tool_schemas():
    schemas = [
        {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    ]

    v = MCPSchemaValidator(schemas)
    v.assert_tool_exists("get_weather")
    v.assert_has_description("get_weather")
    v.assert_parameter_required("get_weather", "city")
    v.assert_parameter_type("get_weather", "city", "string")
    v.assert_parameter_optional("get_weather", "units")
    v.assert_all_tools_have_descriptions()
```

Live MCP server testing (requires `pip install agentspec[mcp]`):

```python
import pytest
from agentspec.mcp import MCPTestHarness

@pytest.mark.asyncio
async def test_live_mcp_server():
    harness = MCPTestHarness(server_cmd=["python", "-m", "my_mcp_server"])
    async with harness:
        harness.assert_tool_available("get_weather")
        result = await harness.call_and_assert(
            "get_weather",
            kwargs={"city": "Shanghai"},
            expect_keys=["temp", "condition"],
        )
        harness.assert_no_errors()
```

### 4. HTML Trace Reports

```python
from agentspec import render_trace_html, save_trace_report, print_trace_summary

# Save as HTML file
save_trace_report(trace, "reports/agent_trace.html")

# Print to terminal
print_trace_summary(trace)
# ╭─── Agent Trace ──────────────────────────────╮
# │ Prompt: What's the weather in Shanghai?       │
# │ Steps: 2 │ LLM Calls: 3 │ Cost: $0.0042      │
# ├──────────────────────────────────────────────-┤
# │ 1. LLM deepseek-chat (420 tok, $0.001, 800ms) │
# │ 2. get_weather(city="Shanghai") -> {...}       │
# │ 3. LLM deepseek-chat (580 tok, $0.002, 650ms) │
# ╰──────────────────────────────────────────────-╯
```

Generate reports automatically in pytest:

```bash
pytest tests/ --agentspec-report=reports/traces.html
```

### 5. CI/CD Integration

Add `--agentspec-ci` to get cost summaries and budget gates:

```bash
# Print cost report + fail if total cost exceeds $1.00
pytest tests/ --agentspec-ci --agentspec-budget 1.0
```

In GitHub Actions, this automatically writes a job summary with cost breakdown.

Copy the ready-made workflow template:

```bash
cp $(python -c "import agentspec; print(agentspec.__file__.replace('__init__.py', 'templates/github_actions.yml'))") \
   .github/workflows/agent-tests.yml
```

Or just create `.github/workflows/agent-tests.yml`:

```yaml
name: Agent Tests
on: [push, pull_request]
jobs:
  test-agents:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install agentspec
      - run: pytest tests/ --agentspec-ci --agentspec-budget 1.0
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Features

### Core

- **AgentHarness** — Give it your agent, mock its tools, get a recorded trace
- **`run_agent_test()`** — One-liner for testing callable agents
- **AgentAdapter protocol** — Any class with `.run(prompt, call_tool)` is testable
- **ExecutionTrace** — Record every tool call, LLM call, cost, and timing
- **MockTool / MockToolRegistry** — Mock any tool with returns, raises, or side effects
- **TraceAssertions** — 12 assertion methods for tool calls, order, cost, output, errors
- **BudgetGuard** — Enforce step, cost, and token limits
- **LoopDetector** — Catch consecutive repeats and pattern loops (A,B,A,B,A,B)

### Integrations

- **OpenAI Patch** — Monkey-patch any OpenAI-compatible client to auto-record traces
- **AgentRunner** — Full tool-use agent loop with automatic recording
- **MCP Testing** — Schema validation, live server testing, test harness
- **Cost Estimation** — Built-in pricing for DeepSeek, GPT-4o/4.1, Claude models

### Developer Experience

- **pytest Plugin** — `agent_harness`, `agent_recorder`, `mock_tools`, `assert_trace`, `mcp_schema_validator` fixtures
- **HTML Reports** — Self-contained trace visualization with timeline, charts, dark/light theme
- **CI Integration** — GitHub Actions summary, cost reports, budget gates
- **Terminal Output** — Compact trace summary with box drawing

## API Reference

### TraceAssertions

| Method | Description |
|--------|-------------|
| `assert_tool_called(name, times=)` | Tool was called (optionally N times) |
| `assert_tool_not_called(name)` | Tool was never called |
| `assert_tool_order(*names)` | Tools called in this order |
| `assert_steps_within(max)` | Total steps under limit |
| `assert_cost_within(max)` | Total cost under limit |
| `assert_tokens_within(max)` | Total tokens under limit |
| `assert_duration_within(max_ms)` | Duration under limit |
| `assert_output_contains(text)` | Final output contains text |
| `assert_output_not_contains(text)` | Final output doesn't contain text |
| `assert_no_errors()` | No tool calls had errors |
| `assert_no_repeated_tool(max)` | No tool called N+ times consecutively |

### MCPSchemaValidator

| Method | Description |
|--------|-------------|
| `assert_tool_exists(name)` | Tool exists in schema |
| `assert_parameter_required(tool, param)` | Parameter is required |
| `assert_parameter_optional(tool, param)` | Parameter is optional |
| `assert_parameter_type(tool, param, type)` | Parameter has expected type |
| `assert_parameter_exists(tool, param)` | Parameter exists |
| `assert_has_description(tool)` | Tool has non-empty description |
| `assert_tool_count(n)` | Exact number of tools |
| `assert_all_tools_have_descriptions()` | All tools documented |

### pytest CLI Options

| Option | Description |
|--------|-------------|
| `--agentspec-report PATH` | Generate HTML trace report |
| `--agentspec-ci` | Enable CI mode (cost summary, GitHub summary) |
| `--agentspec-budget FLOAT` | Max total cost — fails session if exceeded |

## Project Structure

```
agentspec/
├── __init__.py           # Public API
├── harness.py            # AgentHarness + test_agent() — main entry point
├── trace.py              # ExecutionTrace, ToolCall, LLMCall
├── mock.py               # MockTool, MockToolRegistry
├── assertions.py         # TraceAssertions
├── recorder.py           # AgentRecorder (low-level recording)
├── budget.py             # BudgetGuard
├── loop_detector.py      # LoopDetector
├── visualize.py          # HTML reports + terminal output
├── ci.py                 # CI/CD helpers
├── plugin.py             # pytest plugin
├── mcp.py                # MCP server testing
├── adapters/
│   └── base.py           # AgentAdapter / AsyncAgentAdapter protocols
├── integrations/
│   ├── openai_patch.py   # OpenAI client monkey-patch
│   └── agent_runner.py   # Tool-use agent loop
└── templates/
    └── github_actions.yml # CI workflow template
```

## Contributing

```bash
git clone https://github.com/Miasyster/agentspec.git
cd agentspec
pip install -e ".[dev]"
python -m pytest tests/ -x -q -p no:asyncio
```

## License

MIT
