"""Example: testing an MCP server's tools and schemas.

Shows both offline schema validation and live server testing.

Run with: pytest examples/test_mcp_server.py -v
"""

from __future__ import annotations

import pytest

from agentspec import MCPSchemaValidator, TraceAssertions
from agentspec.trace import ExecutionTrace, ToolCall


# ---------------------------------------------------------------------------
# 1. Offline schema validation (no MCP server needed)
# ---------------------------------------------------------------------------

# These schemas could be loaded from a JSON file, extracted from your server
# code, or captured from a previous list_tools() call.
MY_SERVER_SCHEMAS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {
                    "type": "string",
                    "description": "celsius or fahrenheit",
                },
            },
            "required": ["city"],
        },
    },
    {
        "name": "search_hotels",
        "description": "Search for hotels in a city.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "checkin": {"type": "string", "description": "ISO date"},
                "checkout": {"type": "string", "description": "ISO date"},
                "guests": {"type": "integer"},
            },
            "required": ["city", "checkin", "checkout"],
        },
    },
]


class TestSchemaValidation:
    """Validate tool schemas without starting a server."""

    def test_expected_tools_exist(self):
        v = MCPSchemaValidator(MY_SERVER_SCHEMAS)
        v.assert_tool_exists("get_weather")
        v.assert_tool_exists("search_hotels")
        v.assert_tool_count(2)

    def test_all_tools_documented(self):
        v = MCPSchemaValidator(MY_SERVER_SCHEMAS)
        v.assert_all_tools_have_descriptions()

    def test_weather_required_params(self):
        v = MCPSchemaValidator(MY_SERVER_SCHEMAS)
        v.assert_parameter_required("get_weather", "city")
        v.assert_parameter_optional("get_weather", "units")
        v.assert_parameter_type("get_weather", "city", "string")

    def test_hotel_search_params(self):
        v = MCPSchemaValidator(MY_SERVER_SCHEMAS)
        v.assert_parameter_required("search_hotels", "city")
        v.assert_parameter_required("search_hotels", "checkin")
        v.assert_parameter_required("search_hotels", "checkout")
        v.assert_parameter_optional("search_hotels", "guests")
        v.assert_parameter_type("search_hotels", "guests", "integer")


# ---------------------------------------------------------------------------
# 2. Using the mcp_schema_validator fixture (from plugin.py)
# ---------------------------------------------------------------------------

def test_with_fixture(mcp_schema_validator):
    """The fixture provides the MCPSchemaValidator class itself."""
    v = mcp_schema_validator(MY_SERVER_SCHEMAS)
    v.assert_tool_exists("get_weather")
    v.assert_parameter_required("get_weather", "city")


# ---------------------------------------------------------------------------
# 3. Simulated trace-based testing (mock an MCP interaction)
# ---------------------------------------------------------------------------

def test_weather_tool_trace():
    """Simulate an MCP tool call and verify the trace."""
    trace = ExecutionTrace(prompt="What's the weather in Tokyo?")

    # Simulate calling get_weather via MCP
    trace.add_tool_call(ToolCall(
        name="get_weather",
        arguments={"city": "Tokyo", "units": "celsius"},
        result={"temp": 22, "condition": "cloudy", "humidity": 68},
        duration_ms=340.0,
    ))

    trace.finish("The weather in Tokyo is 22C and cloudy.")

    # Verify using TraceAssertions
    a = TraceAssertions(trace)
    a.assert_tool_called("get_weather", times=1)
    a.assert_steps_within(5)
    a.assert_no_errors()
    a.assert_output_contains("22")
    a.assert_output_contains("cloudy")


def test_multi_tool_trace():
    """Simulate a multi-step agent using MCP tools."""
    trace = ExecutionTrace(prompt="Book a hotel in Paris")

    trace.add_tool_call(ToolCall(
        name="get_weather",
        arguments={"city": "Paris"},
        result={"temp": 18, "condition": "sunny"},
        duration_ms=200.0,
    ))
    trace.add_tool_call(ToolCall(
        name="search_hotels",
        arguments={"city": "Paris", "checkin": "2026-06-01", "checkout": "2026-06-05"},
        result=[{"name": "Hotel Lumiere", "price": 180}],
        duration_ms=450.0,
    ))

    trace.finish("Found Hotel Lumiere in Paris for $180/night.")

    a = TraceAssertions(trace)
    a.assert_tool_order("get_weather", "search_hotels")
    a.assert_steps_within(10)
    a.assert_no_errors()


# ---------------------------------------------------------------------------
# 4. Live server testing (requires 'mcp' package + a running server)
#    Skipped by default — run with: pytest -m "not live" to exclude,
#    or remove the skip to test against your actual server.
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Requires a running MCP server and the 'mcp' package")
@pytest.mark.asyncio
async def test_live_mcp_server():
    """Example of testing a live MCP server (stdio transport)."""
    from agentspec import MCPServerUnderTest

    async with MCPServerUnderTest.stdio("python", "-m", "my_mcp_server") as server:
        # Discover tools
        tools = await server.list_tools()
        assert len(tools) > 0

        # Validate schemas
        validator = server.get_schema_validator()
        validator.assert_tool_exists("get_weather")
        validator.assert_parameter_required("get_weather", "city")

        # Call a tool
        result = await server.call_tool("get_weather", city="Shanghai")
        assert result is not None

        # Check the trace
        assert server.trace.total_steps == 1
        assert server.trace.tool_names == ["get_weather"]
        assert server.trace.steps[0].error is None


@pytest.mark.skip(reason="Requires a running MCP server and the 'mcp' package")
@pytest.mark.asyncio
async def test_live_mcp_harness():
    """Example of using MCPTestHarness for integrated testing."""
    from agentspec import MCPTestHarness

    harness = MCPTestHarness(server_cmd=["python", "-m", "my_mcp_server"])
    async with harness:
        harness.assert_tool_available("get_weather")
        harness.assert_all_tools_documented()

        await harness.call_and_assert(
            "get_weather",
            kwargs={"city": "Shanghai"},
        )
        harness.assert_no_errors()
