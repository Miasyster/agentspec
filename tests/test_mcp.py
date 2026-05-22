"""Tests for MCP server testing integration."""

from __future__ import annotations

import pytest

from agentspec.mcp import (
    MCPSchemaValidator,
    MCPServerUnderTest,
    MCPTestHarness,
    MCPToolSchema,
    MCPResource,
)
from agentspec.trace import ExecutionTrace, ToolCall


# ---------------------------------------------------------------------------
# Sample schemas used across tests
# ---------------------------------------------------------------------------

SAMPLE_SCHEMAS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {"type": "string", "description": "Temperature units"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "search_docs",
        "description": "Search documentation by query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "no_desc_tool",
        "description": "",
        "inputSchema": {
            "type": "object",
            "properties": {"x": {"type": "number"}},
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# MCPToolSchema dataclass
# ---------------------------------------------------------------------------

class TestMCPToolSchema:
    def test_required_params(self):
        schema = MCPToolSchema(
            name="t",
            parameters={"properties": {"a": {"type": "string"}}, "required": ["a"]},
        )
        assert schema.required_params == ["a"]

    def test_has_parameter(self):
        schema = MCPToolSchema(
            name="t",
            parameters={"properties": {"city": {"type": "string"}}},
        )
        assert schema.has_parameter("city")
        assert not schema.has_parameter("country")

    def test_parameter_type(self):
        schema = MCPToolSchema(
            name="t",
            parameters={"properties": {"n": {"type": "integer"}}},
        )
        assert schema.parameter_type("n") == "integer"
        assert schema.parameter_type("missing") is None

    def test_empty_parameters(self):
        schema = MCPToolSchema(name="t")
        assert schema.required_params == []
        assert schema.param_properties == {}
        assert not schema.has_parameter("x")


# ---------------------------------------------------------------------------
# MCPResource dataclass
# ---------------------------------------------------------------------------

class TestMCPResource:
    def test_defaults(self):
        r = MCPResource(uri="file:///tmp/a.txt")
        assert r.uri == "file:///tmp/a.txt"
        assert r.name == ""
        assert r.mime_type == ""

    def test_full_init(self):
        r = MCPResource(uri="db://items", name="items", description="All items", mime_type="application/json")
        assert r.name == "items"
        assert r.description == "All items"


# ---------------------------------------------------------------------------
# MCPSchemaValidator — tool existence
# ---------------------------------------------------------------------------

class TestSchemaValidatorToolExistence:
    def test_assert_tool_exists_pass(self):
        v = MCPSchemaValidator(SAMPLE_SCHEMAS)
        v.assert_tool_exists("get_weather")
        v.assert_tool_exists("search_docs")

    def test_assert_tool_exists_fail(self):
        v = MCPSchemaValidator(SAMPLE_SCHEMAS)
        with pytest.raises(AssertionError, match="Tool 'delete_all' not found"):
            v.assert_tool_exists("delete_all")

    def test_tool_names_and_count(self):
        v = MCPSchemaValidator(SAMPLE_SCHEMAS)
        assert set(v.tool_names) == {"get_weather", "search_docs", "no_desc_tool"}
        assert v.tool_count == 3

    def test_assert_tool_count_pass(self):
        v = MCPSchemaValidator(SAMPLE_SCHEMAS)
        v.assert_tool_count(3)

    def test_assert_tool_count_fail(self):
        v = MCPSchemaValidator(SAMPLE_SCHEMAS)
        with pytest.raises(AssertionError, match="Expected 5 tools, found 3"):
            v.assert_tool_count(5)

    def test_get_tool(self):
        v = MCPSchemaValidator(SAMPLE_SCHEMAS)
        tool = v.get_tool("get_weather")
        assert tool is not None
        assert tool.name == "get_weather"
        assert v.get_tool("nonexistent") is None


# ---------------------------------------------------------------------------
# MCPSchemaValidator — parameter assertions
# ---------------------------------------------------------------------------

class TestSchemaValidatorParameters:
    def test_assert_parameter_required_pass(self):
        v = MCPSchemaValidator(SAMPLE_SCHEMAS)
        v.assert_parameter_required("get_weather", "city")

    def test_assert_parameter_required_fail_not_required(self):
        v = MCPSchemaValidator(SAMPLE_SCHEMAS)
        with pytest.raises(AssertionError, match="is not required"):
            v.assert_parameter_required("get_weather", "units")

    def test_assert_parameter_required_fail_no_param(self):
        v = MCPSchemaValidator(SAMPLE_SCHEMAS)
        with pytest.raises(AssertionError, match="has no parameter 'country'"):
            v.assert_parameter_required("get_weather", "country")

    def test_assert_parameter_optional(self):
        v = MCPSchemaValidator(SAMPLE_SCHEMAS)
        v.assert_parameter_optional("get_weather", "units")

    def test_assert_parameter_optional_fail(self):
        v = MCPSchemaValidator(SAMPLE_SCHEMAS)
        with pytest.raises(AssertionError, match="is required, expected it to be optional"):
            v.assert_parameter_optional("get_weather", "city")

    def test_assert_parameter_type_pass(self):
        v = MCPSchemaValidator(SAMPLE_SCHEMAS)
        v.assert_parameter_type("get_weather", "city", "string")
        v.assert_parameter_type("search_docs", "limit", "integer")

    def test_assert_parameter_type_fail(self):
        v = MCPSchemaValidator(SAMPLE_SCHEMAS)
        with pytest.raises(AssertionError, match="has type 'string', expected 'integer'"):
            v.assert_parameter_type("get_weather", "city", "integer")

    def test_assert_parameter_exists(self):
        v = MCPSchemaValidator(SAMPLE_SCHEMAS)
        v.assert_parameter_exists("get_weather", "city")
        with pytest.raises(AssertionError, match="has no parameter"):
            v.assert_parameter_exists("get_weather", "nonexistent")


# ---------------------------------------------------------------------------
# MCPSchemaValidator — description assertions
# ---------------------------------------------------------------------------

class TestSchemaValidatorDescriptions:
    def test_assert_has_description_pass(self):
        v = MCPSchemaValidator(SAMPLE_SCHEMAS)
        v.assert_has_description("get_weather")

    def test_assert_has_description_fail(self):
        v = MCPSchemaValidator(SAMPLE_SCHEMAS)
        with pytest.raises(AssertionError, match="has no description"):
            v.assert_has_description("no_desc_tool")

    def test_assert_all_tools_have_descriptions(self):
        good_schemas = [s for s in SAMPLE_SCHEMAS if s["name"] != "no_desc_tool"]
        v = MCPSchemaValidator(good_schemas)
        v.assert_all_tools_have_descriptions()

    def test_assert_all_tools_have_descriptions_fail(self):
        v = MCPSchemaValidator(SAMPLE_SCHEMAS)
        with pytest.raises(AssertionError, match="Tools without descriptions"):
            v.assert_all_tools_have_descriptions()


# ---------------------------------------------------------------------------
# MCPSchemaValidator — edge cases
# ---------------------------------------------------------------------------

class TestSchemaValidatorEdgeCases:
    def test_empty_schema_list(self):
        v = MCPSchemaValidator([])
        assert v.tool_count == 0
        assert v.tool_names == []

    def test_schema_with_parameters_key_instead_of_inputSchema(self):
        """Some schemas use 'parameters' instead of 'inputSchema'."""
        schemas = [
            {
                "name": "legacy_tool",
                "description": "Uses parameters key.",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "number"}},
                    "required": ["x"],
                },
            },
        ]
        v = MCPSchemaValidator(schemas)
        v.assert_tool_exists("legacy_tool")
        v.assert_parameter_required("legacy_tool", "x")
        v.assert_parameter_type("legacy_tool", "x", "number")

    def test_schema_with_no_parameters(self):
        schemas = [{"name": "ping", "description": "Ping the server."}]
        v = MCPSchemaValidator(schemas)
        v.assert_tool_exists("ping")
        tool = v.get_tool("ping")
        assert tool is not None
        assert tool.required_params == []
        assert tool.param_properties == {}


# ---------------------------------------------------------------------------
# MCPServerUnderTest — initialization and trace recording
# ---------------------------------------------------------------------------

class TestMCPServerUnderTest:
    def test_init_creates_empty_trace(self):
        server = MCPServerUnderTest()
        assert isinstance(server.trace, ExecutionTrace)
        assert server.trace.total_steps == 0

    def test_not_connected_raises_on_list_tools(self):
        server = MCPServerUnderTest()
        with pytest.raises(RuntimeError, match="Not connected"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(server.list_tools())

    def test_not_connected_raises_on_call_tool(self):
        server = MCPServerUnderTest()
        with pytest.raises(RuntimeError, match="Not connected"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(server.call_tool("test"))

    def test_not_connected_raises_on_list_resources(self):
        server = MCPServerUnderTest()
        with pytest.raises(RuntimeError, match="Not connected"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(server.list_resources())

    def test_not_connected_raises_on_read_resource(self):
        server = MCPServerUnderTest()
        with pytest.raises(RuntimeError, match="Not connected"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(server.read_resource("file:///x"))

    def test_get_schema_validator_from_tools(self):
        server = MCPServerUnderTest()
        # Manually populate tools like list_tools() would
        server._tools = [
            MCPToolSchema(
                name="calc",
                description="Calculate something.",
                parameters={
                    "type": "object",
                    "properties": {"expr": {"type": "string"}},
                    "required": ["expr"],
                },
            ),
        ]
        validator = server.get_schema_validator()
        assert isinstance(validator, MCPSchemaValidator)
        validator.assert_tool_exists("calc")
        validator.assert_parameter_required("calc", "expr")


# ---------------------------------------------------------------------------
# Trace recording via MCPServerUnderTest
# ---------------------------------------------------------------------------

class TestMCPTraceRecording:
    def test_trace_records_tool_calls(self):
        """Simulate what happens when call_tool records to trace."""
        trace = ExecutionTrace()
        call = ToolCall(
            name="get_weather",
            arguments={"city": "Shanghai"},
            result={"temp": 28, "condition": "sunny"},
            duration_ms=150.0,
        )
        trace.add_tool_call(call)

        assert trace.total_steps == 1
        assert trace.tool_names == ["get_weather"]
        assert trace.steps[0].arguments == {"city": "Shanghai"}
        assert trace.steps[0].result == {"temp": 28, "condition": "sunny"}

    def test_trace_records_errors(self):
        trace = ExecutionTrace()
        call = ToolCall(
            name="bad_tool",
            arguments={"x": 1},
            error="Connection refused",
            duration_ms=50.0,
        )
        trace.add_tool_call(call)
        assert trace.steps[0].error == "Connection refused"

    def test_trace_records_resource_reads(self):
        trace = ExecutionTrace()
        call = ToolCall(
            name="read_resource(file:///config.json)",
            arguments={"uri": "file:///config.json"},
            result='{"key": "value"}',
            duration_ms=10.0,
        )
        trace.add_tool_call(call)
        assert "read_resource" in trace.tool_names[0]


# ---------------------------------------------------------------------------
# MCPTestHarness — configuration
# ---------------------------------------------------------------------------

class TestMCPTestHarness:
    def test_requires_server_cmd_or_sse_url(self):
        with pytest.raises(ValueError, match="Either server_cmd or sse_url"):
            MCPTestHarness()

    def test_rejects_both_server_cmd_and_sse_url(self):
        with pytest.raises(ValueError, match="Provide either server_cmd or sse_url"):
            MCPTestHarness(server_cmd=["python", "-m", "server"], sse_url="http://x")

    def test_accepts_server_cmd(self):
        harness = MCPTestHarness(server_cmd=["python", "-m", "my_server"])
        assert harness._server_cmd == ["python", "-m", "my_server"]
        assert harness._sse_url is None

    def test_accepts_sse_url(self):
        harness = MCPTestHarness(sse_url="http://localhost:8000/mcp")
        assert harness._sse_url == "http://localhost:8000/mcp"
        assert harness._server_cmd is None

    def test_accepts_env_and_headers(self):
        harness = MCPTestHarness(
            server_cmd=["python", "server.py"],
            env={"API_KEY": "test"},
        )
        assert harness._env == {"API_KEY": "test"}

        harness2 = MCPTestHarness(
            sse_url="http://localhost:8000/mcp",
            headers={"Authorization": "Bearer token"},
        )
        assert harness2._headers == {"Authorization": "Bearer token"}

    def test_server_property_raises_before_connection(self):
        harness = MCPTestHarness(server_cmd=["python", "-m", "server"])
        with pytest.raises(RuntimeError, match="Harness not started"):
            _ = harness.server

    def test_trace_property_raises_before_connection(self):
        harness = MCPTestHarness(server_cmd=["python", "-m", "server"])
        with pytest.raises(RuntimeError, match="Harness not started"):
            _ = harness.trace

    def test_validator_property_raises_before_connection(self):
        harness = MCPTestHarness(server_cmd=["python", "-m", "server"])
        with pytest.raises(RuntimeError, match="No schema validator"):
            _ = harness.validator


# ---------------------------------------------------------------------------
# Integration: schema validator with realistic MCP tool schemas
# ---------------------------------------------------------------------------

class TestRealisticSchemas:
    """Test with schemas resembling real MCP servers."""

    REALISTIC_SCHEMAS = [
        {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path"},
                },
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "description": "Write content to a file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path"},
                    "content": {"type": "string", "description": "File content"},
                    "create_dirs": {"type": "boolean", "description": "Create parent dirs"},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "list_directory",
            "description": "List files and directories at a path.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean"},
                    "pattern": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    ]

    def test_filesystem_server_tools(self):
        v = MCPSchemaValidator(self.REALISTIC_SCHEMAS)
        v.assert_tool_count(3)
        v.assert_tool_exists("read_file")
        v.assert_tool_exists("write_file")
        v.assert_tool_exists("list_directory")
        v.assert_all_tools_have_descriptions()

    def test_write_file_parameters(self):
        v = MCPSchemaValidator(self.REALISTIC_SCHEMAS)
        v.assert_parameter_required("write_file", "path")
        v.assert_parameter_required("write_file", "content")
        v.assert_parameter_optional("write_file", "create_dirs")
        v.assert_parameter_type("write_file", "path", "string")
        v.assert_parameter_type("write_file", "content", "string")
        v.assert_parameter_type("write_file", "create_dirs", "boolean")

    def test_list_directory_parameters(self):
        v = MCPSchemaValidator(self.REALISTIC_SCHEMAS)
        v.assert_parameter_required("list_directory", "path")
        v.assert_parameter_optional("list_directory", "recursive")
        v.assert_parameter_optional("list_directory", "pattern")
        v.assert_parameter_type("list_directory", "recursive", "boolean")
