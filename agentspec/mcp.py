"""MCP Server testing integration.

Provides tools for testing MCP (Model Context Protocol) servers:
- MCPServerUnderTest: connect to an MCP server and test its tools
- MCPSchemaValidator: validate MCP tool schemas offline
- MCPTestHarness: high-level test harness combining server + assertions

The ``mcp`` package is an optional dependency. Schema validation works
without it; live server testing requires ``pip install agentspec[mcp]``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from agentspec.trace import ExecutionTrace, ToolCall


# ---------------------------------------------------------------------------
# Helper: lazy-import guard for the ``mcp`` package
# ---------------------------------------------------------------------------

def _require_mcp() -> None:
    """Raise a clear error if the ``mcp`` package is not installed."""
    try:
        import mcp  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'mcp' package is required for live MCP server testing. "
            "Install it with: pip install agentspec[mcp]"
        ) from None


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class MCPToolSchema:
    """Parsed representation of a single MCP tool schema."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def required_params(self) -> list[str]:
        return self.parameters.get("required", [])

    @property
    def param_properties(self) -> dict[str, Any]:
        return self.parameters.get("properties", {})

    def has_parameter(self, name: str) -> bool:
        return name in self.param_properties

    def parameter_type(self, name: str) -> str | None:
        prop = self.param_properties.get(name)
        if prop is None:
            return None
        return prop.get("type")


@dataclass
class MCPResource:
    """Parsed representation of an MCP resource."""

    uri: str
    name: str = ""
    description: str = ""
    mime_type: str = ""


# ---------------------------------------------------------------------------
# MCPSchemaValidator — works offline, no ``mcp`` package needed
# ---------------------------------------------------------------------------

class MCPSchemaValidator:
    """Validate MCP tool schemas without connecting to a server.

    Usage::

        schemas = [
            {
                "name": "get_weather",
                "description": "Get weather for a city.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        ]
        v = MCPSchemaValidator(schemas)
        v.assert_tool_exists("get_weather")
        v.assert_parameter_required("get_weather", "city")
        v.assert_parameter_type("get_weather", "city", "string")
        v.assert_has_description("get_weather")
    """

    def __init__(self, tool_schemas: list[dict[str, Any]]) -> None:
        self._tools: dict[str, MCPToolSchema] = {}
        for raw in tool_schemas:
            name = raw["name"]
            self._tools[name] = MCPToolSchema(
                name=name,
                description=raw.get("description", ""),
                parameters=raw.get("inputSchema", raw.get("parameters", {})),
            )

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    def get_tool(self, name: str) -> MCPToolSchema | None:
        return self._tools.get(name)

    # -- assertion helpers ------------------------------------------------

    def assert_tool_exists(self, name: str) -> None:
        if name not in self._tools:
            raise AssertionError(
                f"Tool '{name}' not found. Available tools: {self.tool_names}"
            )

    def assert_tool_count(self, expected: int) -> None:
        if self.tool_count != expected:
            raise AssertionError(
                f"Expected {expected} tools, found {self.tool_count}: {self.tool_names}"
            )

    def assert_has_description(self, name: str) -> None:
        self.assert_tool_exists(name)
        tool = self._tools[name]
        if not tool.description:
            raise AssertionError(f"Tool '{name}' has no description")

    def assert_parameter_exists(self, tool_name: str, param_name: str) -> None:
        self.assert_tool_exists(tool_name)
        tool = self._tools[tool_name]
        if not tool.has_parameter(param_name):
            raise AssertionError(
                f"Tool '{tool_name}' has no parameter '{param_name}'. "
                f"Parameters: {list(tool.param_properties.keys())}"
            )

    def assert_parameter_required(self, tool_name: str, param_name: str) -> None:
        self.assert_parameter_exists(tool_name, param_name)
        tool = self._tools[tool_name]
        if param_name not in tool.required_params:
            raise AssertionError(
                f"Parameter '{param_name}' of tool '{tool_name}' is not required. "
                f"Required params: {tool.required_params}"
            )

    def assert_parameter_optional(self, tool_name: str, param_name: str) -> None:
        self.assert_parameter_exists(tool_name, param_name)
        tool = self._tools[tool_name]
        if param_name in tool.required_params:
            raise AssertionError(
                f"Parameter '{param_name}' of tool '{tool_name}' is required, "
                f"expected it to be optional"
            )

    def assert_parameter_type(
        self, tool_name: str, param_name: str, expected_type: str
    ) -> None:
        self.assert_parameter_exists(tool_name, param_name)
        tool = self._tools[tool_name]
        actual = tool.parameter_type(param_name)
        if actual != expected_type:
            raise AssertionError(
                f"Parameter '{param_name}' of tool '{tool_name}' has type "
                f"'{actual}', expected '{expected_type}'"
            )

    def assert_all_tools_have_descriptions(self) -> None:
        missing = [n for n, t in self._tools.items() if not t.description]
        if missing:
            raise AssertionError(
                f"Tools without descriptions: {missing}"
            )


# ---------------------------------------------------------------------------
# MCPServerUnderTest — live server connection (requires ``mcp`` package)
# ---------------------------------------------------------------------------

class MCPServerUnderTest:
    """Connect to an MCP server and test its tools.

    Usage::

        # stdio transport
        async with MCPServerUnderTest.stdio("python", "-m", "my_mcp_server") as server:
            tools = await server.list_tools()
            result = await server.call_tool("get_weather", city="Shanghai")
            trace = server.trace  # ExecutionTrace with all calls recorded

        # SSE transport
        async with MCPServerUnderTest.sse("http://localhost:8000/mcp") as server:
            tools = await server.list_tools()
            ...
    """

    def __init__(self) -> None:
        self.trace = ExecutionTrace()
        self._session: Any = None
        self._transport_cm: Any = None
        self._session_cm: Any = None
        self._tools: list[MCPToolSchema] = []
        self._resources: list[MCPResource] = []

    # -- factory classmethods for transport selection ----------------------

    @classmethod
    def stdio(cls, *command: str, env: dict[str, str] | None = None) -> _StdioConnector:
        """Create a connector for an MCP server over stdio transport."""
        _require_mcp()
        return _StdioConnector(list(command), env=env)

    @classmethod
    def sse(cls, url: str, *, headers: dict[str, str] | None = None) -> _SSEConnector:
        """Create a connector for an MCP server over SSE transport."""
        _require_mcp()
        return _SSEConnector(url, headers=headers)

    # -- tool / resource introspection ------------------------------------

    async def list_tools(self) -> list[MCPToolSchema]:
        """List all tools exposed by the MCP server."""
        if self._session is None:
            raise RuntimeError("Not connected. Use 'async with' to connect first.")
        result = await self._session.list_tools()
        self._tools = [
            MCPToolSchema(
                name=t.name,
                description=getattr(t, "description", "") or "",
                parameters=getattr(t, "inputSchema", {}) or {},
            )
            for t in result.tools
        ]
        return list(self._tools)

    async def list_resources(self) -> list[MCPResource]:
        """List all resources exposed by the MCP server."""
        if self._session is None:
            raise RuntimeError("Not connected. Use 'async with' to connect first.")
        result = await self._session.list_resources()
        self._resources = [
            MCPResource(
                uri=str(r.uri),
                name=getattr(r, "name", "") or "",
                description=getattr(r, "description", "") or "",
                mime_type=getattr(r, "mimeType", "") or "",
            )
            for r in result.resources
        ]
        return list(self._resources)

    async def read_resource(self, uri: str) -> Any:
        """Read a resource by URI."""
        if self._session is None:
            raise RuntimeError("Not connected. Use 'async with' to connect first.")
        start = time.time()
        error_msg = None
        result = None
        try:
            result = await self._session.read_resource(uri)
        except Exception as exc:
            error_msg = str(exc)
            raise
        finally:
            elapsed = (time.time() - start) * 1000
            call = ToolCall(
                name=f"read_resource({uri})",
                arguments={"uri": uri},
                result=result if error_msg is None else None,
                error=error_msg,
                duration_ms=elapsed,
            )
            self.trace.add_tool_call(call)
        return result

    async def call_tool(self, name: str, **kwargs: Any) -> Any:
        """Call a tool on the MCP server and record the interaction."""
        if self._session is None:
            raise RuntimeError("Not connected. Use 'async with' to connect first.")

        start = time.time()
        error_msg = None
        result = None
        try:
            result = await self._session.call_tool(name, arguments=kwargs)
        except Exception as exc:
            error_msg = str(exc)
            raise
        finally:
            elapsed = (time.time() - start) * 1000
            # Extract text content from MCP CallToolResult if possible
            result_value: Any = result
            if result is not None and hasattr(result, "content"):
                texts = []
                for item in result.content:
                    if hasattr(item, "text"):
                        texts.append(item.text)
                if texts:
                    result_value = texts[0] if len(texts) == 1 else texts

            call = ToolCall(
                name=name,
                arguments=kwargs,
                result=result_value if error_msg is None else None,
                error=error_msg,
                duration_ms=elapsed,
            )
            self.trace.add_tool_call(call)
        return result

    def get_schema_validator(self) -> MCPSchemaValidator:
        """Return an MCPSchemaValidator for the tools discovered so far."""
        raw_schemas = [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.parameters,
            }
            for t in self._tools
        ]
        return MCPSchemaValidator(raw_schemas)


# ---------------------------------------------------------------------------
# Transport connectors (async context managers)
# ---------------------------------------------------------------------------

class _StdioConnector:
    """Async context manager that starts an MCP server via stdio."""

    def __init__(self, command: list[str], *, env: dict[str, str] | None = None) -> None:
        self._command = command
        self._env = env
        self._server: MCPServerUnderTest | None = None

    async def __aenter__(self) -> MCPServerUnderTest:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_params = StdioServerParameters(
            command=self._command[0],
            args=self._command[1:] if len(self._command) > 1 else [],
            env=self._env,
        )
        self._server = MCPServerUnderTest()
        self._server._transport_cm = stdio_client(server_params)
        read_stream, write_stream = await self._server._transport_cm.__aenter__()
        self._server._session_cm = ClientSession(read_stream, write_stream)
        self._server._session = await self._server._session_cm.__aenter__()
        await self._server._session.initialize()
        return self._server

    async def __aexit__(self, *args: Any) -> None:
        if self._server is not None:
            if self._server._session_cm is not None:
                await self._server._session_cm.__aexit__(*args)
            if self._server._transport_cm is not None:
                await self._server._transport_cm.__aexit__(*args)
            self._server.trace.finish()


class _SSEConnector:
    """Async context manager that connects to an MCP server via SSE."""

    def __init__(self, url: str, *, headers: dict[str, str] | None = None) -> None:
        self._url = url
        self._headers = headers
        self._server: MCPServerUnderTest | None = None

    async def __aenter__(self) -> MCPServerUnderTest:
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        self._server = MCPServerUnderTest()
        self._server._transport_cm = sse_client(
            self._url, headers=self._headers or {}
        )
        read_stream, write_stream = await self._server._transport_cm.__aenter__()
        self._server._session_cm = ClientSession(read_stream, write_stream)
        self._server._session = await self._server._session_cm.__aenter__()
        await self._server._session.initialize()
        return self._server

    async def __aexit__(self, *args: Any) -> None:
        if self._server is not None:
            if self._server._session_cm is not None:
                await self._server._session_cm.__aexit__(*args)
            if self._server._transport_cm is not None:
                await self._server._transport_cm.__aexit__(*args)
            self._server.trace.finish()


# ---------------------------------------------------------------------------
# MCPTestHarness — high-level test harness
# ---------------------------------------------------------------------------

class MCPTestHarness:
    """High-level test harness combining server testing with assertions.

    Usage::

        harness = MCPTestHarness(server_cmd=["python", "-m", "my_server"])
        async with harness:
            harness.assert_tool_available("get_weather")
            result = await harness.call_and_assert(
                "get_weather",
                kwargs={"city": "Shanghai"},
                expect_keys=["temp", "condition"],
            )
            harness.assert_no_errors()
    """

    def __init__(
        self,
        *,
        server_cmd: list[str] | None = None,
        sse_url: str | None = None,
        env: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        if server_cmd is None and sse_url is None:
            raise ValueError("Either server_cmd or sse_url must be provided")
        if server_cmd is not None and sse_url is not None:
            raise ValueError("Provide either server_cmd or sse_url, not both")

        self._server_cmd = server_cmd
        self._sse_url = sse_url
        self._env = env
        self._headers = headers
        self._server: MCPServerUnderTest | None = None
        self._connector: _StdioConnector | _SSEConnector | None = None
        self._validator: MCPSchemaValidator | None = None

    @property
    def server(self) -> MCPServerUnderTest:
        if self._server is None:
            raise RuntimeError("Harness not started. Use 'async with' first.")
        return self._server

    @property
    def trace(self) -> ExecutionTrace:
        return self.server.trace

    @property
    def validator(self) -> MCPSchemaValidator:
        if self._validator is None:
            raise RuntimeError(
                "No schema validator available. "
                "Call list_tools() or connect to the server first."
            )
        return self._validator

    async def __aenter__(self) -> MCPTestHarness:
        _require_mcp()
        if self._server_cmd is not None:
            self._connector = _StdioConnector(self._server_cmd, env=self._env)
        else:
            assert self._sse_url is not None
            self._connector = _SSEConnector(self._sse_url, headers=self._headers)
        self._server = await self._connector.__aenter__()
        # Eagerly discover tools so validator is ready
        await self._server.list_tools()
        self._validator = self._server.get_schema_validator()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._connector is not None:
            await self._connector.__aexit__(*args)

    # -- convenience methods -----------------------------------------------

    def assert_tool_available(self, name: str) -> None:
        """Assert that a tool is available on the server."""
        self.validator.assert_tool_exists(name)

    def assert_tool_count(self, expected: int) -> None:
        """Assert the server exposes exactly N tools."""
        self.validator.assert_tool_count(expected)

    async def call_and_assert(
        self,
        tool_name: str,
        *,
        kwargs: dict[str, Any] | None = None,
        expect_keys: list[str] | None = None,
        expect_error: bool = False,
    ) -> Any:
        """Call a tool and run common assertions on the result."""
        call_kwargs = kwargs or {}
        error_caught = None
        result = None

        try:
            result = await self.server.call_tool(tool_name, **call_kwargs)
        except Exception as exc:
            error_caught = exc
            if not expect_error:
                raise

        if expect_error and error_caught is None:
            raise AssertionError(
                f"Expected tool '{tool_name}' to raise an error, but it succeeded"
            )

        if expect_keys is not None and result is not None:
            # Check the last recorded tool call's result for keys
            last_call = self.trace.steps[-1] if self.trace.steps else None
            if last_call is not None and isinstance(last_call.result, dict):
                missing = [k for k in expect_keys if k not in last_call.result]
                if missing:
                    raise AssertionError(
                        f"Tool '{tool_name}' result missing keys: {missing}. "
                        f"Got keys: {list(last_call.result.keys())}"
                    )

        return result

    def assert_no_errors(self) -> None:
        """Assert no tool calls in the trace had errors."""
        errors = [s for s in self.trace.steps if s.error is not None]
        if errors:
            msgs = [f"  {s.name}: {s.error}" for s in errors]
            raise AssertionError(
                f"MCP server had {len(errors)} tool errors:\n" + "\n".join(msgs)
            )

    def assert_all_tools_documented(self) -> None:
        """Assert every tool has a description."""
        self.validator.assert_all_tools_have_descriptions()
