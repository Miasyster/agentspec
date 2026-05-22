"""Example: testing agents with AgentHarness.

Demonstrates three patterns:
1. run_agent_test() one-liner
2. AgentHarness with adapter protocol
3. Full workflow with TraceAssertions

Run with: pytest examples/test_harness_demo.py -v
"""

from agentspec import AgentHarness, TraceAssertions
from agentspec.harness import run_agent_test


# ---------------------------------------------------------------------------
# Pattern 1: run_agent_test() one-liner
# ---------------------------------------------------------------------------

def my_search_agent(prompt, call_tool):
    """A simple agent that searches and summarizes."""
    results = call_tool("search", query=prompt)
    summary = call_tool("summarize", text=str(results))
    return f"Summary: {summary}"


def test_search_agent_one_liner():
    """Test an agent in one function call."""
    trace = run_agent_test(
        my_search_agent,
        prompt="python testing frameworks",
        mock_tools={
            "search": [{"title": "pytest", "url": "pytest.org"}],
            "summarize": "pytest is the most popular Python testing framework.",
        },
    )

    a = TraceAssertions(trace)
    a.assert_tool_called("search", times=1)
    a.assert_tool_called("summarize", times=1)
    a.assert_tool_order("search", "summarize")
    a.assert_steps_within(5)
    a.assert_output_contains("Summary")
    a.assert_no_errors()


# ---------------------------------------------------------------------------
# Pattern 2: AgentHarness with adapter class
# ---------------------------------------------------------------------------

class TravelAgent:
    """An agent that plans trips."""

    def run(self, prompt, call_tool, **kwargs):
        weather = call_tool("get_weather", city="Shanghai")
        hotels = call_tool("search_hotels", city="Shanghai", max_price=500)

        if weather["temp"] > 30:
            hotel_type = "with pool"
        else:
            hotel_type = "standard"

        return (
            f"Shanghai: {weather['temp']}°C, {weather['condition']}. "
            f"Recommended: {hotels[0]} ({hotel_type})"
        )


def test_travel_agent_hot_weather():
    """Agent recommends pool hotel when it's hot."""
    harness = AgentHarness()
    harness.mock("get_weather", returns={"temp": 35, "condition": "sunny"})
    harness.mock("search_hotels", returns=["Grand Hyatt", "Park Hyatt"])

    trace = harness.run(TravelAgent(), prompt="Plan a trip to Shanghai")

    a = TraceAssertions(trace)
    a.assert_tool_order("get_weather", "search_hotels")
    a.assert_output_contains("with pool")
    a.assert_no_errors()


def test_travel_agent_cool_weather():
    """Agent recommends standard hotel when it's cool."""
    harness = AgentHarness()
    harness.mock("get_weather", returns={"temp": 18, "condition": "cloudy"})
    harness.mock("search_hotels", returns=["Holiday Inn"])

    trace = harness.run(TravelAgent(), prompt="Plan a trip to Shanghai")

    a = TraceAssertions(trace)
    a.assert_output_contains("standard")
    a.assert_output_not_contains("pool")


# ---------------------------------------------------------------------------
# Pattern 3: Real tools + mock tools mixed
# ---------------------------------------------------------------------------

def calculator_agent(prompt, call_tool):
    """Agent that uses a real calculator and mock data source."""
    data = call_tool("fetch_data", metric="revenue")
    total = call_tool("calculate", expression=f"{data['q1']} + {data['q2']}")
    return f"Total revenue: ${total}"


def test_mixed_real_and_mock():
    """Use real tools for deterministic ops, mock for external deps."""
    harness = AgentHarness()
    harness.mock("fetch_data", returns={"q1": 1000, "q2": 2000})
    harness.register("calculate", lambda expression: eval(expression))

    trace = harness.run(calculator_agent, prompt="Calculate total revenue")

    a = TraceAssertions(trace)
    a.assert_tool_called("fetch_data")
    a.assert_tool_called("calculate")
    a.assert_output_contains("3000")
    a.assert_no_errors()


# ---------------------------------------------------------------------------
# Pattern 4: Error handling
# ---------------------------------------------------------------------------

def resilient_agent(prompt, call_tool):
    """Agent that handles tool errors gracefully."""
    try:
        result = call_tool("primary_api", query=prompt)
    except ConnectionError:
        result = call_tool("fallback_api", query=prompt)
    return f"Result: {result}"


def test_agent_fallback_on_error():
    """Agent should fall back when primary API fails."""
    harness = AgentHarness()
    harness.mock("primary_api", raises=ConnectionError("timeout"))
    harness.mock("fallback_api", returns="backup data")

    trace = harness.run(resilient_agent, prompt="test")

    a = TraceAssertions(trace)
    a.assert_tool_called("primary_api")
    a.assert_tool_called("fallback_api")
    a.assert_output_contains("backup data")
