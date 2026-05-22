"""Example: testing a weather lookup agent workflow.

Run with: pytest examples/ -v
"""

from agentspec import AgentRecorder, TraceAssertions


def test_weather_agent_happy_path():
    """Agent should look up weather and format a response."""
    rec = AgentRecorder(max_steps=10, max_cost=0.10)

    rec.mock_tool("geocode", returns={"lat": 31.23, "lon": 121.47})
    rec.mock_tool("weather_api", returns={
        "temp": 28,
        "humidity": 65,
        "condition": "partly cloudy",
    })
    rec.mock_tool("format_response", returns="Shanghai: 28°C, partly cloudy, humidity 65%")

    rec.record_llm_call(model="gpt-4o-mini", prompt_tokens=80, completion_tokens=30, cost=0.002)
    rec.call_tool("geocode", city="Shanghai")

    rec.record_llm_call(model="gpt-4o-mini", prompt_tokens=120, completion_tokens=40, cost=0.003)
    rec.call_tool("weather_api", lat=31.23, lon=121.47)

    rec.record_llm_call(model="gpt-4o-mini", prompt_tokens=150, completion_tokens=50, cost=0.004)
    rec.call_tool("format_response", temp=28, condition="partly cloudy")

    trace = rec.finish("Shanghai: 28°C, partly cloudy, humidity 65%")

    a = TraceAssertions(trace)
    a.assert_tool_called("geocode", times=1)
    a.assert_tool_called("weather_api", times=1)
    a.assert_tool_order("geocode", "weather_api", "format_response")
    a.assert_steps_within(5)
    a.assert_cost_within(0.05)
    a.assert_output_contains("28°C")
    a.assert_no_errors()
    a.assert_no_repeated_tool()


def test_weather_agent_city_not_found():
    """Agent should handle geocode failure gracefully."""
    rec = AgentRecorder()

    rec.mock_tool("geocode", raises=ValueError("City not found: Atlantis"))
    rec.mock_tool("weather_api", returns={"temp": 0})

    rec.record_llm_call(model="gpt-4o-mini", prompt_tokens=80, completion_tokens=30, cost=0.002)

    try:
        rec.call_tool("geocode", city="Atlantis")
    except ValueError:
        pass

    trace = rec.finish("Sorry, I couldn't find that city.")

    a = TraceAssertions(trace)
    a.assert_tool_called("geocode")
    a.assert_tool_not_called("weather_api")
    a.assert_steps_within(3)
    a.assert_output_contains("couldn't find")


def test_weather_agent_budget_guard():
    """Agent should not exceed cost budget."""
    rec = AgentRecorder(max_cost=0.01)
    rec.mock_tool("weather_api", returns={"temp": 25})

    rec.record_llm_call(model="gpt-4", prompt_tokens=500, completion_tokens=200, cost=0.008)

    from agentspec.budget import BudgetExceeded
    import pytest

    with pytest.raises(BudgetExceeded):
        rec.record_llm_call(model="gpt-4", prompt_tokens=500, completion_tokens=200, cost=0.008)
