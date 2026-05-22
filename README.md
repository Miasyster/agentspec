# agentspec

Testing framework for AI agent workflows. **pytest for agents.**

```python
def test_booking_agent(agent_recorder):
    rec = agent_recorder
    rec.mock_tool("calendar.list_slots", returns={"slots": ["9am", "2pm"]})
    rec.mock_tool("calendar.book", returns={"status": "confirmed"})

    rec.call_tool("calendar.list_slots", date="2026-05-22")
    rec.call_tool("calendar.book", slot="2pm")
    trace = rec.finish("Meeting booked for 2pm")

    a = TraceAssertions(trace)
    a.assert_tool_order("calendar.list_slots", "calendar.book")
    a.assert_steps_within(5)
    a.assert_cost_within(0.05)
    a.assert_no_errors()
```

## Install

```bash
pip install agentspec
```

## Features

- **Mock tools** — Intercept and fake MCP/tool calls
- **Execution traces** — Record every tool call, LLM call, cost, and timing
- **Budget guards** — Stop agents that exceed step/cost/token limits
- **Loop detection** — Catch runaway loops (consecutive repeats and pattern cycles)
- **Rich assertions** — Assert on tool order, call count, cost, output, errors
- **pytest plugin** — `agent_recorder`, `mock_tools`, `assert_trace` fixtures out of the box

## License

MIT
