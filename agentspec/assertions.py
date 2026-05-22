"""Assertion helpers for agent execution traces."""

from __future__ import annotations

from agentspec.trace import ExecutionTrace


class TraceAssertions:
    def __init__(self, trace: ExecutionTrace) -> None:
        self.trace = trace

    def assert_tool_called(self, name: str, *, times: int | None = None) -> None:
        count = self.trace.tool_call_count(name)
        if count == 0:
            raise AssertionError(
                f"Tool '{name}' was never called. "
                f"Tools called: {self.trace.tool_names}"
            )
        if times is not None and count != times:
            raise AssertionError(
                f"Tool '{name}' called {count} times, expected {times}"
            )

    def assert_tool_not_called(self, name: str) -> None:
        count = self.trace.tool_call_count(name)
        if count > 0:
            raise AssertionError(f"Tool '{name}' was called {count} times, expected 0")

    def assert_tool_order(self, *names: str) -> None:
        actual = self.trace.tool_names
        idx = 0
        for name in names:
            found = False
            while idx < len(actual):
                if actual[idx] == name:
                    found = True
                    idx += 1
                    break
                idx += 1
            if not found:
                raise AssertionError(
                    f"Expected tool order {list(names)}, "
                    f"but actual order was {actual}"
                )

    def assert_steps_within(self, max_steps: int) -> None:
        if self.trace.total_steps > max_steps:
            raise AssertionError(
                f"Agent took {self.trace.total_steps} steps, max allowed is {max_steps}"
            )

    def assert_cost_within(self, max_cost: float) -> None:
        if self.trace.total_cost > max_cost:
            raise AssertionError(
                f"Agent cost ${self.trace.total_cost:.4f}, max allowed is ${max_cost:.4f}"
            )

    def assert_tokens_within(self, max_tokens: int) -> None:
        if self.trace.total_tokens > max_tokens:
            raise AssertionError(
                f"Agent used {self.trace.total_tokens} tokens, max allowed is {max_tokens}"
            )

    def assert_duration_within(self, max_ms: float) -> None:
        if self.trace.duration_ms > max_ms:
            raise AssertionError(
                f"Agent took {self.trace.duration_ms:.0f}ms, max allowed is {max_ms:.0f}ms"
            )

    def assert_output_contains(self, text: str) -> None:
        if text not in self.trace.final_output:
            raise AssertionError(
                f"Expected output to contain {text!r}, "
                f"but output was: {self.trace.final_output[:200]!r}"
            )

    def assert_output_not_contains(self, text: str) -> None:
        if text in self.trace.final_output:
            raise AssertionError(f"Expected output NOT to contain {text!r}, but it did")

    def assert_no_errors(self) -> None:
        errors = [s for s in self.trace.steps if s.error is not None]
        if errors:
            msgs = [f"  {s.name}: {s.error}" for s in errors]
            raise AssertionError(f"Agent had {len(errors)} tool errors:\n" + "\n".join(msgs))

    def assert_no_repeated_tool(self, max_consecutive: int = 3) -> None:
        names = self.trace.tool_names
        if len(names) < max_consecutive:
            return
        for i in range(len(names) - max_consecutive + 1):
            window = names[i : i + max_consecutive]
            if len(set(window)) == 1:
                raise AssertionError(
                    f"Tool '{window[0]}' called {max_consecutive}+ times consecutively "
                    f"(possible loop). Steps: {names}"
                )
