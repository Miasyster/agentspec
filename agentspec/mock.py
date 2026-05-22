"""Mock tools for agent testing."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class MockTool:
    name: str
    returns: Any = None
    raises: Exception | None = None
    side_effect: Callable[..., Any] | None = None
    call_log: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, **kwargs: Any) -> Any:
        self.call_log.append(kwargs)
        if self.raises is not None:
            raise self.raises
        if self.side_effect is not None:
            return self.side_effect(**kwargs)
        return self.returns

    @property
    def called(self) -> bool:
        return len(self.call_log) > 0

    @property
    def call_count(self) -> int:
        return len(self.call_log)

    def assert_called(self) -> None:
        if not self.called:
            raise AssertionError(f"MockTool '{self.name}' was never called")

    def assert_called_times(self, n: int) -> None:
        if self.call_count != n:
            raise AssertionError(
                f"MockTool '{self.name}' called {self.call_count} times, expected {n}"
            )

    def assert_called_with(self, **expected: Any) -> None:
        if not self.called:
            raise AssertionError(f"MockTool '{self.name}' was never called")
        last_call = self.call_log[-1]
        for key, val in expected.items():
            if key not in last_call:
                raise AssertionError(
                    f"MockTool '{self.name}' last call missing argument '{key}'"
                )
            if last_call[key] != val:
                raise AssertionError(
                    f"MockTool '{self.name}' argument '{key}': "
                    f"expected {val!r}, got {last_call[key]!r}"
                )

    def reset(self) -> None:
        self.call_log.clear()


class MockToolRegistry:
    def __init__(self) -> None:
        self._mocks: dict[str, MockTool] = {}

    def register(
        self,
        name: str,
        *,
        returns: Any = None,
        raises: Exception | None = None,
        side_effect: Callable[..., Any] | None = None,
    ) -> MockTool:
        mock = MockTool(name=name, returns=returns, raises=raises, side_effect=side_effect)
        self._mocks[name] = mock
        return mock

    def get(self, name: str) -> MockTool | None:
        return self._mocks.get(name)

    def resolve(self, name: str, **kwargs: Any) -> Any:
        mock = self._mocks.get(name)
        if mock is None:
            raise KeyError(f"No mock registered for tool '{name}'")
        return mock(**kwargs)

    def has(self, name: str) -> bool:
        return name in self._mocks

    def reset_all(self) -> None:
        for mock in self._mocks.values():
            mock.reset()

    def clear(self) -> None:
        self._mocks.clear()

    @contextmanager
    def patch(self, name: str, *, returns: Any = None, raises: Exception | None = None):
        prev = self._mocks.get(name)
        self.register(name, returns=returns, raises=raises)
        try:
            yield self._mocks[name]
        finally:
            if prev is not None:
                self._mocks[name] = prev
            else:
                self._mocks.pop(name, None)
