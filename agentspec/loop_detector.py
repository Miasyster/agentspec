"""Detect runaway loops in agent tool calls."""

from __future__ import annotations


class LoopDetected(RuntimeError):
    pass


class LoopDetector:
    def __init__(self, *, max_consecutive: int = 5) -> None:
        self.max_consecutive = max_consecutive
        self._history: list[str] = []

    def check(self, tool_name: str) -> None:
        self._history.append(tool_name)
        if len(self._history) >= self.max_consecutive:
            tail = self._history[-self.max_consecutive :]
            if len(set(tail)) == 1:
                raise LoopDetected(
                    f"Tool '{tool_name}' called {self.max_consecutive} times "
                    f"consecutively — likely stuck in a loop"
                )
        self._detect_pattern_loop()

    def _detect_pattern_loop(self) -> None:
        if len(self._history) < 6:
            return
        for pattern_len in range(2, len(self._history) // 3 + 1):
            if pattern_len > 5:
                break
            tail = self._history[-pattern_len * 3 :]
            if len(tail) < pattern_len * 3:
                continue
            chunks = [
                tail[i : i + pattern_len]
                for i in range(0, pattern_len * 3, pattern_len)
            ]
            if chunks[0] == chunks[1] == chunks[2]:
                raise LoopDetected(
                    f"Repeating pattern detected: {chunks[0]} repeated 3 times"
                )

    def reset(self) -> None:
        self._history.clear()
