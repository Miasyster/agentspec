"""Tests for loop detection."""

import pytest
from agentspec.loop_detector import LoopDetector, LoopDetected


class TestLoopDetector:
    def test_no_loop(self):
        ld = LoopDetector(max_consecutive=3)
        ld.check("a")
        ld.check("b")
        ld.check("a")
        ld.check("b")

    def test_consecutive_loop(self):
        ld = LoopDetector(max_consecutive=3)
        ld.check("retry")
        ld.check("retry")
        with pytest.raises(LoopDetected, match="3 times consecutively"):
            ld.check("retry")

    def test_pattern_loop(self):
        ld = LoopDetector(max_consecutive=10)
        with pytest.raises(LoopDetected, match="Repeating pattern"):
            for tool in ["search", "parse"] * 4:
                ld.check(tool)

    def test_reset(self):
        ld = LoopDetector(max_consecutive=3)
        ld.check("a")
        ld.check("a")
        ld.reset()
        ld.check("a")
        ld.check("a")

    def test_three_tool_pattern(self):
        ld = LoopDetector(max_consecutive=20)
        with pytest.raises(LoopDetected, match="Repeating pattern"):
            for tool in ["fetch", "parse", "store"] * 4:
                ld.check(tool)
