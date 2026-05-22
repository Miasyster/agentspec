"""Tests for tool mocking."""

import pytest
from agentspec.mock import MockTool, MockToolRegistry


class TestMockTool:
    def test_basic_return(self):
        tool = MockTool(name="search", returns={"results": [1, 2, 3]})
        result = tool(q="test")
        assert result == {"results": [1, 2, 3]}
        assert tool.called
        assert tool.call_count == 1

    def test_raises(self):
        tool = MockTool(name="fail", raises=ValueError("bad input"))
        with pytest.raises(ValueError, match="bad input"):
            tool(x=1)
        assert tool.called

    def test_side_effect(self):
        tool = MockTool(name="calc", side_effect=lambda x, y: x + y)
        assert tool(x=3, y=4) == 7

    def test_call_log(self):
        tool = MockTool(name="log_test", returns="ok")
        tool(a=1)
        tool(b=2)
        assert tool.call_count == 2
        assert tool.call_log[0] == {"a": 1}
        assert tool.call_log[1] == {"b": 2}

    def test_assert_called_with(self):
        tool = MockTool(name="t", returns="ok")
        tool(name="alice", age=30)
        tool.assert_called_with(name="alice")

    def test_assert_called_with_wrong_value(self):
        tool = MockTool(name="t", returns="ok")
        tool(name="alice")
        with pytest.raises(AssertionError, match="expected 'bob'"):
            tool.assert_called_with(name="bob")

    def test_reset(self):
        tool = MockTool(name="t", returns="ok")
        tool(x=1)
        assert tool.called
        tool.reset()
        assert not tool.called


class TestMockToolRegistry:
    def test_register_and_resolve(self):
        reg = MockToolRegistry()
        reg.register("search", returns=["a", "b"])
        assert reg.has("search")
        assert reg.resolve("search", q="test") == ["a", "b"]

    def test_resolve_missing(self):
        reg = MockToolRegistry()
        with pytest.raises(KeyError, match="No mock registered"):
            reg.resolve("unknown")

    def test_patch_context(self):
        reg = MockToolRegistry()
        reg.register("tool", returns="original")
        assert reg.resolve("tool") == "original"

        with reg.patch("tool", returns="patched") as mock:
            assert reg.resolve("tool") == "patched"
            assert mock.called

        assert reg.resolve("tool") == "original"

    def test_clear(self):
        reg = MockToolRegistry()
        reg.register("a", returns=1)
        reg.register("b", returns=2)
        reg.clear()
        assert not reg.has("a")
        assert not reg.has("b")
