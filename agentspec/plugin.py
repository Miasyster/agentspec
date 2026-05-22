"""Pytest plugin for agentspec."""

from __future__ import annotations

import pytest

from agentspec.recorder import AgentRecorder
from agentspec.mock import MockToolRegistry
from agentspec.trace import ExecutionTrace
from agentspec.assertions import TraceAssertions


@pytest.fixture
def agent_recorder():
    """Provides an AgentRecorder for capturing agent execution."""
    return AgentRecorder()


@pytest.fixture
def mock_tools():
    """Provides a MockToolRegistry for mocking agent tools."""
    registry = MockToolRegistry()
    yield registry
    registry.clear()


@pytest.fixture
def assert_trace():
    """Returns a factory that creates TraceAssertions for a given trace."""

    def _factory(trace: ExecutionTrace) -> TraceAssertions:
        return TraceAssertions(trace)

    return _factory
