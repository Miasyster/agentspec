"""Budget guard for agent execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentspec.trace import ExecutionTrace


class BudgetExceeded(RuntimeError):
    pass


class BudgetGuard:
    def __init__(
        self,
        *,
        max_steps: int = 50,
        max_cost: float = 1.0,
        max_tokens: int = 0,
    ) -> None:
        self.max_steps = max_steps
        self.max_cost = max_cost
        self.max_tokens = max_tokens

    def check_step(self, trace: ExecutionTrace) -> None:
        if self.max_steps > 0 and trace.total_steps >= self.max_steps:
            raise BudgetExceeded(
                f"Step budget exceeded: {trace.total_steps} >= {self.max_steps}"
            )

    def check_cost(self, trace: ExecutionTrace) -> None:
        if self.max_cost > 0 and trace.total_cost > self.max_cost:
            raise BudgetExceeded(
                f"Cost budget exceeded: ${trace.total_cost:.4f} > ${self.max_cost:.4f}"
            )

    def check_tokens(self, trace: ExecutionTrace) -> None:
        if self.max_tokens > 0 and trace.total_tokens > self.max_tokens:
            raise BudgetExceeded(
                f"Token budget exceeded: {trace.total_tokens} > {self.max_tokens}"
            )

    def check_all(self, trace: ExecutionTrace) -> None:
        self.check_step(trace)
        self.check_cost(trace)
        self.check_tokens(trace)
