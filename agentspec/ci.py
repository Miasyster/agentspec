"""CI/CD integration helpers for agent testing.

Generate CI-friendly output from agent test traces.
Detects CI environment and formats output accordingly.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentspec.trace import ExecutionTrace


class CIReporter:
    """Generate CI-friendly output from agent test traces.

    Detects CI environment (GitHub Actions, GitLab CI, etc.) and formats
    output accordingly. Works outside of CI too — produces plain text.
    """

    @staticmethod
    def detect_ci() -> str | None:
        """Detect CI environment.

        Returns 'github', 'gitlab', 'jenkins', or None if not in CI.
        """
        if os.environ.get("GITHUB_ACTIONS") == "true":
            return "github"
        if os.environ.get("GITLAB_CI") == "true":
            return "gitlab"
        if os.environ.get("JENKINS_URL"):
            return "jenkins"
        return None

    @staticmethod
    def github_summary(traces: list[ExecutionTrace]) -> str:
        """Generate GitHub Actions job summary markdown.

        Produces a markdown table summarizing all traces, suitable for
        writing to $GITHUB_STEP_SUMMARY.
        """
        if not traces:
            return "## Agent Test Results\n\nNo traces recorded.\n"

        total_cost = sum(t.total_cost for t in traces)
        total_tokens = sum(t.total_tokens for t in traces)
        total_steps = sum(t.total_steps for t in traces)

        lines = [
            "## Agent Test Results",
            "",
            f"**{len(traces)} traces** | "
            f"**{total_steps} tool calls** | "
            f"**{total_tokens:,} tokens** | "
            f"**${total_cost:.4f} total cost**",
            "",
            "| # | Prompt | Steps | Tokens | Cost | Duration |",
            "|---|--------|-------|--------|------|----------|",
        ]

        for i, trace in enumerate(traces, 1):
            prompt = trace.prompt[:50] + "..." if len(trace.prompt) > 50 else trace.prompt
            prompt = prompt.replace("|", "\\|")
            duration_s = trace.duration_ms / 1000
            lines.append(
                f"| {i} | {prompt} | {trace.total_steps} | "
                f"{trace.total_tokens:,} | ${trace.total_cost:.4f} | "
                f"{duration_s:.1f}s |"
            )

        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def format_cost_report(traces: list[ExecutionTrace]) -> str:
        """Format cost report suitable for CI output.

        Produces a plain-text summary that works in any CI system or terminal.
        """
        if not traces:
            return "Agent Test Cost Report: No traces recorded."

        total_cost = sum(t.total_cost for t in traces)
        total_tokens = sum(t.total_tokens for t in traces)
        total_steps = sum(t.total_steps for t in traces)

        lines = [
            "=" * 50,
            "Agent Test Cost Report",
            "=" * 50,
            f"  Traces:     {len(traces)}",
            f"  Tool calls: {total_steps}",
            f"  Tokens:     {total_tokens:,}",
            f"  Total cost: ${total_cost:.4f}",
            "-" * 50,
        ]

        for i, trace in enumerate(traces, 1):
            prompt = trace.prompt[:40] + "..." if len(trace.prompt) > 40 else trace.prompt
            lines.append(
                f"  [{i}] {prompt}"
            )
            lines.append(
                f"      steps={trace.total_steps}  "
                f"tokens={trace.total_tokens:,}  "
                f"cost=${trace.total_cost:.4f}"
            )

        lines.append("=" * 50)
        return "\n".join(lines)

    @staticmethod
    def check_budget_gate(
        traces: list[ExecutionTrace],
        max_total_cost: float = 1.0,
    ) -> bool:
        """CI gate: fail if total cost across all traces exceeds budget.

        Returns True if within budget, False if over budget.
        """
        total_cost = sum(t.total_cost for t in traces)
        return total_cost <= max_total_cost

    @staticmethod
    def write_github_summary(traces: list[ExecutionTrace]) -> bool:
        """Write summary to GitHub Actions step summary file.

        Returns True if successfully written, False if not in GitHub Actions
        or if GITHUB_STEP_SUMMARY is not available.
        """
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if not summary_path:
            return False

        markdown = CIReporter.github_summary(traces)
        try:
            with open(summary_path, "a") as f:
                f.write(markdown)
            return True
        except OSError:
            return False
