"""Trace visualization — HTML reports and terminal output."""

from __future__ import annotations

import html
import json
import math
import os
import time
from datetime import datetime, timezone
from typing import Any

from agentspec.trace import ExecutionTrace, LLMCall, ToolCall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_len: int = 200) -> str:
    """Truncate text to *max_len* characters, appending '...' if trimmed."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _format_json(obj: Any, max_len: int = 200) -> str:
    """Pretty-format an object as JSON, truncated."""
    try:
        raw = json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        raw = str(obj)
    return _truncate(raw, max_len)


def _format_ms(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


def _format_cost(cost: float) -> str:
    if cost == 0:
        return "$0"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def _format_tokens(tokens: int) -> str:
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    if tokens >= 1_000:
        return f"{tokens / 1_000:.1f}k"
    return str(tokens)


def _esc(text: str) -> str:
    """HTML-escape a string."""
    return html.escape(str(text))


def _timestamp_display(ts: float) -> str:
    """Format a UNIX timestamp for display."""
    if ts <= 0:
        return "N/A"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ---------------------------------------------------------------------------
# Merged timeline: combine LLM + tool calls sorted by timestamp
# ---------------------------------------------------------------------------

def _build_timeline(trace: ExecutionTrace) -> list[dict[str, Any]]:
    """Build a merged timeline of events from an ExecutionTrace."""
    events: list[dict[str, Any]] = []
    for call in trace.llm_calls:
        events.append({"type": "llm", "call": call, "ts": call.timestamp})
    for call in trace.steps:
        events.append({"type": "tool", "call": call, "ts": call.timestamp})
    events.sort(key=lambda e: e["ts"])
    return events


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------

def _pie_chart_svg(slices: list[tuple[str, float, str]], size: int = 200) -> str:
    """Generate an SVG pie chart.

    *slices* is a list of ``(label, value, colour)`` tuples.
    Returns an SVG string.
    """
    total = sum(v for _, v, _ in slices)
    if total == 0:
        return ""

    cx = cy = size / 2
    r = size / 2 - 4
    paths: list[str] = []
    start_angle = 0.0

    for label, value, colour in slices:
        if value == 0:
            continue
        fraction = value / total
        end_angle = start_angle + fraction * 2 * math.pi
        large_arc = 1 if fraction > 0.5 else 0

        x1 = cx + r * math.cos(start_angle)
        y1 = cy + r * math.sin(start_angle)
        x2 = cx + r * math.cos(end_angle)
        y2 = cy + r * math.sin(end_angle)

        if fraction >= 1.0 - 1e-9:
            # full circle — use two arcs
            mx = cx + r * math.cos(start_angle + math.pi)
            my = cy + r * math.sin(start_angle + math.pi)
            d = (
                f"M {cx},{cy} L {x1},{y1} "
                f"A {r},{r} 0 0 1 {mx},{my} "
                f"A {r},{r} 0 0 1 {x2},{y2} Z"
            )
        else:
            d = (
                f"M {cx},{cy} L {x1},{y1} "
                f"A {r},{r} 0 {large_arc} 1 {x2},{y2} Z"
            )

        paths.append(
            f'<path d="{d}" fill="{colour}">'
            f"<title>{_esc(label)}: {_format_cost(value)}</title>"
            f"</path>"
        )
        start_angle = end_angle

    return (
        f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        + "".join(paths)
        + "</svg>"
    )


def _bar_chart_svg(
    bars: list[tuple[str, float, str]],
    width: int = 400,
    bar_height: int = 28,
    gap: int = 6,
) -> str:
    """Horizontal bar chart as SVG.

    *bars* is ``(label, value, colour)``.
    """
    if not bars:
        return ""
    max_val = max(v for _, v, _ in bars)
    if max_val == 0:
        max_val = 1

    label_width = 120
    chart_width = width - label_width - 50
    height = len(bars) * (bar_height + gap) + gap
    parts: list[str] = []
    for i, (label, value, colour) in enumerate(bars):
        y = gap + i * (bar_height + gap)
        bw = max(value / max_val * chart_width, 2)
        parts.append(
            f'<text x="{label_width - 8}" y="{y + bar_height * 0.7}" '
            f'text-anchor="end" class="bar-label">{_esc(label)}</text>'
        )
        parts.append(
            f'<rect x="{label_width}" y="{y}" width="{bw:.1f}" height="{bar_height}" '
            f'rx="4" fill="{colour}" />'
        )
        parts.append(
            f'<text x="{label_width + bw + 6}" y="{y + bar_height * 0.7}" '
            f'class="bar-value">{int(value)}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        + "".join(parts)
        + "</svg>"
    )


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """\
:root {
  --bg: #ffffff;
  --bg-card: #f8f9fb;
  --bg-code: #f0f2f5;
  --text: #1a1a2e;
  --text-muted: #6b7280;
  --border: #e5e7eb;
  --accent: #3b82f6;
  --accent-light: #dbeafe;
  --error: #ef4444;
  --error-bg: #fef2f2;
  --success: #10b981;
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-mono: "SF Mono", "Fira Code", "Cascadia Code", Menlo, monospace;
}
[data-theme="dark"] {
  --bg: #0f172a;
  --bg-card: #1e293b;
  --bg-code: #1e293b;
  --text: #e2e8f0;
  --text-muted: #94a3b8;
  --border: #334155;
  --accent: #60a5fa;
  --accent-light: #1e3a5f;
  --error: #f87171;
  --error-bg: #2d1b1b;
  --success: #34d399;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0f172a;
    --bg-card: #1e293b;
    --bg-code: #1e293b;
    --text: #e2e8f0;
    --text-muted: #94a3b8;
    --border: #334155;
    --accent: #60a5fa;
    --accent-light: #1e3a5f;
    --error: #f87171;
    --error-bg: #2d1b1b;
    --success: #34d399;
  }
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: var(--font-sans);
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  padding: 2rem;
  max-width: 1100px;
  margin: 0 auto;
}
h1 { font-size: 1.75rem; margin-bottom: 0.25rem; }
h2 {
  font-size: 1.25rem;
  margin: 2rem 0 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid var(--border);
}
.subtitle { color: var(--text-muted); font-size: 0.9rem; margin-bottom: 1.5rem; }
/* Stats grid */
.stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}
.stat-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
  text-align: center;
}
.stat-card .value { font-size: 1.5rem; font-weight: 700; color: var(--accent); }
.stat-card .label { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; }
/* Timeline */
.timeline { position: relative; padding-left: 2rem; }
.timeline::before {
  content: "";
  position: absolute;
  left: 0.75rem;
  top: 0;
  bottom: 0;
  width: 2px;
  background: var(--border);
}
.event {
  position: relative;
  margin-bottom: 1rem;
  padding: 0.75rem 1rem;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
}
.event::before {
  content: "";
  position: absolute;
  left: -1.65rem;
  top: 1rem;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--accent);
  border: 2px solid var(--bg);
}
.event.error { border-color: var(--error); background: var(--error-bg); }
.event.error::before { background: var(--error); }
.event-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 600;
  font-size: 0.95rem;
}
.event-header .icon { font-size: 1.1rem; }
.event-header .badge {
  font-size: 0.75rem;
  padding: 0.1rem 0.5rem;
  border-radius: 9999px;
  font-weight: 500;
}
.badge-llm { background: var(--accent-light); color: var(--accent); }
.badge-tool { background: #ecfdf5; color: #059669; }
[data-theme="dark"] .badge-tool { background: #064e3b; color: #34d399; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) .badge-tool { background: #064e3b; color: #34d399; }
}
.event-meta { font-size: 0.8rem; color: var(--text-muted); margin-top: 0.25rem; }
details { margin-top: 0.5rem; }
summary {
  cursor: pointer;
  font-size: 0.8rem;
  color: var(--accent);
  user-select: none;
}
summary:hover { text-decoration: underline; }
pre {
  background: var(--bg-code);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.5rem 0.75rem;
  font-family: var(--font-mono);
  font-size: 0.8rem;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-word;
  margin-top: 0.25rem;
}
.error-text { color: var(--error); font-weight: 600; }
/* Tables */
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
  margin-top: 0.5rem;
}
th, td {
  padding: 0.5rem 0.75rem;
  border: 1px solid var(--border);
  text-align: left;
}
th {
  background: var(--bg-card);
  font-weight: 600;
  font-size: 0.8rem;
  text-transform: uppercase;
  color: var(--text-muted);
}
/* Charts */
.chart-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  align-items: start;
}
@media (max-width: 700px) { .chart-row { grid-template-columns: 1fr; } }
.chart-container { text-align: center; }
svg .bar-label { font-size: 12px; fill: var(--text); font-family: var(--font-sans); }
svg .bar-value { font-size: 11px; fill: var(--text-muted); font-family: var(--font-mono); }
/* Legend */
.legend { display: flex; flex-wrap: wrap; gap: 0.75rem; margin-top: 0.75rem; justify-content: center; }
.legend-item { display: flex; align-items: center; gap: 0.3rem; font-size: 0.8rem; }
.legend-swatch {
  width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0;
}
/* Theme toggle */
.theme-toggle {
  position: fixed;
  top: 1rem;
  right: 1rem;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.4rem 0.75rem;
  cursor: pointer;
  font-size: 1.1rem;
  color: var(--text);
  z-index: 100;
}
.theme-toggle:hover { background: var(--border); }
/* Multi-trace tabs */
.trace-tabs {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1.5rem;
  flex-wrap: wrap;
}
.trace-tab {
  padding: 0.4rem 1rem;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg-card);
  cursor: pointer;
  font-size: 0.85rem;
}
.trace-tab.active { background: var(--accent); color: #fff; border-color: var(--accent); }
.trace-panel { display: none; }
.trace-panel.active { display: block; }
"""

# ---------------------------------------------------------------------------
# JS (minimal — theme toggle + tab switching)
# ---------------------------------------------------------------------------

_JS = """\
(function() {
  // Theme toggle
  var btn = document.getElementById('theme-toggle');
  if (btn) {
    btn.addEventListener('click', function() {
      var root = document.documentElement;
      var current = root.getAttribute('data-theme');
      if (current === 'dark') {
        root.setAttribute('data-theme', 'light');
        btn.textContent = '\\u263E';
      } else {
        root.setAttribute('data-theme', 'dark');
        btn.textContent = '\\u2600';
      }
    });
  }
  // Tab switching
  document.querySelectorAll('.trace-tab').forEach(function(tab) {
    tab.addEventListener('click', function() {
      var group = tab.getAttribute('data-group');
      document.querySelectorAll('.trace-tab[data-group="' + group + '"]')
        .forEach(function(t) { t.classList.remove('active'); });
      document.querySelectorAll('.trace-panel[data-group="' + group + '"]')
        .forEach(function(p) { p.classList.remove('active'); });
      tab.classList.add('active');
      var target = document.getElementById(tab.getAttribute('data-target'));
      if (target) target.classList.add('active');
    });
  });
})();
"""


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_PALETTE = [
    "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
    "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16",
]


def _colour(index: int) -> str:
    return _PALETTE[index % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Single-trace HTML sections
# ---------------------------------------------------------------------------

def _render_summary_section(trace: ExecutionTrace) -> str:
    """Render the header and quick-stats cards."""
    duration = _format_ms(trace.duration_ms)
    tokens = _format_tokens(trace.total_tokens)
    cost = _format_cost(trace.total_cost)
    error_count = sum(1 for s in trace.steps if s.error is not None)

    stats = [
        ("Steps", str(trace.total_steps)),
        ("LLM Calls", str(len(trace.llm_calls))),
        ("Tokens", tokens),
        ("Cost", cost),
        ("Duration", duration),
    ]
    if error_count:
        stats.append(("Errors", f'<span class="error-text">{error_count}</span>'))

    cards = ""
    for label, value in stats:
        cards += (
            f'<div class="stat-card">'
            f'<div class="value">{value}</div>'
            f'<div class="label">{label}</div>'
            f"</div>"
        )

    prompt_preview = _truncate(trace.prompt, 300)
    return (
        f'<div class="subtitle">Prompt: {_esc(prompt_preview)}</div>'
        f'<div class="stats">{cards}</div>'
    )


def _render_timeline_section(trace: ExecutionTrace) -> str:
    """Render the event timeline."""
    events = _build_timeline(trace)
    if not events:
        return "<p>No events recorded.</p>"

    parts: list[str] = []
    for i, evt in enumerate(events, 1):
        if evt["type"] == "llm":
            call: LLMCall = evt["call"]
            has_error = False
            header = (
                f'<span class="icon">\U0001F916</span>'
                f'<span>{_esc(call.model or "LLM")}</span>'
                f'<span class="badge badge-llm">LLM</span>'
            )
            meta = (
                f'{_format_tokens(call.total_tokens)} tokens &middot; '
                f'{_format_cost(call.cost)} &middot; '
                f'{_format_ms(call.duration_ms)}'
            )
            body = (
                f"<details><summary>Details</summary>"
                f"<pre>Prompt tokens: {call.prompt_tokens}\n"
                f"Completion tokens: {call.completion_tokens}\n"
                f"Total tokens: {call.total_tokens}\n"
                f"Cost: {_format_cost(call.cost)}\n"
                f"Model: {_esc(call.model)}</pre></details>"
            )
        else:
            tc: ToolCall = evt["call"]
            has_error = tc.error is not None
            header = (
                f'<span class="icon">\U0001F527</span>'
                f"<span>{_esc(tc.name)}</span>"
                f'<span class="badge badge-tool">Tool</span>'
            )
            meta = f"{_format_ms(tc.duration_ms)}"
            if has_error:
                meta += f' &middot; <span class="error-text">ERROR</span>'

            body_parts: list[str] = []
            if tc.arguments:
                body_parts.append(
                    f"<details><summary>Arguments</summary>"
                    f"<pre>{_esc(_format_json(tc.arguments))}</pre></details>"
                )
            if tc.error:
                body_parts.append(
                    f'<details open><summary>Error</summary>'
                    f'<pre class="error-text">{_esc(tc.error)}</pre></details>'
                )
            if tc.result is not None and not tc.error:
                body_parts.append(
                    f"<details><summary>Result</summary>"
                    f"<pre>{_esc(_format_json(tc.result))}</pre></details>"
                )
            body = "".join(body_parts)

        error_class = " error" if has_error else ""
        parts.append(
            f'<div class="event{error_class}">'
            f'<div class="event-header">{header}</div>'
            f'<div class="event-meta">#{i} &middot; {meta}</div>'
            f"{body}"
            f"</div>"
        )
    return f'<div class="timeline">{"".join(parts)}</div>'


def _render_cost_section(trace: ExecutionTrace) -> str:
    """Render the cost breakdown: pie chart + table."""
    if not trace.llm_calls:
        return ""

    # Aggregate by model
    model_stats: dict[str, dict[str, float]] = {}
    for call in trace.llm_calls:
        key = call.model or "unknown"
        if key not in model_stats:
            model_stats[key] = {"calls": 0, "tokens": 0, "cost": 0.0}
        model_stats[key]["calls"] += 1
        model_stats[key]["tokens"] += call.total_tokens
        model_stats[key]["cost"] += call.cost

    # Pie chart slices
    slices = [
        (model, stats["cost"], _colour(i))
        for i, (model, stats) in enumerate(model_stats.items())
    ]
    pie = _pie_chart_svg(slices)

    # Legend
    legend_items = "".join(
        f'<div class="legend-item">'
        f'<span class="legend-swatch" style="background:{colour}"></span>'
        f"{_esc(label)}: {_format_cost(value)}"
        f"</div>"
        for label, value, colour in slices
    )
    legend = f'<div class="legend">{legend_items}</div>'

    # Table
    rows = ""
    for model, stats in model_stats.items():
        rows += (
            f"<tr><td>{_esc(model)}</td>"
            f"<td>{int(stats['calls'])}</td>"
            f"<td>{_format_tokens(int(stats['tokens']))}</td>"
            f"<td>{_format_cost(stats['cost'])}</td></tr>"
        )
    table = (
        "<table><thead><tr>"
        "<th>Model</th><th>Calls</th><th>Tokens</th><th>Cost</th>"
        "</tr></thead><tbody>"
        f"{rows}"
        "</tbody></table>"
    )

    return (
        '<h2>Cost Breakdown</h2>'
        '<div class="chart-row">'
        f'<div class="chart-container">{pie}{legend}</div>'
        f"<div>{table}</div>"
        "</div>"
    )


def _render_tool_usage_section(trace: ExecutionTrace) -> str:
    """Render the tool usage: bar chart + table."""
    if not trace.steps:
        return ""

    # Aggregate
    tool_stats: dict[str, dict[str, Any]] = {}
    for call in trace.steps:
        if call.name not in tool_stats:
            tool_stats[call.name] = {"calls": 0, "total_ms": 0.0, "errors": 0}
        tool_stats[call.name]["calls"] += 1
        tool_stats[call.name]["total_ms"] += call.duration_ms
        if call.error:
            tool_stats[call.name]["errors"] += 1

    sorted_tools = sorted(tool_stats.items(), key=lambda x: x[1]["calls"], reverse=True)

    # Bar chart
    bars = [
        (name, float(stats["calls"]), _colour(i))
        for i, (name, stats) in enumerate(sorted_tools)
    ]
    bar_svg = _bar_chart_svg(bars)

    # Table
    rows = ""
    for name, stats in sorted_tools:
        avg_ms = stats["total_ms"] / stats["calls"] if stats["calls"] else 0
        err_cell = (
            f'<span class="error-text">{stats["errors"]}</span>'
            if stats["errors"]
            else "0"
        )
        rows += (
            f"<tr><td>{_esc(name)}</td>"
            f"<td>{stats['calls']}</td>"
            f"<td>{_format_ms(avg_ms)}</td>"
            f"<td>{err_cell}</td></tr>"
        )
    table = (
        "<table><thead><tr>"
        "<th>Tool</th><th>Calls</th><th>Avg Duration</th><th>Errors</th>"
        "</tr></thead><tbody>"
        f"{rows}"
        "</tbody></table>"
    )

    return (
        '<h2>Tool Usage</h2>'
        '<div class="chart-row">'
        f'<div class="chart-container">{bar_svg}</div>'
        f"<div>{table}</div>"
        "</div>"
    )


def _render_errors_section(trace: ExecutionTrace) -> str:
    """Render an errors section (only if errors exist)."""
    error_calls = [s for s in trace.steps if s.error is not None]
    if not error_calls:
        return ""

    items: list[str] = []
    for tc in error_calls:
        items.append(
            f'<div class="event error">'
            f'<div class="event-header">'
            f'<span class="icon">⚠</span>'
            f"<span>{_esc(tc.name)}</span>"
            f"</div>"
            f'<pre class="error-text">{_esc(tc.error or "")}</pre>'
            f"<details><summary>Arguments</summary>"
            f"<pre>{_esc(_format_json(tc.arguments))}</pre></details>"
            f"</div>"
        )
    return f'<h2>Errors</h2>{"".join(items)}'


def _render_output_section(trace: ExecutionTrace) -> str:
    """Render the final output section."""
    if not trace.final_output:
        return ""
    return (
        "<h2>Final Output</h2>"
        f"<pre>{_esc(_truncate(trace.final_output, 2000))}</pre>"
    )


# ---------------------------------------------------------------------------
# Public API — single trace
# ---------------------------------------------------------------------------

def render_trace_html(trace: ExecutionTrace, title: str = "Agent Trace") -> str:
    """Render an ExecutionTrace as a self-contained HTML page."""
    started = _timestamp_display(trace.started_at)

    body = (
        f"<h1>{_esc(title)}</h1>"
        f'<div class="subtitle">{started}</div>'
        + _render_summary_section(trace)
        + "<h2>Timeline</h2>"
        + _render_timeline_section(trace)
        + _render_cost_section(trace)
        + _render_tool_usage_section(trace)
        + _render_errors_section(trace)
        + _render_output_section(trace)
    )

    return (
        "<!DOCTYPE html>"
        '<html lang="en">'
        "<head>"
        '<meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        f"<title>{_esc(title)}</title>"
        f"<style>{_CSS}</style>"
        "</head>"
        "<body>"
        '<button class="theme-toggle" id="theme-toggle">☾</button>'
        f"{body}"
        f"<script>{_JS}</script>"
        "</body></html>"
    )


def save_trace_report(
    trace: ExecutionTrace, path: str, title: str = "Agent Trace"
) -> str:
    """Save trace as HTML file. Returns the file path."""
    html_content = render_trace_html(trace, title=title)
    resolved = os.path.abspath(path)
    os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
    with open(resolved, "w", encoding="utf-8") as f:
        f.write(html_content)
    return resolved


# ---------------------------------------------------------------------------
# Public API — multi-trace comparison
# ---------------------------------------------------------------------------

def render_multi_trace_html(
    traces: list[ExecutionTrace], title: str = "Agent Traces"
) -> str:
    """Render multiple traces for comparison."""
    if not traces:
        body = "<h1>No traces</h1><p>No execution traces to display.</p>"
    elif len(traces) == 1:
        return render_trace_html(traces[0], title=title)
    else:
        # Overview stats table
        overview_rows = ""
        for i, t in enumerate(traces):
            label = _truncate(t.prompt, 60) or f"Trace {i + 1}"
            overview_rows += (
                f"<tr>"
                f"<td>{_esc(label)}</td>"
                f"<td>{t.total_steps}</td>"
                f"<td>{len(t.llm_calls)}</td>"
                f"<td>{_format_tokens(t.total_tokens)}</td>"
                f"<td>{_format_cost(t.total_cost)}</td>"
                f"<td>{_format_ms(t.duration_ms)}</td>"
                f"</tr>"
            )
        overview_table = (
            "<table><thead><tr>"
            "<th>Prompt</th><th>Steps</th><th>LLM Calls</th>"
            "<th>Tokens</th><th>Cost</th><th>Duration</th>"
            "</tr></thead><tbody>"
            f"{overview_rows}"
            "</tbody></table>"
        )

        # Tab buttons
        tabs = ""
        for i, t in enumerate(traces):
            label = _truncate(t.prompt, 30) or f"Trace {i + 1}"
            active = " active" if i == 0 else ""
            tabs += (
                f'<div class="trace-tab{active}" data-group="traces" '
                f'data-target="trace-{i}">{_esc(label)}</div>'
            )

        # Tab panels
        panels = ""
        for i, t in enumerate(traces):
            active = " active" if i == 0 else ""
            panel_content = (
                _render_summary_section(t)
                + "<h2>Timeline</h2>"
                + _render_timeline_section(t)
                + _render_cost_section(t)
                + _render_tool_usage_section(t)
                + _render_errors_section(t)
                + _render_output_section(t)
            )
            panels += (
                f'<div id="trace-{i}" class="trace-panel{active}" data-group="traces">'
                f"{panel_content}</div>"
            )

        body = (
            f"<h1>{_esc(title)}</h1>"
            "<h2>Overview</h2>"
            f"{overview_table}"
            "<h2>Trace Details</h2>"
            f'<div class="trace-tabs">{tabs}</div>'
            f"{panels}"
        )

    return (
        "<!DOCTYPE html>"
        '<html lang="en">'
        "<head>"
        '<meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        f"<title>{_esc(title)}</title>"
        f"<style>{_CSS}</style>"
        "</head>"
        "<body>"
        '<button class="theme-toggle" id="theme-toggle">☾</button>'
        f"{body}"
        f"<script>{_JS}</script>"
        "</body></html>"
    )


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_trace_summary(trace: ExecutionTrace) -> None:
    """Print a compact trace summary to terminal."""
    width = 56

    def _pad(text: str) -> str:
        """Pad a line to fit inside the box."""
        # Account for visible length (strip ANSI if needed)
        visible = text
        padding = max(width - 2 - len(visible), 0)
        return f"│ {visible}{' ' * padding}│"

    top = f"╭{'─' * (width - 2)}╮"
    mid = f"├{'─' * (width - 2)}┤"
    bot = f"╰{'─' * (width - 2)}╯"

    # Header
    lines: list[str] = [top]
    header = f"─── {_truncate(trace.prompt or 'Agent Trace', 40)} ───"
    header_pad = max(width - 2 - len(header), 0)
    lines.append(f"│ {header}{' ' * header_pad}│")

    # Quick stats
    prompt_line = f"Prompt: {_truncate(trace.prompt, width - 12)}"
    lines.append(_pad(prompt_line))

    stats_line = (
        f"Steps: {trace.total_steps} | "
        f"LLM Calls: {len(trace.llm_calls)} | "
        f"Cost: {_format_cost(trace.total_cost)}"
    )
    lines.append(_pad(stats_line))

    tokens_line = (
        f"Tokens: {_format_tokens(trace.total_tokens)} | "
        f"Duration: {_format_ms(trace.duration_ms)}"
    )
    lines.append(_pad(tokens_line))

    # Timeline
    events = _build_timeline(trace)
    if events:
        lines.append(mid)
        for i, evt in enumerate(events, 1):
            if evt["type"] == "llm":
                call: LLMCall = evt["call"]
                line = (
                    f"{i}. \U0001F916 {call.model or 'LLM'} "
                    f"({_format_tokens(call.total_tokens)} tok, "
                    f"{_format_cost(call.cost)}, {_format_ms(call.duration_ms)})"
                )
            else:
                tc: ToolCall = evt["call"]
                args_str = ", ".join(
                    f'{k}={json.dumps(v, ensure_ascii=False, default=str)}'
                    for k, v in list(tc.arguments.items())[:3]
                )
                if len(tc.arguments) > 3:
                    args_str += ", ..."
                args_str = _truncate(args_str, 30)
                result_str = _truncate(str(tc.result or ""), 20)
                suffix = f' !ERR "{_truncate(tc.error, 15)}"' if tc.error else ""
                line = f"{i}. \U0001F527 {tc.name}({args_str})"
                if result_str and not tc.error:
                    line += f" -> {result_str}"
                line += suffix
            lines.append(_pad(_truncate(line, width - 4)))

    # Output
    if trace.final_output:
        lines.append(mid)
        output_line = f"Output: {_truncate(trace.final_output, width - 12)}"
        lines.append(_pad(output_line))

    lines.append(bot)
    print("\n".join(lines))
