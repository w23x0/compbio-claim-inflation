"""Tiny dependency-free SVG score-curve renderer for stage reports."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def write_curve_svg(path: Path, curves: list[dict[str, Any]]) -> None:
    if not curves:
        return
    width, height = 640, 260
    left, right, top, bottom = 55, 20, 20, 45
    plot_w, plot_h = width - left - right, height - top - bottom
    max_y = max(1e-6, max(max(c["best"], c["mean"]) for c in curves))

    def x(i: int) -> float:
        return left + (plot_w * i / max(1, len(curves) - 1))

    def y(v: float) -> float:
        return top + plot_h * (1 - v / max_y)

    best = " ".join(f"{x(i):.1f},{y(c['best']):.1f}" for i, c in enumerate(curves))
    mean = " ".join(f"{x(i):.1f},{y(c['mean']):.1f}" for i, c in enumerate(curves))
    labels = "".join(
        f'<text x="{x(i):.1f}" y="{height - 18}" font-size="11" text-anchor="middle">R{c["round"]}</text>'
        for i, c in enumerate(curves)
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">\n'
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" '
        f'stroke="#777" stroke-width="1"/>\n'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#777" stroke-width="1"/>\n'
        f'<text x="{left - 8}" y="{top + 4}" font-size="10" text-anchor="end">{max_y:.2f}</text>\n'
        f'<text x="{left - 8}" y="{top + plot_h + 4}" font-size="10" text-anchor="end">0</text>\n'
        f'<polyline points="{best}" fill="none" stroke="#1f77b4" stroke-width="2"/>\n'
        f'<polyline points="{mean}" fill="none" stroke="#d62728" stroke-width="2" '
        f'stroke-dasharray="5 3"/>\n'
        f'{labels}\n'
        f'<text x="{left + plot_w - 5}" y="{top + 12}" font-size="11" fill="#1f77b4" '
        f'text-anchor="end">best</text>\n'
        f'<text x="{left + plot_w - 5}" y="{top + 27}" font-size="11" fill="#d62728" '
        f'text-anchor="end">mean</text>\n'
        '</svg>\n'
    )
    path.write_text(svg, encoding="utf-8")
