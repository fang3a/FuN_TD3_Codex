"""Topology overview plot for the coupled electric-gas model."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from project.data.belgian20_data import COMPRESSOR_CONFIGS, GAS_NODES, GAS_PIPES, GAS_SUPPLIERS
from project.data.ieee33_data import (
    ESS_CONFIGS,
    GFG_CONFIGS,
    IEEE33_LINE_DATA,
    P2G_CONFIGS,
    RENEWABLE_CONFIGS,
)


COMPRESSOR_POWER_BUSES = (7, 13, 30)


def save_coupled_topology_overview(
    output_path: str | Path = "project/outputs/topology/coupled_network_overview.png",
) -> Path:
    """Save a static overview of the electric network, gas network, and couplings."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    power_pos = _power_positions()
    gas_pos = _gas_positions()

    fig, ax = plt.subplots(figsize=(18, 10))
    fig.suptitle("IEEE 33-Bus Power Network and Belgian 20-Node Gas Network Coupling", fontsize=17)
    ax.set_title("Dashed links show energy conversion or electric compressor demand", fontsize=11, pad=10)
    ax.axis("off")

    _draw_edges(ax, power_pos, [(u, v) for u, v, _, _ in IEEE33_LINE_DATA], "#6b7280", 1.8, 0.75)
    _draw_edges(ax, gas_pos, [(p.from_junction, p.to_junction) for p in GAS_PIPES], "#0e7490", 2.0, 0.72)
    _draw_compressors(ax, gas_pos)

    _draw_nodes(ax, power_pos, "#dbeafe", "#1d4ed8", "P", node_size=150)
    _draw_nodes(ax, gas_pos, "#cffafe", "#0e7490", "G", node_size=170)

    _highlight_power_devices(ax, power_pos)
    _highlight_gas_devices(ax, gas_pos)
    _draw_couplings(ax, power_pos, gas_pos, FancyArrowPatch)

    ax.text(7.8, 3.4, "Power side: IEEE 33 radial distribution network", ha="center", fontsize=12, weight="bold")
    ax.text(33.0, 3.4, "Gas side: Belgian 20 high-pressure network", ha="center", fontsize=12, weight="bold")

    legend_items = [
        Line2D([0], [0], color="#6b7280", lw=2, label="Power line"),
        Line2D([0], [0], color="#0e7490", lw=2, label="Gas pipe"),
        Line2D([0], [0], color="#7c3aed", lw=2, ls="-.", label="Gas compressor"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#16a34a", markeredgecolor="#166534", label="ESS"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#facc15", markeredgecolor="#a16207", label="PV / wind"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#f97316", markeredgecolor="#9a3412", label="GFG"),
        Line2D([0], [0], marker="h", color="w", markerfacecolor="#22c55e", markeredgecolor="#166534", label="P2G"),
        Line2D([0], [0], color="#f97316", lw=2, ls="--", label="GFG gas-to-power"),
        Line2D([0], [0], color="#22c55e", lw=2, ls="--", label="P2G power-to-gas"),
        Line2D([0], [0], color="#7c3aed", lw=2, ls=":", label="Compressor electric load"),
    ]
    ax.legend(handles=legend_items, loc="lower center", ncol=5, frameon=True, bbox_to_anchor=(0.5, -0.02))

    ax.set_xlim(-1.0, 44.5)
    ax.set_ylim(-5.1, 4.1)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _power_positions() -> dict[int, tuple[float, float]]:
    pos: dict[int, tuple[float, float]] = {}
    for bus in range(18):
        pos[bus] = (float(bus) * 0.82, 0.0)
    for offset, bus in enumerate(range(18, 22), start=1):
        pos[bus] = (0.82 * (1 + offset), -1.55)
    for offset, bus in enumerate(range(22, 25), start=1):
        pos[bus] = (0.82 * (2 + offset), 1.45)
    for offset, bus in enumerate(range(25, 33), start=1):
        pos[bus] = (0.82 * (5 + offset), -3.05)
    return pos


def _gas_positions() -> dict[int, tuple[float, float]]:
    raw = {
        0: (23.0, 2.2),
        1: (25.2, 3.0),
        2: (27.4, 2.8),
        3: (29.6, 2.1),
        4: (23.5, -0.8),
        5: (25.7, -0.9),
        6: (27.9, -0.7),
        7: (30.1, 0.4),
        8: (31.5, -1.6),
        9: (33.5, -1.7),
        10: (35.4, -1.7),
        11: (36.8, -0.5),
        12: (38.3, 0.7),
        13: (35.5, 1.4),
        14: (37.6, 2.2),
        15: (39.8, 2.1),
        16: (36.4, -3.25),
        17: (39.2, -2.7),
        18: (41.0, -2.6),
        19: (42.7, -2.2),
    }
    return {node: (float(x), float(y)) for node, (x, y) in raw.items()}


def _draw_edges(
    ax,
    positions: Mapping[int, tuple[float, float]],
    edges: Iterable[tuple[int, int]],
    color: str,
    linewidth: float,
    alpha: float,
) -> None:
    for u, v in edges:
        x = [positions[u][0], positions[v][0]]
        y = [positions[u][1], positions[v][1]]
        ax.plot(x, y, color=color, lw=linewidth, alpha=alpha, zorder=1)


def _draw_nodes(
    ax,
    positions: Mapping[int, tuple[float, float]],
    face_color: str,
    edge_color: str,
    prefix: str,
    node_size: int,
) -> None:
    xs = [positions[i][0] for i in sorted(positions)]
    ys = [positions[i][1] for i in sorted(positions)]
    ax.scatter(xs, ys, s=node_size, c=face_color, edgecolors=edge_color, linewidths=1.2, zorder=3)
    for node, (x, y) in positions.items():
        label = str(node) if prefix == "P" else str(node + 1)
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5, color="#111827", zorder=4)


def _draw_compressors(ax, gas_pos: Mapping[int, tuple[float, float]]) -> None:
    for idx, comp in enumerate(COMPRESSOR_CONFIGS):
        x1, y1 = gas_pos[comp.from_junction]
        x2, y2 = gas_pos[comp.to_junction]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=2.2, linestyle="-.", mutation_scale=12),
            zorder=2,
        )
        ax.text(0.5 * (x1 + x2), 0.5 * (y1 + y2) + 0.24, f"C{idx}", color="#6d28d9", fontsize=8, weight="bold")


def _highlight_power_devices(ax, power_pos: Mapping[int, tuple[float, float]]) -> None:
    _scatter_device(ax, [ess.bus for ess in ESS_CONFIGS], power_pos, "s", "#16a34a", "#166534", y_offset=0.33)
    _scatter_device(ax, [ren.bus for ren in RENEWABLE_CONFIGS], power_pos, "D", "#facc15", "#a16207", y_offset=-0.33)
    _scatter_device(ax, [gfg.power_bus for gfg in GFG_CONFIGS], power_pos, "^", "#f97316", "#9a3412", y_offset=0.54)
    _scatter_device(ax, [p2g.power_bus for p2g in P2G_CONFIGS], power_pos, "h", "#22c55e", "#166534", y_offset=-0.56)
    _scatter_device(ax, COMPRESSOR_POWER_BUSES, power_pos, "P", "#a78bfa", "#6d28d9", y_offset=0.35)


def _highlight_gas_devices(ax, gas_pos: Mapping[int, tuple[float, float]]) -> None:
    supplier_nodes = [s.supplier_node for s in GAS_SUPPLIERS]
    demand_nodes = [node.node for node in GAS_NODES if node.demand_mm3_per_day > 0.0]
    _scatter_device(ax, supplier_nodes, gas_pos, "*", "#38bdf8", "#0369a1", size=210, y_offset=0.34)
    _scatter_device(ax, demand_nodes, gas_pos, "v", "#94a3b8", "#475569", size=95, y_offset=-0.30)
    _scatter_device(ax, [g.gas_junction for g in GFG_CONFIGS], gas_pos, "^", "#f97316", "#9a3412", y_offset=0.52)
    _scatter_device(ax, [p.gas_junction for p in P2G_CONFIGS], gas_pos, "h", "#22c55e", "#166534", y_offset=-0.50)


def _scatter_device(
    ax,
    nodes: Iterable[int],
    positions: Mapping[int, tuple[float, float]],
    marker: str,
    face: str,
    edge: str,
    size: int = 125,
    y_offset: float = 0.0,
) -> None:
    xs = []
    ys = []
    for node in nodes:
        if node not in positions:
            continue
        xs.append(positions[node][0])
        ys.append(positions[node][1] + y_offset)
    if xs:
        ax.scatter(xs, ys, s=size, marker=marker, c=face, edgecolors=edge, linewidths=1.0, zorder=5)


def _draw_couplings(ax, power_pos: Mapping[int, tuple[float, float]], gas_pos: Mapping[int, tuple[float, float]], patch_cls) -> None:
    for idx, gfg in enumerate(GFG_CONFIGS):
        _curved_arrow(
            ax,
            gas_pos[gfg.gas_junction],
            power_pos[gfg.power_bus],
            "#f97316",
            patch_cls,
            rad=-0.16,
            label=f"GFG {idx}: gas -> power",
        )

    for idx, p2g in enumerate(P2G_CONFIGS):
        _curved_arrow(
            ax,
            power_pos[p2g.power_bus],
            gas_pos[p2g.gas_junction],
            "#22c55e",
            patch_cls,
            rad=0.18,
            label=f"P2G {idx}: power -> gas",
        )

    for idx, (bus, comp) in enumerate(zip(COMPRESSOR_POWER_BUSES, COMPRESSOR_CONFIGS)):
        start = power_pos[bus]
        end = _midpoint(gas_pos[comp.from_junction], gas_pos[comp.to_junction])
        _curved_arrow(
            ax,
            start,
            end,
            "#7c3aed",
            patch_cls,
            rad=0.06,
            linestyle=":",
            label=f"Comp {idx}: electric load",
        )


def _curved_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    patch_cls,
    rad: float,
    label: str,
    linestyle: str = "--",
) -> None:
    arrow = patch_cls(
        start,
        end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=13,
        lw=1.8,
        linestyle=linestyle,
        color=color,
        alpha=0.82,
        zorder=0,
    )
    ax.add_patch(arrow)
    mx, my = _midpoint(start, end)
    if np.isfinite(mx) and np.isfinite(my):
        ax.text(mx, my + 0.10, label, fontsize=7, color=color, alpha=0.95, ha="center")


def _midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
