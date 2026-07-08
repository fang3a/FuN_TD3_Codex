"""Generate modeling-reasonableness figures for the electric-gas microgrid.

This script is intentionally read-only with respect to the simulator and the
hierarchical TD3 implementation. It imports static model data, runs one mild
baseline episode through ``ElectricGasMultiScaleEnv``, and writes figures plus a
short Markdown report that document topology, coupling, time scales,
constraints, and dimensional consistency.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

try:  # NetworkX is used for graph summaries and layout fallback.
    import networkx as nx
except Exception:  # pragma: no cover - optional fallback for minimal installs.
    nx = None  # type: ignore[assignment]

from electric_gas_microgrid_single import (
    COMPRESSOR_CONFIGS,
    COMPRESSOR_POWER_BUSES,
    CONTROLLED_COMPRESSOR_INDICES,
    DEFAULT_CONFIG,
    ESS_CONFIGS,
    GAS_MODEL_NAME,
    GAS_NODES,
    GAS_PIPES,
    GAS_SUPPLIERS,
    GFG_CONFIGS,
    IEEE33_LINE_DATA,
    IEEE33_LOAD_DATA,
    N_GAS_JUNCTIONS,
    N_POWER_BUSES,
    P2G_CONFIGS,
    RENEWABLE_CONFIGS,
    ElectricGasMultiScaleEnv,
    GasConfig,
    PowerConfig,
    TimeConfig,
    calibration_warning_messages,
    gfg_power_to_gas_mdot_kg_s,
    p2g_power_to_gas_mdot_kg_s,
    profile_at,
    validate_belgian20_topology,
)

logging.getLogger("electric_gas_microgrid_single").setLevel(logging.ERROR)


TIME_CONSTANT_NAMES = (
    "FAST_INTERVAL",
    "SLOW_INTERVAL",
    "MANAGER_INTERVAL",
    "EPISODE_STEPS",
    "GOAL_DIM",
)

TIMESERIES_FIELDS = [
    "step",
    "hour",
    "reward",
    "terminated",
    "truncated",
    "solver_failed",
    "power_converged",
    "gas_converged",
    "gas_solved_this_step",
    "gas_solve_count",
    "gas_state_age",
    "slow_action_applied",
    "action_projection_magnitude",
    "vm_min_pu",
    "vm_max_pu",
    "max_line_loading_percent",
    "gas_pressure_min_bar",
    "gas_pressure_max_bar",
    "gas_pressure_mean_bar",
    "max_pipe_velocity_m_per_s",
    "soc_min",
    "soc_max",
    "gfg_total_p_mw",
    "p2g_total_p_mw",
    "compressor_total_p_mw",
    "gfg_total_mdot_kg_s",
    "p2g_total_mdot_kg_s",
    "renewable_available_mw",
    "renewable_used_mw",
    "renewable_curtailment_mw",
    "source_capacity_violation_max_kg_s",
    "ess_soc_0",
    "ess_soc_1",
    "ess_soc_2",
    "exception",
]


@dataclass
class SimulationResult:
    records: list[dict[str, Any]]
    policy_name: str
    error: str | None = None
    missing_fields: set[str] = field(default_factory=set)


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        arr = np.asarray(value)
        if arr.size == 0:
            return default
        out = float(arr.reshape(-1)[0])
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _safe_array(value: Any) -> np.ndarray:
    try:
        if value is None:
            return np.asarray([], dtype=float)
        return np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.asarray([], dtype=float)


def _finite_values(values: Iterable[Any]) -> np.ndarray:
    arr = np.asarray([_safe_float(v) for v in values], dtype=float)
    return arr[np.isfinite(arr)]


def _finite_range(records: Sequence[Mapping[str, Any]], key: str) -> tuple[float, float]:
    finite = _finite_values(row.get(key, np.nan) for row in records)
    if finite.size == 0:
        return np.nan, np.nan
    return float(np.min(finite)), float(np.max(finite))


def _fmt(value: Any, digits: int = 4, suffix: str = "") -> str:
    val = _safe_float(value)
    if not np.isfinite(val):
        return "not available"
    return f"{val:.{digits}f}{suffix}"


def _fmt_range(records: Sequence[Mapping[str, Any]], key: str, digits: int = 4, suffix: str = "") -> str:
    low, high = _finite_range(records, key)
    if not np.isfinite(low) or not np.isfinite(high):
        return "not available"
    return f"{low:.{digits}f} to {high:.{digits}f}{suffix}"


def _arr(records: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray([_safe_float(row.get(key, np.nan)) for row in records], dtype=float)


def _sum_array(value: Any, default: float = np.nan) -> float:
    arr = _safe_array(value)
    if arr.size == 0:
        return default
    return float(np.nansum(arr))


def _max_array(value: Any, default: float = np.nan) -> float:
    arr = _safe_array(value)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return default
    return float(np.max(finite))


def _safe_eval_constant(node: ast.AST, env: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(f"unknown constant name {node.id}")
        return env[node.id]
    if isinstance(node, ast.BinOp):
        left = _safe_eval_constant(node.left, env)
        right = _safe_eval_constant(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
    raise ValueError(f"unsupported constant expression {ast.dump(node)}")


def load_training_time_constants() -> dict[str, Any]:
    """Read TD3 time-scale constants without ever starting training."""

    constants: dict[str, Any] = {name: None for name in TIME_CONSTANT_NAMES}
    constants["source"] = "fallback"
    constants["error"] = None

    try:
        module = importlib.import_module("hierarchical_td3_electric_gas")
        for name in TIME_CONSTANT_NAMES:
            constants[name] = getattr(module, name)
        constants["source"] = "import"
        return constants
    except Exception as exc:
        constants["error"] = repr(exc)

    path = Path(__file__).with_name("hierarchical_td3_electric_gas.py")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        env: dict[str, Any] = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        env[target.id] = _safe_eval_constant(node.value, env)
                    except Exception:
                        pass
        for name in TIME_CONSTANT_NAMES:
            constants[name] = env.get(name)
        constants["source"] = "ast"
    except Exception as exc:
        constants["error"] = f"{constants['error']}; AST fallback failed: {exc!r}"

    return constants


def collect_static_model_summary(training_constants: Mapping[str, Any]) -> dict[str, Any]:
    """Collect topology, coupling, capacity, action-space, and limit summaries."""

    cfg = DEFAULT_CONFIG
    topology = validate_belgian20_topology()

    power_graph_edges = [(u, v) for u, v, _, _ in IEEE33_LINE_DATA]
    gas_pipe_edges = [(pipe.from_junction, pipe.to_junction) for pipe in GAS_PIPES]
    if nx is not None:
        power_graph = nx.Graph()
        power_graph.add_nodes_from(range(N_POWER_BUSES))
        power_graph.add_edges_from(power_graph_edges)
        gas_graph = nx.Graph()
        gas_graph.add_nodes_from(range(N_GAS_JUNCTIONS))
        gas_graph.add_edges_from(gas_pipe_edges)
        power_is_tree = bool(nx.is_tree(power_graph))
        gas_component_count = int(nx.number_connected_components(gas_graph))
    else:
        power_is_tree = len(power_graph_edges) == N_POWER_BUSES - 1
        gas_component_count = -1

    action_decomposition = {
        "ess": len(ESS_CONFIGS),
        "gfg": len(GFG_CONFIGS),
        "p2g": len(P2G_CONFIGS),
        "controlled_compressor": len(CONTROLLED_COMPRESSOR_INDICES),
        "inverter_reactive_power": len(RENEWABLE_CONFIGS),
        "renewable_curtailment": len(RENEWABLE_CONFIGS),
    }
    slow_action_dim = (
        action_decomposition["ess"]
        + action_decomposition["gfg"]
        + action_decomposition["p2g"]
        + action_decomposition["controlled_compressor"]
    )
    fast_action_dim = (
        action_decomposition["inverter_reactive_power"]
        + action_decomposition["renewable_curtailment"]
    )

    return {
        "config": cfg,
        "training_constants": dict(training_constants),
        "counts": {
            "power_buses": N_POWER_BUSES,
            "power_lines": len(IEEE33_LINE_DATA),
            "power_loads": len(IEEE33_LOAD_DATA),
            "gas_junctions": N_GAS_JUNCTIONS,
            "gas_pipes": len(GAS_PIPES),
            "compressors": len(COMPRESSOR_CONFIGS),
            "gas_suppliers": len(GAS_SUPPLIERS),
            "ess": len(ESS_CONFIGS),
            "renewables": len(RENEWABLE_CONFIGS),
            "gfg": len(GFG_CONFIGS),
            "p2g": len(P2G_CONFIGS),
        },
        "topology": {
            "ok": bool(topology.ok),
            "errors": list(topology.errors),
            "warnings": list(topology.warnings),
            "source_component_nodes": sorted(topology.source_component_nodes),
            "power_is_tree": power_is_tree,
            "gas_component_count": gas_component_count,
        },
        "gas_source_nodes": [supplier.supplier_node for supplier in GAS_SUPPLIERS],
        "gas_load_nodes": [node.node for node in GAS_NODES if node.base_mdot_kg_s > 0.0],
        "action_decomposition": action_decomposition,
        "dimensions": {
            "slow_action_dim": slow_action_dim,
            "fast_action_dim": fast_action_dim,
            "total_action_dim": slow_action_dim + fast_action_dim,
            "goal_dim": training_constants.get("GOAL_DIM"),
        },
        "couplings": {
            "gfg": [
                {
                    "name": dev.name,
                    "power_bus": dev.power_bus,
                    "gas_junction": dev.gas_junction,
                    "max_p_mw": dev.max_p_mw,
                    "efficiency": dev.efficiency,
                }
                for dev in GFG_CONFIGS
            ],
            "p2g": [
                {
                    "name": dev.name,
                    "power_bus": dev.power_bus,
                    "gas_junction": dev.gas_junction,
                    "max_p_mw": dev.max_p_mw,
                    "efficiency": dev.efficiency,
                }
                for dev in P2G_CONFIGS
            ],
            "compressors": [
                {
                    "name": comp.name,
                    "electric_bus": COMPRESSOR_POWER_BUSES[i] if i < len(COMPRESSOR_POWER_BUSES) else None,
                    "from_junction": comp.from_junction,
                    "to_junction": comp.to_junction,
                    "controllable": comp.controllable,
                    "max_power_mw": comp.max_power_mw,
                }
                for i, comp in enumerate(COMPRESSOR_CONFIGS)
            ],
        },
        "constraints": {
            "voltage_min_pu": cfg.power.voltage_min_pu,
            "voltage_max_pu": cfg.power.voltage_max_pu,
            "gas_pressure_min_bar": cfg.gas.network_pressure_min_bar,
            "gas_pressure_max_bar": cfg.gas.network_pressure_max_bar,
            "gas_pressure_target_bar": cfg.gas.network_pressure_target_bar,
            "pipe_velocity_max_m_per_s": cfg.gas.max_pipe_velocity_m_per_s,
            "soc_min": min(dev.soc_min for dev in ESS_CONFIGS),
            "soc_max": max(dev.soc_max for dev in ESS_CONFIGS),
            "source_capacity_kg_s": {
                supplier.supplier_node: supplier.max_mdot_kg_s for supplier in GAS_SUPPLIERS
            },
            "hhv_mj_per_kg": cfg.gas.hhv_mj_per_kg,
        },
        "calibration_messages": list(calibration_warning_messages()),
    }


def _power_positions(seed: int = 42) -> dict[int, tuple[float, float]]:
    manual: dict[int, tuple[float, float]] = {}
    for bus in range(18):
        manual[bus] = (float(bus) * 0.82, 0.0)
    for offset, bus in enumerate(range(18, 22), start=1):
        manual[bus] = (0.82 * (1 + offset), -1.55)
    for offset, bus in enumerate(range(22, 25), start=1):
        manual[bus] = (0.82 * (2 + offset), 1.45)
    for offset, bus in enumerate(range(25, 33), start=1):
        manual[bus] = (0.82 * (5 + offset), -3.05)
    return _layout_or_spring(range(N_POWER_BUSES), [(u, v) for u, v, _, _ in IEEE33_LINE_DATA], manual, seed)


def _gas_positions(seed: int = 42) -> dict[int, tuple[float, float]]:
    manual = {
        0: (22.8, 3.0),
        1: (25.0, 3.0),
        2: (27.2, 3.0),
        3: (30.0, 2.35),
        4: (22.8, 0.45),
        5: (25.0, 0.45),
        6: (27.3, 0.75),
        7: (45.4, 1.75),
        8: (43.4, 1.30),
        9: (41.2, 1.25),
        10: (39.0, 1.35),
        11: (36.9, 2.05),
        12: (34.9, 2.35),
        13: (32.5, 2.55),
        14: (34.7, 3.65),
        15: (37.0, 3.80),
        16: (39.0, -0.15),
        17: (41.2, -0.95),
        18: (43.3, -1.10),
        19: (45.2, -1.10),
    }
    return _layout_or_spring(range(N_GAS_JUNCTIONS), [(p.from_junction, p.to_junction) for p in GAS_PIPES], manual, seed)


def _layout_or_spring(
    nodes: Iterable[int],
    edges: Iterable[tuple[int, int]],
    manual: Mapping[int, tuple[float, float]],
    seed: int,
) -> dict[int, tuple[float, float]]:
    nodes = list(nodes)
    if all(node in manual for node in nodes):
        return {node: (float(manual[node][0]), float(manual[node][1])) for node in nodes}
    if nx is None:
        return {node: (float(i), 0.0) for i, node in enumerate(nodes)}
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    pos = nx.spring_layout(graph, seed=seed)
    return {node: (float(pos[node][0]), float(pos[node][1])) for node in nodes}


def _draw_parallel_edges(
    ax: plt.Axes,
    positions: Mapping[int, tuple[float, float]],
    edges: Sequence[tuple[int, int]],
    color: str,
    linewidth: float,
    alpha: float,
) -> None:
    edge_counts = Counter(tuple(sorted(edge)) for edge in edges)
    edge_seen: Counter[tuple[int, int]] = Counter()
    for u, v in edges:
        key = tuple(sorted((u, v)))
        index = edge_seen[key]
        edge_seen[key] += 1
        total = edge_counts[key]
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        offset = 0.0
        if total > 1 and length > 1e-9:
            offset = (index - 0.5 * (total - 1)) * 0.12
        nx_off, ny_off = ((-dy / length, dx / length) if length > 1e-9 else (0.0, 0.0))
        ax.plot(
            [x1 + nx_off * offset, x2 + nx_off * offset],
            [y1 + ny_off * offset, y2 + ny_off * offset],
            color=color,
            lw=linewidth,
            alpha=alpha,
            zorder=1,
        )


def _scatter_nodes(
    ax: plt.Axes,
    positions: Mapping[int, tuple[float, float]],
    face: str,
    edge: str,
    size: int,
) -> None:
    xs = [positions[i][0] for i in sorted(positions)]
    ys = [positions[i][1] for i in sorted(positions)]
    ax.scatter(xs, ys, s=size, c=face, edgecolors=edge, linewidths=1.1, zorder=3)
    for node, (x, y) in positions.items():
        ax.text(x, y, str(node), ha="center", va="center", fontsize=7.5, color="#111827", zorder=4)


def _scatter_device(
    ax: plt.Axes,
    nodes: Iterable[int],
    positions: Mapping[int, tuple[float, float]],
    marker: str,
    face: str,
    edge: str,
    size: int = 120,
    y_offset: float = 0.0,
) -> None:
    xs: list[float] = []
    ys: list[float] = []
    for node in nodes:
        if node not in positions:
            continue
        xs.append(positions[node][0])
        ys.append(positions[node][1] + y_offset)
    if xs:
        ax.scatter(xs, ys, s=size, marker=marker, c=face, edgecolors=edge, linewidths=1.0, zorder=5)


def _midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))


def _curved_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    rad: float,
    label: str,
    linestyle: str = "--",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=13,
        lw=1.7,
        linestyle=linestyle,
        color=color,
        alpha=0.82,
        zorder=0,
    )
    ax.add_patch(arrow)
    if label:
        mx, my = _midpoint(start, end)
        ax.text(mx, my + 0.12, label, fontsize=7.2, color=color, alpha=0.95, ha="center", zorder=6)


def plot_static_topology_coupling(summary: Mapping[str, Any], path: Path, dpi: int, seed: int) -> Path:
    """Plot IEEE33, Belgian-20-derived gas topology, and cross-energy couplings."""

    power_pos = _power_positions(seed)
    gas_pos = _gas_positions(seed)
    fig, ax = plt.subplots(figsize=(19, 10.5), facecolor="white")
    fig.suptitle("Static topology and cross-energy coupling of the electric-gas microgrid", fontsize=17)
    ax.set_title("Explicit mappings: GFG gas-to-power, P2G power-to-gas, and electric-driven gas compression", fontsize=11)
    ax.axis("off")

    _draw_parallel_edges(ax, power_pos, [(u, v) for u, v, _, _ in IEEE33_LINE_DATA], "#64748b", 1.7, 0.72)
    _draw_parallel_edges(ax, gas_pos, [(p.from_junction, p.to_junction) for p in GAS_PIPES], "#0e7490", 2.0, 0.72)

    for idx, comp in enumerate(COMPRESSOR_CONFIGS):
        start = gas_pos[comp.from_junction]
        end = gas_pos[comp.to_junction]
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=2.1, linestyle="-.", mutation_scale=13),
            zorder=2,
        )
        mx, my = _midpoint(start, end)
        role = "fixed" if not comp.controllable else "controlled"
        ax.text(mx, my + 0.26, f"C{idx} {role}", color="#6d28d9", fontsize=8, weight="bold", ha="center")

    _scatter_nodes(ax, power_pos, "#dbeafe", "#1d4ed8", 145)
    _scatter_nodes(ax, gas_pos, "#cffafe", "#0e7490", 165)

    load_buses = [bus for bus, _, _ in IEEE33_LOAD_DATA]
    gas_load_nodes = summary["gas_load_nodes"]
    _scatter_device(ax, load_buses, power_pos, "o", "#f8fafc", "#475569", 56, y_offset=-0.22)
    _scatter_device(ax, [0], power_pos, "*", "#f59e0b", "#92400e", 240, y_offset=0.45)
    _scatter_device(ax, [ess.bus for ess in ESS_CONFIGS], power_pos, "s", "#16a34a", "#166534", 130, 0.35)
    _scatter_device(ax, [ren.bus for ren in RENEWABLE_CONFIGS], power_pos, "D", "#facc15", "#a16207", 118, -0.37)
    _scatter_device(ax, [gfg.power_bus for gfg in GFG_CONFIGS], power_pos, "^", "#f97316", "#9a3412", 135, 0.58)
    _scatter_device(ax, [p2g.power_bus for p2g in P2G_CONFIGS], power_pos, "h", "#22c55e", "#166534", 135, -0.62)
    _scatter_device(ax, COMPRESSOR_POWER_BUSES, power_pos, "P", "#a78bfa", "#6d28d9", 135, 0.34)

    _scatter_device(ax, gas_load_nodes, gas_pos, "v", "#94a3b8", "#475569", 92, -0.32)
    _scatter_device(ax, [supplier.supplier_node for supplier in GAS_SUPPLIERS], gas_pos, "*", "#38bdf8", "#0369a1", 220, 0.34)
    _scatter_device(ax, [gfg.gas_junction for gfg in GFG_CONFIGS], gas_pos, "^", "#f97316", "#9a3412", 135, 0.54)
    _scatter_device(ax, [p2g.gas_junction for p2g in P2G_CONFIGS], gas_pos, "h", "#22c55e", "#166534", 135, -0.54)

    for idx, gfg in enumerate(GFG_CONFIGS):
        _curved_arrow(
            ax,
            gas_pos[gfg.gas_junction],
            power_pos[gfg.power_bus],
            "#f97316",
            -0.14,
            "",
        )
    for idx, p2g in enumerate(P2G_CONFIGS):
        _curved_arrow(
            ax,
            power_pos[p2g.power_bus],
            gas_pos[p2g.gas_junction],
            "#22c55e",
            0.18,
            "",
        )
    for idx, (bus, comp) in enumerate(zip(COMPRESSOR_POWER_BUSES, COMPRESSOR_CONFIGS)):
        _curved_arrow(
            ax,
            power_pos[bus],
            _midpoint(gas_pos[comp.from_junction], gas_pos[comp.to_junction]),
            "#7c3aed",
            0.08,
            "",
            linestyle=":",
        )

    ax.text(power_pos[0][0] - 0.15, power_pos[0][1] + 0.82, "Slack / Utility Grid", ha="left", fontsize=9, color="#92400e")
    ax.text(gas_pos[0][0] - 0.35, gas_pos[0][1] + 0.85, "Main ext_grid / City gate", ha="left", fontsize=9, color="#0369a1")
    ax.text(gas_pos[7][0] - 2.7, gas_pos[7][1] + 0.75, "Auxiliary source", ha="left", fontsize=9, color="#0369a1")
    ax.text(7.4, 4.08, "Power layer: IEEE 33-bus radial distribution network", ha="center", fontsize=12, weight="bold")
    ax.text(34.2, 4.08, "Gas layer: Belgian-20-derived medium-pressure network", ha="center", fontsize=12, weight="bold")

    legend_items = [
        Line2D([0], [0], color="#64748b", lw=2, label="Power line"),
        Line2D([0], [0], color="#0e7490", lw=2, label="Passive gas pipe"),
        Line2D([0], [0], color="#7c3aed", lw=2, ls="-.", label="Gas compressor arc"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#f8fafc", markeredgecolor="#475569", label="Load bus"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#16a34a", markeredgecolor="#166534", label="ESS"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#facc15", markeredgecolor="#a16207", label="PV / WT"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#f97316", markeredgecolor="#9a3412", label="GFG"),
        Line2D([0], [0], marker="h", color="w", markerfacecolor="#22c55e", markeredgecolor="#166534", label="P2G"),
        Line2D([0], [0], marker="P", color="w", markerfacecolor="#a78bfa", markeredgecolor="#6d28d9", label="Compressor electric load"),
        Line2D([0], [0], color="#f97316", lw=2, ls="--", label="GFG gas-to-power link"),
        Line2D([0], [0], color="#22c55e", lw=2, ls="--", label="P2G power-to-gas link"),
        Line2D([0], [0], color="#7c3aed", lw=2, ls=":", label="Electric compressor link"),
    ]
    ax.legend(handles=legend_items, loc="lower center", ncol=4, frameon=True, bbox_to_anchor=(0.5, -0.02))
    ax.set_xlim(-1.1, 47.4)
    ax.set_ylim(-4.85, 4.65)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path


def plot_device_capacity_and_action_space(summary: Mapping[str, Any], path: Path, dpi: int) -> Path:
    """Plot device ratings, conversion rates, and RL action-space decomposition."""

    cfg = summary["config"]
    hhv = cfg.gas.hhv_mj_per_kg
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.8), facecolor="white")
    fig.suptitle("Device capacities and action-space decomposition", fontsize=16)

    ax = axes[0, 0]
    ess_names = [dev.name for dev in ESS_CONFIGS]
    x = np.arange(len(ESS_CONFIGS))
    width = 0.36
    p_vals = [dev.max_p_mw for dev in ESS_CONFIGS]
    e_vals = [dev.capacity_mwh for dev in ESS_CONFIGS]
    ax.bar(x - width / 2, p_vals, width, color="#2563eb", label="Max power (MW)")
    ax.bar(x + width / 2, e_vals, width, color="#16a34a", label="Energy capacity (MWh)")
    for i, dev in enumerate(ESS_CONFIGS):
        ax.text(i, max(p_vals[i], e_vals[i]) + 0.08, f"SOC [{dev.soc_min:.2f}, {dev.soc_max:.2f}]", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(ess_names, rotation=15)
    ax.set_title("ESS power and energy ratings")
    ax.set_ylabel("MW / MWh")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    ren_names = [dev.name for dev in RENEWABLE_CONFIGS]
    x = np.arange(len(RENEWABLE_CONFIGS))
    cap = [dev.capacity_mw for dev in RENEWABLE_CONFIGS]
    s_rated = [dev.s_rated_mva for dev in RENEWABLE_CONFIGS]
    colors = ["#facc15" if dev.kind == "pv" else "#38bdf8" for dev in RENEWABLE_CONFIGS]
    ax.bar(x - width / 2, cap, width, color=colors, edgecolor="#334155", label="Capacity (MW)")
    ax.bar(x + width / 2, s_rated, width, color="#a78bfa", edgecolor="#6d28d9", label="Inverter rating (MVA)")
    for i, dev in enumerate(RENEWABLE_CONFIGS):
        ax.text(i, max(cap[i], s_rated[i]) + 0.05, f"curt {dev.max_curtailment:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ren_names, rotation=35, ha="right")
    ax.set_title("Renewable and inverter capacities")
    ax.set_ylabel("MW / MVA")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    devices = list(GFG_CONFIGS) + list(P2G_CONFIGS)
    names = [dev.name for dev in devices]
    x = np.arange(len(devices))
    power = [dev.max_p_mw for dev in devices]
    mdot = [
        gfg_power_to_gas_mdot_kg_s(dev.max_p_mw, dev.efficiency, hhv)
        if dev in GFG_CONFIGS
        else p2g_power_to_gas_mdot_kg_s(dev.max_p_mw, dev.efficiency, hhv)
        for dev in devices
    ]
    colors = ["#f97316" if dev in GFG_CONFIGS else "#22c55e" for dev in devices]
    ax.bar(x, power, color=colors, edgecolor="#334155", alpha=0.82, label="Electrical power rating (MW)")
    ax2 = ax.twinx()
    ax2.plot(x, mdot, color="#111827", marker="o", lw=1.8, label="Gas flow at Pmax (kg/s)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("MW")
    ax2.set_ylabel("kg/s")
    ax.set_title("GFG / P2G power ratings and gas-flow conversion")
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(
        0.02,
        0.96,
        f"HHV = {hhv:.1f} MJ/kg from GasConfig.hhv_mj_per_kg",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(facecolor="white", edgecolor="#cbd5e1", boxstyle="round,pad=0.3"),
    )
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")

    ax = axes[1, 1]
    counts = summary["action_decomposition"]
    labels = [
        "ESS",
        "GFG",
        "P2G",
        "Ctrl compressor",
        "Inverter Q",
        "Curtailment",
    ]
    values = [
        counts["ess"],
        counts["gfg"],
        counts["p2g"],
        counts["controlled_compressor"],
        counts["inverter_reactive_power"],
        counts["renewable_curtailment"],
    ]
    colors = ["#16a34a", "#f97316", "#22c55e", "#7c3aed", "#2563eb", "#f59e0b"]
    ax.bar(labels, values, color=colors, edgecolor="#334155")
    for i, val in enumerate(values):
        ax.text(i, val + 0.12, str(val), ha="center", fontsize=10, weight="bold")
    dims = summary["dimensions"]
    ax.text(
        0.02,
        0.94,
        f"slow = {dims['slow_action_dim']}   fast = {dims['fast_action_dim']}   total = {dims['total_action_dim']}",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="#cbd5e1", boxstyle="round,pad=0.35"),
    )
    ax.set_title("Action-space dimension breakdown")
    ax.set_ylabel("Action count")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path


def plot_multiscale_timeline(summary: Mapping[str, Any], path: Path, dpi: int) -> Path:
    """Plot fast, slow, and manager decision clocks over a 24-hour day."""

    cfg = summary["config"]
    training = summary["training_constants"]
    dt_minutes = cfg.time.dt_minutes
    steps_per_day = cfg.time.steps_per_day
    slow_interval = cfg.time.slow_action_interval_steps
    manager_interval = training.get("MANAGER_INTERVAL")
    manager_interval = int(manager_interval) if manager_interval is not None else None

    hours = np.arange(0, steps_per_day + 1) * dt_minutes / 60.0
    slow_hours = np.arange(0, steps_per_day + 1, slow_interval) * dt_minutes / 60.0
    manager_hours = (
        np.arange(0, steps_per_day + 1, manager_interval) * dt_minutes / 60.0
        if manager_interval
        else np.asarray([], dtype=float)
    )

    fig, ax = plt.subplots(figsize=(16.5, 5.6), facecolor="white")
    fig.suptitle("Multi-time-scale scheduling timeline", fontsize=16)

    for hour in range(25):
        ax.axvline(hour, color="#cbd5e1", lw=0.8, zorder=0)
        ax.text(hour, 3.12, f"{hour}h", ha="center", va="bottom", fontsize=8, color="#475569")

    ax.scatter(hours, np.full_like(hours, 2.4), s=8, color="#2563eb", alpha=0.35, label=f"Fast step ({dt_minutes} min)")
    ax.scatter(slow_hours, np.full_like(slow_hours, 1.45), marker="s", s=42, color="#16a34a", label=f"Slow action ({slow_interval} steps)")
    if manager_interval:
        ax.scatter(
            manager_hours,
            np.full_like(manager_hours, 0.5),
            marker="D",
            s=52,
            color="#7c3aed",
            label=f"Manager goal ({manager_interval} steps)",
        )
    else:
        ax.text(12.0, 0.5, "Manager interval not available from training file", ha="center", va="center", color="#7c3aed")

    ax.hlines([2.4, 1.45, 0.5], xmin=0, xmax=24, colors=["#bfdbfe", "#bbf7d0", "#ddd6fe"], lw=8, alpha=0.35)
    ax.text(0.15, 2.68, "Fast Worker: inverter Q and renewable curtailment every fast step", fontsize=10, color="#1d4ed8")
    ax.text(0.15, 1.73, "Slow Worker: ESS, GFG, P2G and controllable compressor every hour", fontsize=10, color="#166534")
    ax.text(0.15, 0.78, "Manager: lower-frequency goal generation; no direct physical actuator", fontsize=10, color="#6d28d9")

    interval_text = [
        f"dt_minutes = {dt_minutes}",
        f"steps_per_day = {steps_per_day}",
        f"slow_action_interval_steps = {slow_interval} ({slow_interval * dt_minutes} min)",
    ]
    if manager_interval:
        interval_text.append(f"manager_interval = {manager_interval} ({manager_interval * dt_minutes} min)")
    if training.get("GOAL_DIM") is not None:
        interval_text.append(f"GOAL_DIM = {training.get('GOAL_DIM')}")
    ax.text(
        23.8,
        2.95,
        "\n".join(interval_text),
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="#cbd5e1", boxstyle="round,pad=0.4"),
    )

    ax.set_xlim(-0.25, 24.25)
    ax.set_ylim(0.0, 3.35)
    ax.set_yticks([0.5, 1.45, 2.4])
    ax.set_yticklabels(["Manager", "Slow Worker", "Fast Worker"])
    ax.set_xlabel("Hour of day")
    ax.grid(True, axis="x", alpha=0.18)
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path


def build_mild_baseline_action(env: ElectricGasMultiScaleEnv, hour: float = 0.0) -> np.ndarray:
    """Build a conservative rule-based normalized action vector."""

    values: list[float] = []

    if 7.0 <= hour < 8.0:
        ess_norm = 0.0
        gfg_fraction = 0.15
        p2g_fraction = 0.20
        curtailment_fraction = 0.05
        q_norm = -0.25
    elif 8.0 <= hour < 16.0:
        ess_norm = 0.20
        gfg_fraction = 0.15
        p2g_fraction = 0.30
        curtailment_fraction = 0.10
        q_norm = -0.25
    elif 16.0 <= hour < 18.0:
        ess_norm = 0.0
        gfg_fraction = 0.25
        p2g_fraction = 0.20
        curtailment_fraction = 0.02
        q_norm = 0.0
    elif 18.0 <= hour < 23.0:
        ess_norm = -0.25
        gfg_fraction = 0.35
        p2g_fraction = 0.10
        curtailment_fraction = 0.0
        q_norm = 0.0
    else:
        ess_norm = 0.0
        gfg_fraction = 0.20
        p2g_fraction = 0.20
        curtailment_fraction = 0.02
        q_norm = 0.0

    values.extend([ess_norm] * env.n_ess)
    values.extend([2.0 * gfg_fraction - 1.0] * env.n_gfg)
    values.extend([2.0 * p2g_fraction - 1.0] * env.n_p2g)
    for comp_idx in CONTROLLED_COMPRESSOR_INDICES:
        comp = COMPRESSOR_CONFIGS[comp_idx]
        desired = comp.initial_pressure_ratio
        span = max(comp.max_pressure_ratio - comp.min_pressure_ratio, 1e-9)
        values.append(float(np.clip(2.0 * (desired - comp.min_pressure_ratio) / span - 1.0, -1.0, 1.0)))
    values.extend([q_norm] * env.n_renew)
    values.extend([2.0 * curtailment_fraction - 1.0] * env.n_renew)

    action = np.asarray(values, dtype=np.float32)
    if action.size != env.action_dim:
        fixed = np.zeros(env.action_dim, dtype=np.float32)
        fixed[: min(action.size, env.action_dim)] = action[: min(action.size, env.action_dim)]
        action = fixed
    if hasattr(env.action_space, "low") and hasattr(env.action_space, "high"):
        action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
    return action


def _extract_renewable_used(info: Mapping[str, Any], env: ElectricGasMultiScaleEnv) -> float:
    projections = info.get("inverter_projection")
    if isinstance(projections, list) and projections:
        return float(np.nansum([_safe_float(item.get("p_actual_mw")) for item in projections if isinstance(item, Mapping)]))
    return float(np.nansum([getattr(item, "p_actual_mw", np.nan) for item in env.last_inverter_projection]))


def _extract_renewable_curtailment_mw(
    info: Mapping[str, Any],
    env: ElectricGasMultiScaleEnv,
    available_by_device: np.ndarray,
) -> float:
    projections = info.get("inverter_projection")
    if isinstance(projections, list) and projections and available_by_device.size:
        fractions = np.asarray([
            _safe_float(item.get("curtailment")) for item in projections if isinstance(item, Mapping)
        ])
        return float(np.nansum(available_by_device[: fractions.size] * fractions))
    fractions = np.asarray([getattr(item, "curtailment", np.nan) for item in env.last_inverter_projection], dtype=float)
    if fractions.size == 0 or available_by_device.size == 0:
        return np.nan
    return float(np.nansum(available_by_device[: fractions.size] * fractions))


def extract_step_record(
    step_id: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Mapping[str, Any],
    env: ElectricGasMultiScaleEnv,
    available_by_device: np.ndarray,
) -> tuple[dict[str, Any], set[str]]:
    """Flatten one environment step into CSV-friendly scalar fields."""

    missing: set[str] = set()
    metrics = info.get("constraint_metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}

    def metric(name: str, default: float = np.nan) -> float:
        if name not in metrics:
            missing.add(name)
        return _safe_float(metrics.get(name, default), default)

    physical = getattr(env, "last_physical_slow", {}) or {}
    solve_result = getattr(env, "last_solve_result", None)
    comp_power = metrics.get("compressor_power_mw")
    if comp_power is None and solve_result is not None:
        dispatches = getattr(solve_result, "compressor_dispatches", [])
        comp_power = [getattr(item, "electric_power_mw", np.nan) for item in dispatches]
    if comp_power is None:
        missing.add("compressor_power_mw")

    ess_soc = _safe_array(info.get("ess_soc", getattr(env, "ess_soc", [])))
    row: dict[str, Any] = {
        "step": float(step_id + 1),
        "hour": float((step_id + 1) * env.config.time.dt_hours),
        "reward": float(reward),
        "terminated": float(bool(terminated)),
        "truncated": float(bool(truncated)),
        "solver_failed": float(bool(info.get("solver_failed", False))),
        "power_converged": float(bool(info.get("power_converged", False))),
        "gas_converged": float(bool(info.get("gas_converged", False))),
        "gas_solved_this_step": float(bool(info.get("gas_solved_this_step", False))),
        "gas_solve_count": _safe_float(info.get("gas_solve_count"), 0.0),
        "gas_state_age": _safe_float(info.get("gas_state_age"), 0.0),
        "slow_action_applied": float(bool(info.get("slow_action_applied", False))),
        "action_projection_magnitude": _safe_float(info.get("action_projection_magnitude"), 0.0),
        "vm_min_pu": metric("vm_min_pu"),
        "vm_max_pu": metric("vm_max_pu"),
        "max_line_loading_percent": metric("max_line_loading_percent"),
        "gas_pressure_min_bar": metric("gas_pressure_min_bar"),
        "gas_pressure_max_bar": metric("gas_pressure_max_bar"),
        "gas_pressure_mean_bar": metric("gas_pressure_mean_bar"),
        "max_pipe_velocity_m_per_s": metric("max_pipe_velocity_m_per_s"),
        "soc_min": metric("soc_min"),
        "soc_max": metric("soc_max"),
        "gfg_total_p_mw": _sum_array(physical.get("gfg_p_mw")),
        "p2g_total_p_mw": _sum_array(physical.get("p2g_p_mw")),
        "compressor_total_p_mw": _sum_array(comp_power),
        "gfg_total_mdot_kg_s": _sum_array(getattr(solve_result, "gfg_mdot_kg_s", None)),
        "p2g_total_mdot_kg_s": _sum_array(getattr(solve_result, "p2g_mdot_kg_s", None)),
        "renewable_available_mw": float(np.nansum(available_by_device)) if available_by_device.size else np.nan,
        "renewable_used_mw": _extract_renewable_used(info, env),
        "renewable_curtailment_mw": _extract_renewable_curtailment_mw(info, env, available_by_device),
        "source_capacity_violation_max_kg_s": _max_array(metrics.get("source_capacity_violation_kg_s")),
        "exception": str(info.get("exception", "")),
    }
    for i in range(len(ESS_CONFIGS)):
        row[f"ess_soc_{i}"] = _safe_float(ess_soc[i] if i < ess_soc.size else np.nan)
    if available_by_device.size == 0:
        missing.add("renewable_available_mw")
    if not np.isfinite(row["renewable_used_mw"]):
        missing.add("renewable_used_mw")
    return row, missing


def run_mild_baseline_episode(steps: int, seed: int) -> SimulationResult:
    """Run one mild baseline episode, recording only environment-returned data."""

    records: list[dict[str, Any]] = []
    missing_fields: set[str] = set()
    try:
        env = ElectricGasMultiScaleEnv(DEFAULT_CONFIG)
        env.reset(seed=seed)
    except Exception as exc:
        return SimulationResult(records, "mild_baseline", error=f"Environment reset failed: {exc!r}")

    max_steps = max(0, min(int(steps), env.config.time.steps_per_day))
    error: str | None = None

    for step_id in range(max_steps):
        available_by_device = np.asarray([], dtype=float)
        try:
            hour = float((env.current_step * env.config.time.dt_hours) % 24.0)
            action = build_mild_baseline_action(env, hour)
            if env.profiles is not None:
                profile = profile_at(env.profiles, env.current_step)
                available_by_device = _safe_array(profile.get("renewable_available_mw"))
            _obs, reward, terminated, truncated, info = env.step(action)
            row, missing = extract_step_record(step_id, reward, terminated, truncated, info, env, available_by_device)
            records.append(row)
            missing_fields.update(missing)
            if terminated or truncated:
                break
        except Exception as exc:
            error = f"Step {step_id + 1} failed outside env.step recovery: {exc!r}"
            break

    return SimulationResult(records, "mild_baseline", error=error, missing_fields=missing_fields)


def _write_csv(records: Sequence[Mapping[str, Any]], path: Path, fields: Sequence[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        field_list = sorted({key for row in records for key in row.keys()})
    else:
        extras = sorted({key for row in records for key in row.keys()} - set(fields))
        field_list = list(fields) + extras
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_list)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key, "") for key in field_list})
    return path


def plot_constraint_envelopes(
    sim: SimulationResult,
    summary: Mapping[str, Any],
    path: Path,
    dpi: int,
) -> Path:
    """Plot 24-hour state envelopes and solver/event status."""

    cfg = summary["config"]
    records = sim.records
    fig, axes = plt.subplots(3, 2, figsize=(16.5, 11.5), sharex=True, facecolor="white")
    fig.suptitle("24-hour constraint envelopes under a mild baseline policy", fontsize=16)

    if not records:
        for ax in axes.ravel():
            ax.axis("off")
        msg = sim.error or "No time-series records were produced."
        axes[1, 0].text(
            0.5,
            0.5,
            f"Simulation unavailable\n{msg}\nStatic figures and report were still generated.",
            ha="center",
            va="center",
            fontsize=12,
            transform=axes[1, 0].transAxes,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(path, dpi=dpi, facecolor="white")
        plt.close(fig)
        return path

    x = _arr(records, "hour")

    ax = axes[0, 0]
    ax.plot(x, _arr(records, "vm_min_pu"), label="V min", color="#2563eb", lw=1.7)
    ax.plot(x, _arr(records, "vm_max_pu"), label="V max", color="#dc2626", lw=1.7)
    ax.axhline(cfg.power.voltage_min_pu, color="#16a34a", ls="--", lw=1.2, label="0.95 pu limit")
    ax.axhline(cfg.power.voltage_max_pu, color="#16a34a", ls="--", lw=1.2, label="1.05 pu limit")
    ax.set_title("Voltage constraint envelope")
    ax.set_ylabel("p.u.")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(x, _arr(records, "gas_pressure_min_bar"), label="Gas p min", color="#0891b2", lw=1.7)
    ax.plot(x, _arr(records, "gas_pressure_max_bar"), label="Gas p max", color="#ea580c", lw=1.7)
    ax.axhline(cfg.gas.network_pressure_min_bar, color="#16a34a", ls="--", lw=1.2, label="2.5 bar limit")
    ax.axhline(cfg.gas.network_pressure_max_bar, color="#16a34a", ls="--", lw=1.2, label="5.0 bar limit")
    ax.axhline(cfg.gas.network_pressure_target_bar, color="#111827", ls=":", lw=1.2, label="4.0 bar target")
    ax.set_title("Gas pressure constraint envelope")
    ax.set_ylabel("bar")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for i, dev in enumerate(ESS_CONFIGS):
        key = f"ess_soc_{i}"
        ax.plot(x, _arr(records, key), lw=1.5, label=dev.name)
    ax.axhspan(min(dev.soc_min for dev in ESS_CONFIGS), max(dev.soc_max for dev in ESS_CONFIGS), color="#16a34a", alpha=0.12)
    ax.axhline(cfg.safety.soc_soft_low, color="#64748b", ls=":", lw=1.1, label="soft range")
    ax.axhline(cfg.safety.soc_soft_high, color="#64748b", ls=":", lw=1.1)
    ax.set_title("ESS SOC trajectories")
    ax.set_ylabel("SOC")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(x, _arr(records, "gfg_total_p_mw"), label="GFG total electric output", color="#f97316", lw=1.6)
    ax.plot(x, _arr(records, "p2g_total_p_mw"), label="P2G total electric load", color="#22c55e", lw=1.6)
    ax.plot(x, _arr(records, "compressor_total_p_mw"), label="Compressor electric load", color="#7c3aed", lw=1.6)
    ax.set_title("Coupling-device active power")
    ax.set_ylabel("MW")
    ax.legend(fontsize=8)

    ax = axes[2, 0]
    ax.plot(x, _arr(records, "renewable_available_mw"), label="Available renewable power", color="#0ea5e9", lw=1.6)
    ax.plot(x, _arr(records, "renewable_used_mw"), label="Used renewable power", color="#16a34a", lw=1.6)
    ax.plot(x, _arr(records, "renewable_curtailment_mw"), label="Curtailed renewable power", color="#f59e0b", lw=1.4)
    ax.set_title("Renewable response")
    ax.set_ylabel("MW")
    ax.set_xlabel("Hour")
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    ax.step(x, _arr(records, "gas_state_age"), where="post", color="#475569", lw=1.4, label="Gas state age")
    ax2 = ax.twinx()
    ax2.plot(x, _arr(records, "power_converged"), color="#2563eb", lw=1.1, alpha=0.72, label="Power flow success")
    ax2.plot(x, _arr(records, "gas_converged"), color="#0891b2", lw=1.1, alpha=0.72, label="Gas flow success")
    ax.scatter(
        x[_arr(records, "gas_solved_this_step") > 0.5],
        np.zeros(np.sum(_arr(records, "gas_solved_this_step") > 0.5)),
        s=18,
        color="#0891b2",
        label="Gas solve",
        zorder=4,
    )
    ax.scatter(
        x[_arr(records, "slow_action_applied") > 0.5],
        np.ones(np.sum(_arr(records, "slow_action_applied") > 0.5)),
        s=24,
        color="#16a34a",
        label="Slow decision",
        zorder=4,
    )
    ax.set_title("Solver and event-trigger status")
    ax.set_ylabel("Gas state age / markers")
    ax2.set_ylabel("Success flag")
    ax.set_xlabel("Hour")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8, loc="upper right")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path


def plot_coupling_flow_consistency(summary: Mapping[str, Any], path: Path, dpi: int) -> Path:
    """Plot GFG and P2G power-to-gas-flow dimensional consistency curves."""

    hhv = summary["config"].gas.hhv_mj_per_kg
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.3), facecolor="white")
    fig.suptitle("Dimensional consistency of gas-power conversion models", fontsize=16)

    ax = axes[0]
    for dev in GFG_CONFIGS:
        power = np.linspace(0.0, dev.max_p_mw, 80)
        mdot = [gfg_power_to_gas_mdot_kg_s(p, dev.efficiency, hhv) for p in power]
        ax.plot(power, mdot, lw=2.0, label=f"{dev.name} eta={dev.efficiency:.2f}")
    ax.set_title("GFG: gas-to-power consumption curve")
    ax.set_xlabel("Electric power output (MW)")
    ax.set_ylabel("Gas consumption (kg/s)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.text(
        0.03,
        0.95,
        "mdot_gas = P_e / (eta * HHV)",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="#cbd5e1", boxstyle="round,pad=0.35"),
    )

    ax = axes[1]
    for dev in P2G_CONFIGS:
        power = np.linspace(0.0, dev.max_p_mw, 80)
        mdot = [p2g_power_to_gas_mdot_kg_s(p, dev.efficiency, hhv) for p in power]
        ax.plot(power, mdot, lw=2.0, label=f"{dev.name} eta={dev.efficiency:.2f}")
    ax.set_title("P2G: power-to-gas production curve")
    ax.set_xlabel("Electric power input (MW)")
    ax.set_ylabel("Gas production (kg/s)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.text(
        0.03,
        0.95,
        "mdot_gas = eta * P_e / HHV",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="#cbd5e1", boxstyle="round,pad=0.35"),
    )

    fig.text(
        0.5,
        0.02,
        f"MW = MJ/s, HHV = {hhv:.1f} MJ/kg, so gas-flow units are kg/s. GFG maps gas to power; P2G maps power to gas.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path


def summarize_simulation(sim: SimulationResult, summary: Mapping[str, Any]) -> dict[str, Any]:
    records = sim.records
    cfg = summary["config"]
    if not records:
        return {
            "steps_recorded": 0,
            "power_success_rate": np.nan,
            "gas_success_rate": np.nan,
            "solver_failure_count": 0,
            "major_violation": None,
        }

    solver_failure_count = int(np.nansum(_arr(records, "solver_failed") > 0.5))
    v_min, _ = _finite_range(records, "vm_min_pu")
    _, v_max = _finite_range(records, "vm_max_pu")
    gas_min, _ = _finite_range(records, "gas_pressure_min_bar")
    _, gas_max = _finite_range(records, "gas_pressure_max_bar")
    soc_min, _ = _finite_range(records, "soc_min")
    _, soc_max = _finite_range(records, "soc_max")
    _, pipe_vel_max = _finite_range(records, "max_pipe_velocity_m_per_s")
    _, source_violation_max = _finite_range(records, "source_capacity_violation_max_kg_s")

    major_violation = False
    tol = 1e-6
    if np.isfinite(v_min) and v_min < cfg.power.voltage_min_pu - tol:
        major_violation = True
    if np.isfinite(v_max) and v_max > cfg.power.voltage_max_pu + tol:
        major_violation = True
    if np.isfinite(gas_min) and gas_min < cfg.gas.network_pressure_min_bar - tol:
        major_violation = True
    if np.isfinite(gas_max) and gas_max > cfg.gas.network_pressure_max_bar + tol:
        major_violation = True
    if np.isfinite(soc_min) and soc_min < min(dev.soc_min for dev in ESS_CONFIGS) - tol:
        major_violation = True
    if np.isfinite(soc_max) and soc_max > max(dev.soc_max for dev in ESS_CONFIGS) + tol:
        major_violation = True
    if np.isfinite(pipe_vel_max) and pipe_vel_max > cfg.gas.max_pipe_velocity_m_per_s + tol:
        major_violation = True
    if np.isfinite(source_violation_max) and source_violation_max > tol:
        major_violation = True

    return {
        "steps_recorded": len(records),
        "power_success_rate": float(np.nanmean(_arr(records, "power_converged"))),
        "gas_success_rate": float(np.nanmean(_arr(records, "gas_converged"))),
        "solver_failure_count": solver_failure_count,
        "v_min": v_min,
        "v_max": v_max,
        "gas_min": gas_min,
        "gas_max": gas_max,
        "soc_min": soc_min,
        "soc_max": soc_max,
        "pipe_velocity_max": pipe_vel_max,
        "source_violation_max": source_violation_max,
        "major_violation": major_violation,
    }


def _mapping_lines(rows: Sequence[Mapping[str, Any]], kind: str) -> list[str]:
    lines: list[str] = []
    for row in rows:
        if kind == "compressor":
            lines.append(
                f"- {row['name']}: electric bus {row['electric_bus']} -> gas compressor arc "
                f"{row['from_junction']}->{row['to_junction']} "
                f"({'controllable' if row['controllable'] else 'fixed'})"
            )
        else:
            lines.append(
                f"- {row['name']}: power bus {row['power_bus']} <-> gas junction {row['gas_junction']}, "
                f"Pmax={row['max_p_mw']:.3f} MW, eta={row['efficiency']:.3f}"
            )
    return lines


def write_reasonableness_report(
    summary: Mapping[str, Any],
    sim: SimulationResult,
    paths: Mapping[str, Path],
    path: Path,
) -> Path:
    """Write a concise Markdown report from static data and simulation records."""

    cfg = summary["config"]
    sim_summary = summarize_simulation(sim, summary)
    counts = summary["counts"]
    topology = summary["topology"]
    constraints = summary["constraints"]
    dims = summary["dimensions"]
    training = summary["training_constants"]
    missing = sorted(sim.missing_fields)
    missing_text = ", ".join(missing) if missing else "none"
    manager_text = (
        f"{training.get('MANAGER_INTERVAL')} fast steps"
        if training.get("MANAGER_INTERVAL") is not None
        else "not available from training file"
    )

    lines: list[str] = [
        "# Electric-Gas Coupled Microgrid Modeling Reasonableness Report",
        "",
        "## 1. Static topology consistency",
        f"- Power network bus count: {counts['power_buses']} buses.",
        f"- Power network branch count: {counts['power_lines']} lines; radial tree check: {topology['power_is_tree']}.",
        f"- Gas network junction count: {counts['gas_junctions']} junctions.",
        f"- Passive gas pipe count: {counts['gas_pipes']} pipes.",
        f"- Compressor count: {counts['compressors']} compressor arcs.",
        f"- Gas source nodes: {summary['gas_source_nodes']}.",
        f"- Gas load nodes: {summary['gas_load_nodes']}.",
        f"- Topology validation result: {'passed' if topology['ok'] else 'failed'}.",
    ]
    if topology["errors"]:
        lines.append(f"- Topology validation errors: {topology['errors']}.")
    if topology["warnings"]:
        lines.append(f"- Topology validation warnings: {topology['warnings']}.")

    lines.extend(
        [
            "",
            "## 2. Cross-energy coupling consistency",
            "- GFG mapping: gas junction -> power bus for gas-to-power operation; listed as bidirectional references below.",
            *_mapping_lines(summary["couplings"]["gfg"], "gfg"),
            "- P2G mapping: power bus -> gas junction for power-to-gas operation.",
            *_mapping_lines(summary["couplings"]["p2g"], "p2g"),
            "- Compressor mapping: electric load bus -> gas compressor arc.",
            *_mapping_lines(summary["couplings"]["compressors"], "compressor"),
            f"- Conversion basis: MW = MJ/s and HHV = {cfg.gas.hhv_mj_per_kg:.3f} MJ/kg.",
            "- GFG formula: mdot_gas = P_e / (eta * HHV), giving kg/s gas consumption.",
            "- P2G formula: mdot_gas = eta * P_e / HHV, giving kg/s gas injection.",
            "",
            "## 3. Multi-time-scale consistency",
            f"- Fast step length: {cfg.time.dt_minutes} minutes.",
            f"- Steps per day: {cfg.time.steps_per_day} steps.",
            f"- Slow action interval: {cfg.time.slow_action_interval_steps} fast steps "
            f"({cfg.time.slow_action_interval_steps * cfg.time.dt_minutes} minutes).",
            f"- Manager interval: {manager_text}.",
            f"- Training constants source: {training.get('source')}.",
            f"- Action-space decomposition: slow={dims['slow_action_dim']}, fast={dims['fast_action_dim']}, total={dims['total_action_dim']}.",
            f"- Goal dimension, if available: {dims.get('goal_dim')}.",
            "",
            "## 4. Constraint modeling consistency",
            f"- Voltage limits: {constraints['voltage_min_pu']:.3f} to {constraints['voltage_max_pu']:.3f} p.u.",
            f"- Gas pressure limits: {constraints['gas_pressure_min_bar']:.3f} to {constraints['gas_pressure_max_bar']:.3f} bar; "
            f"target {constraints['gas_pressure_target_bar']:.3f} bar.",
            f"- Pipe velocity limit: {constraints['pipe_velocity_max_m_per_s']:.3f} m/s.",
            f"- SOC limits across ESS devices: {constraints['soc_min']:.3f} to {constraints['soc_max']:.3f}.",
            f"- Source capacity limits: {constraints['source_capacity_kg_s']} kg/s by source node.",
            "",
            "## 5. 24-hour simulation sanity check",
            f"- Baseline policy: {sim.policy_name}; a deterministic mild rule policy with gentle ESS charge/discharge, "
            "small GFG/P2G fractions, initial compressor ratio, limited inverter Q support, and modest renewable curtailment.",
            f"- Steps recorded: {sim_summary['steps_recorded']}.",
            f"- Power flow success rate: {_fmt(sim_summary.get('power_success_rate'), 3)}.",
            f"- Gas flow success rate: {_fmt(sim_summary.get('gas_success_rate'), 3)}.",
            f"- Solver failure count: {sim_summary.get('solver_failure_count')}.",
            f"- Observed voltage range: {_fmt(sim_summary.get('v_min'))} to {_fmt(sim_summary.get('v_max'))} p.u.",
            f"- Observed gas pressure range: {_fmt(sim_summary.get('gas_min'))} to {_fmt(sim_summary.get('gas_max'))} bar.",
            f"- Observed SOC range: {_fmt(sim_summary.get('soc_min'))} to {_fmt(sim_summary.get('soc_max'))}.",
            f"- Major constraint violation under this baseline: {sim_summary.get('major_violation')}.",
            f"- Metrics not read from current info/env fields: {missing_text}.",
        ]
    )
    if sim.error:
        lines.append(f"- Simulation note: {sim.error}")

    lines.extend(
        [
            "",
            "## 6. Modeling scope and caveats",
            f"- Gas model name: {GAS_MODEL_NAME}.",
            "- The gas model is a Belgian-20-derived medium-pressure research calibration, not an exact engineering reproduction of the original Belgian transmission network.",
            "- The gas model is event-triggered quasi-steady-state, not a full transient gas dynamics model.",
            "- This scope is suitable for a reinforcement-learning scheduling testbed for electric-gas coupled microgrid research.",
            "- Calibration notes from the simulator:",
        ]
    )
    lines.extend([f"  - {msg}" for msg in summary["calibration_messages"]])

    lines.extend(
        [
            "",
            "## Generated artifacts",
        ]
    )
    for label, output_path in paths.items():
        lines.append(f"- {label}: `{output_path}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Visualize electric-gas microgrid modeling reasonableness.")
    parser.add_argument("--output-dir", type=Path, default=Path("model_reasonableness_figures"))
    parser.add_argument("--steps", type=int, default=480)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    training_constants = load_training_time_constants()
    summary = collect_static_model_summary(training_constants)

    paths: dict[str, Path] = {}
    paths["static_topology"] = plot_static_topology_coupling(
        summary, output_dir / "01_static_topology_coupling.png", args.dpi, args.seed
    )
    paths["device_capacity_action_space"] = plot_device_capacity_and_action_space(
        summary, output_dir / "02_device_capacity_and_action_space.png", args.dpi
    )
    paths["multiscale_timeline"] = plot_multiscale_timeline(
        summary, output_dir / "03_multiscale_timeline.png", args.dpi
    )

    sim = run_mild_baseline_episode(args.steps, args.seed)
    paths["timeseries_csv"] = _write_csv(
        sim.records,
        output_dir / "model_reasonableness_timeseries.csv",
        TIMESERIES_FIELDS,
    )
    paths["constraint_envelopes"] = plot_constraint_envelopes(
        sim, summary, output_dir / "04_constraint_envelopes_24h.png", args.dpi
    )
    paths["coupling_flow_consistency"] = plot_coupling_flow_consistency(
        summary, output_dir / "05_coupling_flow_consistency.png", args.dpi
    )
    paths["report_md"] = write_reasonableness_report(
        summary,
        sim,
        paths,
        output_dir / "model_reasonableness_report.md",
    )

    print("Generated model reasonableness artifacts:")
    for label, artifact_path in paths.items():
        print(f"  {label}: {artifact_path.resolve()}")


if __name__ == "__main__":
    main()
