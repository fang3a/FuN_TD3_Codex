"""Static parameter visualization for the electric-gas coupled model.

The functions in this module read the project configuration and data tables,
then generate charts and tabular summaries that explain model scale, device
coverage, coupling completeness, constraints, rewards, and calibration status.
They do not run a simulation or require trained RL checkpoints.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from project.config import DEFAULT_CONFIG, calibration_warning_messages
from project.data.belgian20_data import (
    COMPRESSOR_CONFIGS,
    COUPLING_REFERENCES,
    GAS_NODES,
    GAS_PIPES,
    GAS_SUPPLIERS,
    N_GAS_JUNCTIONS,
    STANDARD_GAS_DENSITY_KG_PER_M3,
    TOTAL_GAS_DEMAND_PROFILE_MM3_PER_H,
    mm3_per_day_to_kg_per_s,
)
from project.data.ieee33_data import (
    ESS_CONFIGS,
    GFG_CONFIGS,
    IEEE33_LINE_DATA,
    IEEE33_LOAD_DATA,
    N_POWER_BUSES,
    P2G_CONFIGS,
    RENEWABLE_CONFIGS,
)
from project.networks.topology_validator import validate_belgian20_topology
from project.visualization.topology import COMPRESSOR_POWER_BUSES, save_coupled_topology_overview


def save_model_parameter_artifacts(
    output_dir: str | Path = "project/outputs/model_parameters",
    include_topology: bool = True,
) -> Dict[str, Path]:
    """Generate model-parameter charts, CSV tables, JSON, and a markdown report."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = build_model_parameter_summary()
    paths: Dict[str, Path] = {}

    paths["parameter_summary_csv"] = out / "model_parameter_summary.csv"
    _write_csv(model["parameter_rows"], paths["parameter_summary_csv"])

    paths["device_parameters_csv"] = out / "device_parameters.csv"
    _write_csv(model["device_rows"], paths["device_parameters_csv"])

    paths["coupling_map_csv"] = out / "coupling_map.csv"
    _write_csv(model["coupling_rows"], paths["coupling_map_csv"])

    paths["completeness_csv"] = out / "modeling_completeness.csv"
    _write_csv(model["completeness_rows"], paths["completeness_csv"])

    paths["calibration_items_csv"] = out / "calibration_items.csv"
    _write_csv(model["calibration_rows"], paths["calibration_items_csv"])

    paths["summary_json"] = out / "model_parameter_summary.json"
    _write_json(model, paths["summary_json"])

    paths["overview_dashboard"] = out / "model_parameter_dashboard.png"
    _plot_overview_dashboard(model, paths["overview_dashboard"])

    paths["capacity_dashboard"] = out / "device_capacity_dashboard.png"
    _plot_capacity_dashboard(model, paths["capacity_dashboard"])

    paths["constraint_reward_dashboard"] = out / "constraint_reward_dashboard.png"
    _plot_constraint_reward_dashboard(model, paths["constraint_reward_dashboard"])

    if include_topology:
        paths["topology_overview"] = save_coupled_topology_overview(out / "coupled_topology_overview.png")

    paths["report_md"] = out / "model_parameter_report.md"
    _write_report(model, paths["report_md"], paths)

    return paths


def build_model_parameter_summary() -> Dict[str, Any]:
    """Collect static numerical model parameters from project data files."""

    cfg = DEFAULT_CONFIG
    topology = validate_belgian20_topology()

    base_load_p_mw = float(sum(row[1] for row in IEEE33_LOAD_DATA))
    base_load_q_mvar = float(sum(row[2] for row in IEEE33_LOAD_DATA))
    renewable_capacity_mw = float(sum(dev.capacity_mw for dev in RENEWABLE_CONFIGS))
    renewable_s_mva = float(sum(dev.s_rated_mva for dev in RENEWABLE_CONFIGS))
    ess_power_mw = float(sum(dev.max_p_mw for dev in ESS_CONFIGS))
    ess_energy_mwh = float(sum(dev.capacity_mwh for dev in ESS_CONFIGS))
    gfg_power_mw = float(sum(dev.max_p_mw for dev in GFG_CONFIGS))
    p2g_power_mw = float(sum(dev.max_p_mw for dev in P2G_CONFIGS))
    compressor_power_mw = float(sum(dev.max_power_mw for dev in COMPRESSOR_CONFIGS))

    gas_node_demand_mm3_day = float(sum(node.demand_mm3_per_day for node in GAS_NODES))
    gas_supplier_capacity_mm3_day = float(sum(src.capacity_mm3_per_day for src in GAS_SUPPLIERS))
    gas_demand_kg_s = float(sum(mm3_per_day_to_kg_per_s(node.demand_mm3_per_day) for node in GAS_NODES))
    gas_capacity_kg_s = float(sum(mm3_per_day_to_kg_per_s(src.capacity_mm3_per_day) for src in GAS_SUPPLIERS))

    slow_action_dim = len(ESS_CONFIGS) + len(GFG_CONFIGS) + len(P2G_CONFIGS) + len(COMPRESSOR_CONFIGS)
    fast_action_dim = 2 * len(RENEWABLE_CONFIGS)
    power_obs_dim = N_POWER_BUSES + len(IEEE33_LINE_DATA) + 7 + 3 * len(RENEWABLE_CONFIGS)
    gas_obs_dim = (
        N_GAS_JUNCTIONS
        + len(GFG_CONFIGS)
        + len(GAS_SUPPLIERS)
        + len(COMPRESSOR_CONFIGS)
        + len(GFG_CONFIGS)
        + len(P2G_CONFIGS)
        + 2
        + len(GAS_PIPES)
    )

    system_counts = {
        "power_buses": N_POWER_BUSES,
        "power_lines": len(IEEE33_LINE_DATA),
        "power_loads": len(IEEE33_LOAD_DATA),
        "renewables": len(RENEWABLE_CONFIGS),
        "ess": len(ESS_CONFIGS),
        "gfg": len(GFG_CONFIGS),
        "p2g": len(P2G_CONFIGS),
        "gas_junctions": N_GAS_JUNCTIONS,
        "gas_pipes": len(GAS_PIPES),
        "gas_suppliers": len(GAS_SUPPLIERS),
        "compressors": len(COMPRESSOR_CONFIGS),
        "coupling_references": len(COUPLING_REFERENCES),
    }
    dimensions = {
        "fast_action_dim": fast_action_dim,
        "slow_action_dim": slow_action_dim,
        "total_action_dim": fast_action_dim + slow_action_dim,
        "fast_observation_dim": power_obs_dim,
        "slow_observation_dim": gas_obs_dim,
        "global_observation_dim": power_obs_dim + gas_obs_dim,
    }
    capacity_totals = {
        "base_load_p_mw": base_load_p_mw,
        "base_load_q_mvar": base_load_q_mvar,
        "renewable_capacity_mw": renewable_capacity_mw,
        "renewable_s_rated_mva": renewable_s_mva,
        "ess_power_mw": ess_power_mw,
        "ess_energy_mwh": ess_energy_mwh,
        "gfg_power_mw": gfg_power_mw,
        "p2g_power_mw": p2g_power_mw,
        "compressor_power_mw": compressor_power_mw,
        "gas_node_demand_mm3_day": gas_node_demand_mm3_day,
        "gas_supplier_capacity_mm3_day": gas_supplier_capacity_mm3_day,
        "gas_node_demand_kg_s": gas_demand_kg_s,
        "gas_supplier_capacity_kg_s": gas_capacity_kg_s,
    }
    rationality_metrics = {
        "power_radial_line_ratio": len(IEEE33_LINE_DATA) / max(N_POWER_BUSES - 1, 1),
        "renewable_to_base_load_ratio": renewable_capacity_mw / max(base_load_p_mw, 1e-9),
        "ess_power_to_base_load_ratio": ess_power_mw / max(base_load_p_mw, 1e-9),
        "ess_energy_duration_h": ess_energy_mwh / max(ess_power_mw, 1e-9),
        "gas_supplier_to_demand_ratio": gas_supplier_capacity_mm3_day / max(gas_node_demand_mm3_day, 1e-9),
        "gas_source_reachable_node_ratio": len(topology.source_component_nodes) / max(N_GAS_JUNCTIONS, 1),
    }

    parameter_rows = _build_parameter_rows(
        cfg,
        system_counts,
        dimensions,
        capacity_totals,
        rationality_metrics,
        topology.ok,
    )
    device_rows = _build_device_rows()
    coupling_rows = _build_coupling_rows()
    completeness_rows = _build_completeness_rows(topology.ok, len(topology.source_component_nodes))
    calibration_rows = _build_calibration_rows()

    return {
        "config": asdict(cfg),
        "system_counts": system_counts,
        "dimensions": dimensions,
        "capacity_totals": capacity_totals,
        "rationality_metrics": rationality_metrics,
        "topology": {
            "ok": topology.ok,
            "errors": topology.errors,
            "warnings": topology.warnings,
            "source_component_nodes": sorted(topology.source_component_nodes),
        },
        "parameter_rows": parameter_rows,
        "device_rows": device_rows,
        "coupling_rows": coupling_rows,
        "completeness_rows": completeness_rows,
        "calibration_rows": calibration_rows,
    }


def _build_parameter_rows(
    cfg: Any,
    system_counts: Mapping[str, float],
    dimensions: Mapping[str, float],
    capacity_totals: Mapping[str, float],
    rationality: Mapping[str, float],
    topology_ok: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = [
        _row("time", "fast_step_minutes", cfg.time.dt_minutes, "min", "3-minute fast control step"),
        _row("time", "steps_per_day", cfg.time.steps_per_day, "step", "480 steps cover 24 hours"),
        _row(
            "time",
            "slow_action_interval",
            cfg.time.slow_action_interval_steps,
            "fast step",
            f"{cfg.time.slow_action_interval_steps * cfg.time.dt_minutes} minutes for slow devices",
        ),
        _row("power", "base_voltage", cfg.power.base_kv, "kV", "IEEE33 distribution voltage level"),
        _row("power", "base_power", cfg.power.base_mva, "MVA", "pandapower base power"),
        _row("power", "voltage_band", f"{cfg.power.voltage_min_pu}-{cfg.power.voltage_max_pu}", "pu", "voltage safety band"),
        _row("gas", "high_pressure_band", f"{cfg.gas.high_pressure_min_bar}-{cfg.gas.high_pressure_max_bar}", "bar", "Belgian20 high-pressure bounds"),
        _row("gas", "prs_outlet_band", f"{cfg.gas.prs_outlet_min_bar}-{cfg.gas.prs_outlet_max_bar}", "bar", "low-pressure outlet bounds for GFG supply"),
        _row("gas", "hhv", cfg.gas.hhv_mj_per_kg, "MJ/kg", "gas-to-power and power-to-gas conversion basis"),
        _row("topology", "gas_topology_valid", int(topology_ok), "bool", "all required gas source/sink/coupling nodes are reachable"),
    ]
    for key, value in system_counts.items():
        rows.append(_row("scale", key, value, "count", "static model element count"))
    for key, value in dimensions.items():
        rows.append(_row("rl_interface", key, value, "dimension", "derived from action/state definitions"))
    for key, value in capacity_totals.items():
        unit = _capacity_unit(key)
        rows.append(_row("capacity", key, value, unit, "aggregated installed or reference capacity"))
    for key, value in rationality.items():
        rows.append(_row("rationality", key, value, "ratio", _rationality_note(key, float(value))))
    return rows


def _build_device_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dev in RENEWABLE_CONFIGS:
        rows.append(
            {
                "device_type": "renewable",
                "name": dev.name,
                "power_bus": dev.bus,
                "gas_junction": "",
                "capacity_mw": dev.capacity_mw,
                "energy_mwh": "",
                "s_rated_mva": dev.s_rated_mva,
                "efficiency": "",
                "control_role": "fast inverter P-curtailment and Q support",
                "calibration_status": "implemented",
            }
        )
    for dev in ESS_CONFIGS:
        rows.append(
            {
                "device_type": "ess",
                "name": dev.name,
                "power_bus": dev.bus,
                "gas_junction": "",
                "capacity_mw": dev.max_p_mw,
                "energy_mwh": dev.capacity_mwh,
                "s_rated_mva": "",
                "efficiency": f"charge={dev.eta_charge}, discharge={dev.eta_discharge}",
                "control_role": "slow charge/discharge with SOC projection",
                "calibration_status": "implemented",
            }
        )
    for dev in GFG_CONFIGS:
        rows.append(
            {
                "device_type": "gfg",
                "name": dev.name,
                "power_bus": dev.power_bus,
                "gas_junction": dev.gas_junction,
                "capacity_mw": dev.max_p_mw,
                "energy_mwh": "",
                "s_rated_mva": "",
                "efficiency": dev.efficiency,
                "control_role": "gas-to-power coupling",
                "calibration_status": "implemented",
            }
        )
    for dev in P2G_CONFIGS:
        rows.append(
            {
                "device_type": "p2g",
                "name": dev.name,
                "power_bus": dev.power_bus,
                "gas_junction": dev.gas_junction,
                "capacity_mw": dev.max_p_mw,
                "energy_mwh": "",
                "s_rated_mva": "",
                "efficiency": dev.efficiency,
                "control_role": "power-to-gas coupling",
                "calibration_status": "implemented",
            }
        )
    for power_bus, dev in zip(COMPRESSOR_POWER_BUSES, COMPRESSOR_CONFIGS):
        rows.append(
            {
                "device_type": "compressor",
                "name": dev.name,
                "power_bus": power_bus,
                "gas_junction": f"{dev.from_junction}->{dev.to_junction}",
                "capacity_mw": dev.max_power_mw,
                "energy_mwh": "",
                "s_rated_mva": "",
                "efficiency": dev.isentropic_efficiency,
                "control_role": "electric load coupled to gas pressure ratio",
                "calibration_status": "needs calibration" if dev.needs_calibration else "implemented",
            }
        )
    return rows


def _build_coupling_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dev in GFG_CONFIGS:
        rows.append(
            {
                "coupling_type": "GFG gas-to-power",
                "device": dev.name,
                "power_bus": dev.power_bus,
                "gas_junction": dev.gas_junction,
                "parameter": f"max_p_mw={dev.max_p_mw}, efficiency={dev.efficiency}",
                "modeling_meaning": "gas sink in gas network and generator in power network",
            }
        )
    for dev in P2G_CONFIGS:
        rows.append(
            {
                "coupling_type": "P2G power-to-gas",
                "device": dev.name,
                "power_bus": dev.power_bus,
                "gas_junction": dev.gas_junction,
                "parameter": f"max_p_mw={dev.max_p_mw}, efficiency={dev.efficiency}",
                "modeling_meaning": "electric load in power network and gas source in gas network",
            }
        )
    for power_bus, dev in zip(COMPRESSOR_POWER_BUSES, COMPRESSOR_CONFIGS):
        rows.append(
            {
                "coupling_type": "electric compressor",
                "device": dev.name,
                "power_bus": power_bus,
                "gas_junction": f"{dev.from_junction}->{dev.to_junction}",
                "parameter": f"ratio={dev.min_pressure_ratio}-{dev.max_pressure_ratio}, max_power_mw={dev.max_power_mw}",
                "modeling_meaning": "compressor pressure action creates electric load",
            }
        )
    for ref in COUPLING_REFERENCES:
        rows.append(
            {
                "coupling_type": "Excel reference",
                "device": f"reference_unit_{ref.unit_id}",
                "power_bus": f"IEEE39 bus {ref.ieee39_bus}",
                "gas_junction": ref.gas_node,
                "parameter": f"{ref.efficient_mm3_per_day_per_mw} MMm3/day/MW",
                "modeling_meaning": "reference retained; IEEE39 bus is not inserted into IEEE33",
            }
        )
    return rows


def _build_completeness_rows(topology_ok: bool, reachable_gas_nodes: int) -> List[Dict[str, Any]]:
    rows = [
        ("IEEE33 topology and load table", "implemented", len(IEEE33_LINE_DATA) == N_POWER_BUSES - 1, "33 buses, 32 radial branches, base load table present"),
        ("Belgian20 gas topology", "implemented", topology_ok, f"{reachable_gas_nodes}/{N_GAS_JUNCTIONS} gas nodes reachable from suppliers"),
        ("Multi-timescale interface", "implemented", True, "fast inverter action every 3 minutes; slow devices every 20 fast steps"),
        ("Bidirectional energy coupling", "implemented", bool(GFG_CONFIGS and P2G_CONFIGS), "GFG and P2G connect both networks"),
        ("Electric compressor coupling", "implemented", bool(COMPRESSOR_CONFIGS), "compressor ratios affect gas network and electric load"),
        ("ESS SOC safety projection", "implemented", True, "SOC and power bounds are enforced before stepping"),
        ("Inverter P/Q safety projection", "implemented", True, "P^2+Q^2<=S^2 and curtailment limits are enforced"),
        ("Stability-focused reward", "implemented", True, "voltage and pressure deviation/violation terms are explicit"),
        ("Gas pipe physical calibration", "needs calibration", False, "Wmn/Kmn are retained as references; length/diameter are equivalent first-version parameters"),
        ("Gas quality and HHV calibration", "needs calibration", False, "HHV and gas density should be checked against gas composition"),
        ("Compressor parameter calibration", "needs calibration", False, "flow, efficiency, and pressure limits are first-version assumptions"),
        ("Supplier cost/ramp calibration", "needs calibration", False, "supplier cost and ramp fields are retained but not the current optimization target"),
    ]
    return [
        {
            "item": item,
            "status": status,
            "passes": int(passes),
            "evidence": evidence,
        }
        for item, status, passes, evidence in rows
    ]


def _build_calibration_rows() -> List[Dict[str, Any]]:
    rows = [
        {
            "parameter_group": "gas_quality",
            "affected_parameters": "HHV, density, compressibility",
            "current_value": f"HHV={DEFAULT_CONFIG.gas.hhv_mj_per_kg} MJ/kg, density={STANDARD_GAS_DENSITY_KG_PER_M3} kg/m3",
            "why_it_matters": "affects GFG/P2G mass-flow conversion and gas demand scale",
            "priority": "high",
        },
        {
            "parameter_group": "gas_pipes",
            "affected_parameters": "length_km, diameter_m, roughness_mm",
            "current_value": "equivalent parameters derived from Wmn ranking",
            "why_it_matters": "affects pressure drops and pipeflow feasibility",
            "priority": "high",
        },
        {
            "parameter_group": "compressors",
            "affected_parameters": "nominal_flow_kg_s, efficiency, pressure ratio, max power",
            "current_value": f"{len(COMPRESSOR_CONFIGS)} compressors with needs_calibration=True",
            "why_it_matters": "affects gas pressure control authority and electric compressor demand",
            "priority": "medium",
        },
        {
            "parameter_group": "suppliers",
            "affected_parameters": "capacity, ramping, marginal cost",
            "current_value": f"{len(GAS_SUPPLIERS)} suppliers with needs_calibration=True",
            "why_it_matters": "affects economic dispatch if gas purchase objective is enabled",
            "priority": "medium",
        },
    ]
    rows.extend(
        {
            "parameter_group": "warning",
            "affected_parameters": "startup calibration note",
            "current_value": message,
            "why_it_matters": "documented model limitation",
            "priority": "medium",
        }
        for message in calibration_warning_messages()
    )
    return rows


def _plot_overview_dashboard(model: Mapping[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = model["system_counts"]
    dims = model["dimensions"]
    rationality = model["rationality_metrics"]
    completeness = model["completeness_rows"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Electric-Gas Coupled Microgrid Model Parameter Overview", fontsize=15)

    ax = axes[0, 0]
    labels = ["Power buses", "Power lines", "Gas nodes", "Gas pipes", "Sources", "Coupled devs"]
    values = [
        counts["power_buses"],
        counts["power_lines"],
        counts["gas_junctions"],
        counts["gas_pipes"],
        counts["gas_suppliers"],
        counts["gfg"] + counts["p2g"] + counts["compressors"],
    ]
    ax.bar(labels, values, color=["#2563eb", "#60a5fa", "#0891b2", "#67e8f9", "#16a34a", "#f97316"])
    ax.set_title("Static network and device scale")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[0, 1]
    dim_labels = ["Fast action", "Slow action", "Total action", "Fast obs", "Slow obs", "Global obs"]
    dim_values = [
        dims["fast_action_dim"],
        dims["slow_action_dim"],
        dims["total_action_dim"],
        dims["fast_observation_dim"],
        dims["slow_observation_dim"],
        dims["global_observation_dim"],
    ]
    ax.barh(dim_labels, dim_values, color=["#ea580c", "#7c3aed", "#111827", "#2563eb", "#0891b2", "#4b5563"])
    ax.set_title("RL interface dimensions")
    ax.grid(True, axis="x", alpha=0.25)

    ax = axes[1, 0]
    ratio_labels = [
        "Radial line ratio",
        "Renew/load",
        "ESS power/load",
        "ESS duration h",
        "Gas cap/demand",
        "Reachable gas nodes",
    ]
    ratio_values = [
        rationality["power_radial_line_ratio"],
        rationality["renewable_to_base_load_ratio"],
        rationality["ess_power_to_base_load_ratio"],
        rationality["ess_energy_duration_h"],
        rationality["gas_supplier_to_demand_ratio"],
        rationality["gas_source_reachable_node_ratio"],
    ]
    ax.bar(ratio_labels, ratio_values, color=["#2563eb", "#facc15", "#16a34a", "#0f766e", "#0891b2", "#7c3aed"])
    ax.axhline(1.0, color="#6b7280", ls="--", lw=1)
    ax.set_title("Rationality indicators")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 1]
    status_counts: Dict[str, int] = {}
    for row in completeness:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    ax.pie(
        list(status_counts.values()),
        labels=list(status_counts.keys()),
        autopct="%1.0f",
        startangle=90,
        colors=["#16a34a", "#f97316", "#94a3b8"],
    )
    ax.set_title("Modeling completeness status")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_capacity_dashboard(model: Mapping[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    totals = model["capacity_totals"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Installed Capacity and Coupling Authority", fontsize=15)

    ax = axes[0, 0]
    labels = ["Base load P", "Renewable P", "ESS P", "GFG P", "P2G P", "Compressor P"]
    values = [
        totals["base_load_p_mw"],
        totals["renewable_capacity_mw"],
        totals["ess_power_mw"],
        totals["gfg_power_mw"],
        totals["p2g_power_mw"],
        totals["compressor_power_mw"],
    ]
    ax.bar(labels, values, color=["#4b5563", "#facc15", "#16a34a", "#f97316", "#22c55e", "#7c3aed"])
    ax.set_ylabel("MW")
    ax.set_title("Electric-side controllable capacity")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[0, 1]
    names = [dev.name for dev in RENEWABLE_CONFIGS]
    p_values = [dev.capacity_mw for dev in RENEWABLE_CONFIGS]
    s_values = [dev.s_rated_mva for dev in RENEWABLE_CONFIGS]
    x = np.arange(len(names))
    width = 0.38
    ax.bar(x - width / 2, p_values, width, label="P capacity MW", color="#facc15")
    ax.bar(x + width / 2, s_values, width, label="S rated MVA", color="#2563eb")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_title("Renewable inverter active/reactive sizing")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 0]
    ess_names = [dev.name for dev in ESS_CONFIGS]
    ess_p = [dev.max_p_mw for dev in ESS_CONFIGS]
    ess_e = [dev.capacity_mwh for dev in ESS_CONFIGS]
    duration = [e / max(p, 1e-9) for p, e in zip(ess_p, ess_e)]
    x = np.arange(len(ess_names))
    ax.bar(x - width / 2, ess_p, width, label="Power MW", color="#16a34a")
    ax.bar(x + width / 2, ess_e, width, label="Energy MWh", color="#0f766e")
    for i, d in enumerate(duration):
        ax.text(i, max(ess_e[i], ess_p[i]) + 0.15, f"{d:.1f}h", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(ess_names)
    ax.set_title("ESS power, energy, and duration")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 1]
    gas_labels = ["Node demand", "Supplier capacity"]
    gas_values = [totals["gas_node_demand_mm3_day"], totals["gas_supplier_capacity_mm3_day"]]
    ax.bar(gas_labels, gas_values, color=["#0891b2", "#16a34a"])
    ax.set_ylabel("million m3/day")
    ax.set_title("Gas demand and source capacity")
    ax.grid(True, axis="y", alpha=0.25)
    profile = np.asarray(TOTAL_GAS_DEMAND_PROFILE_MM3_PER_H, dtype=float)
    ax2 = ax.twinx()
    ax2.plot([0, 1], [float(np.min(profile)), float(np.max(profile))], color="#dc2626", marker="o", lw=1.5)
    ax2.set_ylabel("profile range MMm3/h")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_constraint_reward_dashboard(model: Mapping[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = DEFAULT_CONFIG
    reward = asdict(cfg.reward)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Constraints, Safety Bounds, and Reward Weights", fontsize=15)

    ax = axes[0, 0]
    bands = [
        ("Voltage", cfg.power.voltage_min_pu, cfg.power.voltage_target_pu, cfg.power.voltage_max_pu, "pu"),
        ("HP gas", cfg.gas.high_pressure_min_bar, cfg.gas.high_pressure_target_bar, cfg.gas.high_pressure_max_bar, "bar"),
        ("PRS", cfg.gas.prs_outlet_min_bar, cfg.gas.prs_outlet_pressure_bar, cfg.gas.prs_outlet_max_bar, "bar"),
        ("SOC soft", cfg.safety.soc_soft_low, 0.5, cfg.safety.soc_soft_high, "pu"),
    ]
    y = np.arange(len(bands))
    for i, (label, low, target, high, unit) in enumerate(bands):
        target_pos = (target - low) / max(high - low, 1e-9)
        ax.plot([0.0, 1.0], [i, i], color="#94a3b8", lw=9, solid_capstyle="round")
        ax.scatter([target_pos], [i], color="#dc2626", s=65, zorder=3)
        ax.text(1.0, i + 0.16, f"{low:g}-{high:g} {unit}", fontsize=9, ha="right")
    ax.set_yticks(y)
    ax.set_yticklabels([b[0] for b in bands])
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("normalized position inside each band")
    ax.set_title("Operational bands and targets")
    ax.grid(True, axis="x", alpha=0.2)

    ax = axes[0, 1]
    comp_names = [dev.name.replace("COMP_", "") for dev in COMPRESSOR_CONFIGS]
    ratio_min = [dev.min_pressure_ratio for dev in COMPRESSOR_CONFIGS]
    ratio_init = [dev.initial_pressure_ratio for dev in COMPRESSOR_CONFIGS]
    ratio_max = [dev.max_pressure_ratio for dev in COMPRESSOR_CONFIGS]
    x = np.arange(len(comp_names))
    ax.vlines(x, ratio_min, ratio_max, color="#7c3aed", lw=7, alpha=0.7)
    ax.scatter(x, ratio_init, color="#111827", s=50, label="initial")
    ax.set_xticks(x)
    ax.set_xticklabels(comp_names, rotation=20, ha="right")
    ax.set_ylabel("pressure ratio")
    ax.set_title("Compressor pressure-ratio authority")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 0]
    ordered = sorted(reward.items(), key=lambda item: float(item[1]), reverse=True)
    labels = [key for key, _ in ordered]
    values = [float(value) for _, value in ordered]
    colors = ["#dc2626" if "violation" in label or "failure" in label else "#2563eb" for label in labels]
    ax.barh(labels, values, color=colors)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.invert_yaxis()
    ax.set_title("Reward/cost weights")
    ax.grid(True, axis="x", alpha=0.25)

    ax = axes[1, 1]
    complete = model["completeness_rows"]
    labels = [row["item"] for row in complete]
    values = [1 if row["status"] == "implemented" else 0.5 for row in complete]
    colors = ["#16a34a" if row["status"] == "implemented" else "#f97316" for row in complete]
    ax.barh(labels, values, color=colors)
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.set_title("Implemented vs calibration-needed items")
    ax.grid(True, axis="x", alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _write_report(model: Mapping[str, Any], path: Path, artifacts: Mapping[str, Path]) -> None:
    counts = model["system_counts"]
    dims = model["dimensions"]
    totals = model["capacity_totals"]
    rationality = model["rationality_metrics"]
    topology = model["topology"]

    implemented = sum(1 for row in model["completeness_rows"] if row["status"] == "implemented")
    needs_calibration = sum(1 for row in model["completeness_rows"] if row["status"] == "needs calibration")

    lines = [
        "# 电-气耦合微电网建模参数可视化报告",
        "",
        "## 1. 建模规模",
        "",
        f"- 电网侧采用 IEEE 33 节点辐射型配电网：{counts['power_buses']} 个母线、{counts['power_lines']} 条线路、{counts['power_loads']} 个基础负荷。",
        f"- 气网侧采用 Belgian 20 高压气网：{counts['gas_junctions']} 个节点、{counts['gas_pipes']} 条管道、{counts['gas_suppliers']} 个气源、{counts['compressors']} 台压缩机。",
        f"- 耦合设备包含 {counts['gfg']} 台 GFG、{counts['p2g']} 台 P2G 和 {counts['compressors']} 台电驱压缩机，能够覆盖 gas-to-power、power-to-gas 和电驱压缩三类耦合路径。",
        "",
        "## 2. 强化学习接口完整性",
        "",
        f"- 快速动作维度为 {dims['fast_action_dim']}，对应每台新能源逆变器的无功控制和有功削减率。",
        f"- 慢速动作维度为 {dims['slow_action_dim']}，对应 ESS、GFG、P2G 和压缩机压力比。",
        f"- 全局观测维度为 {dims['global_observation_dim']}，由电网状态、气网状态、设备状态和时间特征组成。",
        "",
        "## 3. 容量与物理合理性指标",
        "",
        f"- 基础有功负荷合计 {totals['base_load_p_mw']:.3f} MW；新能源装机 {totals['renewable_capacity_mw']:.3f} MW，新能源/基础负荷比约 {rationality['renewable_to_base_load_ratio']:.2f}。",
        f"- ESS 合计功率 {totals['ess_power_mw']:.3f} MW、能量 {totals['ess_energy_mwh']:.3f} MWh，等效时长约 {rationality['ess_energy_duration_h']:.2f} h。",
        f"- GFG 最大出力 {totals['gfg_power_mw']:.3f} MW，P2G 最大用电 {totals['p2g_power_mw']:.3f} MW，二者共同构成双向能量耦合。",
        f"- 气网节点需求合计 {totals['gas_node_demand_mm3_day']:.3f} million m3/day，供应商容量合计 {totals['gas_supplier_capacity_mm3_day']:.3f} million m3/day，气源容量/节点需求比约 {rationality['gas_supplier_to_demand_ratio']:.2f}。",
        f"- 电网线路数/辐射网理论线路数比为 {rationality['power_radial_line_ratio']:.2f}；气网从气源可达节点比例为 {rationality['gas_source_reachable_node_ratio']:.2f}。",
        "",
        "## 4. 建模完成度判断",
        "",
        f"- 已实现条目：{implemented} 项；待校准条目：{needs_calibration} 项。",
        f"- 气网拓扑校验结果：{'通过' if topology['ok'] else '未通过'}。",
        "- 当前模型完成了拓扑、设备、动作边界、安全投影、耦合换算、稳定性奖励和时序接口的主要闭环。",
        "- 需要谨慎说明的是：Belgian 20 管道长度/管径/粗糙度、气体 HHV/密度、压缩机流量和供应商经济参数仍属于第一版待校准参数，适合用于算法验证和建模流程展示，不宜直接声称为工程实测参数。",
        "",
        "## 5. 输出文件",
        "",
    ]
    for key, artifact_path in artifacts.items():
        lines.append(f"- `{key}`: `{artifact_path}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(data), f, indent=2, ensure_ascii=False)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _row(category: str, metric: str, value: Any, unit: str, interpretation: str) -> Dict[str, Any]:
    return {
        "category": category,
        "metric": metric,
        "value": value,
        "unit": unit,
        "interpretation": interpretation,
    }


def _rationality_note(key: str, value: float) -> str:
    notes = {
        "power_radial_line_ratio": "1.0 indicates a radial IEEE33 topology",
        "renewable_to_base_load_ratio": "distributed renewable penetration relative to base load",
        "ess_power_to_base_load_ratio": "storage active-power authority relative to base load",
        "ess_energy_duration_h": "aggregate ESS energy divided by aggregate ESS power",
        "gas_supplier_to_demand_ratio": "gas source capacity margin against static node demand",
        "gas_source_reachable_node_ratio": "fraction of gas nodes reachable from suppliers",
    }
    return f"{notes.get(key, 'derived model sanity indicator')}; value={value:.3g}"


def _capacity_unit(key: str) -> str:
    if key.endswith("_mw"):
        return "MW"
    if key.endswith("_mwh"):
        return "MWh"
    if key.endswith("_mvar"):
        return "MVAr"
    if key.endswith("_mva"):
        return "MVA"
    if "mm3_day" in key:
        return "MMm3/day"
    if key.endswith("_kg_s"):
        return "kg/s"
    return ""
