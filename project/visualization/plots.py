"""随机策略和训练回放的可视化输出。

可视化不是环境动力学的一部分，只用于把每一步记录的 metrics/components
变成 CSV 和 PNG，帮助人快速判断电压、气压、SOC 和奖励成本是否异常。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def save_episode_artifacts(
    records: Sequence[Mapping[str, Any]],
    output_dir: str | Path = "project/outputs/random_policy",
    prefix: str = "random_policy",
) -> dict[str, Path]:
    """保存 CSV 与图表。

    图表面向调试：突出越限、求解触发、SOC 边界和购能趋势。records 中的
    字段由 run_random_policy.py 生成，后续训练脚本也可复用这个接口。
    """

    if not records:
        raise ValueError("records 为空，无法生成可视化")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{prefix}_timeseries.csv"
    dashboard_path = out / f"{prefix}_dashboard.png"
    cost_path = out / f"{prefix}_costs.png"

    _write_csv(records, csv_path)
    _plot_dashboard(records, dashboard_path)
    _plot_costs(records, cost_path)
    return {"csv": csv_path, "dashboard": dashboard_path, "costs": cost_path}


def _write_csv(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    """把每步记录写成宽表 CSV；列名来自 records 中出现过的所有 key。"""

    fieldnames = sorted({key for row in records for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _plot_dashboard(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    """画一张环境运行状态总览图。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = _arr(records, "hour")
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True)
    fig.suptitle("Electric-Gas Coupled Microgrid Random Policy Overview", fontsize=15)

    ax = axes[0, 0]
    ax.plot(x, _arr(records, "vm_min_pu"), color="#2563eb", lw=1.6, label="V min")
    ax.plot(x, _arr(records, "vm_max_pu"), color="#dc2626", lw=1.6, label="V max")
    ax.axhspan(0.95, 1.05, color="#16a34a", alpha=0.10, label="0.95-1.05 pu")
    ax.set_ylabel("Voltage [pu]")
    ax.set_title("Bus Voltage Envelope")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[0, 1]
    ax.plot(x, _arr(records, "max_line_loading_percent"), color="#7c3aed", lw=1.6)
    ax.axhline(100.0, color="#dc2626", ls="--", lw=1.0)
    ax.set_ylabel("Loading [%]")
    ax.set_title("Maximum Line Loading")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    ax.plot(x, _arr(records, "high_pressure_min_bar"), color="#0891b2", lw=1.5, label="HP min")
    ax.plot(x, _arr(records, "high_pressure_max_bar"), color="#ea580c", lw=1.5, label="HP max")
    ax.axhspan(30.0, 70.0, color="#16a34a", alpha=0.10, label="30-70 bar")
    ax.set_ylabel("Pressure [bar]")
    ax.set_title("High-Pressure Gas Network")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1, 1]
    ax.plot(x, _arr(records, "prs_pressure_min_bar"), color="#0f766e", lw=1.5, label="PRS min")
    ax.plot(x, _arr(records, "prs_pressure_max_bar"), color="#be123c", lw=1.5, label="PRS max")
    ax.axhspan(1.35, 1.65, color="#16a34a", alpha=0.10, label="1.35-1.65 bar")
    ax.set_ylabel("Pressure [bar]")
    ax.set_title("PRS Outlet Pressure")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[2, 0]
    ax.plot(x, _arr(records, "soc_min"), color="#2563eb", lw=1.5, label="SOC min")
    ax.plot(x, _arr(records, "soc_max"), color="#dc2626", lw=1.5, label="SOC max")
    ax.axhspan(0.10, 0.95, color="#16a34a", alpha=0.10, label="hard bounds")
    ax.set_ylabel("SOC")
    ax.set_xlabel("Hour")
    ax.set_title("ESS SOC Envelope")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[2, 1]
    ax.step(x, _arr(records, "gas_state_age"), where="post", color="#4b5563", lw=1.4, label="gas_state_age")
    gas_solved = _arr(records, "gas_solved_this_step")
    slow_applied = _arr(records, "slow_action_applied")
    ax.scatter(x[gas_solved > 0.5], np.zeros(np.sum(gas_solved > 0.5)), color="#0891b2", s=14, label="pipeflow")
    ax.scatter(x[slow_applied > 0.5], np.ones(np.sum(slow_applied > 0.5)), color="#ea580c", s=18, label="slow action")
    ax.set_ylabel("Age / event")
    ax.set_xlabel("Hour")
    ax.set_title("Event-Driven Gas Solves")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_costs(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    """画奖励和成本分解，帮助定位策略主要被哪类惩罚影响。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = _arr(records, "hour")
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle("Reward and Stability Diagnostics", fontsize=15)

    ax = axes[0, 0]
    ax.plot(x, _arr(records, "reward"), color="#111827", lw=1.3)
    ax.set_title("Step Reward")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    ax.plot(x, _arr(records, "voltage_rms_deviation_pu") * 100.0, color="#2563eb", lw=1.5, label="Voltage RMS x100")
    ax.set_title("Voltage RMS Deviation")
    ax.set_ylabel("pu x100")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1, 0]
    ax.plot(x, _arr(records, "high_pressure_rms_deviation_bar"), color="#0891b2", lw=1.5, label="HP RMS")
    ax.plot(x, _arr(records, "prs_pressure_rms_deviation_bar"), color="#0f766e", lw=1.5, label="PRS RMS")
    ax.set_title("Gas Pressure RMS Deviation")
    ax.set_ylabel("bar")
    ax.set_xlabel("Hour")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1, 1]
    names = [
        "voltage_violation_cost",
        "voltage_deviation_cost",
        "high_pressure_violation_cost",
        "high_pressure_deviation_cost",
        "prs_pressure_violation_cost",
        "prs_pressure_deviation_cost",
        "line_overload_cost",
    ]
    values = [float(np.nansum(_arr(records, name))) for name in names]
    labels = [
        "V viol",
        "V dev",
        "HP viol",
        "HP dev",
        "PRS viol",
        "PRS dev",
        "Line",
    ]
    ax.bar(labels, values, color=["#1d4ed8", "#60a5fa", "#0891b2", "#67e8f9", "#15803d", "#86efac", "#7c3aed"])
    ax.set_title("Penalty / Cost Breakdown")
    ax.set_ylabel("Weighted cost")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _arr(records: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, np.nan)) for row in records], dtype=float)
