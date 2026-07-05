"""运行一天随机安全策略，输出摘要并保存可视化结果。

这是环境的 smoke test，而不是训练算法。随机策略的作用是确认：
动作空间能采样、环境能 step、求解器能收敛、info 字段能被记录和画图。
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from project.envs.electric_gas_multiscale_env import ElectricGasMultiScaleEnv
from project.visualization import save_episode_artifacts


@dataclass
class EpisodeStats:
    """一日仿真统计。"""

    power_success: list[bool] = field(default_factory=list)
    gas_success: list[bool] = field(default_factory=list)
    vm_min: list[float] = field(default_factory=list)
    vm_max: list[float] = field(default_factory=list)
    voltage_violation_count: int = 0
    max_line_loading: list[float] = field(default_factory=list)
    high_pressure_min: list[float] = field(default_factory=list)
    high_pressure_max: list[float] = field(default_factory=list)
    prs_pressure_min: list[float] = field(default_factory=list)
    prs_pressure_max: list[float] = field(default_factory=list)
    soc_min: list[float] = field(default_factory=list)
    soc_max: list[float] = field(default_factory=list)
    total_power_loss_cost: float = 0.0
    total_curtailment_cost: float = 0.0
    total_grid_purchase_mwh: float = 0.0
    total_gas_purchase_kg: float = 0.0
    gas_solve_count_last: int = 0
    slow_action_count: int = 0
    records: list[dict[str, float]] = field(default_factory=list)


def run_episode(seed: int = 42) -> EpisodeStats:
    """运行一个随机安全策略 episode。

    强化学习训练前可先跑这个函数，检查环境本身是否健康。
    """

    env = ElectricGasMultiScaleEnv()
    obs, info = env.reset(seed=seed)
    del obs, info
    stats = EpisodeStats()

    terminated = False
    truncated = False
    step_id = 0
    dt_hours = env.config.time.dt_hours
    while not (terminated or truncated):
        # 这里直接采样随机动作，不使用任何 Actor 网络。
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        del obs
        metrics = info.get("constraint_metrics", {})
        components = info.get("reward_components", {})

        power_ok = bool(info.get("power_converged", False))
        gas_ok = bool(info.get("gas_converged", False))
        vm_min = float(metrics.get("vm_min_pu", np.nan))
        vm_max = float(metrics.get("vm_max_pu", np.nan))
        line_loading = float(metrics.get("max_line_loading_percent", np.nan))
        hp_min = float(metrics.get("high_pressure_min_bar", np.nan))
        hp_max = float(metrics.get("high_pressure_max_bar", np.nan))
        prs_min = float(metrics.get("prs_pressure_min_bar", np.nan))
        prs_max = float(metrics.get("prs_pressure_max_bar", np.nan))
        soc_min = float(metrics.get("soc_min", np.nan))
        soc_max = float(metrics.get("soc_max", np.nan))
        grid_purchase_mwh = float(metrics.get("grid_purchase_mwh", 0.0))
        gas_purchase_kg = float(metrics.get("gas_purchase_kg", 0.0))

        stats.power_success.append(power_ok)
        stats.gas_success.append(gas_ok)
        stats.vm_min.append(vm_min)
        stats.vm_max.append(vm_max)
        if vm_min < 0.95 or vm_max > 1.05:
            stats.voltage_violation_count += 1
        stats.max_line_loading.append(line_loading)
        stats.high_pressure_min.append(hp_min)
        stats.high_pressure_max.append(hp_max)
        stats.prs_pressure_min.append(prs_min)
        stats.prs_pressure_max.append(prs_max)
        stats.soc_min.append(soc_min)
        stats.soc_max.append(soc_max)
        stats.total_power_loss_cost += float(components.get("power_loss", 0.0))
        stats.total_curtailment_cost += float(components.get("renewable_curtailment", 0.0))
        stats.total_grid_purchase_mwh += grid_purchase_mwh
        stats.total_gas_purchase_kg += gas_purchase_kg
        stats.gas_solve_count_last = int(info.get("gas_solve_count", 0))
        if info.get("slow_action_applied", False):
            stats.slow_action_count += 1

        stats.records.append(_build_record(step_id, dt_hours, reward, info, metrics, components))
        step_id += 1

    return stats


def _build_record(
    step_id: int,
    dt_hours: float,
    reward: float,
    info: dict[str, Any],
    metrics: dict[str, Any],
    components: dict[str, Any],
) -> dict[str, float]:
    """将 info 中的嵌套指标展平成可画图的一行。"""

    return {
        "step": float(step_id + 1),
        "hour": float((step_id + 1) * dt_hours),
        "reward": float(reward),
        "power_converged": float(bool(info.get("power_converged", False))),
        "gas_converged": float(bool(info.get("gas_converged", False))),
        "gas_solved_this_step": float(bool(info.get("gas_solved_this_step", False))),
        "slow_action_applied": float(bool(info.get("slow_action_applied", False))),
        "gas_state_age": float(info.get("gas_state_age", 0)),
        "equivalent_linepack_indicator": float(info.get("equivalent_linepack_indicator", 0.0)),
        "vm_min_pu": float(metrics.get("vm_min_pu", np.nan)),
        "vm_max_pu": float(metrics.get("vm_max_pu", np.nan)),
        "voltage_mean_abs_deviation_pu": float(metrics.get("voltage_mean_abs_deviation_pu", np.nan)),
        "voltage_rms_deviation_pu": float(metrics.get("voltage_rms_deviation_pu", np.nan)),
        "max_line_loading_percent": float(metrics.get("max_line_loading_percent", np.nan)),
        "high_pressure_min_bar": float(metrics.get("high_pressure_min_bar", np.nan)),
        "high_pressure_max_bar": float(metrics.get("high_pressure_max_bar", np.nan)),
        "high_pressure_mean_abs_deviation_bar": float(metrics.get("high_pressure_mean_abs_deviation_bar", np.nan)),
        "high_pressure_rms_deviation_bar": float(metrics.get("high_pressure_rms_deviation_bar", np.nan)),
        "prs_pressure_min_bar": float(metrics.get("prs_pressure_min_bar", np.nan)),
        "prs_pressure_max_bar": float(metrics.get("prs_pressure_max_bar", np.nan)),
        "prs_pressure_mean_abs_deviation_bar": float(metrics.get("prs_pressure_mean_abs_deviation_bar", np.nan)),
        "prs_pressure_rms_deviation_bar": float(metrics.get("prs_pressure_rms_deviation_bar", np.nan)),
        "soc_min": float(metrics.get("soc_min", np.nan)),
        "soc_max": float(metrics.get("soc_max", np.nan)),
        "grid_purchase_mwh": float(metrics.get("grid_purchase_mwh", 0.0)),
        "gas_purchase_kg": float(metrics.get("gas_purchase_kg", 0.0)),
        "voltage_deviation_cost": float(components.get("voltage_deviation", 0.0)),
        "voltage_violation_cost": float(components.get("voltage_violation", 0.0)),
        "high_pressure_deviation_cost": float(components.get("high_pressure_deviation", 0.0)),
        "high_pressure_violation_cost": float(components.get("high_pressure_violation", 0.0)),
        "prs_pressure_deviation_cost": float(components.get("prs_pressure_deviation", 0.0)),
        "prs_pressure_violation_cost": float(components.get("prs_pressure_violation", 0.0)),
        "line_overload_cost": float(components.get("line_overload", 0.0)),
        "power_loss_cost": float(components.get("power_loss", 0.0)),
        "grid_purchase_cost": float(components.get("grid_purchase", 0.0)),
        "gas_purchase_cost": float(components.get("gas_purchase", 0.0)),
        "renewable_curtailment_cost": float(components.get("renewable_curtailment", 0.0)),
        "compressor_energy_cost": float(components.get("compressor_energy", 0.0)),
        "soc_soft_cost": float(components.get("soc_soft", 0.0)),
        "solver_failure_cost": float(components.get("solver_failure", 0.0)),
    }


def _fmt_range(values: list[float]) -> str:
    arr = np.asarray(values, dtype=float)
    return f"{np.nanmin(arr):.4f} / {np.nanmax(arr):.4f}"


def _print_summary(stats: EpisodeStats, artifacts: dict[str, Path] | None) -> None:
    n = max(len(stats.power_success), 1)
    print("随机安全策略一日仿真结果")
    print(f"潮流成功率: {100.0 * np.mean(stats.power_success):.2f}%")
    print(f"气流成功率: {100.0 * np.mean(stats.gas_success):.2f}%")
    print(f"最小/最大母线电压: {_fmt_range(stats.vm_min + stats.vm_max)} pu")
    print(f"电压越限次数: {stats.voltage_violation_count}")
    print(f"最大线路负载率: {np.nanmax(np.asarray(stats.max_line_loading, dtype=float)):.2f}%")
    print(f"最小/最大高压气网压力: {_fmt_range(stats.high_pressure_min + stats.high_pressure_max)} bar")
    print(f"调压站出口压力范围: {_fmt_range(stats.prs_pressure_min + stats.prs_pressure_max)} bar")
    print(f"最小/最大SOC: {_fmt_range(stats.soc_min + stats.soc_max)}")
    print(f"总网损成本指标: {stats.total_power_loss_cost:.4f}")
    print(f"总新能源削减成本指标: {stats.total_curtailment_cost:.4f}")
    print(f"总购电量: {stats.total_grid_purchase_mwh:.4f} MWh")
    print(f"总购气量: {stats.total_gas_purchase_kg:.4f} kg")
    print(f"气网实际求解次数: {stats.gas_solve_count_last}")
    print(f"慢速动作执行次数: {stats.slow_action_count} / {n}")
    if artifacts:
        print("可视化输出:")
        for name, path in artifacts.items():
            print(f"  {name}: {path}")


def main() -> None:
    """命令行入口：运行随机策略并可选保存图表。"""

    parser = argparse.ArgumentParser(description="Run one random-policy day for the coupled electric-gas environment.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("project/outputs/random_policy"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    stats = run_episode(seed=args.seed)
    artifacts = None
    if not args.no_plots:
        artifacts = save_episode_artifacts(stats.records, output_dir=args.output_dir)
    _print_summary(stats, artifacts)


if __name__ == "__main__":
    main()
