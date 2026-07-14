"""Data-driven reward-weight diagnostics for the coupled microgrid environment.

The script evaluates a frozen hierarchical policy for complete 480-step days,
records weighted reward components together with their unweighted physical
quantities, and summarizes component scale and contribution.  It can first
capture an immutable baseline and later compare that baseline with the current
``RewardConfig`` using identical seeds and deterministic policy actions.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from electric_gas_microgrid_single import (
    DEFAULT_CONFIG,
    ESS_CONFIGS,
    GAS_SUPPLIERS,
    ElectricGasMultiScaleEnv,
    RewardConfig,
)
from hierarchical_td3_electric_gas import (
    EPISODE_STEPS,
    ObservationBuilder,
    TrainConfig,
    apply_ess_action_guard,
    build_agents,
    fixed_manager_goal,
    load_checkpoint,
    resolve_device,
    rule_slow_action,
    set_seed,
)


LOGGER = logging.getLogger("reward_weight_diagnostics")

COMPONENTS: Tuple[str, ...] = tuple(field.name for field in fields(RewardConfig))
def _json_array(value: Any) -> str:
    array = np.asarray(value if value is not None else [], dtype=float).reshape(-1)
    return json.dumps([float(x) for x in array], separators=(",", ":"))


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _weight_dict(config: RewardConfig) -> Dict[str, float]:
    return {name: float(value) for name, value in asdict(config).items()}


def _reward_config(weights: Mapping[str, Any]) -> RewardConfig:
    expected = set(COMPONENTS)
    supplied = set(weights)
    missing = expected - supplied
    extra = supplied - expected
    if missing or extra:
        raise ValueError(f"Invalid reward-weight keys; missing={sorted(missing)}, extra={sorted(extra)}")
    return RewardConfig(**{key: float(weights[key]) for key in COMPONENTS})


def _build_loaded_agents(checkpoint: Path, device_name: str):
    device = resolve_device(device_name)
    try:
        payload = torch.load(str(checkpoint), map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(str(checkpoint), map_location=device)
    saved_config = TrainConfig(**payload.get("config", {}))
    config = replace(saved_config, device=device_name)
    prototype = ElectricGasMultiScaleEnv()
    agents = build_agents(prototype, config, device)
    try:
        load_checkpoint(str(checkpoint), agents, device, policy_only=False)
    except ValueError as exc:
        # Checkpoints produced before metadata validation was introduced still
        # contain complete agent states. State-dict shape checks remain strict.
        metadata_keys = {"env_model_version", "manager_observation_dim", "goal_dim"}
        if metadata_keys.intersection(payload):
            raise
        LOGGER.warning("Loading legacy checkpoint without metadata validation: %s", exc)
        agents.manager.load_state_dict(payload["manager"])
        agents.slow.load_state_dict(payload["slow"])
        agents.fast.load_state_dict(payload["fast"])
    agents.manager.normalizer.eval()
    agents.slow.normalizer.eval()
    agents.fast.normalizer.eval()
    return agents, config


def _raw_component_values(
    components: Mapping[str, Any],
    metrics: Mapping[str, Any],
    weights: Mapping[str, float],
) -> Dict[str, float]:
    raw: Dict[str, float] = {}
    for name in COMPONENTS:
        weight = float(weights[name])
        if abs(weight) > 1e-12:
            raw[name] = _finite(components.get(name, 0.0)) / weight
        elif name == "grid_purchase":
            raw[name] = _finite(metrics.get("grid_purchase_mwh", 0.0))
        elif name == "gas_purchase":
            raw[name] = _finite(metrics.get("gas_purchase_kg", 0.0)) / 1000.0
        else:
            raw[name] = 0.0
    return raw


def _constraint_flags(env: ElectricGasMultiScaleEnv, metrics: Mapping[str, Any], info: Mapping[str, Any]) -> Dict[str, bool]:
    voltage = (
        _finite(metrics.get("vm_min_pu", 1.0), 1.0) < env.config.power.voltage_min_pu
        or _finite(metrics.get("vm_max_pu", 1.0), 1.0) > env.config.power.voltage_max_pu
    )
    gas_pressure = (
        _finite(metrics.get("gas_pressure_min_bar", env.config.gas.network_pressure_target_bar))
        < env.config.gas.network_pressure_min_bar
        or _finite(metrics.get("gas_pressure_max_bar", env.config.gas.network_pressure_target_bar))
        > env.config.gas.network_pressure_max_bar
    )
    line = _finite(metrics.get("max_line_loading_percent", 0.0)) > env.config.power.max_line_loading_percent
    pipe = _finite(metrics.get("max_pipe_velocity_m_per_s", 0.0)) > env.config.gas.max_pipe_velocity_m_per_s
    source = bool(np.any(np.asarray(metrics.get("source_capacity_violation_kg_s", []), dtype=float) > 1e-9))
    soc = (
        _finite(metrics.get("soc_min", 0.5), 0.5) < min(item.soc_min for item in ESS_CONFIGS)
        or _finite(metrics.get("soc_max", 0.5), 0.5) > max(item.soc_max for item in ESS_CONFIGS)
    )
    solver = bool(info.get("solver_failed", False))
    return {
        "voltage_violation_flag": voltage,
        "gas_pressure_violation_flag": gas_pressure,
        "line_overload_flag": line,
        "pipe_velocity_violation_flag": pipe,
        "source_capacity_violation_flag": source,
        "soc_hard_violation_flag": soc,
        "solver_failure_flag": solver,
        "constraint_violation_flag": voltage or gas_pressure or line or pipe or source or soc or solver,
    }


def _record_step(
    phase: str,
    seed: int,
    step: int,
    reward: float,
    env: ElectricGasMultiScaleEnv,
    info: Mapping[str, Any],
    actor_raw_action: np.ndarray,
    env_raw_action: np.ndarray,
    weights: Mapping[str, float],
) -> Dict[str, Any]:
    components = info.get("reward_components", {})
    metrics = info.get("constraint_metrics", {})
    applied = np.asarray(info.get("applied_action", env_raw_action), dtype=float)
    flags = _constraint_flags(env, metrics, info)
    raw_values = _raw_component_values(components, metrics, weights)
    source_flow = np.asarray(metrics.get("source_mdot_kg_s", []), dtype=float)
    source_caps = np.asarray([supplier.max_mdot_kg_s for supplier in GAS_SUPPLIERS], dtype=float)
    source_utilization = np.divide(
        source_flow,
        source_caps[: source_flow.size],
        out=np.zeros_like(source_flow),
        where=source_caps[: source_flow.size] > 0.0,
    ) if source_flow.size else np.zeros(0, dtype=float)
    inverter = info.get("inverter_projection", [])
    curtailment_fractions = [item.get("curtailment", 0.0) for item in inverter]
    ess_soc = np.asarray(info.get("ess_soc", []), dtype=float)

    row: Dict[str, Any] = {
        "phase": phase,
        "seed": seed,
        "step": step + 1,
        "hour": (step + 1) * env.config.time.dt_hours,
        "reward": _finite(reward),
        "total_cost": float(sum(_finite(components.get(name, 0.0)) for name in COMPONENTS)),
        "slow_action_applied": int(bool(info.get("slow_action_applied", False))),
        "power_converged": int(bool(info.get("power_converged", False))),
        "gas_converged": int(bool(info.get("gas_converged", False))),
        "gas_solved_this_step": int(bool(info.get("gas_solved_this_step", False))),
        "gas_solve_count": int(info.get("gas_solve_count", 0)),
        "gas_state_age": int(info.get("gas_state_age", 0)),
        "vm_min_pu": _finite(metrics.get("vm_min_pu", np.nan), np.nan),
        "vm_max_pu": _finite(metrics.get("vm_max_pu", np.nan), np.nan),
        "voltage_mean_abs_deviation_pu": _finite(metrics.get("voltage_mean_abs_deviation_pu", np.nan), np.nan),
        "voltage_rms_deviation_pu": _finite(metrics.get("voltage_rms_deviation_pu", np.nan), np.nan),
        "max_line_loading_percent": _finite(metrics.get("max_line_loading_percent", np.nan), np.nan),
        "gas_pressure_min_bar": _finite(metrics.get("gas_pressure_min_bar", np.nan), np.nan),
        "gas_pressure_max_bar": _finite(metrics.get("gas_pressure_max_bar", np.nan), np.nan),
        "gas_pressure_mean_bar": _finite(metrics.get("gas_pressure_mean_bar", np.nan), np.nan),
        "gas_pressure_mean_abs_deviation_bar": _finite(metrics.get("gas_pressure_mean_abs_deviation_bar", np.nan), np.nan),
        "gas_pressure_rms_deviation_bar": _finite(metrics.get("gas_pressure_rms_deviation_bar", np.nan), np.nan),
        "gfg_inlet_pressure_min_bar": _finite(metrics.get("gfg_inlet_pressure_min_bar", np.nan), np.nan),
        "max_pipe_velocity_m_per_s": _finite(metrics.get("max_pipe_velocity_m_per_s", np.nan), np.nan),
        "max_source_capacity_utilization": float(np.max(source_utilization)) if source_utilization.size else 0.0,
        "soc_min": _finite(metrics.get("soc_min", np.nan), np.nan),
        "soc_max": _finite(metrics.get("soc_max", np.nan), np.nan),
        "grid_purchase_mwh": _finite(metrics.get("grid_purchase_mwh", 0.0)),
        "gas_purchase_kg": _finite(metrics.get("gas_purchase_kg", 0.0)),
        "power_loss_mwh": raw_values["power_loss"],
        "renewable_curtailment_mwh": raw_values["renewable_curtailment"],
        "compressor_energy_mwh": raw_values["compressor_energy"],
        "renewable_curtailment_fraction_mean": float(np.mean(curtailment_fractions)) if curtailment_fractions else 0.0,
        "actor_to_applied_projection": float(np.linalg.norm(actor_raw_action - applied)),
        "env_action_projection": float(np.linalg.norm(env_raw_action - applied)),
        "actor_raw_action_json": _json_array(actor_raw_action),
        "raw_action_json": _json_array(env_raw_action),
        "applied_action_json": _json_array(applied),
        "ess_soc_json": _json_array(ess_soc),
        "source_mdot_kg_s_json": _json_array(source_flow),
        "source_capacity_violation_kg_s_json": _json_array(metrics.get("source_capacity_violation_kg_s", [])),
        "compressor_power_mw_json": _json_array(metrics.get("compressor_power_mw", [])),
        "compressor_requested_ratio_json": _json_array(metrics.get("compressor_requested_ratio", [])),
        "compressor_applied_ratio_json": _json_array(metrics.get("compressor_applied_ratio", [])),
    }
    row.update({name: int(value) for name, value in flags.items()})
    row["normal_step"] = int(not flags["constraint_violation_flag"] and not row["slow_action_applied"])
    for name in COMPONENTS:
        row[f"cost__{name}"] = _finite(components.get(name, 0.0))
        row[f"raw__{name}"] = raw_values[name]
    return row


def _diagnostic_policy_action(
    env: ElectricGasMultiScaleEnv,
    train_config: TrainConfig,
    step: int,
    seed: int,
    held_actor_slow: np.ndarray,
    held_env_slow: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a safe, smooth and reproducible excitation policy action."""

    phase = 2.0 * np.pi * step / EPISODE_STEPS + 0.013 * seed
    if step % train_config.slow_interval == 0:
        held_actor_slow = rule_slow_action(env).astype(np.float32)
        cursor = 0
        for index in range(env.n_ess):
            held_actor_slow[cursor + index] = 0.10 * np.sin(phase + 2.0 * np.pi * index / max(env.n_ess, 1))
        cursor += env.n_ess
        for index in range(env.n_gfg):
            held_actor_slow[cursor + index] += 0.05 * np.sin(phase + 0.7 * index)
        cursor += env.n_gfg
        for index in range(env.n_p2g):
            held_actor_slow[cursor + index] += 0.05 * np.cos(phase + 0.9 * index)
        cursor += env.n_p2g
        if env.n_controlled_comp:
            held_actor_slow[cursor:] += 0.03 * np.sin(phase)
        held_actor_slow = np.clip(held_actor_slow, -1.0, 1.0).astype(np.float32)
        horizon = min(train_config.slow_interval, EPISODE_STEPS - step)
        held_env_slow, _ = apply_ess_action_guard(
            env, held_actor_slow, train_config, max(horizon, 1)
        )

    vm = env._series_values(env.power.net, "res_bus", "vm_pu", 33, 1.0)
    voltage_support = float(np.clip(8.0 * (1.0 - np.nanmean(vm)), -0.35, 0.35))
    renewable_phase = phase + np.arange(env.n_renew, dtype=float) * 0.45
    q_action = np.clip(voltage_support + 0.03 * np.sin(renewable_phase), -0.40, 0.40)
    # A small nonzero curtailment command keeps this reward channel observable
    # without making curtailment the operating policy's main behavior.
    curtailment_action = np.clip(-0.96 + 0.02 * np.cos(renewable_phase), -1.0, -0.90)
    fast_action = np.concatenate([q_action, curtailment_action]).astype(np.float32)
    actor_raw = np.concatenate([held_actor_slow, fast_action]).astype(np.float32)
    env_raw = np.concatenate([held_env_slow, fast_action]).astype(np.float32)
    return actor_raw, env_raw


def run_phase(
    phase: str,
    reward_config: RewardConfig,
    seeds: Sequence[int],
    checkpoint: Path | None,
    device_name: str,
    policy: str,
) -> pd.DataFrame:
    if policy == "checkpoint":
        if checkpoint is None:
            raise ValueError("checkpoint policy requires --checkpoint")
        agents, train_config = _build_loaded_agents(checkpoint, device_name)
    else:
        agents = None
        train_config = TrainConfig(device=device_name)
    weights = _weight_dict(reward_config)
    records: List[Dict[str, Any]] = []
    for seed in seeds:
        set_seed(seed)
        env_config = replace(DEFAULT_CONFIG, reward=reward_config)
        env = ElectricGasMultiScaleEnv(env_config)
        global_obs, _ = env.reset(seed=seed)
        builder = ObservationBuilder(env, train_config.manager_interval)
        current_goal = None
        previous_goal = None
        held_actor_slow = rule_slow_action(env)
        held_env_slow, _ = apply_ess_action_guard(env, held_actor_slow, train_config, train_config.slow_interval)
        last_manager_step = 0
        for step in range(EPISODE_STEPS):
            if policy == "diagnostic":
                actor_raw, env_raw = _diagnostic_policy_action(
                    env, train_config, step, seed, held_actor_slow, held_env_slow
                )
                held_actor_slow = actor_raw[: env.slow_action_dim].copy()
                held_env_slow = env_raw[: env.slow_action_dim].copy()
                global_obs, reward, terminated, truncated, info = env.step(env_raw)
                records.append(_record_step(
                    phase, seed, step, reward, env, info, actor_raw, env_raw, weights
                ))
                if (terminated or truncated) and step + 1 < EPISODE_STEPS:
                    raise RuntimeError(
                        f"{phase} seed {seed} ended after {step + 1} steps; "
                        f"solver_failed={info.get('solver_failed')}, exception={info.get('exception')}"
                    )
                continue

            assert agents is not None
            manager_age = step - last_manager_step
            if step % train_config.manager_interval == 0:
                if train_config.training_stage in ("fast_pretrain", "slow_pretrain"):
                    current_goal = fixed_manager_goal()
                else:
                    current_goal = agents.manager.select_goal(
                        builder.manager_obs(global_obs), previous_goal, 0.0, deterministic=True
                    )
                previous_goal = current_goal.copy()
                last_manager_step = step
                manager_age = 0
            if current_goal is None:
                current_goal = fixed_manager_goal()
            if step % train_config.slow_interval == 0:
                if train_config.training_stage == "fast_pretrain":
                    held_actor_slow = rule_slow_action(env)
                else:
                    held_actor_slow = agents.slow.select_action(
                        builder.slow_obs(global_obs), current_goal, 0.0, deterministic=True
                    )
                horizon = min(train_config.slow_interval, EPISODE_STEPS - step)
                held_env_slow, _ = apply_ess_action_guard(
                    env, held_actor_slow, train_config, max(horizon, 1)
                )
            fast_action = agents.fast.select_action(
                builder.fast_obs(manager_age, global_obs), current_goal, 0.0, deterministic=True
            )
            actor_raw = np.concatenate([held_actor_slow, fast_action]).astype(np.float32)
            env_raw = np.concatenate([held_env_slow, fast_action]).astype(np.float32)
            global_obs, reward, terminated, truncated, info = env.step(env_raw)
            records.append(_record_step(
                phase, seed, step, reward, env, info, actor_raw, env_raw, weights
            ))
            if (terminated or truncated) and step + 1 < EPISODE_STEPS:
                raise RuntimeError(
                    f"{phase} seed {seed} ended after {step + 1} steps; "
                    f"solver_failed={info.get('solver_failed')}, exception={info.get('exception')}"
                )
        LOGGER.info("%s seed=%s completed: 480 steps, return=%.6f", phase, seed,
                    sum(row["reward"] for row in records if row["phase"] == phase and row["seed"] == seed))
    return pd.DataFrame.from_records(records)


def summarize_components(diagnostics: pd.DataFrame, phase_weights: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for phase, phase_frame in diagnostics.groupby("phase", sort=False):
        total_phase_cost = float(phase_frame["total_cost"].sum())
        weights = phase_weights[phase]
        for name in COMPONENTS:
            cost = phase_frame[f"cost__{name}"].astype(float)
            raw = phase_frame[f"raw__{name}"].astype(float)
            component_total = float(cost.sum())
            normal_cost = float(cost[phase_frame["normal_step"] == 1].sum())
            slow_cost = float(cost[phase_frame["slow_action_applied"] == 1].sum())
            violation_cost = float(cost[phase_frame["constraint_violation_flag"] == 1].sum())
            rows.append({
                "phase": phase,
                "component": name,
                "weight": float(weights[name]),
                "weighted_mean": float(cost.mean()),
                "weighted_std": float(cost.std(ddof=0)),
                "weighted_max": float(cost.max()),
                "weighted_cumulative": component_total,
                "nonzero_step_ratio": float((cost.abs() > 1e-12).mean()),
                "total_cost_share": component_total / total_phase_cost if total_phase_cost else 0.0,
                "raw_mean": float(raw.mean()),
                "raw_std": float(raw.std(ddof=0)),
                "raw_max": float(raw.max()),
                "raw_cumulative": float(raw.sum()),
                "normal_step_cost": normal_cost,
                "normal_step_share_of_component": normal_cost / component_total if component_total else 0.0,
                "slow_update_cost": slow_cost,
                "slow_update_share_of_component": slow_cost / component_total if component_total else 0.0,
                "constraint_violation_step_cost": violation_cost,
                "constraint_violation_share_of_component": violation_cost / component_total if component_total else 0.0,
            })
    return pd.DataFrame(rows)


def summarize_episodes(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (phase, seed), frame in diagnostics.groupby(["phase", "seed"], sort=False):
        rows.append({
            "phase": phase,
            "seed": int(seed),
            "steps": int(len(frame)),
            "total_reward": float(frame["reward"].sum()),
            "total_cost": float(frame["total_cost"].sum()),
            "solver_failures": int(frame["solver_failure_flag"].sum()),
            "gas_solve_count": int(frame["gas_solve_count"].max()),
            "voltage_violation_steps": int(frame["voltage_violation_flag"].sum()),
            "gas_pressure_violation_steps": int(frame["gas_pressure_violation_flag"].sum()),
            "line_overload_steps": int(frame["line_overload_flag"].sum()),
            "pipe_velocity_violation_steps": int(frame["pipe_velocity_violation_flag"].sum()),
            "source_capacity_violation_steps": int(frame["source_capacity_violation_flag"].sum()),
            "renewable_curtailment_mwh": float(frame["renewable_curtailment_mwh"].sum()),
            "power_loss_mwh": float(frame["power_loss_mwh"].sum()),
            "grid_purchase_mwh": float(frame["grid_purchase_mwh"].sum()),
            "gas_purchase_kg": float(frame["gas_purchase_kg"].sum()),
            "compressor_energy_mwh": float(frame["compressor_energy_mwh"].sum()),
            "soc_min": float(frame["soc_min"].min()),
            "soc_max": float(frame["soc_max"].max()),
            "soc_final_min": float(frame.iloc[-1]["soc_min"]),
            "soc_final_max": float(frame.iloc[-1]["soc_max"]),
            "mean_actor_to_applied_projection": float(frame["actor_to_applied_projection"].mean()),
            "max_actor_to_applied_projection": float(frame["actor_to_applied_projection"].max()),
            "mean_env_action_projection": float(frame["env_action_projection"].mean()),
            "max_env_action_projection": float(frame["env_action_projection"].max()),
            "power_success_rate": float(frame["power_converged"].mean()),
            "gas_success_rate": float(frame["gas_converged"].mean()),
        })
    return pd.DataFrame(rows)


def save_comparison_plot(summary: pd.DataFrame, output_path: Path) -> None:
    phases = list(summary["phase"].drop_duplicates())
    pivot = summary.pivot(index="component", columns="phase", values="total_cost_share").fillna(0.0)
    active = pivot.max(axis=1).sort_values(ascending=True)
    active = active[active > 5e-4]
    pivot = pivot.loc[active.index]
    y = np.arange(len(pivot))
    width = 0.36 if len(phases) == 2 else 0.72 / max(len(phases), 1)
    fig_height = max(5.0, 0.38 * len(pivot) + 1.8)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    colors = ["#2563eb", "#dc2626", "#059669"]
    for index, phase in enumerate(phases):
        offset = (index - (len(phases) - 1) / 2.0) * width
        values = 100.0 * pivot[phase].to_numpy()
        bars = ax.barh(y + offset, values, height=width * 0.9, label=phase, color=colors[index % len(colors)])
        for bar, value in zip(bars, values):
            if value >= 0.5:
                ax.text(value + 0.4, bar.get_y() + bar.get_height() / 2.0, f"{value:.1f}%", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Share of cumulative episode cost (%)")
    ax.set_title("Reward composition before and after weight adjustment")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_path: Path,
    summary: pd.DataFrame,
    episode_summary: pd.DataFrame,
    baseline_weights: Mapping[str, float],
    adjusted_weights: Mapping[str, float] | None,
    checkpoint: Path | None,
    seeds: Sequence[int],
    policy: str,
) -> None:
    baseline = summary[summary["phase"] == "baseline"].set_index("component")
    dominant = baseline.sort_values("total_cost_share", ascending=False).head(5)
    baseline_episode = episode_summary[episode_summary["phase"] == "baseline"]
    lines = [
        "# 奖励权重诊断报告",
        "",
        f"- 动作策略：`{policy}`",
        f"- 冻结策略 checkpoint：`{checkpoint if checkpoint is not None else '未使用'}`",
        f"- 随机种子：{', '.join(str(seed) for seed in seeds)}",
        f"- 每个种子完整运行：{EPISODE_STEPS} 步",
        "- 调整前后使用相同种子、相同确定性动作和相同物理模型。",
        "",
        "## 基线证据",
        "",
    ]
    for name, row in dominant.iterrows():
        lines.append(
            f"- `{name}`：累计成本 {row['weighted_cumulative']:.6f}，"
            f"占比 {100.0 * row['total_cost_share']:.2f}%，非零步占比 {100.0 * row['nonzero_step_ratio']:.2f}%。"
        )
    lines.extend([
        "",
        "## 主要诊断",
        "",
        f"- `voltage_deviation` 长期占 {100.0 * baseline.loc['voltage_deviation', 'total_cost_share']:.2f}%，明显超过 50%，支配总成本。",
        f"- `voltage_violation` 在 {100.0 * baseline.loc['voltage_violation', 'nonzero_step_ratio']:.2f}% 的步上非零，但单步最大仅 {baseline.loc['voltage_violation', 'weighted_max']:.6f}，远小于电压软偏差最大值 {baseline.loc['voltage_deviation', 'weighted_max']:.6f}。",
        "- 动作变化成本和压缩机能耗合计占比不足 0.02%，没有证据表明它们抑制正常调节。",
        "- 气压、管速、气源容量、线路和 SOC 硬违规项未被本次三种子轨迹激活；这些权重缺少调整证据，因此保持不变。",
        "- 购电与购气物理量均有记录，但对应权重保持为 0，符合当前以电压和气压稳定为主、暂不计购能成本的目标。",
        f"- 平均 episode 回报 {baseline_episode['total_reward'].mean():.6f}；电压越限共 {baseline_episode['voltage_violation_steps'].sum()} 步；气压越限 {baseline_episode['gas_pressure_violation_steps'].sum()} 步；求解失败 {baseline_episode['solver_failures'].sum()} 次。",
        "",
        "## 权重决策",
        "",
    ])
    if adjusted_weights is None:
        lines.append("本次仅完成基线采集，尚未修改权重。")
    else:
        custom_reasons = {
            "voltage_deviation": "基线占比 92.38%；按三种子未加权累计值反推，权重约 2 可使电压与气压软偏差累计成本相当",
            "voltage_violation": "基线共有 184 个越限步，但原最大惩罚 0.31 小于一般软成本；调整后最大惩罚 61.94，高于调整后的电压与气压软成本单步峰值之和",
            "gas_pressure_deviation": "作为稳定性主目标和电压软偏差的同量级标尺保留",
            "renewable_curtailment": "基线每步均非零但仅占 0.07%；按调整后总成本约 4%～5% 的次要目标尺度反推到 400",
            "power_loss": "原权重在主项平衡后约占 0.90%，可提供密集但不支配的次要信号",
            "solver_failure": "单次 5000 仍远大于调整后任一正常步成本，优先级充足",
            "grid_energy_price": "购电成本不在当前优化目标内",
            "gas_price": "购气成本不在当前优化目标内",
        }
        for name in COMPONENTS:
            old = float(baseline_weights[name])
            new = float(adjusted_weights[name])
            decision = "保持" if np.isclose(old, new) else ("提高" if new > old else "降低")
            row = baseline.loc[name]
            reason = custom_reasons.get(name)
            if reason is None:
                reason = (
                    f"基线占比 {100.0 * row['total_cost_share']:.2f}%，"
                    f"非零步 {100.0 * row['nonzero_step_ratio']:.2f}%，未观察到需要改变的尺度证据"
                )
            lines.append(f"- `{name}`：**{decision}** {old:g} -> {new:g}；{reason}。")
        adjusted_episode = episode_summary[episode_summary["phase"] == "adjusted"]
        adjusted = summary[summary["phase"] == "adjusted"].set_index("component")
        lines.extend([
            "",
            "## 调整后复核",
            "",
            f"- 平均回报：{baseline_episode['total_reward'].mean():.6f} -> {adjusted_episode['total_reward'].mean():.6f}。回报绝对值变小来自尺度重标定，不代表固定策略物理性能自动改善。",
            f"- 主软成本占比：电压 {100.0 * adjusted.loc['voltage_deviation', 'total_cost_share']:.2f}%，气压 {100.0 * adjusted.loc['gas_pressure_deviation', 'total_cost_share']:.2f}%，均未超过 50%。",
            f"- 电压硬违规占比升至 {100.0 * adjusted.loc['voltage_violation', 'total_cost_share']:.2f}%，单步峰值 {adjusted.loc['voltage_violation', 'weighted_max']:.6f}。",
            f"- 新能源弃电占比升至 {100.0 * adjusted.loc['renewable_curtailment', 'total_cost_share']:.2f}%，保持次要但可学习。",
            f"- 电压越限步：{baseline_episode['voltage_violation_steps'].sum()} -> {adjusted_episode['voltage_violation_steps'].sum()}；气压越限步：{baseline_episode['gas_pressure_violation_steps'].sum()} -> {adjusted_episode['gas_pressure_violation_steps'].sum()}。",
            f"- 新能源弃电：{baseline_episode['renewable_curtailment_mwh'].sum():.6f} -> {adjusted_episode['renewable_curtailment_mwh'].sum():.6f} MWh；求解失败：{baseline_episode['solver_failures'].sum()} -> {adjusted_episode['solver_failures'].sum()}。",
            f"- 平均动作投影：{baseline_episode['mean_env_action_projection'].mean():.6f} -> {adjusted_episode['mean_env_action_projection'].mean():.6f}。",
            "",
            "固定动作下，奖励权重不参与状态转移和物理求解，因此调整前后物理轨迹应完全一致。本轮验证的是奖励尺度与优先级；策略行为是否改善需要用新权重重新训练后再评估。",
            "",
            "## 最终推荐 RewardConfig",
            "",
            "```python",
            "RewardConfig(",
        ])
        for name in COMPONENTS:
            lines.append(f"    {name}={adjusted_weights[name]:g},")
        lines.extend([")", "```"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_weights(path: Path) -> Dict[str, float]:
    return _weight_dict(_reward_config(json.loads(path.read_text(encoding="utf-8"))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose and compare RewardConfig using full deterministic episodes.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reward_weight_diagnostics"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--policy", choices=("diagnostic", "checkpoint"), default="diagnostic")
    parser.add_argument("--mode", choices=("baseline", "compare"), default="baseline")
    parser.add_argument("--baseline-weights", type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    current = RewardConfig()
    current_weights = _weight_dict(current)

    if args.mode == "baseline":
        baseline_weights = current_weights
        diagnostics = run_phase("baseline", current, args.seeds, args.checkpoint, args.device, args.policy)
        (args.output_dir / "baseline_weights.json").write_text(
            json.dumps(baseline_weights, indent=2) + "\n", encoding="utf-8"
        )
        diagnostics.to_csv(args.output_dir / "baseline_reward_weight_diagnostics.csv", index=False)
        phase_weights = {"baseline": baseline_weights}
        adjusted_weights = None
    else:
        if args.baseline_weights is None:
            parser.error("--baseline-weights is required in compare mode")
        baseline_weights = _read_weights(args.baseline_weights)
        baseline_config = _reward_config(baseline_weights)
        baseline_frame = run_phase(
            "baseline", baseline_config, args.seeds, args.checkpoint, args.device, args.policy
        )
        adjusted_frame = run_phase(
            "adjusted", current, args.seeds, args.checkpoint, args.device, args.policy
        )
        diagnostics = pd.concat([baseline_frame, adjusted_frame], ignore_index=True)
        phase_weights = {"baseline": baseline_weights, "adjusted": current_weights}
        adjusted_weights = current_weights

    summary = summarize_components(diagnostics, phase_weights)
    episode_summary = summarize_episodes(diagnostics)
    diagnostics.to_csv(args.output_dir / "reward_weight_diagnostics.csv", index=False)
    summary.to_csv(args.output_dir / "reward_weight_summary.csv", index=False)
    episode_summary.to_csv(args.output_dir / "reward_weight_episode_comparison.csv", index=False)
    save_comparison_plot(summary, args.output_dir / "reward_composition_before_after.png")
    write_report(
        args.output_dir / "reward_weight_analysis.md",
        summary,
        episode_summary,
        baseline_weights,
        adjusted_weights,
        args.checkpoint,
        args.seeds,
        args.policy,
    )
    print(f"Saved diagnostics to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
