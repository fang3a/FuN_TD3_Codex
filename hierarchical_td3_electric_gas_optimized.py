"""投影感知、物理引导的分层 SMDP-TD3。

本文件保留 ``hierarchical_td3_electric_gas.py`` 的 Manager-Slow-Fast 网络、
环境交互和 checkpoint 结构，只替换算法层的配置、Replay、TD target、动作语义、
物理 goal 与阶段训练控制。电-气网络拓扑、设备参数、动作映射和奖励公式仍由
``electric_gas_microgrid_single.py`` 提供，未在这里复制或修改。


Typical quick check:

    & 'D:\\anaconda\\anaconda\\envs\\python_3_8\\python.exe' hierarchical_td3_electric_gas_optimized.py --run-tests

Short debug training:

    & 'D:\\anaconda\\anaconda\\envs\\python_3_8\\python.exe' hierarchical_td3_electric_gas_optimized.py --episodes 1 --episode-steps 20 --batch-size 16 --learning-starts 8
"""

from __future__ import annotations

import argparse
import copy
import inspect
import logging
import math
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import hierarchical_td3_electric_gas as legacy
from electric_gas_microgrid_single import (
    DEFAULT_CONFIG,
    ENV_MODEL_VERSION,
    ESS_CONFIGS,
    GAS_PIPES,
    GAS_SUPPLIERS,
    IEEE33_LINE_DATA,
    N_GAS_JUNCTIONS,
    RENEWABLE_CONFIGS,
    ElectricGasMultiScaleEnv,
)


LOGGER = logging.getLogger("hierarchical_td3_electric_gas_optimized")

FAST_INTERVAL = legacy.FAST_INTERVAL
SLOW_INTERVAL = legacy.SLOW_INTERVAL
MANAGER_INTERVAL = legacy.MANAGER_INTERVAL
EPISODE_STEPS = legacy.EPISODE_STEPS
GOAL_DIM = legacy.GOAL_DIM
GOAL_PHYSICAL_SLICE = slice(24, 32)

ObservationBuilder = legacy.ObservationBuilder
AgentBundle = legacy.AgentBundle
RunningMeanStd = legacy.RunningMeanStd

_LEGACY_RUN_TRAINING = legacy.run_training
_LEGACY_EVALUATE_POLICY = legacy.evaluate_policy
_LEGACY_SAFE_ENV_STEP = legacy.safe_env_step
_LEGACY_APPLY_ESS_ACTION_GUARD = legacy.apply_ess_action_guard
_LEGACY_LOAD_CHECKPOINT = legacy.load_checkpoint
_LEGACY_FIXED_MANAGER_GOAL = legacy.fixed_manager_goal
_LEGACY_SAVE_CHECKPOINT = legacy.save_checkpoint
_LEGACY_SAVE_BEST_FILES = legacy.save_best_files
_LEGACY_CHECKPOINT_METADATA = legacy.checkpoint_metadata
_LEGACY_VALIDATE_CHECKPOINT = legacy.validate_checkpoint_compatibility


# =============================================================================
# Configuration and compatibility
# =============================================================================


@dataclass
class TrainConfig(legacy.TrainConfig):
    """优化版训练配置。

    ``batch_size``、``learning_starts`` 和 ``updates_per_step`` 仅保留为旧命令行与
    checkpoint 兼容字段；实际更新全部读取对应层级的独立参数。
    """

    # 旧训练循环的外层门控保持开启，真正门控由各层 update 内部完成。
    batch_size: int = 1
    learning_starts: int = 0
    updates_per_step: int = 1
    slow_update_interval_steps: int = 1
    manager_update_interval_steps: int = 1

    fast_batch_size: int = 256
    slow_batch_size: int = 64
    manager_batch_size: int = 32
    fast_learning_starts: int = 5000
    slow_learning_starts: int = 256
    manager_learning_starts: int = 128
    fast_updates_per_step: int = 1
    slow_updates_per_boundary: int = 2
    manager_updates_per_boundary: int = 4

    fast_lr: float = 3e-4
    slow_lr: float = 1e-4
    manager_lr: float = 5e-5
    joint_worker_lr: float = 1e-4
    joint_manager_lr: float = 2.5e-5

    projection_imitation_weight: float = 0.10  # 旧字段，保存兼容性。
    projection_imitation_initial_weight: float = 0.10
    projection_imitation_final_weight: float = 0.01
    projection_imitation_decay_updates: int = 100_000
    projection_imitation_threshold: float = 1e-3
    projection_behavior_match_threshold: float = 0.05

    use_prioritized_replay: bool = True
    priority_alpha: float = 0.6
    priority_beta_initial: float = 0.4
    priority_beta_final: float = 1.0
    priority_beta_anneal_steps: int = 200_000
    priority_epsilon: float = 1e-5
    constraint_priority_weight: float = 1.0
    projection_priority_weight: float = 1.0

    fast_pretrain_episodes: int = 50
    slow_pretrain_episodes: int = 50
    manager_train_episodes: int = 50
    joint_finetune_episodes: int = 100
    eval_episodes: int = 3
    eval_seeds: Tuple[int, ...] = (20_026, 20_027, 20_028)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TrainConfig":
        """读取新旧 checkpoint 配置；旧配置缺少的字段使用本类默认值。"""

        valid = {item.name for item in fields(cls)}
        filtered = {key: value for key, value in payload.items() if key in valid}
        aliases = (
            ("batch_size", ("fast_batch_size", "slow_batch_size", "manager_batch_size")),
            ("learning_starts", ("fast_learning_starts", "slow_learning_starts", "manager_learning_starts")),
            ("updates_per_step", ("fast_updates_per_step", "slow_updates_per_boundary", "manager_updates_per_boundary")),
        )
        for old_name, new_names in aliases:
            if old_name in filtered and not any(name in filtered for name in new_names):
                for name in new_names:
                    filtered[name] = filtered[old_name]
                LOGGER.warning("Migrated deprecated checkpoint config %s to %s.", old_name, ", ".join(new_names))
        for old_name in ("batch_size", "learning_starts", "updates_per_step",
                         "slow_update_interval_steps", "manager_update_interval_steps"):
            filtered.pop(old_name, None)
        if "eval_seeds" in filtered:
            filtered["eval_seeds"] = tuple(int(x) for x in filtered["eval_seeds"])
        return cls(**filtered)

    def __post_init__(self) -> None:
        # 这些旧字段只为反序列化兼容；非规范值必须显式告警，不能静默覆盖。
        legacy_values = (self.batch_size, self.learning_starts, self.updates_per_step,
                         self.slow_update_interval_steps, self.manager_update_interval_steps)
        if legacy_values != (1, 0, 1, 1, 1):
            warnings.warn(
                "Deprecated shared training fields are ignored; use per-level batch size, "
                "learning starts and update-count fields.", FutureWarning, stacklevel=2,
            )
        self.batch_size = 1
        self.learning_starts = 0
        self.updates_per_step = 1
        self.slow_update_interval_steps = 1
        self.manager_update_interval_steps = 1
        for name in ("fast_batch_size", "slow_batch_size", "manager_batch_size"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "fast_learning_starts", "slow_learning_starts", "manager_learning_starts",
            "fast_updates_per_step", "slow_updates_per_boundary", "manager_updates_per_boundary",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self.priority_beta_initial <= self.priority_beta_final <= 1.0:
            raise ValueError("priority beta must satisfy 0 <= initial <= final <= 1")
        if self.projection_behavior_match_threshold < 0.0:
            raise ValueError("projection_behavior_match_threshold must be non-negative")
        if self.reachability_weight > 0.0:
            raise ValueError(
                "reachability_weight must remain 0 until a differentiable projection model is available: "
                "the Transition Model is trained on executed actions, not unconstrained raw actions."
            )


def smdp_discount(gamma_fast: float, duration_steps: torch.Tensor) -> torch.Tensor:
    """按每条样本真实持续时间计算 SMDP 折扣 ``gamma_fast ** duration``。"""

    gamma = torch.as_tensor(gamma_fast, dtype=duration_steps.dtype, device=duration_steps.device)
    return torch.pow(gamma, duration_steps.clamp_min(0.0))


def build_smdp_target(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    q_next: torch.Tensor,
    gamma_fast: float,
    duration_steps: torch.Tensor,
) -> torch.Tensor:
    """SMDP TD target；片段奖励已经区间折扣累计，不再整体乘折扣。"""

    return rewards + (1.0 - dones) * smdp_discount(gamma_fast, duration_steps) * q_next


def projection_imitation_weight(cfg: TrainConfig, update_step: int) -> float:
    """从初始权重线性退火到最终权重，避免模仿项长期压制策略梯度。"""

    horizon = max(int(cfg.projection_imitation_decay_updates), 1)
    fraction = min(max(float(update_step) / horizon, 0.0), 1.0)
    return float(
        cfg.projection_imitation_initial_weight
        + fraction * (cfg.projection_imitation_final_weight - cfg.projection_imitation_initial_weight)
    )


def projection_imitation_masks(
    current_actor_actions: torch.Tensor,
    historical_raw_actions: torch.Tensor,
    executed_actions: torch.Tensor,
    projection_threshold: float,
    behavior_match_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """仅对仍接近历史行为动作的投影样本启用离线模仿监督。"""

    projection_rms = torch.sqrt(
        torch.mean((historical_raw_actions - executed_actions).pow(2), dim=-1) + 1e-12
    )
    behavior_rms = torch.sqrt(
        torch.mean((current_actor_actions.detach() - historical_raw_actions).pow(2), dim=-1) + 1e-12
    )
    projection_mask = projection_rms > float(projection_threshold)
    behavior_match_mask = behavior_rms < float(behavior_match_threshold)
    return projection_mask, behavior_match_mask, projection_mask & behavior_match_mask


# =============================================================================
# Centralized physical state layout and goal semantics
# =============================================================================


@dataclass(frozen=True)
class PhysicalStateLayout:
    """集中描述 Worker 局部状态中的物理切片，避免散落魔法索引。"""

    power_bus_count: int = 33
    line_count: int = len(IEEE33_LINE_DATA)
    renewable_count: int = len(RENEWABLE_CONFIGS)
    ess_count: int = len(ESS_CONFIGS)
    gas_junction_count: int = N_GAS_JUNCTIONS
    gas_source_count: int = len(GAS_SUPPLIERS)
    pipe_count: int = len(GAS_PIPES)

    @property
    def fast_voltage(self) -> slice:
        return slice(0, self.power_bus_count)

    @property
    def fast_line_loading(self) -> slice:
        start = self.power_bus_count
        return slice(start, start + self.line_count)

    @property
    def fast_renewable_available(self) -> slice:
        start = self.power_bus_count + self.line_count + 3
        return slice(start, start + self.renewable_count)

    @property
    def fast_renewable_actual(self) -> slice:
        start = self.fast_renewable_available.stop
        return slice(start, start + self.renewable_count)

    @property
    def fast_grid_purchase(self) -> int:
        return self.fast_renewable_actual.stop + self.renewable_count

    @property
    def fast_required_size(self) -> int:
        return self.fast_grid_purchase + 1

    @property
    def slow_soc(self) -> slice:
        return slice(0, self.ess_count)

    @property
    def slow_gas_pressure(self) -> slice:
        start = 3 * self.ess_count
        return slice(start, start + self.gas_junction_count)

    @property
    def slow_source_loading(self) -> slice:
        start = self.slow_gas_pressure.stop + 3 + 2
        return slice(start, start + self.gas_source_count)

    @property
    def slow_linepack(self) -> int:
        return self.slow_source_loading.stop + 3 + 3 + 1

    @property
    def slow_required_size(self) -> int:
        return self.slow_linepack + 1


PHYSICAL_LAYOUT = PhysicalStateLayout()
PHYSICAL_GOAL_NAMES: Tuple[str, ...] = (
    "voltage_deviation",       # goal[24]
    "max_line_loading",        # goal[25]
    "renewable_utilization",   # goal[26]
    "grid_purchase",           # goal[27]
    "mean_soc",                # goal[28]
    "gas_pressure_deviation",  # goal[29]
    "gas_source_loading",      # goal[30]
    "linepack",                # goal[31]
)


class PhysicalGoalFeatureExtractor:
    """把当前 Worker 物理状态转换为与 goal[24:32] 同域的八维 ``[-1, 1]`` 特征。"""

    FAST_MASK = np.array([True, True, True, True, False, False, False, False])
    SLOW_MASK = ~FAST_MASK

    def __init__(self, layout: PhysicalStateLayout = PHYSICAL_LAYOUT):
        self.layout = layout
        self.renewable_capacities = np.asarray(
            [item.capacity_mw for item in RENEWABLE_CONFIGS], dtype=np.float32
        )

    @staticmethod
    def _unit_to_bipolar(value: float) -> float:
        return float(2.0 * np.clip(value, 0.0, 1.0) - 1.0)

    def extract_fast(self, obs: np.ndarray) -> np.ndarray:
        values = np.asarray(obs, dtype=np.float32).reshape(-1)
        features = np.zeros(8, dtype=np.float32)
        if values.size < self.layout.fast_required_size:
            return features
        voltage_deviation = float(np.mean(np.abs(values[self.layout.fast_voltage])))
        max_loading_pu = float(np.max(values[self.layout.fast_line_loading]))
        available_pu = values[self.layout.fast_renewable_available]
        actual_pu = values[self.layout.fast_renewable_actual]
        available_mw = float(np.sum(available_pu * self.renewable_capacities))
        actual_mw = float(np.sum(actual_pu * self.renewable_capacities))
        utilization = actual_mw / max(available_mw, 1e-6)
        grid_purchase_pu = float(values[self.layout.fast_grid_purchase])
        features[0] = self._unit_to_bipolar(voltage_deviation / 1.0)
        features[1] = self._unit_to_bipolar(max_loading_pu / 1.5)
        features[2] = self._unit_to_bipolar(utilization)
        features[3] = self._unit_to_bipolar(grid_purchase_pu)
        return np.clip(features, -1.0, 1.0)

    def extract_slow(self, obs: np.ndarray) -> np.ndarray:
        values = np.asarray(obs, dtype=np.float32).reshape(-1)
        features = np.zeros(8, dtype=np.float32)
        if values.size < self.layout.slow_required_size:
            return features
        mean_soc = float(np.mean(values[self.layout.slow_soc]))
        pressure_deviation_bar = float(np.mean(np.abs(values[self.layout.slow_gas_pressure])))
        max_source_loading = float(np.max(values[self.layout.slow_source_loading]))
        linepack_scaled = float(values[self.layout.slow_linepack])
        features[4] = self._unit_to_bipolar(mean_soc)
        features[5] = self._unit_to_bipolar(pressure_deviation_bar / 2.5)
        features[6] = self._unit_to_bipolar(max_source_loading)
        features[7] = float(np.clip(linepack_scaled, -1.0, 1.0))
        return np.clip(features, -1.0, 1.0)

    @staticmethod
    def distance(features: np.ndarray, physical_goal: np.ndarray, mask: np.ndarray) -> float:
        delta = np.asarray(features, dtype=np.float32)[mask] - np.asarray(physical_goal, dtype=np.float32)[mask]
        return float(np.sqrt(np.mean(np.square(delta)) + 1e-12))

    def progress(self, obs: np.ndarray, next_obs: np.ndarray, goal: np.ndarray, role: str) -> float:
        physical_goal = np.asarray(goal, dtype=np.float32)[GOAL_PHYSICAL_SLICE]
        if role == "fast":
            before, after, mask = self.extract_fast(obs), self.extract_fast(next_obs), self.FAST_MASK
        elif role == "slow":
            before, after, mask = self.extract_slow(obs), self.extract_slow(next_obs), self.SLOW_MASK
        else:
            raise ValueError(f"unknown physical feature role: {role}")
        return self.distance(before, physical_goal, mask) - self.distance(after, physical_goal, mask)


PHYSICAL_FEATURES = PhysicalGoalFeatureExtractor()


def fast_physical_progress(obs: np.ndarray, next_obs: np.ndarray, goal: np.ndarray) -> float:
    return PHYSICAL_FEATURES.progress(obs, next_obs, goal, "fast")


def slow_physical_progress(obs: np.ndarray, next_obs: np.ndarray, goal: np.ndarray) -> float:
    return PHYSICAL_FEATURES.progress(obs, next_obs, goal, "slow")


def fixed_manager_goal() -> np.ndarray:
    """预训练的固定 goal：低偏差/低购电、高新能源利用率、SOC 约 0.5。"""

    goal = _LEGACY_FIXED_MANAGER_GOAL()
    goal[GOAL_PHYSICAL_SLICE] = np.array(
        [-1.0, -0.30, 1.0, -1.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32
    )
    return goal.astype(np.float32)


# =============================================================================
# Constraint score and prioritized replay
# =============================================================================


def normalized_constraint_score(info: Mapping[str, Any], env: Any) -> float:
    """把多种安全指标归一到 ``[0,1]``，solver failure 只作为有界一项。"""

    metrics = info.get("constraint_metrics", {})
    cfg = getattr(env, "config", DEFAULT_CONFIG)

    def value(name: str, default: float) -> float:
        raw = metrics.get(name, default)
        try:
            result = float(raw)
        except (TypeError, ValueError):
            return default
        return result if np.isfinite(result) else default

    voltage = max(
        cfg.power.voltage_min_pu - value("vm_min_pu", cfg.power.voltage_target_pu),
        value("vm_max_pu", cfg.power.voltage_target_pu) - cfg.power.voltage_max_pu,
        0.0,
    ) / 0.05
    pressure = max(
        cfg.gas.network_pressure_min_bar - value("gas_pressure_min_bar", cfg.gas.network_pressure_target_bar),
        value("gas_pressure_max_bar", cfg.gas.network_pressure_target_bar) - cfg.gas.network_pressure_max_bar,
        0.0,
    ) / max(cfg.gas.network_pressure_max_bar - cfg.gas.network_pressure_min_bar, 1e-6)
    line = max(value("max_line_loading_percent", 0.0) - cfg.power.max_line_loading_percent, 0.0) / 100.0
    pipe = max(value("max_pipe_velocity_m_per_s", 0.0) - cfg.gas.max_pipe_velocity_m_per_s, 0.0) / max(
        cfg.gas.max_pipe_velocity_m_per_s, 1e-6
    )
    source_violation = np.asarray(metrics.get("source_capacity_violation_kg_s", []), dtype=float)
    source_caps = np.asarray([item.max_mdot_kg_s for item in GAS_SUPPLIERS], dtype=float)
    source = 0.0
    if source_violation.size:
        source = float(np.max(source_violation / np.maximum(source_caps[: source_violation.size], 1e-6)))
    soc = max(
        min(item.soc_min for item in ESS_CONFIGS) - value("soc_min", 0.5),
        value("soc_max", 0.5) - max(item.soc_max for item in ESS_CONFIGS),
        0.0,
    ) / 0.10
    solver = float(bool(info.get("solver_failed", False)))
    terms = np.clip(np.asarray([voltage, pressure, line, pipe, source, soc, solver]), 0.0, 1.0)
    return float(np.clip(0.5 * np.max(terms) + 0.5 * np.mean(terms), 0.0, 1.0))


class _PrioritizedReplayMixin:
    priorities: np.ndarray
    td_priorities: np.ndarray
    constraint_scores: np.ndarray
    projection_scores: np.ndarray
    cfg: TrainConfig
    sample_calls: int

    def _init_priority(self, capacity: int, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.td_priorities = np.zeros(capacity, dtype=np.float32)
        self.constraint_scores = np.zeros(capacity, dtype=np.float32)
        self.projection_scores = np.zeros(capacity, dtype=np.float32)
        self.sample_calls = 0

    def _initial_td_priority(self, size_before_add: int) -> float:
        if size_before_add <= 0:
            return 1.0
        return float(max(np.max(self.td_priorities[:size_before_add]), self.cfg.priority_epsilon))

    @staticmethod
    def _normalize_projection_mse(raw_action: np.ndarray, executed_action: np.ndarray) -> float:
        # 两个动作均在 [-1,1]，逐维最大平方误差为 4。
        mse = float(np.mean(np.square(raw_action - executed_action)))
        return float(np.clip(mse / 4.0, 0.0, 1.0))

    def _compose_priority(self, indices: np.ndarray) -> np.ndarray:
        """TD、约束和投影三个独立分量只在此处组合一次。"""

        return (
            self.td_priorities[indices]
            + self.cfg.constraint_priority_weight * self.constraint_scores[indices]
            + self.cfg.projection_priority_weight * self.projection_scores[indices]
            + self.cfg.priority_epsilon
        )

    def _set_new_priority(self, slot: int, size_before_add: int) -> None:
        self.td_priorities[slot] = self._initial_td_priority(size_before_add)
        indices = np.asarray([slot], dtype=np.int64)
        self.priorities[slot] = self._compose_priority(indices)[0]

    def _sample_indices(self, batch_size: int, size: int) -> Tuple[np.ndarray, np.ndarray]:
        self.sample_calls += 1
        if not self.cfg.use_prioritized_replay:
            indices = np.random.randint(0, size, size=batch_size)
            return indices, np.ones((batch_size, 1), dtype=np.float32)
        scaled = np.power(np.maximum(self.priorities[:size], self.cfg.priority_epsilon), self.cfg.priority_alpha)
        probabilities = scaled / max(float(np.sum(scaled)), 1e-12)
        indices = np.random.choice(size, size=batch_size, replace=True, p=probabilities)
        anneal = min(self.sample_calls / max(self.cfg.priority_beta_anneal_steps, 1), 1.0)
        beta = self.cfg.priority_beta_initial + anneal * (
            self.cfg.priority_beta_final - self.cfg.priority_beta_initial
        )
        weights = np.power(size * probabilities[indices], -beta)
        weights /= max(float(np.max(weights)), 1e-12)
        return indices, weights.reshape(-1, 1).astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        flat_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        flat_errors = np.asarray(td_errors, dtype=np.float32).reshape(-1)
        if flat_indices.size != flat_errors.size:
            raise ValueError("priority indices and TD errors must have the same length")
        if flat_indices.size == 0:
            return
        unique, inverse = np.unique(flat_indices, return_inverse=True)
        max_abs_errors = np.zeros(unique.size, dtype=np.float32)
        np.maximum.at(max_abs_errors, inverse, np.abs(flat_errors))
        # log1p 控制灾难样本的尺度，同时保持 TD error 的单调排序。
        self.td_priorities[unique] = np.log1p(max_abs_errors)
        self.priorities[unique] = self._compose_priority(unique)


_ACTIVE_CONFIG: Optional[TrainConfig] = None


class FastReplayBuffer(_PrioritizedReplayMixin, legacy.FastReplayBuffer):
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, goal_dim: int,
                 device: torch.device, cfg: Optional[TrainConfig] = None):
        super().__init__(capacity, obs_dim, action_dim, goal_dim, device)
        # legacy 数组不再承载在线 Encoder 生成的陈旧 intrinsic/total reward。
        del self.reward_intrinsic
        del self.reward_total
        self._init_priority(capacity, cfg or _ACTIVE_CONFIG or TrainConfig())
        self.physical_progress = np.zeros((capacity, 1), dtype=np.float32)
        self.projection_penalty = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs: np.ndarray, next_obs: np.ndarray, raw_action: np.ndarray, executed_action: np.ndarray,
            reward_external: float, physical_progress: float, projection_penalty: float, goal: np.ndarray,
            next_goal: np.ndarray, done: bool, constraint_score: float = 0.0,
            reward_clipped: bool = False) -> None:
        del reward_clipped
        slot = self.idx % self.capacity
        size_before = len(self)
        self.obs[slot] = obs
        self.next_obs[slot] = next_obs
        self.raw_actions[slot] = raw_action
        self.executed_actions[slot] = executed_action
        self.reward_external[slot, 0] = reward_external
        self.physical_progress[slot, 0] = physical_progress
        self.projection_penalty[slot, 0] = projection_penalty
        self.goals[slot] = goal
        self.next_goals[slot] = next_goal
        self.goal_changed[slot, 0] = float(np.linalg.norm(goal - next_goal) > 1e-6)
        self.dones[slot, 0] = float(done)
        self.idx += 1
        self.full = self.full or self.idx >= self.capacity
        self.constraint_scores[slot] = float(np.clip(constraint_score, 0.0, 1.0))
        self.projection_scores[slot] = self._normalize_projection_mse(raw_action, executed_action)
        self._set_new_priority(slot, size_before)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        size = len(self)
        indices, is_weights = self._sample_indices(batch_size, size)
        return {
            "obs": legacy.to_tensor(self.obs[indices], self.device),
            "next_obs": legacy.to_tensor(self.next_obs[indices], self.device),
            "raw_actions": legacy.to_tensor(self.raw_actions[indices], self.device),
            "executed_actions": legacy.to_tensor(self.executed_actions[indices], self.device),
            "reward_external": legacy.to_tensor(self.reward_external[indices], self.device),
            "physical_progress": legacy.to_tensor(self.physical_progress[indices], self.device),
            "projection_penalty": legacy.to_tensor(self.projection_penalty[indices], self.device),
            "goals": legacy.to_tensor(self.goals[indices], self.device),
            "next_goals": legacy.to_tensor(self.next_goals[indices], self.device),
            "goal_changed": legacy.to_tensor(self.goal_changed[indices], self.device),
            "dones": legacy.to_tensor(self.dones[indices], self.device),
            "duration_steps": torch.ones((batch_size, 1), dtype=torch.float32, device=self.device),
            "constraint_scores": legacy.to_tensor(self.constraint_scores[indices, None], self.device),
            "projection_scores": legacy.to_tensor(self.projection_scores[indices, None], self.device),
            "indices": torch.as_tensor(indices, dtype=torch.long, device=self.device),
            "is_weights": legacy.to_tensor(is_weights, self.device),
        }


class SlowReplayBuffer(_PrioritizedReplayMixin, legacy.SlowReplayBuffer):
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, goal_dim: int,
                 device: torch.device, cfg: Optional[TrainConfig] = None):
        super().__init__(capacity, obs_dim, action_dim, goal_dim, device)
        self._init_priority(capacity, cfg or _ACTIVE_CONFIG or TrainConfig())
        self.external_reward = np.zeros((capacity, 1), dtype=np.float32)
        self.physical_progress = np.zeros((capacity, 1), dtype=np.float32)
        self.projection_penalty = np.zeros((capacity, 1), dtype=np.float32)
        self.segment_constraint_mean = np.zeros((capacity, 1), dtype=np.float32)
        self.segment_constraint_max = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs_start: np.ndarray, obs_end: np.ndarray, raw_action: np.ndarray,
            executed_action: np.ndarray, discounted_external_reward: float,
            physical_progress: float, projection_penalty: float, goal: np.ndarray,
            next_goal: np.ndarray, done: bool, duration_steps: int,
            constraint_score: float = 0.0, segment_constraint_mean: float = 0.0,
            segment_constraint_max: float = 0.0, reward_clipped: bool = False) -> None:
        del reward_clipped
        slot = self.idx % self.capacity
        size_before = len(self)
        self.obs_start[slot] = obs_start
        self.obs_end[slot] = obs_end
        self.raw_actions[slot] = raw_action
        self.executed_actions[slot] = executed_action
        self.discounted_reward[slot, 0] = discounted_external_reward
        self.external_reward[slot, 0] = discounted_external_reward
        self.physical_progress[slot, 0] = physical_progress
        self.projection_penalty[slot, 0] = projection_penalty
        self.goals[slot] = goal
        self.next_goals[slot] = next_goal
        self.dones[slot, 0] = float(done)
        self.duration_steps[slot, 0] = float(duration_steps)
        self.idx += 1
        self.full = self.full or self.idx >= self.capacity
        self.constraint_scores[slot] = float(np.clip(constraint_score, 0.0, 1.0))
        self.segment_constraint_mean[slot, 0] = float(np.clip(segment_constraint_mean, 0.0, 1.0))
        self.segment_constraint_max[slot, 0] = float(np.clip(segment_constraint_max, 0.0, 1.0))
        self.projection_scores[slot] = self._normalize_projection_mse(raw_action, executed_action)
        self._set_new_priority(slot, size_before)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        size = len(self)
        indices, is_weights = self._sample_indices(batch_size, size)
        return {
            "obs": legacy.to_tensor(self.obs_start[indices], self.device),
            "next_obs": legacy.to_tensor(self.obs_end[indices], self.device),
            "raw_actions": legacy.to_tensor(self.raw_actions[indices], self.device),
            "executed_actions": legacy.to_tensor(self.executed_actions[indices], self.device),
            "reward_external": legacy.to_tensor(self.external_reward[indices], self.device),
            "physical_progress": legacy.to_tensor(self.physical_progress[indices], self.device),
            "projection_penalty": legacy.to_tensor(self.projection_penalty[indices], self.device),
            "goals": legacy.to_tensor(self.goals[indices], self.device),
            "next_goals": legacy.to_tensor(self.next_goals[indices], self.device),
            "dones": legacy.to_tensor(self.dones[indices], self.device),
            "duration_steps": legacy.to_tensor(self.duration_steps[indices], self.device),
            "constraint_scores": legacy.to_tensor(self.constraint_scores[indices, None], self.device),
            "projection_scores": legacy.to_tensor(self.projection_scores[indices, None], self.device),
            "segment_constraint_mean": legacy.to_tensor(self.segment_constraint_mean[indices], self.device),
            "segment_constraint_max": legacy.to_tensor(self.segment_constraint_max[indices], self.device),
            "indices": torch.as_tensor(indices, dtype=torch.long, device=self.device),
            "is_weights": legacy.to_tensor(is_weights, self.device),
        }


class ManagerReplayBuffer(legacy.ManagerReplayBuffer):
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        data = super().sample(batch_size)
        data["indices"] = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        data["is_weights"] = torch.ones((batch_size, 1), dtype=torch.float32, device=self.device)
        return data


# =============================================================================
# Projection-aware pending transitions
# =============================================================================


_LAST_CONSTRAINT_SCORE = 0.0
_LAST_SLOW_ACTOR_RAW: Optional[np.ndarray] = None
_CURRENT_SLOW_PENDING: Optional["PendingSlowSegment"] = None


@dataclass
class PendingFastTransition:
    obs: np.ndarray
    raw_action: np.ndarray
    executed_action: np.ndarray
    reward_external: float
    projection: float
    goal: np.ndarray
    done: bool
    constraint_score: float = field(init=False)

    def __post_init__(self) -> None:
        self.constraint_score = float(_LAST_CONSTRAINT_SCORE)


@dataclass
class PendingSlowSegment:
    obs_start: np.ndarray
    goal: np.ndarray
    raw_action: np.ndarray
    executed_action: Optional[np.ndarray] = None
    discounted_reward: float = 0.0
    projection_penalty_sum: float = 0.0
    duration_steps: int = 0
    constraint_scores: List[float] = field(default_factory=list)
    solver_failure_seen: bool = False

    def __post_init__(self) -> None:
        global _CURRENT_SLOW_PENDING
        if _LAST_SLOW_ACTOR_RAW is not None and _LAST_SLOW_ACTOR_RAW.shape == self.raw_action.shape:
            self.raw_action = _LAST_SLOW_ACTOR_RAW.copy()
        _CURRENT_SLOW_PENDING = self


def safe_env_step(env: ElectricGasMultiScaleEnv, action: np.ndarray, last_obs: np.ndarray):
    global _LAST_CONSTRAINT_SCORE
    result = _LEGACY_SAFE_ENV_STEP(env, action, last_obs)
    info = result[4]
    _LAST_CONSTRAINT_SCORE = normalized_constraint_score(info, env)
    if _CURRENT_SLOW_PENDING is not None:
        _CURRENT_SLOW_PENDING.constraint_scores.append(_LAST_CONSTRAINT_SCORE)
        _CURRENT_SLOW_PENDING.solver_failure_seen = (
            _CURRENT_SLOW_PENDING.solver_failure_seen or bool(info.get("solver_failed", False))
        )
    return result


def apply_ess_action_guard(env: ElectricGasMultiScaleEnv, slow_action: np.ndarray,
                           cfg: TrainConfig, horizon_steps: int):
    global _LAST_SLOW_ACTOR_RAW
    _LAST_SLOW_ACTOR_RAW = np.asarray(slow_action, dtype=np.float32).copy()
    return _LEGACY_APPLY_ESS_ACTION_GUARD(env, slow_action, cfg, horizon_steps)


def projection_penalty(raw_action: np.ndarray, executed_action: np.ndarray, scale: float) -> float:
    actor_raw = np.asarray(raw_action, dtype=np.float32)
    if _LAST_SLOW_ACTOR_RAW is not None and actor_raw.shape == _LAST_SLOW_ACTOR_RAW.shape:
        actor_raw = _LAST_SLOW_ACTOR_RAW
    return -float(scale * np.mean(np.square(actor_raw - np.asarray(executed_action, dtype=np.float32))))


def aggregate_segment_constraint(scores: Sequence[float], solver_failure_seen: bool) -> Tuple[float, float, float]:
    """保留慢片段峰值风险；求解失败不能被正常步平均稀释。"""

    if scores:
        values = np.clip(np.asarray(scores, dtype=np.float32), 0.0, 1.0)
        mean_score = float(np.mean(values))
        max_score = float(np.max(values))
    else:
        mean_score = 0.0
        max_score = 0.0
    final_score = 0.5 * max_score + 0.5 * mean_score
    if solver_failure_seen:
        final_score = max(final_score, 1.0)
    return mean_score, max_score, float(np.clip(final_score, 0.0, 1.0))


def finalize_fast_transition(pending: PendingFastTransition, next_obs: np.ndarray, next_goal: np.ndarray,
                             fast_agent: "WorkerTD3", buffer: FastReplayBuffer,
                             cfg: TrainConfig) -> Dict[str, float]:
    latent = fast_agent.latent_goal_reward(pending.obs, next_obs, pending.goal, cfg.delta_z_min)
    physical = fast_physical_progress(pending.obs, next_obs, pending.goal)
    total_raw = legacy.build_worker_reward(
        pending.reward_external, latent, physical, pending.projection, cfg
    )
    total, clipped = legacy.clip_reward_value(total_raw, cfg.worker_reward_clip_abs)
    buffer.add(
        pending.obs, next_obs, pending.raw_action, pending.executed_action,
        pending.reward_external, physical, pending.projection, pending.goal, next_goal, pending.done,
        constraint_score=pending.constraint_score, reward_clipped=clipped,
    )
    return {
        "fast_external": pending.reward_external, "fast_latent": latent,
        "fast_physical": physical, "fast_total_raw": total_raw, "fast_total": total,
        "fast_reward_clipped": float(clipped), "projection": pending.projection,
        "fast_constraint_score": pending.constraint_score,
    }


def finalize_slow_segment(pending: PendingSlowSegment, obs_end: np.ndarray, next_goal: np.ndarray,
                          slow_agent: "WorkerTD3", buffer: SlowReplayBuffer, cfg: TrainConfig,
                          done: bool) -> Dict[str, float]:
    global _CURRENT_SLOW_PENDING
    executed = pending.executed_action if pending.executed_action is not None else pending.raw_action.copy()
    latent = slow_agent.latent_goal_reward(pending.obs_start, obs_end, pending.goal, cfg.delta_z_min)
    physical = slow_physical_progress(pending.obs_start, obs_end, pending.goal)
    total_raw = legacy.build_worker_reward(
        pending.discounted_reward, latent, physical, pending.projection_penalty_sum, cfg
    )
    total, clipped = legacy.clip_reward_value(total_raw, cfg.worker_reward_clip_abs)
    segment_mean, segment_max, constraint_score = aggregate_segment_constraint(
        pending.constraint_scores, pending.solver_failure_seen
    )
    buffer.add(
        pending.obs_start, obs_end, pending.raw_action, executed, pending.discounted_reward,
        physical, pending.projection_penalty_sum, pending.goal, next_goal, done,
        pending.duration_steps, constraint_score=constraint_score,
        segment_constraint_mean=segment_mean, segment_constraint_max=segment_max,
        reward_clipped=clipped,
    )
    if _CURRENT_SLOW_PENDING is pending:
        _CURRENT_SLOW_PENDING = None
    return {
        "slow_external": pending.discounted_reward, "slow_latent": latent,
        "slow_physical": physical, "slow_total_raw": total_raw, "slow_total": total,
        "slow_projection": pending.projection_penalty_sum,
        "slow_reward_clipped": float(clipped),
        "slow_segment_constraint_mean": segment_mean,
        "slow_segment_constraint_max": segment_max,
        "slow_constraint_score": constraint_score,
    }


# =============================================================================
# SMDP-TD3 agents
# =============================================================================


def _clip_grad_norm(parameters: Any, max_norm: float) -> float:
    params = [parameter for parameter in parameters if parameter.grad is not None]
    if not params:
        return 0.0
    norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    return float(norm.detach().cpu() if torch.is_tensor(norm) else norm)


def _mean_logs(logs: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not logs:
        return {}
    keys = set().union(*(item.keys() for item in logs))
    return {key: float(np.mean([item[key] for item in logs if key in item])) for key in keys}


class ManagerTD3(legacy.ManagerTD3):
    def __init__(self, obs_dim: int, cfg: TrainConfig, device: torch.device):
        manager_lr = cfg.manager_lr
        if cfg.training_stage == "joint_finetune":
            cfg.manager_lr = cfg.joint_manager_lr
        try:
            super().__init__(obs_dim, cfg, device)
        finally:
            cfg.manager_lr = manager_lr
        self._last_boundary_buffer_size = 0

    def _update_once(self, buffer: ManagerReplayBuffer, batch_size: int) -> Dict[str, float]:
        data = buffer.sample(batch_size)
        obs = legacy.to_tensor(self.normalizer.normalize(data["obs"].cpu().numpy()), self.device)
        next_obs = legacy.to_tensor(self.normalizer.normalize(data["next_obs"].cpu().numpy()), self.device)
        goals, rewards, dones = data["goals"], data["rewards"], data["dones"]
        durations = data["duration_steps"]

        z = self.encoder(obs)
        with torch.no_grad():
            next_z = self.target_encoder(next_obs)
            next_goal = self.target_actor(next_z)
            noise = (torch.randn_like(next_goal) * self.cfg.target_noise).clamp(
                -self.cfg.target_noise_clip, self.cfg.target_noise_clip
            )
            next_goal = legacy.normalize_goal_tensor(next_goal + noise)
            q1_next, q2_next = self.target_critic(next_z, next_goal)
            target_q = build_smdp_target(
                rewards, dones, torch.minimum(q1_next, q2_next), self.cfg.gamma_fast, durations
            )
            if self.cfg.target_q_clip_abs > 0.0:
                target_q = target_q.clamp(-self.cfg.target_q_clip_abs, self.cfg.target_q_clip_abs)

        q1, q2 = self.critic(z, goals)
        critic_loss = F.smooth_l1_loss(q1, target_q) + F.smooth_l1_loss(q2, target_q)
        td_error = 0.5 * ((q1 - target_q).abs() + (q2 - target_q).abs())
        encoder_loss = critic_loss + self.cfg.lambda_latent_norm * z.pow(2).mean()
        self.critic_optim.zero_grad()
        self.encoder_optim.zero_grad()
        encoder_loss.backward()
        critic_grad = _clip_grad_norm(self.critic.parameters(), self.cfg.gradient_clip)
        encoder_grad = _clip_grad_norm(self.encoder.parameters(), self.cfg.gradient_clip)
        self.critic_optim.step()
        self.encoder_optim.step()

        actor_loss_value = 0.0
        actor_grad = 0.0
        if self.total_updates % self.cfg.policy_frequency == 0:
            z_pi = self.encoder(obs).detach()
            actor_loss = -self.critic.q_min(z_pi, self.actor(z_pi)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            actor_grad = _clip_grad_norm(self.actor.parameters(), self.cfg.gradient_clip)
            self.actor_optim.step()
            actor_loss_value = float(actor_loss.detach().cpu())
            legacy.soft_update(self.target_actor, self.actor, self.cfg.tau)
            legacy.soft_update(self.target_critic, self.critic, self.cfg.tau)
            legacy.soft_update(self.target_encoder, self.encoder, self.cfg.tau)
        self.total_updates += 1
        clipped_ratio = float((rewards.abs() >= self.cfg.manager_reward_clip_abs - 1e-6).float().mean())
        return {
            "manager/critic_loss": float(critic_loss.detach().cpu()),
            "manager/actor_loss": actor_loss_value,
            "manager/target_q_mean": float(target_q.mean().detach().cpu()),
            "manager/target_q_std": float(target_q.std(unbiased=False).detach().cpu()),
            "manager/target_q_abs_max": float(target_q.abs().max().detach().cpu()),
            "manager/q1_mean": float(q1.mean().detach().cpu()),
            "manager/q2_mean": float(q2.mean().detach().cpu()),
            "manager/td_error": float(td_error.mean().detach().cpu()),
            "manager/critic_grad_norm": critic_grad,
            "manager/encoder_grad_norm": encoder_grad,
            "manager/actor_grad_norm": actor_grad,
            "manager/reward_clipping_ratio": clipped_ratio,
            "manager/mean_duration_steps": float(durations.mean().detach().cpu()),
        }

    def update(self, buffer: ManagerReplayBuffer, batch_size: int = 0) -> Dict[str, float]:
        del batch_size
        if len(buffer) < max(self.cfg.manager_learning_starts, self.cfg.manager_batch_size):
            return {}
        if len(buffer) == self._last_boundary_buffer_size:
            return {}
        self._last_boundary_buffer_size = len(buffer)
        logs = [self._update_once(buffer, self.cfg.manager_batch_size)
                for _ in range(self.cfg.manager_updates_per_boundary)]
        return _mean_logs(logs)

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["last_boundary_buffer_size"] = self._last_boundary_buffer_size
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        self._last_boundary_buffer_size = int(state.get("last_boundary_buffer_size", 0))


class WorkerTD3(legacy.WorkerTD3):
    def __init__(self, role: str, obs_dim: int, action_dim: int, latent_dim: int,
                 lr: float, cfg: TrainConfig, device: torch.device):
        super().__init__(role, obs_dim, action_dim, latent_dim, lr, cfg, device)
        self._last_boundary_buffer_size = 0

    def select_action(self, obs: np.ndarray, goal: np.ndarray, noise_std: float,
                      deterministic: bool = False) -> np.ndarray:
        # 阶段训练中被冻结的 Worker 仍参与控制，但不应继续注入探索噪声。
        actor_trainable = any(parameter.requires_grad for parameter in self.actor.parameters())
        return super().select_action(
            obs, goal, noise_std if actor_trainable else 0.0,
            deterministic=deterministic or not actor_trainable,
        )

    def _role_parameters(self) -> Tuple[int, int, int]:
        if self.role == "fast":
            return (
                self.cfg.fast_batch_size,
                self.cfg.fast_learning_starts,
                self.cfg.fast_updates_per_step,
            )
        return (
            self.cfg.slow_batch_size,
            self.cfg.slow_learning_starts,
            self.cfg.slow_updates_per_boundary,
        )

    def _target_latent_reward(self, obs: torch.Tensor, next_obs: torch.Tensor,
                              goals: torch.Tensor) -> torch.Tensor:
        """用稳定 target encoder 对采样 transition 动态重算 FuN latent reward。"""

        with torch.no_grad():
            z = self.target_encoder(obs)
            next_z = self.target_encoder(next_obs)
            delta = next_z - z
            direction = legacy.expanded_goal_direction_tensor(goals, self.role, self.latent_dim)
            delta_norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
            cosine = F.cosine_similarity(delta, direction, dim=-1, eps=1e-8).unsqueeze(-1)
            valid = delta_norm >= float(self.cfg.delta_z_min)
            return torch.where(valid, cosine, torch.zeros_like(cosine))

    def _update_once(self, buffer: Any, batch_size: int) -> Dict[str, float]:
        data = buffer.sample(batch_size)
        obs = legacy.to_tensor(self.normalizer.normalize(data["obs"].cpu().numpy()), self.device)
        next_obs = legacy.to_tensor(self.normalizer.normalize(data["next_obs"].cpu().numpy()), self.device)
        raw_actions = data["raw_actions"]
        executed_actions = data["executed_actions"]
        reward_external = data["reward_external"]
        reward_physical = data["physical_progress"]
        reward_projection = data["projection_penalty"]
        dones = data["dones"]
        goals, next_goals = data["goals"], data["next_goals"]
        durations = data.get("duration_steps", torch.ones_like(reward_external))
        is_weights = data.get("is_weights", torch.ones_like(reward_external))
        wg = legacy.worker_goal_tensor(goals, self.role)
        next_wg = legacy.worker_goal_tensor(next_goals, self.role)

        reward_latent = self._target_latent_reward(obs, next_obs, goals)
        reward_raw = (
            self.cfg.alpha_external * reward_external
            + self.cfg.beta_latent * reward_latent
            + self.cfg.beta_physical * reward_physical
            + reward_projection
        )
        if self.cfg.worker_reward_clip_abs > 0.0:
            rewards = reward_raw.clamp(-self.cfg.worker_reward_clip_abs, self.cfg.worker_reward_clip_abs)
            reward_clipped = (rewards != reward_raw).float()
        else:
            rewards = reward_raw
            reward_clipped = torch.zeros_like(rewards)

        z = self.encoder(obs)
        with torch.no_grad():
            next_z = self.target_encoder(next_obs)
            next_actions = self.target_actor(next_z, next_wg)
            noise = (torch.randn_like(next_actions) * self.cfg.target_noise).clamp(
                -self.cfg.target_noise_clip, self.cfg.target_noise_clip
            )
            next_actions = (next_actions + noise).clamp(-1.0, 1.0)
            q1_next, q2_next = self.target_critic(next_z, next_wg, next_actions)
            target_q = build_smdp_target(
                rewards, dones, torch.minimum(q1_next, q2_next), self.cfg.gamma_fast, durations
            )
            if self.cfg.target_q_clip_abs > 0.0:
                target_q = target_q.clamp(-self.cfg.target_q_clip_abs, self.cfg.target_q_clip_abs)

        # Critic 与 Actor 统一在 raw action 域；环境投影属于转移函数。
        q1, q2 = self.critic(z, wg, raw_actions)
        loss1 = F.smooth_l1_loss(q1, target_q, reduction="none")
        loss2 = F.smooth_l1_loss(q2, target_q, reduction="none")
        critic_loss = (is_weights * (loss1 + loss2)).mean()
        td_error = 0.5 * ((q1 - target_q).abs() + (q2 - target_q).abs())

        transition_encoder_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        if self.transition_model is not None:
            with torch.no_grad():
                target_delta_for_encoder = self.target_encoder(next_obs) - z.detach()
            legacy.set_requires_grad(self.transition_model, False)
            try:
                # Transition Model 预测真实 executed action 造成的状态变化。
                pred_delta_for_encoder = self.transition_model(z, executed_actions)
                transition_encoder_loss = F.mse_loss(pred_delta_for_encoder, target_delta_for_encoder)
            finally:
                legacy.set_requires_grad(self.transition_model, True)

        encoder_loss = (
            critic_loss
            + self.cfg.lambda_transition * transition_encoder_loss
            + self.cfg.lambda_latent_norm * z.pow(2).mean()
        )
        self.critic_optim.zero_grad()
        self.encoder_optim.zero_grad()
        encoder_loss.backward()
        critic_grad = _clip_grad_norm(self.critic.parameters(), self.cfg.gradient_clip)
        encoder_grad = _clip_grad_norm(self.encoder.parameters(), self.cfg.gradient_clip)
        self.critic_optim.step()
        self.encoder_optim.step()

        if hasattr(buffer, "update_priorities"):
            buffer.update_priorities(
                data["indices"].detach().cpu().numpy(), td_error.detach().cpu().numpy()
            )

        transition_loss_value = 0.0
        transition_grad = 0.0
        if self.transition_model is not None and self.transition_optim is not None:
            with torch.no_grad():
                z_detached = self.encoder(obs).detach()
                target_delta = self.target_encoder(next_obs) - z_detached
            pred_delta = self.transition_model(z_detached, executed_actions)
            transition_loss = F.mse_loss(pred_delta, target_delta.detach())
            self.transition_optim.zero_grad()
            transition_loss.backward()
            transition_grad = _clip_grad_norm(self.transition_model.parameters(), self.cfg.gradient_clip)
            self.transition_optim.step()
            transition_loss_value = float(transition_loss.detach().cpu())

        actor_loss_value = 0.0
        actor_grad = 0.0
        imitation_loss_value = 0.0
        projection_mask_ratio = 0.0
        behavior_match_ratio = 0.0
        imitation_mask_ratio = 0.0
        imitation_weight_value = projection_imitation_weight(self.cfg, self.total_updates)
        if self.total_updates % self.cfg.policy_frequency == 0:
            z_pi = self.encoder(obs).detach()
            actor_raw_actions = self.actor(z_pi, wg)
            actor_loss = -self.critic.q_min(z_pi, wg, actor_raw_actions).mean()
            actor_loss = actor_loss + self.cfg.worker_action_l2_weight * actor_raw_actions.pow(2).mean()
            projection_mask, behavior_match_mask, imitation_mask = projection_imitation_masks(
                actor_raw_actions, raw_actions, executed_actions,
                self.cfg.projection_imitation_threshold,
                self.cfg.projection_behavior_match_threshold,
            )
            projection_mask_ratio = float(projection_mask.float().mean().detach().cpu())
            behavior_match_ratio = float(behavior_match_mask.float().mean().detach().cpu())
            imitation_mask_ratio = float(imitation_mask.float().mean().detach().cpu())
            if bool(imitation_mask.any()) and imitation_weight_value > 0.0:
                imitation_loss = F.mse_loss(actor_raw_actions[imitation_mask], executed_actions[imitation_mask])
                actor_loss = actor_loss + imitation_weight_value * imitation_loss
                imitation_loss_value = float(imitation_loss.detach().cpu())
            if self.transition_model is not None and self.cfg.reachability_weight > 0.0:
                predicted_delta = self.transition_model(z_pi, actor_raw_actions)
                direction = legacy.expanded_goal_direction_tensor(goals, self.role, self.latent_dim)
                reachability = 1.0 - F.cosine_similarity(
                    predicted_delta, direction, dim=-1, eps=1e-8
                ).mean()
                actor_loss = actor_loss + self.cfg.reachability_weight * reachability
            self.actor_optim.zero_grad()
            actor_loss.backward()
            actor_grad = _clip_grad_norm(self.actor.parameters(), self.cfg.gradient_clip)
            self.actor_optim.step()
            actor_loss_value = float(actor_loss.detach().cpu())
            legacy.soft_update(self.target_actor, self.actor, self.cfg.tau)
            legacy.soft_update(self.target_critic, self.critic, self.cfg.tau)
            legacy.soft_update(self.target_encoder, self.encoder, self.cfg.tau)

        self.total_updates += 1
        prefix = f"{self.role}/"
        reward_clipping_ratio = float(reward_clipped.mean().detach().cpu())
        return {
            prefix + "critic_loss": float(critic_loss.detach().cpu()),
            prefix + "actor_loss": actor_loss_value,
            prefix + "projection_imitation_loss": imitation_loss_value,
            prefix + "projection_mask_ratio": projection_mask_ratio,
            prefix + "behavior_match_ratio": behavior_match_ratio,
            prefix + "projection_imitation_mask_ratio": imitation_mask_ratio,
            prefix + "projection_imitation_weight": imitation_weight_value,
            prefix + "sample_projection_mse": float((raw_actions - executed_actions).pow(2).mean().detach().cpu()),
            prefix + "transition_loss": transition_loss_value,
            prefix + "transition_encoder_loss": float(transition_encoder_loss.detach().cpu()),
            prefix + "target_q_mean": float(target_q.mean().detach().cpu()),
            prefix + "target_q_std": float(target_q.std(unbiased=False).detach().cpu()),
            prefix + "target_q_abs_max": float(target_q.abs().max().detach().cpu()),
            prefix + "q1_mean": float(q1.mean().detach().cpu()),
            prefix + "q2_mean": float(q2.mean().detach().cpu()),
            prefix + "td_error": float(td_error.mean().detach().cpu()),
            prefix + "critic_grad_norm": critic_grad,
            prefix + "encoder_grad_norm": encoder_grad,
            prefix + "actor_grad_norm": actor_grad,
            prefix + "transition_grad_norm": transition_grad,
            prefix + "reward_clipping_ratio": reward_clipping_ratio,
            prefix + "reward_external_mean": float(reward_external.mean().detach().cpu()),
            prefix + "reward_latent_mean": float(reward_latent.mean().detach().cpu()),
            prefix + "reward_physical_mean": float(reward_physical.mean().detach().cpu()),
            prefix + "reward_projection_mean": float(reward_projection.mean().detach().cpu()),
            prefix + "batch_reward_mean": float(rewards.mean().detach().cpu()),
            prefix + "mean_duration_steps": float(durations.mean().detach().cpu()),
            prefix + "critic_raw_action_mean": float(raw_actions.mean().detach().cpu()),
            prefix + "executed_action_mean": float(executed_actions.mean().detach().cpu()),
        }

    def update(self, buffer: Any, batch_size: int = 0, gamma: float = 0.0) -> Dict[str, float]:
        del batch_size, gamma
        role_batch, learning_starts, update_count = self._role_parameters()
        if len(buffer) < max(learning_starts, role_batch):
            return {}
        if self.role == "slow":
            if len(buffer) == self._last_boundary_buffer_size:
                return {}
            self._last_boundary_buffer_size = len(buffer)
        logs = [self._update_once(buffer, role_batch) for _ in range(update_count)]
        return _mean_logs(logs)

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["last_boundary_buffer_size"] = self._last_boundary_buffer_size
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        self._last_boundary_buffer_size = int(state.get("last_boundary_buffer_size", 0))


# =============================================================================
# Agent construction, freezing, checkpoint and evaluation
# =============================================================================


def _set_agent_trainable(agent: Any, enabled: bool) -> None:
    for name in ("encoder", "actor", "critic", "transition_model"):
        module = getattr(agent, name, None)
        if module is not None:
            legacy.set_requires_grad(module, enabled)


def validate_time_scale(env: ElectricGasMultiScaleEnv, cfg: TrainConfig) -> None:
    """在任何训练或评估交互前强制校验三层时钟。"""

    env_slow = int(env.config.time.slow_action_interval_steps)
    actual = {
        "slow_interval": int(cfg.slow_interval),
        "manager_interval": int(cfg.manager_interval),
        "episode_steps": int(cfg.episode_steps),
        "env_slow_interval": env_slow,
    }
    if cfg.slow_interval <= 0 or cfg.manager_interval <= 0 or cfg.episode_steps <= 0:
        raise ValueError(f"Time-scale values must be > 0; actual={actual}")
    if int(cfg.slow_interval) != env_slow:
        raise ValueError(
            "slow_interval does not match environment slow-action clock: "
            f"actual cfg.slow_interval={cfg.slow_interval}, expected env={env_slow}"
        )
    if int(cfg.manager_interval) % int(cfg.slow_interval) != 0:
        raise ValueError(
            "manager_interval must be a positive integer multiple of slow_interval so a goal "
            "cannot change inside an unfinished slow segment: "
            f"actual manager_interval={cfg.manager_interval}, expected k*{cfg.slow_interval} (k>=1)"
        )


def validate_legacy_api() -> None:
    """尽早拒绝与本优化层调用契约不一致的 legacy 文件。"""

    expected_parameters = {
        "run_training": (legacy.run_training, ("cfg",)),
        "safe_env_step": (legacy.safe_env_step, ("env", "action", "last_obs")),
        "evaluate_policy": (legacy.evaluate_policy, ("agents", "cfg", "episodes", "max_steps", "seed")),
        "load_checkpoint": (legacy.load_checkpoint, ("path", "agents", "map_location", "policy_only")),
    }
    problems: List[str] = []
    for name, (function, names) in expected_parameters.items():
        actual = tuple(inspect.signature(function).parameters)
        if actual[:len(names)] != names:
            problems.append(f"{name}{actual} expected prefix {names}")
    legacy_version = getattr(legacy, "ENV_MODEL_VERSION", None)
    if legacy_version != ENV_MODEL_VERSION:
        problems.append(f"environment version actual={legacy_version!r}, expected={ENV_MODEL_VERSION!r}")
    if problems:
        raise RuntimeError("Incompatible legacy API: " + " / ".join(problems))


def validate_environment_contract(env: ElectricGasMultiScaleEnv, cfg: TrainConfig) -> None:
    validate_time_scale(env, cfg)
    action_shape = tuple(getattr(env.action_space, "shape", ()))
    observation_shape = tuple(getattr(env.observation_space, "shape", ()))
    problems: List[str] = []
    if env.action_dim != env.slow_action_dim + env.fast_action_dim:
        problems.append(
            f"action_dim={env.action_dim}, slow+fast={env.slow_action_dim + env.fast_action_dim}"
        )
    if action_shape != (env.action_dim,):
        problems.append(f"action_space.shape={action_shape}, expected={(env.action_dim,)}")
    if observation_shape != (env.global_state_dim,):
        problems.append(f"observation_space.shape={observation_shape}, expected={(env.global_state_dim,)}")
    if problems:
        raise ValueError("Environment action/observation contract is incompatible: " + " / ".join(problems))


def build_agents(env: ElectricGasMultiScaleEnv, cfg: TrainConfig, device: torch.device) -> AgentBundle:
    validate_environment_contract(env, cfg)
    obs, _ = env.reset(seed=cfg.seed)
    builder = ObservationBuilder(env, cfg.manager_interval)
    manager_obs = builder.manager_obs(obs)
    fast_obs = builder.fast_obs(0, obs)
    slow_obs = builder.slow_obs(obs)
    slow_lr = cfg.joint_worker_lr if cfg.training_stage == "joint_finetune" else cfg.slow_lr
    fast_lr = cfg.joint_worker_lr if cfg.training_stage == "joint_finetune" else cfg.fast_lr
    agents = AgentBundle(
        manager=ManagerTD3(manager_obs.size, cfg, device),
        slow=WorkerTD3("slow", slow_obs.size, env.slow_action_dim, cfg.slow_latent_dim, slow_lr, cfg, device),
        fast=WorkerTD3("fast", fast_obs.size, env.fast_action_dim, cfg.fast_latent_dim, fast_lr, cfg, device),
    )
    flags = legacy.stage_flags(cfg.training_stage)
    _set_agent_trainable(agents.manager, flags["manager"])
    _set_agent_trainable(agents.slow, flags["slow"])
    _set_agent_trainable(agents.fast, flags["fast"])
    agents.environment_metadata = {
        "env_model_version": ENV_MODEL_VERSION,
        "global_state_dim": int(env.global_state_dim),
        "slow_action_dim": int(env.slow_action_dim),
        "fast_action_dim": int(env.fast_action_dim),
        "total_action_dim": int(env.action_dim),
    }
    return agents


def _reset_optimizer_lr(optimizer: Any, learning_rate: float) -> None:
    if optimizer is None:
        return
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)


def _apply_stage_learning_rates(agents: AgentBundle, cfg: TrainConfig) -> None:
    manager_lr = cfg.joint_manager_lr if cfg.training_stage == "joint_finetune" else cfg.manager_lr
    slow_lr = cfg.joint_worker_lr if cfg.training_stage == "joint_finetune" else cfg.slow_lr
    fast_lr = cfg.joint_worker_lr if cfg.training_stage == "joint_finetune" else cfg.fast_lr
    for optimizer in (agents.manager.encoder_optim, agents.manager.actor_optim, agents.manager.critic_optim):
        _reset_optimizer_lr(optimizer, manager_lr)
    for agent, lr in ((agents.slow, slow_lr), (agents.fast, fast_lr)):
        for optimizer in (agent.encoder_optim, agent.actor_optim, agent.critic_optim, agent.transition_optim):
            _reset_optimizer_lr(optimizer, lr)


def _critical_optimization_config(cfg: TrainConfig) -> Dict[str, Any]:
    names = (
        "gamma_fast", "manager_latent_dim", "slow_latent_dim", "fast_latent_dim",
        "hidden_dim", "manager_hidden_dim", "critic_hidden_dim", "use_transition_model",
        "alpha_external", "beta_latent", "beta_physical", "lambda_projection",
        "use_prioritized_replay", "priority_alpha", "constraint_priority_weight",
        "projection_priority_weight", "projection_behavior_match_threshold",
    )
    return {name: getattr(cfg, name) for name in names}


def checkpoint_metadata(agents: AgentBundle, cfg: TrainConfig) -> Dict[str, Any]:
    env_meta = getattr(agents, "environment_metadata", {})
    return {
        "checkpoint_schema_version": 2,
        "training_stage": cfg.training_stage,
        "env_model_version": ENV_MODEL_VERSION,
        "slow_interval": int(cfg.slow_interval),
        "manager_interval": int(cfg.manager_interval),
        "global_state_dim": int(env_meta.get("global_state_dim", agents.manager.obs_dim)),
        "manager_observation_dim": agents.manager.obs_dim,
        "slow_observation_dim": agents.slow.obs_dim,
        "fast_observation_dim": agents.fast.obs_dim,
        "slow_action_dim": agents.slow.action_dim,
        "fast_action_dim": agents.fast.action_dim,
        "total_action_dim": agents.slow.action_dim + agents.fast.action_dim,
        "goal_dim": GOAL_DIM,
        "optimization_config": _critical_optimization_config(cfg),
    }


def save_checkpoint(path: Path, cfg: TrainConfig, agents: AgentBundle, episode: int,
                    global_step: int, best_return: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **checkpoint_metadata(agents, cfg),
        "observation_dim": agents.manager.obs_dim,
        "config": asdict(cfg),
        "manager": agents.manager.state_dict(),
        "slow": agents.slow.state_dict(),
        "fast": agents.fast.state_dict(),
        "episode": episode,
        "global_step": global_step,
        "best_return": best_return,
    }
    torch.save(payload, str(path))


def validate_checkpoint_compatibility(payload: Dict[str, Any], agents: AgentBundle) -> None:
    cfg = agents.fast.cfg
    expected = checkpoint_metadata(agents, cfg)
    required = (
        "env_model_version", "manager_observation_dim", "slow_observation_dim",
        "fast_observation_dim", "slow_action_dim", "fast_action_dim", "total_action_dim", "goal_dim",
    )
    problems: List[str] = []
    for key in required:
        if key not in payload:
            problems.append(f"missing required metadata {key}")
        elif payload[key] != expected[key]:
            problems.append(f"{key}: checkpoint={payload[key]!r}, current={expected[key]!r}")

    old_config = payload.get("config", {}) if isinstance(payload.get("config", {}), Mapping) else {}
    for key in ("slow_interval", "manager_interval"):
        actual = payload.get(key, old_config.get(key))
        if actual is None:
            LOGGER.warning("Legacy checkpoint has no %s; using current value %s as explicit migration fallback.",
                           key, expected[key])
        elif int(actual) != int(expected[key]):
            problems.append(f"{key}: checkpoint={actual!r}, current={expected[key]!r}")
    if "global_state_dim" not in payload:
        LOGGER.warning("Legacy checkpoint has no global_state_dim; network observation dimensions remain strictly checked.")
    elif int(payload["global_state_dim"]) != int(expected["global_state_dim"]):
        problems.append(
            f"global_state_dim: checkpoint={payload['global_state_dim']!r}, current={expected['global_state_dim']!r}"
        )
    if "optimization_config" not in payload:
        LOGGER.warning("Legacy checkpoint has no optimization_config; missing fields use TrainConfig migration defaults.")
    else:
        architecture_keys = (
            "manager_latent_dim", "slow_latent_dim", "fast_latent_dim", "hidden_dim",
            "manager_hidden_dim", "critic_hidden_dim", "use_transition_model",
        )
        saved_optimization = payload["optimization_config"]
        for key in architecture_keys:
            if key in saved_optimization and saved_optimization[key] != expected["optimization_config"][key]:
                problems.append(
                    f"optimization_config.{key}: checkpoint={saved_optimization[key]!r}, "
                    f"current={expected['optimization_config'][key]!r}"
                )
    if problems:
        raise ValueError("Incompatible checkpoint: " + " / ".join(problems))


def load_checkpoint(path: str, agents: AgentBundle, map_location: torch.device,
                    policy_only: bool = False) -> Dict[str, Any]:
    payload = legacy.trusted_torch_load(path, map_location=map_location)
    validate_checkpoint_compatibility(payload, agents)
    if policy_only:
        legacy.load_agent_policy_state(agents.manager, payload["manager"])
        legacy.load_agent_policy_state(agents.slow, payload["slow"])
        legacy.load_agent_policy_state(agents.fast, payload["fast"])
    else:
        agents.manager.load_state_dict(payload["manager"])
        agents.slow.load_state_dict(payload["slow"])
        agents.fast.load_state_dict(payload["fast"])
    _apply_stage_learning_rates(agents, agents.fast.cfg)
    return payload


_STAGE_BEST_FILES = {
    "fast_pretrain": "best_fast.pt",
    "slow_pretrain": "best_slow.pt",
    "manager_train": "best_manager.pt",
    "joint_finetune": "best_joint.pt",
}


def save_best_files(root: Path, agents: AgentBundle, cfg: TrainConfig, episode: int,
                    global_step: int, best_return: float) -> None:
    filename = _STAGE_BEST_FILES.get(cfg.training_stage)
    if filename is None:
        raise ValueError(f"Cannot choose best checkpoint filename for stage={cfg.training_stage!r}")
    save_checkpoint(root / "latest_checkpoint.pt", cfg, agents, episode, global_step, best_return)
    save_checkpoint(root / filename, cfg, agents, episode, global_step, best_return)


def evaluation_control_spec(stage: str) -> Dict[str, str]:
    specs = {
        "fast_pretrain": {"manager": "fixed_goal", "slow": "rule", "fast": "deterministic_policy"},
        "slow_pretrain": {"manager": "fixed_goal", "slow": "deterministic_policy", "fast": "deterministic_policy"},
        "manager_train": {"manager": "deterministic_policy", "slow": "deterministic_policy", "fast": "deterministic_policy"},
        "joint_finetune": {"manager": "deterministic_policy", "slow": "deterministic_policy", "fast": "deterministic_policy"},
    }
    if stage not in specs:
        raise ValueError(f"Unknown evaluation stage: {stage}")
    return specs[stage]


def _resolve_eval_seeds(cfg: TrainConfig, episodes: int, seed: int) -> Tuple[int, ...]:
    if episodes <= 0:
        raise ValueError(f"evaluation episodes must be > 0, got {episodes}")
    configured = tuple(int(item) for item in cfg.eval_seeds)
    if configured:
        anchor = configured[0]
        resolved = [int(seed) + (item - anchor) for item in configured[:episodes]]
    else:
        resolved = []
    next_offset = 0
    while len(resolved) < episodes:
        candidate = int(seed) + next_offset
        next_offset += 1
        if candidate not in resolved:
            resolved.append(candidate)
    return tuple(resolved)


@contextmanager
def _deterministic_evaluation_mode(agents: AgentBundle):
    modules: List[torch.nn.Module] = []
    normalizers: List[RunningMeanStd] = []
    for agent in (agents.manager, agents.slow, agents.fast):
        normalizers.append(agent.normalizer)
        for name in ("encoder", "target_encoder", "actor", "target_actor", "critic", "target_critic",
                     "transition_model"):
            module = getattr(agent, name, None)
            if isinstance(module, torch.nn.Module):
                modules.append(module)
    module_modes = [module.training for module in modules]
    normalizer_modes = [normalizer.training for normalizer in normalizers]
    for module in modules:
        module.eval()
    for normalizer in normalizers:
        normalizer.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for module, mode in zip(modules, module_modes):
            module.train(mode)
        for normalizer, mode in zip(normalizers, normalizer_modes):
            normalizer.train() if mode else normalizer.eval()


def evaluate_policy(agents: AgentBundle, cfg: TrainConfig, episodes: int = 1,
                    max_steps: int = EPISODE_STEPS, seed: int = 12345) -> Dict[str, float]:
    evaluation_control_spec(cfg.training_stage)
    probe_env = ElectricGasMultiScaleEnv()
    validate_environment_contract(probe_env, cfg)
    seeds = _resolve_eval_seeds(cfg, int(episodes), int(seed))
    global _LAST_CONSTRAINT_SCORE, _LAST_SLOW_ACTOR_RAW, _CURRENT_SLOW_PENDING
    saved_pending = (_LAST_CONSTRAINT_SCORE, _LAST_SLOW_ACTOR_RAW, _CURRENT_SLOW_PENDING)
    try:
        _LAST_CONSTRAINT_SCORE, _LAST_SLOW_ACTOR_RAW, _CURRENT_SLOW_PENDING = 0.0, None, None
        with _deterministic_evaluation_mode(agents):
            results = [
                _LEGACY_EVALUATE_POLICY(agents, cfg, episodes=1, max_steps=max_steps, seed=item)
                for item in seeds
            ]
    finally:
        _LAST_CONSTRAINT_SCORE, _LAST_SLOW_ACTOR_RAW, _CURRENT_SLOW_PENDING = saved_pending
    returns = np.asarray([item["mean_return"] for item in results], dtype=float)
    total_steps = float(sum(item["steps"] for item in results))
    aggregate: Dict[str, float] = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "solver_failures": float(sum(item["solver_failures"] for item in results)),
        "steps": total_steps,
        "episodes": float(len(results)),
        "power_success_rate": float(sum(item["power_success_rate"] * item["steps"] for item in results) / max(total_steps, 1.0)),
        "gas_success_rate": float(sum(item["gas_success_rate"] * item["steps"] for item in results) / max(total_steps, 1.0)),
        "mean_voltage_rms_deviation_pu": float(np.mean([item["mean_voltage_rms_deviation_pu"] for item in results])),
        "mean_gas_pressure_rms_deviation_bar": float(np.mean([item["mean_gas_pressure_rms_deviation_bar"] for item in results])),
    }
    component_keys = set().union(*(item.keys() for item in results))
    for key in component_keys:
        if key.endswith("_cost"):
            aggregate[key] = float(sum(item.get(key, 0.0) for item in results))
    return aggregate


_RUNTIME_ACTIVE = False


def _reset_runtime_state() -> None:
    global _ACTIVE_CONFIG, _LAST_CONSTRAINT_SCORE, _LAST_SLOW_ACTOR_RAW, _CURRENT_SLOW_PENDING
    _ACTIVE_CONFIG = None
    _LAST_CONSTRAINT_SCORE = 0.0
    _LAST_SLOW_ACTOR_RAW = None
    _CURRENT_SLOW_PENDING = None


@contextmanager
def runtime_overrides(cfg: TrainConfig):
    """临时安装 optimized 实现，并在成功或异常后完整恢复 legacy 模块。"""

    global _ACTIVE_CONFIG, _RUNTIME_ACTIVE
    if _RUNTIME_ACTIVE:
        raise RuntimeError("Concurrent or nested optimized training runs are not allowed")
    validate_legacy_api()
    replacements = {
        "TrainConfig": TrainConfig, "FastReplayBuffer": FastReplayBuffer,
        "SlowReplayBuffer": SlowReplayBuffer, "ManagerReplayBuffer": ManagerReplayBuffer,
        "PendingFastTransition": PendingFastTransition, "PendingSlowSegment": PendingSlowSegment,
        "ManagerTD3": ManagerTD3, "WorkerTD3": WorkerTD3, "build_agents": build_agents,
        "fast_physical_progress": fast_physical_progress, "slow_physical_progress": slow_physical_progress,
        "fixed_manager_goal": fixed_manager_goal, "safe_env_step": safe_env_step,
        "apply_ess_action_guard": apply_ess_action_guard, "projection_penalty": projection_penalty,
        "finalize_fast_transition": finalize_fast_transition,
        "finalize_slow_segment": finalize_slow_segment, "load_checkpoint": load_checkpoint,
        "evaluate_policy": evaluate_policy, "checkpoint_metadata": checkpoint_metadata,
        "validate_checkpoint_compatibility": validate_checkpoint_compatibility,
        "save_checkpoint": save_checkpoint, "save_best_files": save_best_files,
    }
    original = {name: getattr(legacy, name) for name in replacements}
    _RUNTIME_ACTIVE = True
    _ACTIVE_CONFIG = cfg
    _reset_runtime_state()
    _ACTIVE_CONFIG = cfg
    try:
        for name, value in replacements.items():
            setattr(legacy, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(legacy, name, value)
        _reset_runtime_state()
        _RUNTIME_ACTIVE = False


def run_training(cfg: TrainConfig) -> Dict[str, Any]:
    with runtime_overrides(cfg):
        return _LEGACY_RUN_TRAINING(cfg)


def run_all_stages(cfg: TrainConfig) -> Dict[str, Any]:
    stages = (
        ("fast_pretrain", cfg.fast_pretrain_episodes),
        ("slow_pretrain", cfg.slow_pretrain_episodes),
        ("manager_train", cfg.manager_train_episodes),
        ("joint_finetune", cfg.joint_finetune_episodes),
    )
    root = Path(cfg.checkpoint_dir) / ("optimized_all_stages_" + time.strftime("%Y%m%d_%H%M%S"))
    previous_checkpoint = cfg.load_checkpoint
    initial_checkpoint = cfg.load_checkpoint
    results: Dict[str, Any] = {"stages": []}
    for stage, count in stages:
        if count <= 0:
            continue
        stage_cfg = copy.deepcopy(cfg)
        stage_cfg.training_stage = stage
        stage_cfg.episodes = int(count)
        stage_cfg.checkpoint_dir = str(root / stage)
        stage_cfg.load_checkpoint = previous_checkpoint
        stage_cfg.load_policy_only = bool(
            cfg.load_policy_only
            and initial_checkpoint
            and previous_checkpoint == initial_checkpoint
        )
        LOGGER.info("Starting optimized stage %s for %s episodes", stage, count)
        result = run_training(stage_cfg)
        result["stage"] = stage
        results["stages"].append(result)
        run_root = Path(result["run_root"])
        candidate = run_root / _STAGE_BEST_FILES[stage]
        previous_checkpoint = str(candidate if candidate.exists() else run_root / "latest_checkpoint.pt")
    results["latest_checkpoint"] = previous_checkpoint
    return results


# =============================================================================
# Tests
# =============================================================================


def _assert_finite(logs: Mapping[str, float]) -> None:
    for key, value in logs.items():
        assert np.isfinite(value), f"{key} is NaN/Inf"


def _test_smdp_discount() -> None:
    gamma = 0.99
    durations = torch.tensor([[1.0], [20.0], [40.0], [7.0]])
    rewards = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    dones = torch.tensor([[0.0], [0.0], [0.0], [1.0]])
    q_next = torch.full_like(rewards, 10.0)
    target = build_smdp_target(rewards, dones, q_next, gamma, durations)
    assert torch.allclose(target[:3], rewards[:3] + torch.pow(torch.tensor(gamma), durations[:3]) * 10.0)
    assert torch.allclose(target[3], rewards[3]), "提前结束片段不应 bootstrap"


def _test_projection_schedule_and_mask() -> None:
    cfg = TrainConfig()
    assert math.isclose(projection_imitation_weight(cfg, 0), 0.10)
    assert math.isclose(projection_imitation_weight(cfg, cfg.projection_imitation_decay_updates), 0.01)
    historical_raw = torch.tensor([[0.5, -0.5], [0.5, -0.5], [0.0, 0.0]])
    current = torch.tensor([[0.52, -0.52], [-0.5, 0.5], [0.0, 0.0]], requires_grad=True)
    executed = torch.zeros_like(historical_raw)
    projection_mask, behavior_mask, imitation_mask = projection_imitation_masks(
        current, historical_raw, executed, 0.01, 0.05
    )
    assert projection_mask.tolist() == [True, True, False]
    assert behavior_mask.tolist() == [True, False, True]
    assert imitation_mask.tolist() == [True, False, False]
    empty_loss = torch.zeros((), dtype=torch.float32) if not bool((imitation_mask & False).any()) else current.sum()
    assert torch.isfinite(empty_loss) and float(empty_loss) == 0.0


def _test_physical_features() -> None:
    fast = np.zeros(PHYSICAL_LAYOUT.fast_required_size, dtype=np.float32)
    fast[PHYSICAL_LAYOUT.fast_renewable_available] = 0.8
    fast[PHYSICAL_LAYOUT.fast_renewable_actual] = 0.6
    fast_features = PHYSICAL_FEATURES.extract_fast(fast)
    slow = np.zeros(PHYSICAL_LAYOUT.slow_required_size, dtype=np.float32)
    slow[PHYSICAL_LAYOUT.slow_soc] = np.array([0.3, 0.5, 0.7], dtype=np.float32)
    slow[PHYSICAL_LAYOUT.slow_source_loading] = 0.5
    slow_features = PHYSICAL_FEATURES.extract_slow(slow)
    assert fast_features.shape == (8,) and slow_features.shape == (8,)
    assert np.all(np.abs(fast_features) <= 1.0) and np.all(np.abs(slow_features) <= 1.0)
    target = np.ones(8, dtype=np.float32)
    for index in range(8):
        before = np.zeros(8, dtype=np.float32)
        after = before.copy()
        after[index] = 1.0
        mask = PHYSICAL_FEATURES.FAST_MASK if index < 4 else PHYSICAL_FEATURES.SLOW_MASK
        progress = PHYSICAL_FEATURES.distance(before, target, mask) - PHYSICAL_FEATURES.distance(after, target, mask)
        assert progress > 0.0, f"physical goal feature {index} was not used"


def _test_time_scale_validation() -> None:
    env = ElectricGasMultiScaleEnv()
    validate_environment_contract(env, TrainConfig(slow_interval=20, manager_interval=40, episode_steps=40))
    try:
        validate_time_scale(env, TrainConfig(slow_interval=10, manager_interval=40, episode_steps=40))
    except ValueError as exc:
        assert "actual cfg.slow_interval=10" in str(exc) and "expected env=20" in str(exc)
    else:
        raise AssertionError("slow interval mismatch must fail")
    try:
        validate_time_scale(env, TrainConfig(slow_interval=20, manager_interval=30, episode_steps=40))
    except ValueError as exc:
        assert "actual manager_interval=30" in str(exc) and "k*20" in str(exc)
    else:
        raise AssertionError("manager interval inside slow segment must fail")
    try:
        TrainConfig(reachability_weight=0.1)
    except ValueError as exc:
        assert "differentiable projection model" in str(exc)
    else:
        raise AssertionError("unsafe raw-action reachability must be rejected")


def _test_cli_contract() -> None:
    cfg = parse_args([
        "--batch-size", "8", "--learning-starts", "6", "--updates-per-step", "2",
        "--projection-imitation-weight", "0.03",
        "--projection-behavior-match-threshold", "0.04",
    ])
    assert (cfg.fast_batch_size, cfg.slow_batch_size, cfg.manager_batch_size) == (8, 8, 8)
    assert (cfg.fast_learning_starts, cfg.slow_learning_starts, cfg.manager_learning_starts) == (6, 6, 6)
    assert (cfg.fast_updates_per_step, cfg.slow_updates_per_boundary, cfg.manager_updates_per_boundary) == (2, 2, 2)
    assert cfg.projection_imitation_initial_weight == cfg.projection_imitation_final_weight == 0.03
    assert cfg.projection_behavior_match_threshold == 0.04
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parse_args(["--slow-update-interval-steps", "5"])
    assert any("deprecated and ignored" in str(item.message) for item in caught)


def _test_per_initialization_and_sampling(device: torch.device) -> None:
    cfg = TrainConfig(use_prioritized_replay=True, priority_alpha=1.0)
    buffer = FastReplayBuffer(32, 4, 2, GOAL_DIM, device, cfg)
    goal = fixed_manager_goal()
    zero_obs = np.zeros(4, dtype=np.float32)
    raw = np.ones(2, dtype=np.float32)
    executed = -np.ones(2, dtype=np.float32)
    inserted_priorities: List[float] = []
    for _ in range(10):
        buffer.add(zero_obs, zero_obs, raw, executed, -1.0, 0.0, -0.1, goal, goal, False,
                   constraint_score=1.0)
        inserted_priorities.append(float(buffer.priorities[len(buffer) - 1]))
    assert np.allclose(inserted_priorities, inserted_priorities[0]), (
        "high-constraint insertions recursively amplified initial priority"
    )
    assert np.allclose(buffer.td_priorities[:10], 1.0)

    probability_buffer = FastReplayBuffer(8, 4, 2, GOAL_DIM, device, cfg)
    for score in (0.0, 1.0):
        probability_buffer.add(zero_obs, zero_obs, np.zeros(2, dtype=np.float32),
                               np.zeros(2, dtype=np.float32), -1.0, 0.0, 0.0,
                               goal, goal, False, constraint_score=score)
    counts = np.zeros(2, dtype=np.int64)
    for _ in range(1000):
        index, _ = probability_buffer._sample_indices(1, 2)
        counts[index[0]] += 1
    assert counts[1] > counts[0], f"constraint-aware sampling ineffective: counts={counts.tolist()}"
    probability_buffer.update_priorities(np.array([0, 0, 1]), np.array([1.0, 10.0, 0.5]))
    assert math.isclose(float(probability_buffer.td_priorities[0]), math.log1p(10.0), rel_tol=1e-6)
    assert np.all(np.isfinite(probability_buffer.priorities[:2]))


def _fill_worker_buffer(buffer: Any, obs_dim: int, action_dim: int, count: int,
                        goal: np.ndarray, slow: bool = False) -> None:
    for index in range(count):
        obs = np.full(obs_dim, 0.01 * index, dtype=np.float32)
        next_obs = obs + 0.02
        raw = np.full(action_dim, 0.75, dtype=np.float32)
        executed = np.full(action_dim, -0.25, dtype=np.float32)
        if slow:
            buffer.add(obs, next_obs, raw, executed, -1.0, 0.1, -0.1, goal, goal, False, 7,
                       constraint_score=0.5, reward_clipped=False)
        else:
            buffer.add(obs, next_obs, raw, executed, -1.0, 0.1, -0.1, goal, goal, False,
                       constraint_score=0.5, reward_clipped=False)


def _test_raw_critic_executed_transition_and_per(device: torch.device) -> None:
    cfg = TrainConfig(
        hidden_dim=32, critic_hidden_dim=32, fast_latent_dim=8,
        fast_batch_size=4, fast_learning_starts=4, fast_updates_per_step=1,
        use_transition_model=True, policy_frequency=100,
    )
    buffer = FastReplayBuffer(32, 10, 2, GOAL_DIM, device, cfg)
    goal = fixed_manager_goal()
    _fill_worker_buffer(buffer, 10, 2, 8, goal)
    worker = WorkerTD3("fast", 10, 2, 8, 1e-3, cfg, device)
    critic_actions: List[torch.Tensor] = []
    transition_actions: List[torch.Tensor] = []
    latent_recompute_calls: List[int] = []
    original_critic_forward = worker.critic.forward
    original_transition_forward = worker.transition_model.forward  # type: ignore[union-attr]
    original_latent_reward = worker._target_latent_reward

    def critic_forward(z: torch.Tensor, worker_goal: torch.Tensor, action: torch.Tensor):
        critic_actions.append(action.detach().cpu())
        return original_critic_forward(z, worker_goal, action)

    def transition_forward(z: torch.Tensor, action: torch.Tensor):
        transition_actions.append(action.detach().cpu())
        return original_transition_forward(z, action)

    def latent_reward(obs: torch.Tensor, next_obs: torch.Tensor, goals: torch.Tensor):
        latent_recompute_calls.append(obs.shape[0])
        return original_latent_reward(obs, next_obs, goals)

    worker.critic.forward = critic_forward  # type: ignore[method-assign]
    worker.transition_model.forward = transition_forward  # type: ignore[union-attr,method-assign]
    worker._target_latent_reward = latent_reward  # type: ignore[method-assign]
    logs = worker.update(buffer)
    _assert_finite(logs)
    assert torch.allclose(critic_actions[0], torch.full_like(critic_actions[0], 0.75))
    assert all(torch.allclose(action, torch.full_like(action, -0.25)) for action in transition_actions)
    sampled = buffer.sample(4)
    assert "indices" in sampled and "is_weights" in sampled
    assert "rewards" not in sampled and "reward_intrinsic" not in sampled
    assert latent_recompute_calls, "sampled transitions must recompute latent reward with target encoder"
    for key in ("fast/reward_external_mean", "fast/reward_latent_mean", "fast/reward_physical_mean",
                "fast/reward_projection_mean", "fast/batch_reward_mean"):
        assert key in logs and np.isfinite(logs[key])
    old = buffer.priorities.copy()
    buffer.update_priorities(np.array([0, 1]), np.array([10.0, 0.01]))
    assert not np.allclose(old[:2], buffer.priorities[:2])
    assert buffer.priorities[0] > buffer.priorities[1]


def _test_independent_parameters_and_slow_duration(device: torch.device) -> None:
    cfg = TrainConfig(
        hidden_dim=32, critic_hidden_dim=32, slow_latent_dim=8,
        fast_batch_size=7, slow_batch_size=3, manager_batch_size=2,
        fast_learning_starts=11, slow_learning_starts=3, manager_learning_starts=2,
        slow_updates_per_boundary=2, manager_updates_per_boundary=4,
    )
    assert (cfg.fast_batch_size, cfg.slow_batch_size, cfg.manager_batch_size) == (7, 3, 2)
    assert (cfg.fast_learning_starts, cfg.slow_learning_starts, cfg.manager_learning_starts) == (11, 3, 2)
    goal = fixed_manager_goal()
    buffer = SlowReplayBuffer(16, 12, 2, GOAL_DIM, device, cfg)
    _fill_worker_buffer(buffer, 12, 2, 3, goal, slow=True)
    worker = WorkerTD3("slow", 12, 2, 8, 1e-3, cfg, device)
    logs = worker.update(buffer)
    _assert_finite(logs)
    assert math.isclose(logs["slow/mean_duration_steps"], 7.0)
    assert worker.total_updates == 2
    assert worker.update(buffer) == {}, "没有新边界样本时不应重复更新 Slow"

    manager_buffer = ManagerReplayBuffer(8, 6, GOAL_DIM, device)
    manager = ManagerTD3(6, cfg, device)
    for duration in (13, 40):
        manager_buffer.add(
            np.zeros(6, dtype=np.float32), np.ones(6, dtype=np.float32),
            goal, -1.0, False, duration,
        )
    manager_logs = manager.update(manager_buffer)
    _assert_finite(manager_logs)
    assert manager.total_updates == 4
    assert 13.0 <= manager_logs["manager/mean_duration_steps"] <= 40.0

    mean_score, max_score, final_score = aggregate_segment_constraint([0.0] * 19 + [0.2], False)
    assert math.isclose(mean_score, 0.01, rel_tol=1e-5)
    assert math.isclose(max_score, 0.2, rel_tol=1e-5)
    assert math.isclose(final_score, 0.105, rel_tol=1e-5)
    _, _, failed_score = aggregate_segment_constraint([0.0] * 20, True)
    assert failed_score == 1.0, "single solver failure must not be diluted by segment averaging"


def _test_checkpoint_and_environment(device: torch.device) -> None:
    cfg = TrainConfig(
        hidden_dim=32, manager_hidden_dim=32, critic_hidden_dim=32,
        manager_latent_dim=8, slow_latent_dim=8, fast_latent_dim=8,
        eval_episodes=1, eval_seeds=(42,), no_tensorboard=True,
    )
    env = ElectricGasMultiScaleEnv()
    obs, _ = env.reset(seed=42)
    assert env.slow_action_dim == 10 and env.fast_action_dim == 16 and env.action_dim == 26
    next_obs, reward, _, truncated, info = env.step(np.zeros(env.action_dim, dtype=np.float32))
    assert next_obs.shape == obs.shape and np.isfinite(reward) and not truncated
    assert "reward_components" in info and "constraint_metrics" in info
    agents = build_agents(env, cfg, device)
    root = Path("hierarchical_td3_test_runs")
    root.mkdir(exist_ok=True)
    path = root / "optimized_checkpoint_test.pt"
    save_checkpoint(path, cfg, agents, 0, 0, -1.0)
    payload = legacy.trusted_torch_load(str(path), device)
    for key in ("training_stage", "env_model_version", "slow_interval", "manager_interval",
                "global_state_dim", "slow_action_dim", "fast_action_dim", "total_action_dim",
                "goal_dim", "optimization_config"):
        assert key in payload
    restored_cfg = TrainConfig.from_mapping(payload["config"])
    assert restored_cfg.fast_batch_size == cfg.fast_batch_size
    legacy_config = {
        "batch_size": 128, "learning_starts": 1000, "updates_per_step": 3,
        "slow_update_interval_steps": 5, "manager_update_interval_steps": 20,
        "fast_lr": 3e-4,
    }
    migrated = TrainConfig.from_mapping(legacy_config)
    assert migrated.slow_batch_size == 128 and migrated.manager_learning_starts == 1000
    assert migrated.manager_updates_per_boundary == 3
    assert (migrated.batch_size, migrated.learning_starts, migrated.updates_per_step) == (1, 0, 1)
    restored_agents = build_agents(env, cfg, device)
    load_checkpoint(str(path), restored_agents, device)
    builder = ObservationBuilder(env, cfg.manager_interval)
    goal = fixed_manager_goal()
    a = agents.fast.select_action(builder.fast_obs(0, obs), goal, 0.0, deterministic=True)
    b = restored_agents.fast.select_action(builder.fast_obs(0, obs), goal, 0.0, deterministic=True)
    assert np.allclose(a, b, atol=1e-5)

    legacy_payload = copy.deepcopy(payload)
    for key in ("checkpoint_schema_version", "training_stage", "slow_interval", "manager_interval",
                "global_state_dim", "optimization_config"):
        legacy_payload.pop(key, None)
    legacy_path = root / "optimized_legacy_checkpoint_test.pt"
    torch.save(legacy_payload, str(legacy_path))
    load_checkpoint(str(legacy_path), build_agents(env, cfg, device), device)
    incompatible = copy.deepcopy(payload)
    incompatible["slow_interval"] = 10
    incompatible_path = root / "optimized_incompatible_checkpoint_test.pt"
    torch.save(incompatible, str(incompatible_path))
    try:
        load_checkpoint(str(incompatible_path), build_agents(env, cfg, device), device)
    except ValueError as exc:
        assert "slow_interval" in str(exc)
    else:
        raise AssertionError("incompatible checkpoint interval must fail")

    frozen_cfg = copy.deepcopy(cfg)
    frozen_cfg.training_stage = "fast_pretrain"
    frozen_agents = build_agents(env, frozen_cfg, device)
    assert any(parameter.requires_grad for parameter in frozen_agents.fast.actor.parameters())
    assert not any(parameter.requires_grad for parameter in frozen_agents.slow.actor.parameters())
    assert not any(parameter.requires_grad for parameter in frozen_agents.manager.actor.parameters())
    slow_obs = builder.slow_obs(obs)
    frozen_agents.slow.normalizer.eval()
    action1 = frozen_agents.slow.select_action(slow_obs, goal, 1.0, deterministic=False)
    action2 = frozen_agents.slow.select_action(slow_obs, goal, 1.0, deterministic=False)
    assert np.allclose(action1, action2), "冻结 Worker 不应继续注入探索噪声"


def _test_stage_evaluation_and_runtime_restore(device: torch.device) -> None:
    expected_calls = {
        "fast_pretrain": (0, 0, 3),
        "slow_pretrain": (0, 1, 1),
        "manager_train": (1, 1, 1),
        "joint_finetune": (1, 1, 1),
    }
    for stage, expected in expected_calls.items():
        cfg = TrainConfig(
            training_stage=stage, episode_steps=40, eval_seeds=(100,),
            hidden_dim=16, manager_hidden_dim=16, critic_hidden_dim=16,
            manager_latent_dim=8, slow_latent_dim=8, fast_latent_dim=8,
        )
        env = ElectricGasMultiScaleEnv()
        agents = build_agents(env, cfg, device)
        counts = {"manager": 0, "slow": 0, "fast": 0}
        originals = {
            "manager": agents.manager.select_goal,
            "slow": agents.slow.select_action,
            "fast": agents.fast.select_action,
        }

        def manager_call(*args: Any, **kwargs: Any) -> np.ndarray:
            counts["manager"] += 1
            return originals["manager"](*args, **kwargs)

        def slow_call(*args: Any, **kwargs: Any) -> np.ndarray:
            counts["slow"] += 1
            return originals["slow"](*args, **kwargs)

        def fast_call(*args: Any, **kwargs: Any) -> np.ndarray:
            counts["fast"] += 1
            return originals["fast"](*args, **kwargs)

        agents.manager.select_goal = manager_call  # type: ignore[method-assign]
        agents.slow.select_action = slow_call  # type: ignore[method-assign]
        agents.fast.select_action = fast_call  # type: ignore[method-assign]
        normalizer_counts = tuple(agent.normalizer.count for agent in (agents.manager, agents.slow, agents.fast))
        episodes = 3 if stage == "fast_pretrain" else 1
        stats = evaluate_policy(agents, cfg, episodes=episodes, max_steps=1, seed=777)
        multiplier = episodes if stage == "fast_pretrain" else 1
        assert (counts["manager"], counts["slow"], counts["fast"]) == expected
        assert stats["episodes"] == episodes and stats["steps"] == episodes
        assert _resolve_eval_seeds(cfg, episodes, 777)[0] == 777
        assert normalizer_counts == tuple(
            agent.normalizer.count for agent in (agents.manager, agents.slow, agents.fast)
        )
        assert multiplier >= 1

    cfg = TrainConfig(episode_steps=20)
    names = ("FastReplayBuffer", "safe_env_step", "save_checkpoint", "evaluate_policy")
    original = {name: getattr(legacy, name) for name in names}
    try:
        with runtime_overrides(cfg):
            assert legacy.FastReplayBuffer is FastReplayBuffer
            raise RuntimeError("intentional runtime restoration test")
    except RuntimeError as exc:
        assert "intentional" in str(exc)
    for name in names:
        assert getattr(legacy, name) is original[name], f"legacy.{name} was not restored"
    assert _ACTIVE_CONFIG is None and _CURRENT_SLOW_PENDING is None


def _test_short_training() -> None:
    common = dict(
        device="cpu", hidden_dim=32, manager_hidden_dim=32, critic_hidden_dim=32,
        manager_latent_dim=8, slow_latent_dim=8, fast_latent_dim=8,
        fast_batch_size=4, fast_learning_starts=4, fast_updates_per_step=1,
        slow_batch_size=1, slow_learning_starts=1, slow_updates_per_boundary=2,
        manager_batch_size=1, manager_learning_starts=1,
        eval_interval=1, eval_episodes=1, eval_seeds=(54_321,), no_tensorboard=True,
        fast_buffer_size=64, slow_buffer_size=16, manager_buffer_size=8,
    )
    fast_cfg = TrainConfig(
        episodes=1, episode_steps=20, training_stage="fast_pretrain", device="cpu",
        checkpoint_dir="hierarchical_td3_test_runs/optimized_fast_short",
        **{key: value for key, value in common.items() if key != "device"},
    )
    fast_result = run_training(fast_cfg)
    assert fast_result["global_step"] == 20 and np.isfinite(fast_result["best_eval_return"])
    assert (Path(fast_result["run_root"]) / "best_fast.pt").exists()

    slow_cfg = TrainConfig(
        episodes=1, episode_steps=40, training_stage="slow_pretrain",
        checkpoint_dir="hierarchical_td3_test_runs/optimized_slow_short", **common,
    )
    slow_result = run_training(slow_cfg)
    assert slow_result["global_step"] == 40 and np.isfinite(slow_result["best_eval_return"])
    assert slow_result["slow_buffer_size"] == 2
    assert (Path(slow_result["run_root"]) / "best_slow.pt").exists()
    assert _ACTIVE_CONFIG is None and _CURRENT_SLOW_PENDING is None


def run_minimum_tests() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("Running optimized SMDP-TD3 tests...")
    device = torch.device("cpu")
    legacy.set_seed(42)
    _test_smdp_discount()
    _test_projection_schedule_and_mask()
    _test_physical_features()
    _test_time_scale_validation()
    _test_cli_contract()
    _test_per_initialization_and_sampling(device)
    _test_raw_critic_executed_transition_and_per(device)
    _test_independent_parameters_and_slow_duration(device)
    _test_checkpoint_and_environment(device)
    _test_stage_evaluation_and_runtime_restore(device)
    _test_short_training()
    LOGGER.info("All optimized SMDP-TD3 tests passed.")


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv: Optional[Sequence[str]] = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description="Projection-aware physical hierarchical SMDP-TD3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--episode-steps", type=int, default=EPISODE_STEPS)
    parser.add_argument("--manager-interval", type=int, default=MANAGER_INTERVAL)
    parser.add_argument("--slow-interval", type=int, default=SLOW_INTERVAL)
    parser.add_argument("--training-stage", default="joint_finetune",
                        choices=("fast_pretrain", "slow_pretrain", "manager_train", "joint_finetune", "all"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gamma-fast", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--target-noise", type=float, default=0.10)
    parser.add_argument("--target-noise-clip", type=float, default=0.30)
    parser.add_argument("--target-q-clip-abs", type=float, default=200_000.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--fast-batch-size", type=int, default=256)
    parser.add_argument("--slow-batch-size", type=int, default=64)
    parser.add_argument("--manager-batch-size", type=int, default=32)
    parser.add_argument("--fast-learning-starts", type=int, default=5000)
    parser.add_argument("--slow-learning-starts", type=int, default=256)
    parser.add_argument("--manager-learning-starts", type=int, default=128)
    parser.add_argument("--fast-updates-per-step", type=int, default=1)
    parser.add_argument("--slow-updates-per-boundary", type=int, default=2)
    parser.add_argument("--manager-updates-per-boundary", type=int, default=4)
    # 旧参数仍可用；显式传入时同时设置三个层级。
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-starts", type=int)
    parser.add_argument("--updates-per-step", type=int)
    parser.add_argument("--slow-update-interval-steps", type=int)
    parser.add_argument("--manager-update-interval-steps", type=int)
    parser.add_argument("--fast-lr", type=float, default=3e-4)
    parser.add_argument("--slow-lr", type=float, default=1e-4)
    parser.add_argument("--manager-lr", type=float, default=5e-5)
    parser.add_argument("--joint-worker-lr", type=float, default=1e-4)
    parser.add_argument("--joint-manager-lr", type=float, default=2.5e-5)
    parser.add_argument("--fast-buffer-size", type=int, default=200_000)
    parser.add_argument("--slow-buffer-size", type=int, default=50_000)
    parser.add_argument("--manager-buffer-size", type=int, default=10_000)
    parser.add_argument("--projection-imitation-initial-weight", type=float, default=0.10)
    parser.add_argument("--projection-imitation-final-weight", type=float, default=0.01)
    parser.add_argument("--projection-imitation-decay-updates", type=int, default=100_000)
    parser.add_argument("--projection-imitation-threshold", type=float, default=1e-3)
    parser.add_argument("--projection-behavior-match-threshold", type=float, default=0.05)
    parser.add_argument("--projection-imitation-weight", type=float)
    parser.add_argument("--disable-prioritized-replay", action="store_true")
    parser.add_argument("--priority-alpha", type=float, default=0.6)
    parser.add_argument("--priority-beta-initial", type=float, default=0.4)
    parser.add_argument("--priority-beta-final", type=float, default=1.0)
    parser.add_argument("--priority-beta-anneal-steps", type=int, default=200_000)
    parser.add_argument("--constraint-priority-weight", type=float, default=1.0)
    parser.add_argument("--projection-priority-weight", type=float, default=1.0)
    parser.add_argument("--fast-pretrain-episodes", type=int, default=50)
    parser.add_argument("--slow-pretrain-episodes", type=int, default=50)
    parser.add_argument("--manager-train-episodes", type=int, default=50)
    parser.add_argument("--joint-finetune-episodes", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[20_026, 20_027, 20_028])
    parser.add_argument("--checkpoint-dir", default="hierarchical_td3_optimized_runs")
    parser.add_argument("--load-checkpoint", default="")
    parser.add_argument("--load-policy-only", action="store_true")
    parser.add_argument("--use-transition-model", action="store_true")
    parser.add_argument("--fast-exploration-noise", type=float, default=0.15)
    parser.add_argument("--slow-exploration-noise", type=float, default=0.10)
    parser.add_argument("--manager-exploration-noise", type=float, default=0.05)
    parser.add_argument("--min-fast-exploration-noise", type=float, default=0.02)
    parser.add_argument("--min-slow-exploration-noise", type=float, default=0.02)
    parser.add_argument("--min-manager-exploration-noise", type=float, default=0.01)
    parser.add_argument("--noise-decay-episodes", type=int, default=200)
    parser.add_argument("--goal-smoothing", type=float, default=0.20)
    parser.add_argument("--goal-change-penalty-weight", type=float, default=0.05)
    parser.add_argument("--lambda-projection", type=float, default=0.10)
    parser.add_argument("--worker-reward-clip-abs", type=float, default=5_000.0)
    parser.add_argument("--manager-reward-clip-abs", type=float, default=25_000.0)
    parser.add_argument("--worker-action-l2-weight", type=float, default=1e-3)
    parser.add_argument("--disable-ess-action-guard", action="store_true")
    parser.add_argument("--reachability-weight", type=float, default=0.0)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args(argv)

    if args.slow_update_interval_steps is not None or args.manager_update_interval_steps is not None:
        warnings.warn(
            "--slow-update-interval-steps and --manager-update-interval-steps are deprecated and ignored; "
            "use --slow-updates-per-boundary and --manager-updates-per-boundary.",
            FutureWarning, stacklevel=2,
        )

    fast_batch, slow_batch, manager_batch = args.fast_batch_size, args.slow_batch_size, args.manager_batch_size
    if args.batch_size is not None:
        fast_batch = slow_batch = manager_batch = args.batch_size
    fast_starts, slow_starts, manager_starts = (
        args.fast_learning_starts, args.slow_learning_starts, args.manager_learning_starts
    )
    if args.learning_starts is not None:
        fast_starts = slow_starts = manager_starts = args.learning_starts
    fast_updates, slow_updates, manager_updates = (
        args.fast_updates_per_step, args.slow_updates_per_boundary, args.manager_updates_per_boundary
    )
    if args.updates_per_step is not None:
        fast_updates = slow_updates = manager_updates = args.updates_per_step
    imitation_initial = args.projection_imitation_initial_weight
    imitation_final = args.projection_imitation_final_weight
    imitation_decay = args.projection_imitation_decay_updates
    if args.projection_imitation_weight is not None:
        imitation_initial = imitation_final = args.projection_imitation_weight
        imitation_decay = 1

    return TrainConfig(
        seed=args.seed, episodes=args.episodes, episode_steps=args.episode_steps,
        manager_interval=args.manager_interval, slow_interval=args.slow_interval,
        training_stage=args.training_stage, device=args.device, gamma_fast=args.gamma_fast,
        tau=args.tau, target_noise=args.target_noise, target_noise_clip=args.target_noise_clip,
        target_q_clip_abs=args.target_q_clip_abs, gradient_clip=args.gradient_clip,
        fast_batch_size=fast_batch, slow_batch_size=slow_batch, manager_batch_size=manager_batch,
        fast_learning_starts=fast_starts, slow_learning_starts=slow_starts,
        manager_learning_starts=manager_starts, fast_updates_per_step=fast_updates,
        slow_updates_per_boundary=slow_updates, manager_updates_per_boundary=manager_updates,
        fast_lr=args.fast_lr, slow_lr=args.slow_lr, manager_lr=args.manager_lr,
        joint_worker_lr=args.joint_worker_lr, joint_manager_lr=args.joint_manager_lr,
        fast_buffer_size=args.fast_buffer_size, slow_buffer_size=args.slow_buffer_size,
        manager_buffer_size=args.manager_buffer_size,
        projection_imitation_initial_weight=imitation_initial,
        projection_imitation_final_weight=imitation_final,
        projection_imitation_decay_updates=imitation_decay,
        projection_imitation_threshold=args.projection_imitation_threshold,
        projection_behavior_match_threshold=args.projection_behavior_match_threshold,
        use_prioritized_replay=not args.disable_prioritized_replay,
        priority_alpha=args.priority_alpha, priority_beta_initial=args.priority_beta_initial,
        priority_beta_final=args.priority_beta_final,
        priority_beta_anneal_steps=args.priority_beta_anneal_steps,
        constraint_priority_weight=args.constraint_priority_weight,
        projection_priority_weight=args.projection_priority_weight,
        fast_pretrain_episodes=args.fast_pretrain_episodes,
        slow_pretrain_episodes=args.slow_pretrain_episodes,
        manager_train_episodes=args.manager_train_episodes,
        joint_finetune_episodes=args.joint_finetune_episodes,
        eval_interval=args.eval_interval, eval_episodes=args.eval_episodes,
        eval_seeds=tuple(args.eval_seeds), checkpoint_dir=args.checkpoint_dir,
        load_checkpoint=args.load_checkpoint, load_policy_only=args.load_policy_only,
        use_transition_model=args.use_transition_model,
        fast_exploration_noise=args.fast_exploration_noise,
        slow_exploration_noise=args.slow_exploration_noise,
        manager_exploration_noise=args.manager_exploration_noise,
        min_fast_exploration_noise=args.min_fast_exploration_noise,
        min_slow_exploration_noise=args.min_slow_exploration_noise,
        min_manager_exploration_noise=args.min_manager_exploration_noise,
        noise_decay_episodes=args.noise_decay_episodes, goal_smoothing=args.goal_smoothing,
        goal_change_penalty_weight=args.goal_change_penalty_weight,
        lambda_projection=args.lambda_projection, worker_reward_clip_abs=args.worker_reward_clip_abs,
        manager_reward_clip_abs=args.manager_reward_clip_abs,
        worker_action_l2_weight=args.worker_action_l2_weight,
        use_ess_action_guard=not args.disable_ess_action_guard,
        reachability_weight=args.reachability_weight,
        no_tensorboard=args.no_tensorboard, run_tests=args.run_tests,
    )


def main() -> None:
    cfg = parse_args()
    if cfg.run_tests:
        run_minimum_tests()
    elif cfg.training_stage == "all":
        run_all_stages(cfg)
    else:
        run_training(cfg)


if __name__ == "__main__":
    main()
