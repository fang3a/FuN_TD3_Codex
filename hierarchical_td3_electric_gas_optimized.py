"""投影感知、物理引导的分层 SMDP-TD3。

本文件保留 ``hierarchical_td3_electric_gas.py`` 的 Manager-Slow-Fast 网络、
环境交互和 checkpoint 结构，只替换算法层的配置、Replay、TD target、动作语义、
物理 goal 与阶段训练控制。电-气网络拓扑、设备参数、动作映射和奖励公式仍由
``electric_gas_microgrid_single.py`` 提供，未在这里复制或修改。

Worker 动作语义是显式的：raw_action 是 Actor 请求并作为 Critic
``Q(s, raw_request_action)`` 的动作输入；guarded_action 是训练脚本 ESS
前瞻保护后的请求；executed_action 是环境逐步安全投影后的真实
执行动作，用于转移监督、投影惩罚、投影模仿和诊断，不是 Critic 输入。


Run with the tested conda environment:
    & 'D:\\anaconda\\anaconda\\envs\\python_3_8\\python.exe' electric_gas_microgrid_single.py --mode both
"""

from __future__ import annotations

import argparse
import copy
import inspect
import logging
import math
import os
import random
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
    COMPRESSOR_CONFIGS,
    CONTROLLED_COMPRESSOR_INDICES,
    DEFAULT_CONFIG,
    ENV_MODEL_VERSION,
    ESS_CONFIGS,
    GFG_CONFIGS,
    GAS_PIPES,
    GAS_SUPPLIERS,
    IEEE33_LOAD_DATA,
    IEEE33_LINE_DATA,
    N_GAS_JUNCTIONS,
    P2G_CONFIGS,
    PhysicalActions,
    RENEWABLE_CONFIGS,
    SLOW_SAFETY_SCHEMA_VERSION,
    ElectricGasMultiScaleEnv,
    default_compressor_ratios,
)


LOGGER = logging.getLogger("hierarchical_td3_electric_gas_optimized")
REQUIRED_LEGACY_ALGORITHM_API_VERSION = 5
REQUIRED_SLOW_SAFETY_SCHEMA_VERSION = 2

FAST_INTERVAL = legacy.FAST_INTERVAL
SLOW_INTERVAL = legacy.SLOW_INTERVAL
MANAGER_INTERVAL = legacy.MANAGER_INTERVAL
EPISODE_STEPS = legacy.EPISODE_STEPS
GOAL_DIM = legacy.GOAL_DIM
GOAL_PHYSICAL_SLICE = slice(24, 32)
ALGORITHM_VERSION = "safety-smdp-td3-v11-guard-credit-calibrated"

ObservationBuilder = legacy.ObservationBuilder
AgentBundle = legacy.AgentBundle
RunningMeanStd = legacy.RunningMeanStd
execute_manager_goal_np = legacy.execute_manager_goal_np
execute_manager_goal_tensor = legacy.execute_manager_goal_tensor

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
_LEGACY_MANAGER_REPLAY_BUFFER_CLASS = legacy.ManagerReplayBuffer


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
    joint_worker_lr: float = 5e-5
    joint_manager_lr: float = 1e-5
    joint_policy_frequency: int = 6

    projection_imitation_weight: float = 0.10  # 旧字段，保存兼容性。
    projection_imitation_initial_weight: float = 0.10
    projection_imitation_final_weight: float = 0.02
    projection_imitation_decay_updates: int = 100_000
    projection_imitation_threshold: float = 1e-3
    projection_behavior_match_threshold: float = 0.05
    slow_guard_imitation_multiplier: float = 5.0

    use_prioritized_replay: bool = True
    priority_alpha: float = 0.6
    priority_beta_initial: float = 0.4
    priority_beta_final: float = 1.0
    priority_beta_anneal_steps: int = 200_000
    fast_priority_beta_anneal_updates: int = 200_000
    slow_priority_beta_anneal_updates: int = 50_000
    manager_priority_beta_anneal_updates: int = 25_000
    priority_epsilon: float = 1e-5
    constraint_priority_weight: float = 1.0
    projection_priority_weight: float = 1.0
    max_td_error_for_priority: float = 1e6
    max_replay_priority: float = 1e6
    prioritized_sample_fraction: float = 0.75
    manager_use_prioritized_replay: bool = False
    manager_constraint_priority_weight: float = 1.0
    manager_solver_priority_weight: float = 2.0
    slow_shaping_duration_mode: str = "terminal"
    priority_component_normalization: str = "running_scale"
    nonfinite_update_policy: str = "raise"
    full_resume_checkpoint_interval: int = 5
    strict_resume_required: bool = True
    bootstrap_on_time_limit: bool = False
    unexpected_env_exception_policy: str = "raise"
    fast_random_warmup_steps: int = 2_000
    slow_random_warmup_segments: int = 128
    manager_random_warmup_segments: int = 64
    warmup_blend_fraction: float = 0.20
    strict_stage_sample_validation: bool = True
    debug_terminal_soc_penalty: bool = True
    reward_component_transform: str = "log1p_reference"
    reward_scale_profile: str = "safety_calibrated_20260716_v2"
    slow_role_specific_reward_scale: float = 25.0
    shaping_reference_floor: float = 1.0
    adaptive_auxiliary_loss_scaling: bool = False
    auxiliary_loss_scale_max: float = 1_000_000.0
    auxiliary_loss_coefficient_max: float = 0.05
    worker_component_clip_abs: float = 50.0
    fast_global_safety_weight: float = 0.50
    fast_role_specific_weight: float = 0.50
    slow_global_safety_weight: float = 0.50
    slow_role_specific_weight: float = 0.50
    health_warning_patience_episodes: int = 3
    health_clip_warning_ratio: float = 0.05
    health_action_saturation_warning_ratio: float = 0.30
    health_projection_warning_rms: float = 0.20
    health_actor_collapse_warning_abs_mean: float = 0.03

    # Joint fine-tuning starts from the transferred safety candidate.  Keeping the
    # Manager deterministic initially reduces simultaneous three-policy drift.
    inherit_stage_best_on_joint_transfer: bool = True
    joint_freeze_manager_actor_episodes: int = 10
    joint_early_stop_patience_evaluations: int = 3
    joint_early_stop_min_episodes: int = 20

    fast_pretrain_episodes: int = 50
    slow_pretrain_episodes: int = 50
    manager_train_episodes: int = 50
    joint_finetune_episodes: int = 30
    eval_episodes: int = 3
    eval_seeds: Tuple[int, ...] = (20_026, 20_027, 20_028)
    eval_seed_mode: str = "fixed"
    checkpoint_load_mode: str = "resume"
    best_model_metric: str = "feasible_then_return"
    min_power_success_rate: float = 0.999
    min_gas_success_rate: float = 0.999
    max_soc_violation_rate: float = 0.0
    max_hard_constraint_violation_rate: float = 0.0
    max_voltage_rms_deviation_pu: float = 0.05
    max_gas_pressure_rms_deviation_bar: float = 0.5
    metric_comparison_tolerance: float = 1e-6

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
            migrated_names: List[str] = []
            if old_name in filtered:
                for name in new_names:
                    if name not in filtered:
                        filtered[name] = filtered[old_name]
                        migrated_names.append(name)
            if migrated_names:
                LOGGER.warning("Migrated deprecated checkpoint config %s to missing fields %s.",
                               old_name, ", ".join(migrated_names))
        beta_names = (
            "fast_priority_beta_anneal_updates", "slow_priority_beta_anneal_updates",
            "manager_priority_beta_anneal_updates",
        )
        if "priority_beta_anneal_steps" in filtered:
            migrated_beta: List[str] = []
            for name in beta_names:
                if name not in filtered:
                    filtered[name] = filtered["priority_beta_anneal_steps"]
                    migrated_beta.append(name)
            if migrated_beta:
                LOGGER.warning(
                    "Migrated deprecated priority_beta_anneal_steps to per-level fields %s.",
                    ", ".join(migrated_beta),
                )
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
        if self.load_policy_only and self.checkpoint_load_mode != "policy_only":
            warnings.warn("load_policy_only is deprecated; checkpoint_load_mode='policy_only' is used.",
                          FutureWarning, stacklevel=2)
            self.checkpoint_load_mode = "policy_only"
        self.load_policy_only = self.checkpoint_load_mode == "policy_only"

        def require(name: str, condition: bool, expectation: str) -> None:
            value = getattr(self, name)
            if not condition:
                raise ValueError(f"{name}={value!r} must satisfy {expectation}")

        require("gamma_fast", 0.0 < self.gamma_fast <= 1.0, "0 < gamma_fast <= 1")
        require("tau", 0.0 < self.tau <= 1.0, "0 < tau <= 1")
        require("policy_frequency", int(self.policy_frequency) > 0, "policy_frequency > 0")
        require("joint_policy_frequency", int(self.joint_policy_frequency) > 0,
                "joint_policy_frequency > 0")
        require("priority_alpha", np.isfinite(self.priority_alpha) and self.priority_alpha >= 0.0,
                "priority_alpha must be finite and >= 0")
        require("priority_epsilon", np.isfinite(self.priority_epsilon) and self.priority_epsilon > 0.0,
                "priority_epsilon must be finite and > 0")
        for name in ("constraint_priority_weight", "projection_priority_weight",
                     "manager_constraint_priority_weight", "manager_solver_priority_weight"):
            require(name, np.isfinite(float(getattr(self, name))) and float(getattr(self, name)) >= 0.0,
                    f"{name} must be finite and >= 0")
        require("max_td_error_for_priority", np.isfinite(self.max_td_error_for_priority)
                and self.max_td_error_for_priority > 0.0,
                "max_td_error_for_priority must be finite and > 0")
        require("max_replay_priority", np.isfinite(self.max_replay_priority)
                and self.max_replay_priority >= self.priority_epsilon,
                "max_replay_priority must be finite and >= priority_epsilon")
        require("prioritized_sample_fraction", 0.0 <= self.prioritized_sample_fraction <= 1.0,
                "0 <= prioritized_sample_fraction <= 1")
        for name in ("projection_imitation_weight", "projection_imitation_initial_weight",
                     "projection_imitation_final_weight", "slow_guard_imitation_multiplier"):
            require(name, np.isfinite(float(getattr(self, name))) and float(getattr(self, name)) >= 0.0,
                    f"{name} must be finite and >= 0")
        require("projection_imitation_decay_updates", int(self.projection_imitation_decay_updates) > 0,
                "projection_imitation_decay_updates > 0")
        for name in ("fast_buffer_size", "slow_buffer_size", "manager_buffer_size",
                     "fast_batch_size", "slow_batch_size", "manager_batch_size"):
            require(name, int(getattr(self, name)) > 0, f"{name} > 0")
        require("eval_interval", int(self.eval_interval) > 0, "eval_interval > 0")
        require("eval_episodes", int(self.eval_episodes) > 0, "eval_episodes > 0")
        for name in ("worker_reward_clip_abs", "manager_reward_clip_abs", "target_q_clip_abs"):
            require(name, np.isfinite(float(getattr(self, name))) and float(getattr(self, name)) >= 0.0,
                    f"{name} must be finite and >= 0")
        if self.reward_component_transform not in ("none", "log1p_reference"):
            raise ValueError(
                f"reward_component_transform={self.reward_component_transform!r} must be "
                "'none' or 'log1p_reference'"
            )
        if self.reward_scale_profile != "safety_calibrated_20260716_v2":
            raise ValueError(
                f"Unsupported reward_scale_profile={self.reward_scale_profile!r}; "
                "expected 'safety_calibrated_20260716_v2'"
            )
        require("shaping_reference_floor",
                np.isfinite(self.shaping_reference_floor) and self.shaping_reference_floor > 0.0,
                "shaping_reference_floor must be finite and > 0")
        require("auxiliary_loss_scale_max",
                np.isfinite(self.auxiliary_loss_scale_max) and self.auxiliary_loss_scale_max > 0.0,
                "auxiliary_loss_scale_max must be finite and > 0")
        require("auxiliary_loss_coefficient_max",
                np.isfinite(self.auxiliary_loss_coefficient_max)
                and self.auxiliary_loss_coefficient_max > 0.0,
                "auxiliary_loss_coefficient_max must be finite and > 0")
        require("episode_steps", int(self.episode_steps) > 0, "episode_steps > 0")
        require("episode_steps", int(self.episode_steps) <= int(DEFAULT_CONFIG.time.steps_per_day),
                f"episode_steps <= steps_per_day ({DEFAULT_CONFIG.time.steps_per_day}); cross-day episodes are unsupported")
        require("episodes", int(self.episodes) >= 0, "episodes >= 0")
        if self.run_mode not in ("formal", "debug"):
            raise ValueError(f"run_mode={self.run_mode!r} must be 'formal' or 'debug'")
        if self.run_mode == "formal" and self.best_model_metric != "feasible_then_return":
            raise ValueError(
                "Formal safety training requires best_model_metric='feasible_then_return'"
            )
        for name in ("fast_pretrain_episodes", "slow_pretrain_episodes", "manager_train_episodes",
                     "joint_finetune_episodes"):
            require(name, int(getattr(self, name)) >= 0, f"{name} >= 0")
        for name in ("fast_lr", "slow_lr", "manager_lr", "joint_worker_lr", "joint_manager_lr"):
            require(name, np.isfinite(float(getattr(self, name))) and float(getattr(self, name)) > 0.0,
                    f"{name} must be finite and > 0")
        require("gradient_clip", np.isfinite(self.gradient_clip) and self.gradient_clip > 0.0,
                "gradient_clip must be finite and > 0")
        for name in ("target_noise", "target_noise_clip", "fast_exploration_noise",
                     "slow_exploration_noise", "manager_exploration_noise",
                     "min_fast_exploration_noise", "min_slow_exploration_noise",
                     "min_manager_exploration_noise"):
            require(name, np.isfinite(float(getattr(self, name))) and float(getattr(self, name)) >= 0.0,
                    f"{name} must be finite and >= 0")
        require("noise_decay_episodes", int(self.noise_decay_episodes) > 0, "noise_decay_episodes > 0")
        require("goal_smoothing", np.isfinite(self.goal_smoothing) and 0.0 <= self.goal_smoothing <= 1.0,
                "0 <= goal_smoothing <= 1")
        for name in ("alpha_external", "beta_latent", "beta_physical", "lambda_projection",
                     "worker_action_l2_weight", "lambda_transition", "lambda_latent_norm",
                     "reachability_weight", "goal_change_penalty_weight"):
            require(name, np.isfinite(float(getattr(self, name))) and float(getattr(self, name)) >= 0.0,
                    f"{name} must be finite and >= 0")
        for name in ("worker_component_clip_abs", "fast_global_safety_weight",
                     "fast_role_specific_weight", "slow_global_safety_weight",
                     "slow_role_specific_weight", "slow_role_specific_reward_scale"):
            require(name, np.isfinite(float(getattr(self, name))) and float(getattr(self, name)) >= 0.0,
                    f"{name} must be finite and >= 0")
        if self.fast_global_safety_weight + self.fast_role_specific_weight <= 0.0:
            raise ValueError("fast_global_safety_weight + fast_role_specific_weight must be > 0")
        if self.slow_global_safety_weight + self.slow_role_specific_weight <= 0.0:
            raise ValueError("slow_global_safety_weight + slow_role_specific_weight must be > 0")
        if self.run_mode == "formal" and self.fast_global_safety_weight <= 0.0:
            raise ValueError(
                "Formal safety training requires fast_global_safety_weight > 0"
            )
        if self.run_mode == "formal" and self.slow_global_safety_weight <= 0.0:
            raise ValueError(
                "Formal safety training requires slow_global_safety_weight > 0"
            )
        for name in (
            "fast_learning_starts", "slow_learning_starts", "manager_learning_starts",
            "fast_updates_per_step", "slow_updates_per_boundary", "manager_updates_per_boundary",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self.priority_beta_initial <= self.priority_beta_final <= 1.0:
            raise ValueError("priority beta must satisfy 0 <= initial <= final <= 1")
        for name in ("priority_beta_anneal_steps", "fast_priority_beta_anneal_updates",
                     "slow_priority_beta_anneal_updates", "manager_priority_beta_anneal_updates"):
            require(name, int(getattr(self, name)) > 0, f"{name} > 0")
        if self.projection_behavior_match_threshold < 0.0:
            raise ValueError(
                f"projection_behavior_match_threshold={self.projection_behavior_match_threshold!r} must be non-negative"
            )
        if self.checkpoint_load_mode not in ("resume", "stage_transfer", "policy_only"):
            raise ValueError(f"checkpoint_load_mode={self.checkpoint_load_mode!r} must be resume, stage_transfer or policy_only")
        if self.eval_seed_mode not in ("fixed", "offset"):
            raise ValueError(f"eval_seed_mode={self.eval_seed_mode!r} must be fixed or offset")
        if self.best_model_metric not in ("feasible_then_return", "return"):
            raise ValueError(
                f"best_model_metric={self.best_model_metric!r} must be feasible_then_return or return"
            )
        if self.slow_shaping_duration_mode not in ("terminal", "normalized"):
            raise ValueError(
                f"slow_shaping_duration_mode={self.slow_shaping_duration_mode!r} must be terminal or normalized"
            )
        if self.priority_component_normalization not in ("none", "running_scale", "rank"):
            raise ValueError(
                f"priority_component_normalization={self.priority_component_normalization!r} "
                "must be none, running_scale or rank"
            )
        if self.nonfinite_update_policy not in ("raise", "skip_batch"):
            raise ValueError(
                f"nonfinite_update_policy={self.nonfinite_update_policy!r} must be raise or skip_batch"
            )
        if self.bootstrap_on_time_limit:
            raise ValueError(
                "bootstrap_on_time_limit=True is incompatible with this environment because reset "
                "starts a new exogenous scenario"
            )
        if self.unexpected_env_exception_policy not in ("raise", "truncate"):
            raise ValueError(
                f"unexpected_env_exception_policy={self.unexpected_env_exception_policy!r} "
                "must be raise or truncate"
            )
        for name in ("fast_random_warmup_steps", "slow_random_warmup_segments",
                     "manager_random_warmup_segments"):
            require(name, int(getattr(self, name)) >= 0, f"{name} >= 0")
        require("warmup_blend_fraction",
                np.isfinite(self.warmup_blend_fraction)
                and 0.0 <= self.warmup_blend_fraction <= 1.0,
                "warmup_blend_fraction must be finite and in [0,1]")
        require("health_warning_patience_episodes", int(self.health_warning_patience_episodes) > 0,
                "health_warning_patience_episodes > 0")
        for name in ("health_clip_warning_ratio", "health_action_saturation_warning_ratio"):
            require(name, np.isfinite(float(getattr(self, name))) and 0.0 <= float(getattr(self, name)) <= 1.0,
                    f"{name} must be finite and in [0,1]")
        require("health_projection_warning_rms",
                np.isfinite(self.health_projection_warning_rms) and self.health_projection_warning_rms >= 0.0,
                "health_projection_warning_rms must be finite and >= 0")
        require("health_actor_collapse_warning_abs_mean",
                np.isfinite(self.health_actor_collapse_warning_abs_mean)
                and self.health_actor_collapse_warning_abs_mean >= 0.0,
                "health_actor_collapse_warning_abs_mean must be finite and >= 0")
        require("joint_freeze_manager_actor_episodes",
                int(self.joint_freeze_manager_actor_episodes) >= 0,
                "joint_freeze_manager_actor_episodes must be >= 0")
        require("joint_early_stop_patience_evaluations",
                int(self.joint_early_stop_patience_evaluations) > 0,
                "joint_early_stop_patience_evaluations must be > 0")
        require("joint_early_stop_min_episodes",
                int(self.joint_early_stop_min_episodes) >= 0,
                "joint_early_stop_min_episodes must be >= 0")
        require("full_resume_checkpoint_interval", int(self.full_resume_checkpoint_interval) > 0,
                "full_resume_checkpoint_interval > 0")
        for name in ("min_power_success_rate", "min_gas_success_rate", "max_soc_violation_rate",
                     "max_hard_constraint_violation_rate"):
            require(name, np.isfinite(float(getattr(self, name))) and 0.0 <= float(getattr(self, name)) <= 1.0,
                    f"{name} must be finite and in [0,1]")
        for name in ("max_voltage_rms_deviation_pu", "max_gas_pressure_rms_deviation_bar",
                     "metric_comparison_tolerance"):
            require(name, np.isfinite(float(getattr(self, name))) and float(getattr(self, name)) >= 0.0,
                    f"{name} must be finite and >= 0")
        if self.reachability_weight > 0.0:
            raise ValueError(
                "reachability_weight must remain 0 until a differentiable projection model is available: "
                "the Transition Model is trained on executed actions, not unconstrained raw actions."
            )
        for batch_name, buffer_name in (
            ("fast_batch_size", "fast_buffer_size"),
            ("slow_batch_size", "slow_buffer_size"),
            ("manager_batch_size", "manager_buffer_size"),
        ):
            if int(getattr(self, batch_name)) > int(getattr(self, buffer_name)):
                raise ValueError(
                    f"{batch_name}={getattr(self, batch_name)} must be <= "
                    f"{buffer_name}={getattr(self, buffer_name)}"
                )
        for starts_name, buffer_name in (
            ("fast_learning_starts", "fast_buffer_size"),
            ("slow_learning_starts", "slow_buffer_size"),
            ("manager_learning_starts", "manager_buffer_size"),
        ):
            if int(getattr(self, starts_name)) > int(getattr(self, buffer_name)):
                raise ValueError(
                    f"{starts_name}={getattr(self, starts_name)} must be <= "
                    f"{buffer_name}={getattr(self, buffer_name)}"
                )
        active_flags = legacy.stage_flags(self.training_stage)
        update_fields = {
            "fast": "fast_updates_per_step", "slow": "slow_updates_per_boundary",
            "manager": "manager_updates_per_boundary",
        }
        for role, field_name in update_fields.items():
            if active_flags[role] and int(getattr(self, field_name)) <= 0:
                raise ValueError(
                    f"{field_name}={getattr(self, field_name)} must be > 0 while {role} is trainable"
                )
        # Replay-budget validation is intentionally performed by run_training().
        # At that point strict-resume Replay sizes are known and can be included.


def estimate_replay_samples(cfg: TrainConfig, episodes: Optional[int] = None) -> Dict[str, int]:
    """Return exact upper-bound insertions produced by the configured horizon."""

    count = int(cfg.episodes if episodes is None else episodes)
    return {
        "fast": count * int(cfg.episode_steps),
        "slow": count * math.ceil(int(cfg.episode_steps) / int(cfg.slow_interval)),
        "manager": count * math.ceil(int(cfg.episode_steps) / int(cfg.manager_interval)),
    }


def _resume_replay_sizes(cfg: TrainConfig) -> Tuple[Dict[str, int], int, Dict[str, int]]:
    sizes = {"fast": 0, "slow": 0, "manager": 0}
    update_counts = {
        role + suffix: 0
        for role in ("fast", "slow", "manager")
        for suffix in ("_critic", "_actor")
    }
    next_episode = 0
    if not cfg.load_checkpoint or cfg.checkpoint_load_mode != "resume":
        return sizes, next_episode, update_counts
    payload = legacy.trusted_torch_load(cfg.load_checkpoint, map_location=torch.device("cpu"))
    next_episode = int(payload.get("next_episode", int(payload.get("episode", -1)) + 1))
    for role in sizes:
        replay = payload.get(role + "_replay", {})
        sizes[role] = int(replay.get("valid_size", replay.get("capacity", 0) if replay.get("full") else replay.get("idx", 0)))
        state = payload.get(role, {})
        total = int(state.get("total_updates", 0))
        update_counts[role + "_critic"] = int(state.get("critic_updates", total))
        update_counts[role + "_actor"] = int(state.get("actor_updates", total // max(int(cfg.policy_frequency), 1)))
    return sizes, next_episode, update_counts


def validate_training_contract(cfg: TrainConfig) -> Dict[str, Dict[str, int]]:
    """Fail before environment startup when a requested training stage cannot update."""

    if cfg.run_mode not in ("formal", "debug"):
        raise ValueError(f"run_mode={cfg.run_mode!r} must be 'formal' or 'debug'")
    if cfg.run_mode == "formal" and int(cfg.episode_steps) != int(DEFAULT_CONFIG.time.steps_per_day):
        raise ValueError(
            "Formal training requires a complete day: "
            f"episode_steps={cfg.episode_steps}, steps_per_day={DEFAULT_CONFIG.time.steps_per_day}. "
            "Use run_mode='debug' only for short diagnostics/tests."
        )
    existing, next_episode, _ = _resume_replay_sizes(cfg)
    remaining_episodes = max(int(cfg.episodes) - next_episode, 0)
    expected_new = estimate_replay_samples(cfg, remaining_episodes)
    flags = legacy.stage_flags(cfg.training_stage)
    thresholds = {
        "fast": max(int(cfg.fast_learning_starts), int(cfg.fast_batch_size)),
        "slow": max(int(cfg.slow_learning_starts), int(cfg.slow_batch_size)),
        "manager": max(int(cfg.manager_learning_starts), int(cfg.manager_batch_size)),
    }
    warmups = {
        "fast": int(cfg.fast_random_warmup_steps),
        "slow": int(cfg.slow_random_warmup_segments),
        "manager": int(cfg.manager_random_warmup_segments),
    }
    report: Dict[str, Dict[str, int]] = {}
    for role in ("fast", "slow", "manager"):
        available = existing[role] + expected_new[role]
        required = max(thresholds[role], warmups[role])
        report[role] = {
            "existing": existing[role], "expected_new": expected_new[role],
            "expected_total": available, "learning_threshold": thresholds[role],
            "warmup_threshold": warmups[role], "required": required,
        }
        if flags[role] and remaining_episodes > 0 and available < required:
            raise ValueError(
                f"Invalid {cfg.training_stage} training budget for {role}: expected Replay "
                f"total={available} (existing={existing[role]}, new={expected_new[role]}) is below "
                f"required={required} (learning_starts/batch={thresholds[role]}, warmup={warmups[role]})."
            )
    return report


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


def adaptive_auxiliary_weight(primary_loss: torch.Tensor, auxiliary_loss: torch.Tensor,
                              relative_weight: float, cfg: TrainConfig) -> float:
    """Convert a desired relative actor-loss share into a detached coefficient."""

    if relative_weight <= 0.0:
        return 0.0
    if not cfg.adaptive_auxiliary_loss_scaling:
        return float(relative_weight)
    primary = float(primary_loss.detach().abs().cpu())
    auxiliary = float(auxiliary_loss.detach().abs().cpu())
    scale = primary / max(auxiliary, 1e-8)
    coefficient = float(relative_weight * min(scale, cfg.auxiliary_loss_scale_max))
    # Without an absolute coefficient cap, an auxiliary loss approaching zero
    # produces an increasingly strong gradient despite a small loss contribution.
    # That feedback caused both Workers to collapse to normalized zero actions.
    return float(min(coefficient, cfg.auxiliary_loss_coefficient_max))


def worker_action_regularization_reference(
    raw_observations: torch.Tensor,
    role: str,
    action_dim: int,
) -> torch.Tensor:
    """Return the reference action in the Actor's exact normalized coordinates.

    Fast action zero is not physically neutral: the last half controls renewable
    curtailment and zero requests half of the allowed curtailment.  Its neutral
    reference is therefore ``[Q=0, curtailment=-1]``.  Slow actions are regularized
    relative to the action already held by the environment, which is present in the
    compact Markov observation, rather than toward midpoint GFG/P2G/compressor output.
    """

    if raw_observations.ndim != 2:
        raise ValueError(
            f"raw_observations must be [batch, obs_dim], got shape={tuple(raw_observations.shape)}"
        )
    if role == "fast":
        if action_dim <= 0 or action_dim % 2 != 0:
            raise ValueError(f"Fast action_dim must be a positive even value, got {action_dim}")
        reference = torch.zeros(
            (raw_observations.shape[0], action_dim),
            dtype=raw_observations.dtype,
            device=raw_observations.device,
        )
        reference[:, action_dim // 2:] = -1.0
        return reference
    if role == "slow":
        held_slice = legacy.SLOW_OBSERVATION_LAYOUT.slices["held_slow_action"]
        if held_slice.stop is None or held_slice.stop > raw_observations.shape[1]:
            expected_action_dim = held_slice.stop - held_slice.start
            if action_dim != expected_action_dim:
                # Compact synthetic-network tests intentionally use a smaller action
                # space and do not represent the real environment contract.
                return torch.zeros(
                    (raw_observations.shape[0], action_dim),
                    dtype=raw_observations.dtype,
                    device=raw_observations.device,
                )
            raise ValueError(
                "Slow observation cannot provide held_slow_action: "
                f"obs_dim={raw_observations.shape[1]}, expected_stop={held_slice.stop}"
            )
        reference = raw_observations[:, held_slice]
        if reference.shape[1] != action_dim:
            raise ValueError(
                f"held_slow_action size={reference.shape[1]}, expected action_dim={action_dim}"
            )
        require_finite_tensor("slow/action_regularization_reference", reference)
        return reference.clamp(-1.0, 1.0)
    raise ValueError(f"Unknown Worker role={role!r}")


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


def projection_imitation_element_mask(
    current_actor_actions: torch.Tensor,
    historical_raw_actions: torch.Tensor,
    guarded_actions: torch.Tensor,
    projection_threshold: float,
    behavior_match_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select only action dimensions actually changed by the safety guard.

    The old sample-level mask imitated all ten Slow dimensions whenever any ESS
    dimension was guarded.  That copied unrelated GFG/P2G/compressor behavior and
    diluted the ESS safety signal.  The element mask keeps the conservative
    behavior-support check per sample but supervises only projected dimensions.
    """

    if not (
        current_actor_actions.shape == historical_raw_actions.shape == guarded_actions.shape
    ):
        raise ValueError(
            "Projection imitation tensors must share shape, got "
            f"actor={tuple(current_actor_actions.shape)}, "
            f"raw={tuple(historical_raw_actions.shape)}, "
            f"guarded={tuple(guarded_actions.shape)}"
        )
    projected_elements = (
        historical_raw_actions - guarded_actions
    ).abs() > float(projection_threshold)
    behavior_rms = torch.sqrt(
        torch.mean(
            (current_actor_actions.detach() - historical_raw_actions).pow(2), dim=-1
        ) + 1e-12
    )
    behavior_supported = behavior_rms < float(behavior_match_threshold)
    imitation_elements = projected_elements & behavior_supported[:, None]
    return projected_elements, behavior_supported, imitation_elements


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
        return legacy.SLOW_OBSERVATION_LAYOUT.slices["soc"]

    @property
    def slow_gas_pressure(self) -> slice:
        return legacy.SLOW_OBSERVATION_LAYOUT.slices["gas_pressure_summary"]

    @property
    def slow_source_loading(self) -> slice:
        return legacy.SLOW_OBSERVATION_LAYOUT.slices["source_utilization"]

    @property
    def slow_linepack(self) -> int:
        return int(legacy.SLOW_OBSERVATION_LAYOUT.slices["linepack"].start)

    @property
    def slow_required_size(self) -> int:
        return legacy.SLOW_OBSERVATION_LAYOUT.dimension


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
        return legacy.slow_physical_goal_features(obs)

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
    """安全预训练 goal：低越限/低功率缺额、高新能源利用率、SOC 居中。"""

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
            array = np.asarray(raw, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError, OverflowError):
            return default
        if array.size == 0:
            return default
        array = np.nan_to_num(array, nan=default, posinf=default, neginf=default)
        return float(array[0])

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
    try:
        source_violation = np.asarray(
            metrics.get("source_capacity_violation_kg_s", []), dtype=np.float64
        ).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        source_violation = np.empty(0, dtype=np.float64)
    source_violation = np.nan_to_num(
        source_violation, nan=0.0, posinf=np.finfo(np.float32).max, neginf=0.0
    )
    source_caps = np.asarray([item.max_mdot_kg_s for item in GAS_SUPPLIERS], dtype=np.float64).reshape(-1)
    source_caps = np.nan_to_num(source_caps, nan=1e-6, posinf=1e6, neginf=1e-6)
    source = 0.0
    aligned = min(source_violation.size, source_caps.size)
    if aligned > 0:
        normalized = source_violation[:aligned] / np.maximum(source_caps[:aligned], 1e-6)
        source = float(np.max(np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)))
    soc = max(
        min(item.soc_min for item in ESS_CONFIGS) - value("soc_min", 0.5),
        value("soc_max", 0.5) - max(item.soc_max for item in ESS_CONFIGS),
        0.0,
    ) / 0.10
    solver = float(bool(info.get("solver_failed", False)))
    terms = np.clip(np.asarray([voltage, pressure, line, pipe, source, soc, solver]), 0.0, 1.0)
    score = float(np.clip(0.5 * np.max(terms) + 0.5 * np.mean(terms), 0.0, 1.0))
    if solver > 0.0:
        score = 1.0
    return score if np.isfinite(score) else solver


class _PrioritizedReplayMixin:
    priorities: np.ndarray
    td_priorities: np.ndarray
    constraint_scores: np.ndarray
    projection_scores: np.ndarray
    cfg: TrainConfig
    sample_calls: int
    priority_fallback_count: int
    priority_role: str

    def _init_priority(self, capacity: int, cfg: TrainConfig, role: str) -> None:
        self.cfg = cfg
        self.priority_role = role
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.td_priorities = np.zeros(capacity, dtype=np.float32)
        self.constraint_scores = np.zeros(capacity, dtype=np.float32)
        self.projection_scores = np.zeros(capacity, dtype=np.float32)
        self.sample_calls = 0
        self.priority_fallback_count = 0
        self._force_uniform_sampling = False
        self.priority_running_scales = {"td": 1.0, "constraint": 1.0, "projection": 1.0, "solver": 1.0}
        self.last_sampling_diagnostics: Dict[str, float] = {}

    def _initial_td_priority(self, size_before_add: int) -> float:
        if size_before_add <= 0:
            return 1.0
        maximum = float(np.nan_to_num(
            np.max(self.td_priorities[:size_before_add]), nan=1.0,
            posinf=self.cfg.max_replay_priority, neginf=self.cfg.priority_epsilon,
        ))
        return float(np.clip(maximum, self.cfg.priority_epsilon, self.cfg.max_replay_priority))

    @staticmethod
    def _normalize_projection_mse(raw_action: np.ndarray, executed_action: np.ndarray) -> float:
        # 两个动作均在 [-1,1]，逐维最大平方误差为 4。
        mse = float(np.mean(np.square(raw_action - executed_action)))
        return float(np.clip(mse / 4.0, 0.0, 1.0))

    def _compose_priority(self, indices: np.ndarray) -> np.ndarray:
        """TD、约束和投影三个独立分量只在此处组合一次。"""

        combined = (
            self._normalized_priority_component(self.td_priorities, indices, "td")
            + self._constraint_priority_weight()
            * self._normalized_priority_component(self.constraint_scores, indices, "constraint")
            + self.cfg.projection_priority_weight
            * self._normalized_priority_component(self.projection_scores, indices, "projection")
            + self.cfg.priority_epsilon
        )
        combined = np.nan_to_num(
            combined, nan=self.cfg.priority_epsilon,
            posinf=self.cfg.max_replay_priority, neginf=self.cfg.priority_epsilon,
        )
        return np.clip(combined, self.cfg.priority_epsilon, self.cfg.max_replay_priority)

    def _normalized_priority_component(self, values: np.ndarray, indices: np.ndarray,
                                       component: str) -> np.ndarray:
        mode = self.cfg.priority_component_normalization
        selected = np.asarray(values[indices], dtype=np.float64)
        if mode == "none":
            return selected
        valid_size = max(1, len(self))
        valid = np.nan_to_num(np.asarray(values[:valid_size], dtype=np.float64), nan=0.0,
                              posinf=self.cfg.max_replay_priority, neginf=0.0)
        if mode == "running_scale":
            observed_scale = max(float(np.percentile(np.abs(valid), 90)), self.cfg.priority_epsilon)
            old_scale = float(self.priority_running_scales.get(component, 1.0))
            scale = 0.99 * old_scale + 0.01 * observed_scale
            self.priority_running_scales[component] = scale
            return selected / max(scale, self.cfg.priority_epsilon)
        order = np.argsort(np.argsort(valid, kind="stable"), kind="stable").astype(np.float64)
        ranks = (order + 1.0) / max(float(valid_size), 1.0)
        return ranks[indices]

    def _constraint_priority_weight(self) -> float:
        if self.priority_role == "manager":
            return float(self.cfg.manager_constraint_priority_weight)
        return float(self.cfg.constraint_priority_weight)

    def _prioritized_enabled(self) -> bool:
        if self.priority_role == "manager":
            return bool(self.cfg.manager_use_prioritized_replay)
        return bool(self.cfg.use_prioritized_replay)

    def _beta_anneal_updates(self) -> int:
        return int(getattr(self.cfg, f"{self.priority_role}_priority_beta_anneal_updates"))

    def _priority_probability(self, size: int) -> np.ndarray:
        if self.cfg.priority_component_normalization != "none":
            valid_indices = np.arange(size, dtype=np.int64)
            # running_scale/rank depend on the current complete Replay. Cached
            # final priorities would otherwise retain obsolete scales or ranks.
            self.priorities[:size] = self._compose_priority(valid_indices)
        values = np.asarray(self.priorities[:size], dtype=np.float64)
        if not np.all(np.isfinite(values)) or float(np.max(values)) <= 0.0:
            self.priority_fallback_count += 1
            LOGGER.warning(
                "%s Replay priorities are non-finite or all zero; falling back to uniform sampling "
                "(fallback_count=%s).", self.priority_role, self.priority_fallback_count,
            )
            return np.full(size, 1.0 / size, dtype=np.float64)
        scaled = np.power(np.maximum(values, self.cfg.priority_epsilon), self.cfg.priority_alpha)
        total = float(np.sum(scaled))
        if (not np.all(np.isfinite(scaled))) or not np.isfinite(total) or total <= 0.0:
            self.priority_fallback_count += 1
            LOGGER.warning(
                "%s Replay priority distribution invalid; falling back to uniform sampling "
                "(fallback_count=%s).", self.priority_role, self.priority_fallback_count,
            )
            return np.full(size, 1.0 / size, dtype=np.float64)
        probabilities = scaled / total
        if not np.all(np.isfinite(probabilities)) or not np.isclose(float(probabilities.sum()), 1.0):
            self.priority_fallback_count += 1
            LOGGER.warning(
                "%s Replay probability normalization invalid; falling back to uniform sampling "
                "(fallback_count=%s).", self.priority_role, self.priority_fallback_count,
            )
            return np.full(size, 1.0 / size, dtype=np.float64)
        return probabilities

    def _set_new_priority(self, slot: int, size_before_add: int) -> None:
        self.td_priorities[slot] = self._initial_td_priority(size_before_add)
        indices = np.asarray([slot], dtype=np.int64)
        self.priorities[slot] = self._compose_priority(indices)[0]

    def _sample_indices(self, batch_size: int, size: int) -> Tuple[np.ndarray, np.ndarray]:
        if size <= 0:
            raise ValueError(f"Cannot sample from empty {self.priority_role} Replay")
        self.sample_calls += 1
        if self._force_uniform_sampling or not self._prioritized_enabled():
            indices = np.random.randint(0, size, size=batch_size)
            weights = np.ones((batch_size, 1), dtype=np.float32)
            self._record_sampling_diagnostics(indices, weights, size)
            return indices, weights
        priority_probability = self._priority_probability(size)
        fraction = float(self.cfg.prioritized_sample_fraction)
        # 每个 batch 位置独立选择采样源，任何小 batch 下边际分布都严格等于 p_mix。
        use_priority = np.random.random(batch_size) < fraction
        indices = np.empty(batch_size, dtype=np.int64)
        priority_count = int(np.count_nonzero(use_priority))
        if priority_count:
            indices[use_priority] = np.random.choice(
                size, size=priority_count, replace=True, p=priority_probability
            )
        if priority_count < batch_size:
            indices[~use_priority] = np.random.randint(0, size, size=batch_size - priority_count)
        probabilities = fraction * priority_probability + (1.0 - fraction) / size
        anneal = min(self.sample_calls / max(self._beta_anneal_updates(), 1), 1.0)
        beta = self.cfg.priority_beta_initial + anneal * (
            self.cfg.priority_beta_final - self.cfg.priority_beta_initial
        )
        weights = np.power(size * probabilities[indices], -beta)
        weights /= max(float(np.max(weights)), 1e-12)
        weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
        weights = weights.reshape(-1, 1).astype(np.float32)
        self._record_sampling_diagnostics(indices, weights, size)
        return indices, weights

    def sample_uniform(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Draw an actor batch uniformly, independently of PER TD-error priorities."""

        previous = self._force_uniform_sampling
        self._force_uniform_sampling = True
        try:
            return self.sample(batch_size)
        finally:
            self._force_uniform_sampling = previous

    @staticmethod
    def _summary(values: np.ndarray, prefix: str) -> Dict[str, float]:
        finite = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return {
            prefix + "mean": float(np.mean(finite)) if finite.size else 0.0,
            prefix + "p50": float(np.percentile(finite, 50)) if finite.size else 0.0,
            prefix + "p90": float(np.percentile(finite, 90)) if finite.size else 0.0,
            prefix + "p99": float(np.percentile(finite, 99)) if finite.size else 0.0,
        }

    def _record_sampling_diagnostics(self, indices: np.ndarray, weights: np.ndarray, size: int) -> None:
        valid_indices = np.arange(size, dtype=np.int64)
        td = self._normalized_priority_component(self.td_priorities, valid_indices, "td")
        constraint = self._constraint_priority_weight() * self._normalized_priority_component(
            self.constraint_scores, valid_indices, "constraint"
        )
        projection = self.cfg.projection_priority_weight * self._normalized_priority_component(
            self.projection_scores, valid_indices, "projection"
        )
        solver = self.cfg.manager_solver_priority_weight * self._normalized_priority_component(
            getattr(self, "solver_failure_seen", np.zeros((size, 1), dtype=np.float32)).reshape(-1),
            valid_indices, "solver",
        ) if self.priority_role == "manager" else np.zeros(size, dtype=np.float64)
        final = self.priorities[:size]
        flat_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        diagnostics: Dict[str, float] = {}
        diagnostics.update(self._summary(td, "td_priority_"))
        diagnostics.update(self._summary(constraint, "constraint_priority_"))
        diagnostics.update(self._summary(projection, "projection_priority_"))
        diagnostics.update(self._summary(solver, "solver_priority_"))
        diagnostics.update(self._summary(final, "final_priority_"))
        denominator = float(np.sum(np.square(flat_weights)))
        diagnostics.update({
            "solver_sample_ratio": float(np.mean(
                getattr(self, "solver_failure_seen", np.zeros((size, 1)))[indices]
            )) if size and indices.size else 0.0,
            "effective_sample_size": float(np.square(np.sum(flat_weights)) / max(denominator, 1e-12)),
            "is_weight_mean": float(np.mean(flat_weights)),
            "is_weight_min": float(np.min(flat_weights)),
            "is_weight_max": float(np.max(flat_weights)),
            "sampled_unique_ratio": float(np.unique(indices).size / max(indices.size, 1)),
            "priority_fallback_count": float(self.priority_fallback_count),
        })
        self.last_sampling_diagnostics = diagnostics

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        flat_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        flat_errors = np.asarray(td_errors, dtype=np.float32).reshape(-1)
        if flat_indices.size != flat_errors.size:
            raise ValueError("priority indices and TD errors must have the same length")
        if flat_indices.size == 0:
            return
        valid = (flat_indices >= 0) & (flat_indices < self.capacity)
        if not np.all(valid):
            self.priority_fallback_count += 1
            LOGGER.warning(
                "%s Replay ignored %s invalid priority indices (fallback_count=%s).",
                self.priority_role, int(np.count_nonzero(~valid)), self.priority_fallback_count,
            )
            flat_indices = flat_indices[valid]
            flat_errors = flat_errors[valid]
        if flat_indices.size == 0:
            return
        flat_errors = np.nan_to_num(
            flat_errors, nan=0.0, posinf=self.cfg.max_td_error_for_priority,
            neginf=-self.cfg.max_td_error_for_priority,
        )
        flat_errors = np.clip(np.abs(flat_errors), 0.0, self.cfg.max_td_error_for_priority)
        unique, inverse = np.unique(flat_indices, return_inverse=True)
        max_abs_errors = np.zeros(unique.size, dtype=np.float32)
        np.maximum.at(max_abs_errors, inverse, flat_errors)
        # log1p 控制灾难样本的尺度，同时保持 TD error 的单调排序。
        self.td_priorities[unique] = np.log1p(max_abs_errors)
        self.priorities[unique] = self._compose_priority(unique)

    def _priority_state_dict(self) -> Dict[str, Any]:
        stored_size = self.capacity if self.full else len(self)
        return {
            "priorities": self.priorities[:stored_size].copy(),
            "td_priorities": self.td_priorities[:stored_size].copy(),
            "constraint_scores": self.constraint_scores[:stored_size].copy(),
            "projection_scores": self.projection_scores[:stored_size].copy(),
            "sample_calls": int(self.sample_calls),
            "priority_fallback_count": int(self.priority_fallback_count),
            "priority_role": self.priority_role,
            "priority_running_scales": dict(self.priority_running_scales),
            "last_sampling_diagnostics": dict(self.last_sampling_diagnostics),
        }

    def _load_priority_state_dict(self, state: Mapping[str, Any]) -> None:
        schema = int(state.get("replay_schema_version", 1))
        stored_size = self.capacity if schema == 1 or bool(state.get("full", False)) else int(
            state.get("valid_size", len(self))
        )
        for name in ("priorities", "td_priorities", "constraint_scores", "projection_scores"):
            if name not in state:
                raise ValueError(f"Replay priority state is missing {name!r}")
            target = getattr(self, name)
            raw_value = np.asarray(state[name])
            if schema >= 2 and raw_value.dtype != target.dtype:
                raise ValueError(
                    f"Replay priority array {name} dtype mismatch: "
                    f"checkpoint={raw_value.dtype}, current={target.dtype}"
                )
            value = raw_value.astype(target.dtype, copy=False)
            expected_shape = (stored_size,)
            if value.shape != expected_shape:
                raise ValueError(
                    f"Replay priority array {name} shape mismatch: "
                    f"checkpoint={value.shape}, expected={expected_shape}"
                )
            target.fill(0.0)
            target[:stored_size] = value
        self.sample_calls = int(state.get("sample_calls", 0))
        self.priority_fallback_count = int(state.get("priority_fallback_count", 0))
        self.priority_running_scales.update(state.get("priority_running_scales", {}))
        self.last_sampling_diagnostics = dict(state.get("last_sampling_diagnostics", {}))


_ACTIVE_CONFIG: Optional[TrainConfig] = None


class FastReplayBuffer(_PrioritizedReplayMixin, legacy.FastReplayBuffer):
    """PER Fast Replay with raw Critic action and projected transition action."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, goal_dim: int,
                 device: torch.device, cfg: Optional[TrainConfig] = None):
        super().__init__(capacity, obs_dim, action_dim, goal_dim, device)
        # legacy 数组不再承载在线 Encoder 生成的陈旧 intrinsic/total reward。
        del self.reward_intrinsic
        del self.reward_total
        self._init_priority(capacity, cfg or _ACTIVE_CONFIG or TrainConfig(), "fast")
        self.physical_progress = np.zeros((capacity, 1), dtype=np.float32)
        self.projection_penalty = np.zeros((capacity, 1), dtype=np.float32)
        self.reward_global_safety = np.zeros((capacity, 1), dtype=np.float32)
        self.reward_role_specific = np.zeros((capacity, 1), dtype=np.float32)
        self.reward_raw_total = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs: np.ndarray, next_obs: np.ndarray, raw_action: np.ndarray, executed_action: np.ndarray,
            reward_external: float, physical_progress: float, projection_penalty: float, goal: np.ndarray,
            next_goal: np.ndarray, done: bool, constraint_score: float = 0.0,
            reward_clipped: bool = False, reward_global_safety: float = 0.0,
            reward_role_specific: float = 0.0, reward_raw_total: float = 0.0) -> None:
        slot = self.idx % self.capacity
        size_before = len(self)
        self.obs[slot] = obs
        self.next_obs[slot] = next_obs
        self.raw_actions[slot] = raw_action
        self.executed_actions[slot] = executed_action
        self.reward_external[slot, 0] = reward_external
        self.physical_progress[slot, 0] = physical_progress
        self.projection_penalty[slot, 0] = projection_penalty
        self.reward_global_safety[slot, 0] = reward_global_safety
        self.reward_role_specific[slot, 0] = reward_role_specific
        self.reward_raw_total[slot, 0] = reward_raw_total
        self.goals[slot] = goal
        self.next_goals[slot] = next_goal
        self.goal_changed[slot, 0] = float(np.linalg.norm(goal - next_goal) > 1e-6)
        self.dones[slot, 0] = float(done)
        self.duration_steps[slot, 0] = 1.0
        self.idx += 1
        self.full = self.full or self.idx >= self.capacity
        self.total_insertions += 1
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
            "reward_global_safety": legacy.to_tensor(self.reward_global_safety[indices], self.device),
            "reward_role_specific": legacy.to_tensor(self.reward_role_specific[indices], self.device),
            "reward_raw_total": legacy.to_tensor(self.reward_raw_total[indices], self.device),
            "goals": legacy.to_tensor(self.goals[indices], self.device),
            "next_goals": legacy.to_tensor(self.next_goals[indices], self.device),
            "goal_changed": legacy.to_tensor(self.goal_changed[indices], self.device),
            "dones": legacy.to_tensor(self.dones[indices], self.device),
            "duration_steps": legacy.to_tensor(self.duration_steps[indices], self.device),
            "constraint_scores": legacy.to_tensor(self.constraint_scores[indices, None], self.device),
            "projection_scores": legacy.to_tensor(self.projection_scores[indices, None], self.device),
            "indices": torch.as_tensor(indices, dtype=torch.long, device=self.device),
            "is_weights": legacy.to_tensor(is_weights, self.device),
        }

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state.update(self._priority_state_dict())
        stored_size = self.capacity if self.full else len(self)
        state.update({
            "physical_progress": self.physical_progress[:stored_size].copy(),
            "projection_penalty": self.projection_penalty[:stored_size].copy(),
            "reward_global_safety": self.reward_global_safety[:stored_size].copy(),
            "reward_role_specific": self.reward_role_specific[:stored_size].copy(),
            "reward_raw_total": self.reward_raw_total[:stored_size].copy(),
        })
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        schema = int(state.get("replay_schema_version", 1))
        stored_size = self.capacity if schema == 1 or self.full else len(self)
        for name in ("physical_progress", "projection_penalty", "reward_global_safety",
                     "reward_role_specific", "reward_raw_total"):
            target = getattr(self, name)
            raw_value = np.asarray(state[name])
            if schema >= 2 and raw_value.dtype != target.dtype:
                raise ValueError(f"Fast Replay {name} dtype mismatch: {raw_value.dtype}")
            value = raw_value.astype(target.dtype, copy=False)
            expected_shape = (stored_size,) + target.shape[1:]
            if value.shape != expected_shape:
                raise ValueError(f"Fast Replay {name} shape mismatch: {value.shape}, expected={expected_shape}")
            target.fill(0.0)
            target[:stored_size] = value
        self._load_priority_state_dict(state)


class SlowReplayBuffer(_PrioritizedReplayMixin, legacy.SlowReplayBuffer):
    """SMDP Slow Replay with raw/guarded and per-step executed action semantics.

    ``raw_actions`` feed the Worker Critic. ``guarded_actions`` are script-side
    ESS-forecast-guarded requests. Executed summaries/sequences describe the
    environment-projected transition and support imitation and diagnostics.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, goal_dim: int,
                 device: torch.device, cfg: Optional[TrainConfig] = None):
        super().__init__(capacity, obs_dim, action_dim, goal_dim, device)
        resolved_cfg = cfg or _ACTIVE_CONFIG or TrainConfig()
        self._init_priority(capacity, resolved_cfg, "slow")
        self.max_sequence_steps = int(resolved_cfg.slow_interval)
        self.external_reward = np.zeros((capacity, 1), dtype=np.float32)
        self.physical_progress = np.zeros((capacity, 1), dtype=np.float32)
        self.projection_penalty = np.zeros((capacity, 1), dtype=np.float32)
        self.segment_constraint_mean = np.zeros((capacity, 1), dtype=np.float32)
        self.segment_constraint_max = np.zeros((capacity, 1), dtype=np.float32)
        self.guarded_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.first_executed_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.last_executed_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.executed_action_variance = np.zeros((capacity, action_dim), dtype=np.float32)
        self.max_dynamic_projection = np.zeros((capacity, 1), dtype=np.float32)
        self.raw_to_guard_projection_rms = np.zeros((capacity, 1), dtype=np.float32)
        self.guard_to_executed_projection_rms = np.zeros((capacity, 1), dtype=np.float32)
        self.mean_executed_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.executed_action_sequences = np.zeros(
            (capacity, self.max_sequence_steps, action_dim), dtype=np.float32
        )
        self.executed_action_sequence_lengths = np.zeros((capacity, 1), dtype=np.int32)
        self.reward_global_safety = np.zeros((capacity, 1), dtype=np.float32)
        self.reward_role_specific = np.zeros((capacity, 1), dtype=np.float32)
        self.reward_raw_total = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs_start: np.ndarray, obs_end: np.ndarray, raw_action: np.ndarray,
            executed_action: np.ndarray, discounted_external_reward: float,
            physical_progress: float, projection_penalty: float, goal: np.ndarray,
            next_goal: np.ndarray, done: bool, duration_steps: int,
            constraint_score: float = 0.0, segment_constraint_mean: float = 0.0,
            segment_constraint_max: float = 0.0, reward_clipped: bool = False,
            guarded_action: Optional[np.ndarray] = None,
            first_executed_action: Optional[np.ndarray] = None,
            last_executed_action: Optional[np.ndarray] = None,
            executed_action_variance: Optional[np.ndarray] = None,
            max_dynamic_projection: float = 0.0,
            mean_executed_action: Optional[np.ndarray] = None,
            executed_action_sequence: Optional[np.ndarray] = None,
            reward_global_safety: float = 0.0,
            reward_role_specific: float = 0.0,
            reward_raw_total: float = 0.0) -> None:
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
        self.total_insertions += 1
        self.constraint_scores[slot] = float(np.clip(constraint_score, 0.0, 1.0))
        self.segment_constraint_mean[slot, 0] = float(np.clip(segment_constraint_mean, 0.0, 1.0))
        self.segment_constraint_max[slot, 0] = float(np.clip(segment_constraint_max, 0.0, 1.0))
        guarded = np.asarray(guarded_action if guarded_action is not None else executed_action,
                             dtype=np.float32)
        first_executed = np.asarray(
            first_executed_action if first_executed_action is not None else executed_action,
            dtype=np.float32,
        )
        last_executed = np.asarray(
            last_executed_action if last_executed_action is not None else executed_action,
            dtype=np.float32,
        )
        variance = np.asarray(
            executed_action_variance if executed_action_variance is not None
            else np.zeros_like(executed_action), dtype=np.float32,
        )
        self.guarded_actions[slot] = guarded
        self.first_executed_actions[slot] = first_executed
        self.last_executed_actions[slot] = last_executed
        self.executed_action_variance[slot] = np.maximum(variance, 0.0)
        self.max_dynamic_projection[slot, 0] = max(float(max_dynamic_projection), 0.0)
        self.mean_executed_actions[slot] = np.asarray(
            mean_executed_action if mean_executed_action is not None else executed_action,
            dtype=np.float32,
        )
        if executed_action_sequence is None:
            sequence = np.repeat(
                np.asarray(executed_action, dtype=np.float32)[None, :],
                int(duration_steps), axis=0,
            )
        else:
            sequence = np.asarray(executed_action_sequence, dtype=np.float32)
        if sequence.ndim != 2 or sequence.shape[1] != self.raw_actions.shape[1]:
            raise ValueError(
                f"Slow executed action sequence shape={sequence.shape} must be "
                f"(duration_steps, {self.raw_actions.shape[1]})"
            )
        if sequence.shape[0] != int(duration_steps):
            raise ValueError(
                f"Slow executed action sequence length={sequence.shape[0]} does not match "
                f"duration_steps={duration_steps}"
            )
        if not 0 < sequence.shape[0] <= self.max_sequence_steps:
            raise ValueError(
                f"Slow executed action sequence length={sequence.shape[0]} must be in "
                f"[1,{self.max_sequence_steps}]"
            )
        self.executed_action_sequences[slot].fill(0.0)
        self.executed_action_sequences[slot, :sequence.shape[0]] = sequence
        self.executed_action_sequence_lengths[slot, 0] = sequence.shape[0]
        self.reward_global_safety[slot, 0] = reward_global_safety
        self.reward_role_specific[slot, 0] = reward_role_specific
        self.reward_raw_total[slot, 0] = reward_raw_total
        raw_guard_mse = float(np.mean(np.square(np.asarray(raw_action) - guarded)))
        guard_executed_mse = float(np.mean(np.square(guarded - np.asarray(executed_action))))
        self.raw_to_guard_projection_rms[slot, 0] = math.sqrt(max(raw_guard_mse, 0.0))
        self.guard_to_executed_projection_rms[slot, 0] = math.sqrt(
            max(guard_executed_mse, 0.0)
        )
        # PER must see the largest safety intervention.  In the diagnosed run
        # raw->ESS-guard RMS was 0.475 while guard->environment was ~0, so using
        # only the latter made projection priority effectively vanish.
        self.projection_scores[slot] = max(
            self._normalize_projection_mse(raw_action, guarded),
            self._normalize_projection_mse(guarded, executed_action),
        )
        self._set_new_priority(slot, size_before)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        size = len(self)
        indices, is_weights = self._sample_indices(batch_size, size)
        return {
            "obs": legacy.to_tensor(self.obs_start[indices], self.device),
            "next_obs": legacy.to_tensor(self.obs_end[indices], self.device),
            "raw_actions": legacy.to_tensor(self.raw_actions[indices], self.device),
            "executed_actions": legacy.to_tensor(self.executed_actions[indices], self.device),
            "guarded_actions": legacy.to_tensor(self.guarded_actions[indices], self.device),
            "first_executed_actions": legacy.to_tensor(self.first_executed_actions[indices], self.device),
            "last_executed_actions": legacy.to_tensor(self.last_executed_actions[indices], self.device),
            "executed_action_variance": legacy.to_tensor(self.executed_action_variance[indices], self.device),
            "max_dynamic_projection": legacy.to_tensor(self.max_dynamic_projection[indices], self.device),
            "raw_to_guard_projection_rms": legacy.to_tensor(
                self.raw_to_guard_projection_rms[indices], self.device
            ),
            "guard_to_executed_projection_rms": legacy.to_tensor(
                self.guard_to_executed_projection_rms[indices], self.device
            ),
            "mean_executed_actions": legacy.to_tensor(self.mean_executed_actions[indices], self.device),
            "executed_action_sequences": legacy.to_tensor(
                self.executed_action_sequences[indices], self.device
            ),
            "executed_action_sequence_lengths": torch.as_tensor(
                self.executed_action_sequence_lengths[indices], dtype=torch.long, device=self.device
            ),
            "executed_action_sequence_mask": torch.arange(
                self.max_sequence_steps, device=self.device
            )[None, :] < torch.as_tensor(
                self.executed_action_sequence_lengths[indices], dtype=torch.long, device=self.device
            ),
            "reward_external": legacy.to_tensor(self.external_reward[indices], self.device),
            "physical_progress": legacy.to_tensor(self.physical_progress[indices], self.device),
            "projection_penalty": legacy.to_tensor(self.projection_penalty[indices], self.device),
            "reward_global_safety": legacy.to_tensor(self.reward_global_safety[indices], self.device),
            "reward_role_specific": legacy.to_tensor(self.reward_role_specific[indices], self.device),
            "reward_raw_total": legacy.to_tensor(self.reward_raw_total[indices], self.device),
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

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state.update(self._priority_state_dict())
        stored_size = self.capacity if self.full else len(self)
        for name in ("external_reward", "physical_progress", "projection_penalty",
                     "segment_constraint_mean", "segment_constraint_max", "guarded_actions",
                     "first_executed_actions", "last_executed_actions",
                     "executed_action_variance", "max_dynamic_projection",
                     "raw_to_guard_projection_rms", "guard_to_executed_projection_rms",
                     "mean_executed_actions",
                     "executed_action_sequences", "executed_action_sequence_lengths",
                     "reward_global_safety", "reward_role_specific", "reward_raw_total"):
            state[name] = getattr(self, name)[:stored_size].copy()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        schema = int(state.get("replay_schema_version", 1))
        stored_size = self.capacity if schema == 1 or self.full else len(self)
        for name in ("external_reward", "physical_progress", "projection_penalty",
                     "segment_constraint_mean", "segment_constraint_max", "guarded_actions",
                     "first_executed_actions", "last_executed_actions",
                     "executed_action_variance", "max_dynamic_projection",
                     "raw_to_guard_projection_rms", "guard_to_executed_projection_rms",
                     "mean_executed_actions",
                     "executed_action_sequences", "executed_action_sequence_lengths",
                     "reward_global_safety", "reward_role_specific", "reward_raw_total"):
            target = getattr(self, name)
            raw_value = np.asarray(state[name])
            if schema >= 2 and raw_value.dtype != target.dtype:
                raise ValueError(f"Slow Replay {name} dtype mismatch: {raw_value.dtype}")
            value = raw_value.astype(target.dtype, copy=False)
            expected_shape = (stored_size,) + target.shape[1:]
            if value.shape != expected_shape:
                raise ValueError(f"Slow Replay {name} shape mismatch: {value.shape}, expected={expected_shape}")
            target.fill(0.0)
            target[:stored_size] = value
        self._load_priority_state_dict(state)


class ManagerReplayBuffer(_PrioritizedReplayMixin, _LEGACY_MANAGER_REPLAY_BUFFER_CLASS):
    def __init__(self, capacity: int, obs_dim: int, goal_dim: int, device: torch.device,
                 cfg: Optional[TrainConfig] = None):
        _LEGACY_MANAGER_REPLAY_BUFFER_CLASS.__init__(self, capacity, obs_dim, goal_dim, device)
        self._init_priority(capacity, cfg or _ACTIVE_CONFIG or TrainConfig(), "manager")
        self.segment_constraint_mean = np.zeros((capacity, 1), dtype=np.float32)
        self.segment_constraint_max = np.zeros((capacity, 1), dtype=np.float32)
        self.solver_failure_seen = np.zeros((capacity, 1), dtype=np.float32)

    def _compose_priority(self, indices: np.ndarray) -> np.ndarray:
        combined = (
            self._normalized_priority_component(self.td_priorities, indices, "td")
            + self.cfg.manager_constraint_priority_weight
            * self._normalized_priority_component(self.constraint_scores, indices, "constraint")
            + self.cfg.manager_solver_priority_weight
            * self._normalized_priority_component(self.solver_failure_seen[:, 0], indices, "solver")
            + self.cfg.priority_epsilon
        )
        combined = np.nan_to_num(
            combined, nan=self.cfg.priority_epsilon,
            posinf=self.cfg.max_replay_priority, neginf=self.cfg.priority_epsilon,
        )
        return np.clip(combined, self.cfg.priority_epsilon, self.cfg.max_replay_priority)

    def add(self, global_obs_start: np.ndarray, global_obs_end: np.ndarray, manager_goal: np.ndarray,
            discounted_external_reward: float, done: bool, duration_steps: int,
            segment_constraint_mean: float = 0.0, segment_constraint_max: float = 0.0,
            solver_failure_seen: bool = False, raw_goal: Optional[np.ndarray] = None,
            previous_executed_goal: Optional[np.ndarray] = None) -> None:
        slot = self.idx % self.capacity
        size_before = len(self)
        _LEGACY_MANAGER_REPLAY_BUFFER_CLASS.add(
            self, global_obs_start, global_obs_end, manager_goal,
            discounted_external_reward, done, duration_steps,
            segment_constraint_mean, segment_constraint_max, solver_failure_seen,
            raw_goal, previous_executed_goal,
        )
        mean_score = float(np.clip(np.nan_to_num(segment_constraint_mean, nan=0.0), 0.0, 1.0))
        max_score = float(np.clip(np.nan_to_num(segment_constraint_max, nan=0.0), 0.0, 1.0))
        self.segment_constraint_mean[slot, 0] = mean_score
        self.segment_constraint_max[slot, 0] = max_score
        self.solver_failure_seen[slot, 0] = float(bool(solver_failure_seen))
        self.constraint_scores[slot] = max(0.5 * mean_score + 0.5 * max_score,
                                           float(bool(solver_failure_seen)))
        self.projection_scores[slot] = 0.0
        self._set_new_priority(slot, size_before)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        size = len(self)
        indices, is_weights = self._sample_indices(batch_size, size)
        return {
            "obs": legacy.to_tensor(self.global_obs_start[indices], self.device),
            "next_obs": legacy.to_tensor(self.global_obs_end[indices], self.device),
            "goals": legacy.to_tensor(self.manager_goals[indices], self.device),
            "executed_goals": legacy.to_tensor(self.manager_goals[indices], self.device),
            "raw_goals": legacy.to_tensor(self.raw_goals[indices], self.device),
            "previous_executed_goals": legacy.to_tensor(self.previous_executed_goals[indices], self.device),
            "rewards": legacy.to_tensor(self.discounted_external_reward[indices], self.device),
            "dones": legacy.to_tensor(self.dones[indices], self.device),
            "duration_steps": legacy.to_tensor(self.duration_steps[indices], self.device),
            "segment_constraint_mean": legacy.to_tensor(self.segment_constraint_mean[indices], self.device),
            "segment_constraint_max": legacy.to_tensor(self.segment_constraint_max[indices], self.device),
            "solver_failure_seen": legacy.to_tensor(self.solver_failure_seen[indices], self.device),
            "indices": torch.as_tensor(indices, dtype=torch.long, device=self.device),
            "is_weights": legacy.to_tensor(is_weights, self.device),
        }

    def state_dict(self) -> Dict[str, Any]:
        state = _LEGACY_MANAGER_REPLAY_BUFFER_CLASS.state_dict(self)
        state.update(self._priority_state_dict())
        stored_size = self.capacity if self.full else len(self)
        for name in ("segment_constraint_mean", "segment_constraint_max", "solver_failure_seen"):
            state[name] = getattr(self, name)[:stored_size].copy()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        _LEGACY_MANAGER_REPLAY_BUFFER_CLASS.load_state_dict(self, state)
        schema = int(state.get("replay_schema_version", 1))
        stored_size = self.capacity if schema == 1 or self.full else len(self)
        for name in ("segment_constraint_mean", "segment_constraint_max", "solver_failure_seen"):
            target = getattr(self, name)
            raw_value = np.asarray(state[name])
            if schema >= 2 and raw_value.dtype != target.dtype:
                raise ValueError(f"Manager Replay {name} dtype mismatch: {raw_value.dtype}")
            value = raw_value.astype(target.dtype, copy=False)
            expected_shape = (stored_size,) + target.shape[1:]
            if value.shape != expected_shape:
                raise ValueError(f"Manager Replay {name} shape mismatch: {value.shape}, expected={expected_shape}")
            target.fill(0.0)
            target[:stored_size] = value
        self._load_priority_state_dict(state)


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
    reward_global_safety: float = 0.0
    reward_role_specific: float = 0.0
    reward_raw_total: float = 0.0
    reward_component_clipped: float = 0.0
    constraint_score: float = field(init=False)

    def __post_init__(self) -> None:
        self.constraint_score = float(_LAST_CONSTRAINT_SCORE)


@dataclass
class PendingSlowSegment:
    obs_start: np.ndarray
    goal: np.ndarray
    raw_action: np.ndarray
    guarded_action: np.ndarray = field(init=False)
    executed_action: Optional[np.ndarray] = None
    first_executed_action: Optional[np.ndarray] = None
    last_executed_action: Optional[np.ndarray] = None
    executed_sum: Optional[np.ndarray] = None
    executed_square_sum: Optional[np.ndarray] = None
    executed_plain_sum: Optional[np.ndarray] = None
    executed_plain_square_sum: Optional[np.ndarray] = None
    executed_count: int = 0
    executed_actions_by_step: List[np.ndarray] = field(default_factory=list)
    executed_weight_sum: float = 0.0
    max_dynamic_projection: float = 0.0
    discounted_reward: float = 0.0
    discounted_global_safety_reward: float = 0.0
    discounted_role_specific_reward: float = 0.0
    discounted_raw_total_reward: float = 0.0
    component_clipped_steps: int = 0
    projection_penalty_sum: float = 0.0
    duration_steps: int = 0
    constraint_scores: List[float] = field(default_factory=list)
    solver_failure_seen: bool = False

    def __post_init__(self) -> None:
        global _CURRENT_SLOW_PENDING
        self.guarded_action = np.asarray(self.raw_action, dtype=np.float32).copy()
        if _LAST_SLOW_ACTOR_RAW is not None and _LAST_SLOW_ACTOR_RAW.shape == self.raw_action.shape:
            self.raw_action = _LAST_SLOW_ACTOR_RAW.copy()
        _CURRENT_SLOW_PENDING = self

    def record_executed_action(self, action: np.ndarray, gamma_fast: float) -> None:
        executed = np.asarray(action, dtype=np.float32).reshape(self.guarded_action.shape)
        if self.first_executed_action is None:
            self.first_executed_action = executed.copy()
        self.last_executed_action = executed.copy()
        if self.executed_sum is None:
            self.executed_sum = np.zeros_like(executed)
            self.executed_square_sum = np.zeros_like(executed)
            self.executed_plain_sum = np.zeros_like(executed)
            self.executed_plain_square_sum = np.zeros_like(executed)
        weight = float(gamma_fast ** self.duration_steps)
        self.executed_sum += weight * executed
        assert self.executed_square_sum is not None
        self.executed_square_sum += weight * np.square(executed)
        self.executed_weight_sum += weight
        assert self.executed_plain_sum is not None and self.executed_plain_square_sum is not None
        self.executed_plain_sum += executed
        self.executed_plain_square_sum += np.square(executed)
        self.executed_count += 1
        self.executed_actions_by_step.append(executed.copy())
        dynamic_rms = float(np.sqrt(np.mean(np.square(self.guarded_action - executed))))
        self.max_dynamic_projection = max(self.max_dynamic_projection, dynamic_rms)

    def executed_summary(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.executed_sum is None or self.executed_weight_sum <= 0.0:
            mean = self.guarded_action.copy()
            return mean, mean.copy(), np.zeros_like(mean)
        discounted_mean = self.executed_sum / self.executed_weight_sum
        assert self.executed_plain_sum is not None and self.executed_plain_square_sum is not None
        mean = self.executed_plain_sum / max(self.executed_count, 1)
        variance = np.maximum(
            self.executed_plain_square_sum / max(self.executed_count, 1) - np.square(mean), 0.0
        )
        return discounted_mean.astype(np.float32), mean.astype(np.float32), variance.astype(np.float32)


def safe_env_step(env: ElectricGasMultiScaleEnv, action: np.ndarray, last_obs: np.ndarray):
    global _LAST_CONSTRAINT_SCORE
    result = _LEGACY_SAFE_ENV_STEP(env, action, last_obs)
    info = result[4]
    _LAST_CONSTRAINT_SCORE = normalized_constraint_score(info, env)
    info["_normalized_constraint_score"] = _LAST_CONSTRAINT_SCORE
    if _CURRENT_SLOW_PENDING is not None:
        _CURRENT_SLOW_PENDING.constraint_scores.append(_LAST_CONSTRAINT_SCORE)
        _CURRENT_SLOW_PENDING.solver_failure_seen = (
            _CURRENT_SLOW_PENDING.solver_failure_seen or bool(info.get("solver_failed", False))
        )
        applied = np.asarray(info.get("applied_action", action), dtype=np.float32)
        if applied.shape == np.asarray(action).shape:
            _CURRENT_SLOW_PENDING.record_executed_action(
                applied[:env.slow_action_dim],
                float(_ACTIVE_CONFIG.gamma_fast if _ACTIVE_CONFIG is not None else 0.99),
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
    del fast_agent, cfg
    physical = fast_physical_progress(pending.obs, next_obs, pending.goal)
    buffer.add(
        pending.obs, next_obs, pending.raw_action, pending.executed_action,
        pending.reward_external, physical, pending.projection, pending.goal, next_goal, pending.done,
        constraint_score=pending.constraint_score,
        reward_global_safety=pending.reward_global_safety,
        reward_role_specific=pending.reward_role_specific,
        reward_raw_total=pending.reward_raw_total,
    )
    return {
        "interaction/fast/external_reward": pending.reward_external,
        "interaction/fast/physical_progress": physical,
        "interaction/fast/projection_penalty": pending.projection,
        "interaction/fast/constraint_score": pending.constraint_score,
    }


def finalize_slow_segment(pending: PendingSlowSegment, obs_end: np.ndarray, next_goal: np.ndarray,
                          slow_agent: "WorkerTD3", buffer: SlowReplayBuffer, cfg: TrainConfig,
                          done: bool) -> Dict[str, float]:
    global _CURRENT_SLOW_PENDING
    executed_mean, ordinary_mean, executed_variance = pending.executed_summary()
    first_executed = (
        pending.first_executed_action.copy()
        if pending.first_executed_action is not None else pending.guarded_action.copy()
    )
    last_executed = (
        pending.last_executed_action.copy()
        if pending.last_executed_action is not None else first_executed.copy()
    )
    del slow_agent, cfg
    physical = slow_physical_progress(pending.obs_start, obs_end, pending.goal)
    segment_mean, segment_max, constraint_score = aggregate_segment_constraint(
        pending.constraint_scores, pending.solver_failure_seen
    )
    buffer.add(
        pending.obs_start, obs_end, pending.raw_action, executed_mean, pending.discounted_reward,
        physical, pending.projection_penalty_sum, pending.goal, next_goal, done,
        pending.duration_steps, constraint_score=constraint_score,
        segment_constraint_mean=segment_mean, segment_constraint_max=segment_max,
        guarded_action=pending.guarded_action,
        first_executed_action=first_executed,
        last_executed_action=last_executed,
        executed_action_variance=executed_variance,
        max_dynamic_projection=pending.max_dynamic_projection,
        mean_executed_action=ordinary_mean,
        executed_action_sequence=np.stack(pending.executed_actions_by_step, axis=0),
        reward_global_safety=pending.discounted_global_safety_reward,
        reward_role_specific=pending.discounted_role_specific_reward,
        reward_raw_total=pending.discounted_raw_total_reward,
    )
    if _CURRENT_SLOW_PENDING is pending:
        _CURRENT_SLOW_PENDING = None
    return {
        "interaction/slow/external_reward": pending.discounted_reward,
        "interaction/slow/physical_progress": physical,
        "interaction/slow/projection_penalty": pending.projection_penalty_sum,
        "interaction/slow/segment_constraint_mean": segment_mean,
        "interaction/slow/segment_constraint_max": segment_max,
        "interaction/slow/constraint_score": constraint_score,
        "interaction/slow/raw_to_guarded_projection_rms": float(np.sqrt(np.mean(
            np.square(pending.raw_action - pending.guarded_action)
        ))),
        "interaction/slow/guarded_to_executed_projection_rms": float(np.sqrt(np.mean(
            np.square(pending.guarded_action - executed_mean)
        ))),
        "interaction/slow/max_dynamic_projection_rms": pending.max_dynamic_projection,
        "interaction/slow/ordinary_mean_executed_action_norm": float(np.linalg.norm(ordinary_mean)),
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


def require_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not torch.is_tensor(tensor) or not bool(torch.isfinite(tensor).all()):
        finite = tensor[torch.isfinite(tensor)] if torch.is_tensor(tensor) else torch.empty(0)
        finite_range = (
            (float(finite.min().detach().cpu()), float(finite.max().detach().cpu()))
            if finite.numel() else (None, None)
        )
        raise FloatingPointError(f"non-finite tensor {name}; finite_range={finite_range}")


def require_finite_gradients(name: str, module: torch.nn.Module) -> float:
    squared_norm = torch.zeros((), dtype=torch.float64)
    for parameter_name, parameter in module.named_parameters():
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(f"non-finite gradient {name}.{parameter_name}")
        squared_norm += parameter.grad.detach().double().pow(2).sum().cpu()
    norm = float(torch.sqrt(squared_norm))
    if not np.isfinite(norm):
        raise FloatingPointError(f"non-finite gradient norm {name}={norm}")
    return norm


def _clear_agent_gradients(agent: Any) -> None:
    for name in ("encoder_optim", "actor_optim", "critic_optim", "transition_optim"):
        optimizer = getattr(agent, name, None)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)


def _clear_non_actor_gradients(agent: Any) -> None:
    for name in ("encoder_optim", "critic_optim", "transition_optim"):
        optimizer = getattr(agent, name, None)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)


def _handle_nonfinite_update(agent: Any, role: str, data: Mapping[str, Any],
                             error: FloatingPointError) -> Dict[str, float]:
    _clear_agent_gradients(agent)
    agent.nonfinite_batch_count = int(getattr(agent, "nonfinite_batch_count", 0)) + 1
    indices = data.get("indices")
    index_values = (
        indices.detach().cpu().tolist() if torch.is_tensor(indices) else []
    )
    LOGGER.error("Skipped non-finite %s update=%s indices=%s: %s",
                 role, agent.total_updates, index_values, error)
    if agent.cfg.nonfinite_update_policy == "raise":
        root = Path(agent.cfg.checkpoint_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"emergency_{role}_{agent.total_updates}.pt"
        temp_path = path.with_name(path.name + f".tmp-{os.getpid()}")
        payload = {
            "role": role,
            "update_count": int(agent.total_updates),
            "error": str(error),
            "replay_indices": index_values,
            "agent": agent.state_dict(),
        }
        try:
            torch.save(payload, str(temp_path))
            os.replace(str(temp_path), str(path))
        finally:
            if temp_path.exists():
                temp_path.unlink()
        raise error
    return {
        f"training_batch/{role}/nonfinite_batch_skipped": 1.0,
        f"training_batch/{role}/nonfinite_batch_count": float(agent.nonfinite_batch_count),
    }


@contextmanager
def _frozen_parameters(module: torch.nn.Module):
    flags = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in zip(module.parameters(), flags):
            parameter.requires_grad_(enabled)


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
        self._last_update_insertion_id = 0
        self.nonfinite_batch_count = 0
        self.critic_updates = 0
        self.actor_updates = 0

    def select_goal_pair(self, obs: np.ndarray, previous_goal: Optional[np.ndarray], noise_std: float,
                         deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        actor_trainable = any(parameter.requires_grad for parameter in self.actor.parameters())
        return super().select_goal_pair(
            obs, previous_goal, noise_std if actor_trainable else 0.0,
            deterministic=deterministic or not actor_trainable,
        )

    def select_goal(self, obs: np.ndarray, previous_goal: Optional[np.ndarray], noise_std: float,
                    deterministic: bool = False) -> np.ndarray:
        actor_trainable = any(parameter.requires_grad for parameter in self.actor.parameters())
        return super().select_goal(
            obs, previous_goal, noise_std if actor_trainable else 0.0,
            deterministic=deterministic or not actor_trainable,
        )

    def _update_once_impl(self, buffer: ManagerReplayBuffer, batch_size: int) -> Dict[str, float]:
        data = buffer.sample(batch_size)
        obs = legacy.to_tensor(self.normalizer.normalize(data["obs"].cpu().numpy()), self.device)
        next_obs = legacy.to_tensor(self.normalizer.normalize(data["next_obs"].cpu().numpy()), self.device)
        goals = data.get("executed_goals", data["goals"])
        previous_goals = data.get("previous_executed_goals", torch.zeros_like(goals))
        raw_goals = data.get("raw_goals", goals)
        rewards, dones = data["rewards"], data["dones"]
        durations = data["duration_steps"]
        is_weights = data["is_weights"]
        for name, tensor in (
            ("normalized_obs", obs), ("normalized_next_obs", next_obs), ("goal", goals),
            ("reward", rewards), ("done", dones), ("duration", durations),
            ("is_weights", is_weights),
        ):
            require_finite_tensor(f"manager/{name}", tensor)

        z = self.encoder(obs)
        with torch.no_grad():
            next_z = self.target_encoder(next_obs)
            next_raw_goal = self.target_actor(next_z)
            noise = (torch.randn_like(next_raw_goal) * self.cfg.target_noise).clamp(
                -self.cfg.target_noise_clip, self.cfg.target_noise_clip
            )
            if self.cfg.target_noise > 0.0:
                next_raw_goal = legacy.normalize_goal_tensor(next_raw_goal + noise)
            next_goal = legacy.execute_manager_goal_tensor(
                next_raw_goal, goals, self.cfg.goal_smoothing
            )
            q1_next, q2_next = self.target_critic(next_z, next_goal)
            target_q_unclipped = build_smdp_target(
                rewards, dones, torch.minimum(q1_next, q2_next), self.cfg.gamma_fast, durations
            )
            target_q = target_q_unclipped
            if self.cfg.target_q_clip_abs > 0.0:
                target_q = target_q.clamp(-self.cfg.target_q_clip_abs, self.cfg.target_q_clip_abs)
            target_q_clipping_ratio = float((target_q != target_q_unclipped).float().mean().cpu())

        q1, q2 = self.critic(z, goals)
        critic_loss = (
            is_weights * (
                F.smooth_l1_loss(q1, target_q, reduction="none")
                + F.smooth_l1_loss(q2, target_q, reduction="none")
            )
        ).mean()
        td_error = 0.5 * ((q1 - target_q).abs() + (q2 - target_q).abs())
        encoder_loss = critic_loss + self.cfg.lambda_latent_norm * z.pow(2).mean()
        for name, tensor in (("target_q", target_q), ("q1", q1), ("q2", q2),
                             ("critic_loss", critic_loss), ("encoder_loss", encoder_loss)):
            require_finite_tensor(f"manager/{name}", tensor)
        self.critic_optim.zero_grad()
        self.encoder_optim.zero_grad()
        encoder_loss.backward()
        require_finite_gradients("manager/critic", self.critic)
        require_finite_gradients("manager/encoder", self.encoder)
        critic_grad = _clip_grad_norm(self.critic.parameters(), self.cfg.gradient_clip)
        encoder_grad = _clip_grad_norm(self.encoder.parameters(), self.cfg.gradient_clip)
        if not np.isfinite(critic_grad) or not np.isfinite(encoder_grad):
            raise FloatingPointError(
                f"manager non-finite clipped gradient norm critic={critic_grad}, encoder={encoder_grad}"
            )

        actor_loss_value = 0.0
        actor_grad = 0.0
        actor_saturation_ratio = 0.0
        actor_update = (
            any(parameter.requires_grad for parameter in self.actor.parameters())
            and self.total_updates % self.cfg.policy_frequency == 0
        )
        if actor_update:
            actor_data = buffer.sample_uniform(batch_size)
            actor_obs = legacy.to_tensor(
                self.normalizer.normalize(actor_data["obs"].cpu().numpy()), self.device
            )
            actor_previous_goals = actor_data.get(
                "previous_executed_goals", torch.zeros_like(actor_data["goals"])
            )
            z_pi = self.encoder(actor_obs).detach()
            self.actor_optim.zero_grad()
            with _frozen_parameters(self.critic):
                actor_raw_goal = self.actor(z_pi)
                actor_goal = legacy.execute_manager_goal_tensor(
                    actor_raw_goal, actor_previous_goals, self.cfg.goal_smoothing
                )
                actor_saturation_ratio = float((actor_raw_goal.abs() >= 0.98).float().mean().detach().cpu())
                actor_loss = -self.critic.q_min(z_pi, actor_goal).mean()
                require_finite_tensor("manager/actor_loss", actor_loss)
                actor_loss.backward()
            require_finite_gradients("manager/actor", self.actor)
            actor_grad = _clip_grad_norm(self.actor.parameters(), self.cfg.gradient_clip)
            if not np.isfinite(actor_grad):
                raise FloatingPointError(f"manager non-finite actor gradient norm={actor_grad}")

        # All losses and gradients are finite. Optimizers now commit as one batch.
        self.critic_optim.step()
        self.encoder_optim.step()
        if actor_update:
            self.actor_optim.step()
            actor_loss_value = float(actor_loss.detach().cpu())
            legacy.soft_update(self.target_actor, self.actor, self.cfg.tau)
            legacy.soft_update(self.target_critic, self.critic, self.cfg.tau)
            legacy.soft_update(self.target_encoder, self.encoder, self.cfg.tau)
            self.actor_updates += 1
        if self.cfg.manager_use_prioritized_replay:
            buffer.update_priorities(
                data["indices"].detach().cpu().numpy(), td_error.detach().cpu().numpy()
            )
        _clear_non_actor_gradients(self)
        self.critic_updates += 1
        self.total_updates += 1
        clipped_ratio = 0.0
        if self.cfg.manager_reward_clip_abs > 0.0:
            clipped_ratio = float(
                (rewards.abs() >= self.cfg.manager_reward_clip_abs - 1e-6).float().mean()
            )
        prefix = "training_batch/manager/"
        logs = {
            prefix + "critic_loss": float(critic_loss.detach().cpu()),
            prefix + "actor_loss": actor_loss_value,
            prefix + "target_q_mean": float(target_q.mean().detach().cpu()),
            prefix + "target_q_std": float(target_q.std(unbiased=False).detach().cpu()),
            prefix + "target_q_abs_max": float(target_q.abs().max().detach().cpu()),
            prefix + "q1_mean": float(q1.mean().detach().cpu()),
            prefix + "q2_mean": float(q2.mean().detach().cpu()),
            prefix + "td_error": float(td_error.mean().detach().cpu()),
            prefix + "critic_grad_norm": critic_grad,
            prefix + "encoder_grad_norm": encoder_grad,
            prefix + "actor_grad_norm": actor_grad,
            prefix + "reward_clipping_ratio": clipped_ratio,
            prefix + "target_q_clipping_ratio": target_q_clipping_ratio,
            prefix + "actor_action_saturation_ratio": actor_saturation_ratio,
            prefix + "mean_duration_steps": float(durations.mean().detach().cpu()),
            prefix + "priority_fallback_count": float(buffer.priority_fallback_count),
            prefix + "replay_raw_to_executed_goal_rms": float(
                torch.sqrt(torch.mean((raw_goals - goals).pow(2))).detach().cpu()
            ),
        }
        logs.update({prefix + "replay/" + key: value
                     for key, value in buffer.last_sampling_diagnostics.items()})
        return logs

    def _update_once(self, buffer: ManagerReplayBuffer, batch_size: int) -> Dict[str, float]:
        data_holder: Dict[str, Any] = {}
        original_sample = buffer.sample

        def tracked_sample(size: int) -> Dict[str, torch.Tensor]:
            sampled = original_sample(size)
            data_holder.update(sampled)
            return sampled

        buffer.sample = tracked_sample  # type: ignore[assignment]
        try:
            return self._update_once_impl(buffer, batch_size)
        except FloatingPointError as error:
            return _handle_nonfinite_update(self, "manager", data_holder, error)
        finally:
            buffer.sample = original_sample  # type: ignore[assignment]

    def update(self, buffer: ManagerReplayBuffer, batch_size: int = 0) -> Dict[str, float]:
        del batch_size
        if len(buffer) < max(self.cfg.manager_learning_starts, self.cfg.manager_batch_size):
            return {}
        insertion_id = int(getattr(buffer, "total_insertions", buffer.idx))
        if insertion_id == self._last_update_insertion_id:
            return {}
        logs = [self._update_once(buffer, self.cfg.manager_batch_size)
                for _ in range(self.cfg.manager_updates_per_boundary)]
        self._last_update_insertion_id = insertion_id
        return _mean_logs(logs)

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["nonfinite_batch_count"] = int(self.nonfinite_batch_count)
        state["last_update_insertion_id"] = int(self._last_update_insertion_id)
        state["critic_updates"] = int(self.critic_updates)
        state["actor_updates"] = int(self.actor_updates)
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        self._last_update_insertion_id = int(state.get("last_update_insertion_id", 0))
        self.nonfinite_batch_count = int(state.get("nonfinite_batch_count", 0))
        self.critic_updates = int(state.get("critic_updates", state.get("total_updates", 0)))
        self.actor_updates = int(state.get("actor_updates", self.critic_updates // self.cfg.policy_frequency))


class WorkerTD3(legacy.WorkerTD3):
    """Worker TD3 whose Critic is Q(state, goal, raw_request_action)."""

    def __init__(self, role: str, obs_dim: int, action_dim: int, latent_dim: int,
                 lr: float, cfg: TrainConfig, device: torch.device):
        super().__init__(role, obs_dim, action_dim, latent_dim, lr, cfg, device)
        self._last_update_insertion_id = 0
        self.nonfinite_batch_count = 0
        self.critic_updates = 0
        self.actor_updates = 0

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

    def _target_transition_delta(self, obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        """Transition 监督两端都来自同一套无梯度 target encoder 表示。"""

        with torch.no_grad():
            target_z = self.target_encoder(obs)
            target_next_z = self.target_encoder(next_obs)
            return target_next_z - target_z

    def _update_once_impl(self, buffer: Any, batch_size: int) -> Dict[str, float]:
        data = buffer.sample(batch_size)
        obs = legacy.to_tensor(self.normalizer.normalize(data["obs"].cpu().numpy()), self.device)
        next_obs = legacy.to_tensor(self.normalizer.normalize(data["next_obs"].cpu().numpy()), self.device)
        raw_actions = data["raw_actions"]
        executed_actions = data["executed_actions"]
        imitation_actions = data.get("guarded_actions", executed_actions)
        reward_external = data["reward_external"]
        reward_physical = data["physical_progress"]
        reward_projection = data["projection_penalty"]
        reward_global_safety = data.get("reward_global_safety", torch.zeros_like(reward_external))
        reward_role_specific = data.get("reward_role_specific", reward_external)
        reward_interaction_raw = data.get("reward_raw_total", reward_external)
        dones = data["dones"]
        goals, next_goals = data["goals"], data["next_goals"]
        durations = data.get("duration_steps", torch.ones_like(reward_external))
        is_weights = data.get("is_weights", torch.ones_like(reward_external))
        wg = legacy.worker_goal_tensor(goals, self.role)
        next_wg = legacy.worker_goal_tensor(next_goals, self.role)
        for name, tensor in (
            ("normalized_obs", obs), ("normalized_next_obs", next_obs),
            ("raw_action", raw_actions), ("executed_action", executed_actions),
            ("imitation_action", imitation_actions), ("reward_external", reward_external),
            ("reward_physical", reward_physical), ("reward_projection", reward_projection),
            ("done", dones), ("goal", goals), ("next_goal", next_goals),
            ("duration", durations), ("is_weights", is_weights),
        ):
            require_finite_tensor(f"{self.role}/{name}", tensor)

        reward_latent = self._target_latent_reward(obs, next_obs, goals)
        shaping_scale = torch.ones_like(durations)
        if self.role == "slow" and self.cfg.slow_shaping_duration_mode == "normalized":
            if abs(1.0 - self.cfg.gamma_fast) < 1e-12:
                shaping_scale = durations / float(self.cfg.slow_interval)
            else:
                shaping_scale = (
                    1.0 - torch.pow(torch.as_tensor(
                        self.cfg.gamma_fast, dtype=durations.dtype, device=durations.device
                    ), durations)
                ) / (1.0 - self.cfg.gamma_fast ** self.cfg.slow_interval)
        # external/projection 已在片段内逐快速步折扣累计；latent/physical 是端点 shaping。
        # normalized 模式只缩放端点 shaping，绝不能再次缩放区间累计奖励。
        reward_latent_used = reward_latent * shaping_scale
        reward_physical_used = reward_physical * shaping_scale
        reward_projection_used = reward_projection
        shaping_reference = reward_external.abs().clamp_min(self.cfg.shaping_reference_floor)
        reward_latent_contribution = (
            self.cfg.beta_latent * shaping_reference * torch.tanh(reward_latent_used)
        )
        reward_physical_contribution = (
            self.cfg.beta_physical * shaping_reference * torch.tanh(reward_physical_used)
        )
        reward_raw = (
            self.cfg.alpha_external * reward_external
            + reward_latent_contribution
            + reward_physical_contribution
            + reward_projection_used
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
            target_q_unclipped = build_smdp_target(
                rewards, dones, torch.minimum(q1_next, q2_next), self.cfg.gamma_fast, durations
            )
            target_q = target_q_unclipped
            if self.cfg.target_q_clip_abs > 0.0:
                target_q = target_q.clamp(-self.cfg.target_q_clip_abs, self.cfg.target_q_clip_abs)
            target_q_clipping_ratio = float((target_q != target_q_unclipped).float().mean().cpu())

        # Critic 与 Actor 统一在 raw action 域；环境投影属于转移函数。
        q1, q2 = self.critic(z, wg, raw_actions)
        loss1 = F.smooth_l1_loss(q1, target_q, reduction="none")
        loss2 = F.smooth_l1_loss(q2, target_q, reduction="none")
        critic_loss = (is_weights * (loss1 + loss2)).mean()
        td_error = 0.5 * ((q1 - target_q).abs() + (q2 - target_q).abs())

        transition_encoder_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        if self.transition_model is not None:
            target_delta_for_encoder = self._target_transition_delta(obs, next_obs)
            with _frozen_parameters(self.transition_model):
                # Transition Model 预测真实 executed action 造成的状态变化。
                pred_delta_for_encoder = self.transition_model(z, executed_actions)
                transition_encoder_loss = F.mse_loss(pred_delta_for_encoder, target_delta_for_encoder)

        encoder_loss = (
            critic_loss
            + self.cfg.lambda_transition * transition_encoder_loss
            + self.cfg.lambda_latent_norm * z.pow(2).mean()
        )
        for name, tensor in (
            ("reward_latent", reward_latent), ("reward_raw", reward_raw),
            ("reward", rewards), ("target_q", target_q), ("q1", q1), ("q2", q2),
            ("critic_loss", critic_loss), ("encoder_loss", encoder_loss),
            ("transition_encoder_loss", transition_encoder_loss),
        ):
            require_finite_tensor(f"{self.role}/{name}", tensor)
        self.critic_optim.zero_grad()
        self.encoder_optim.zero_grad()
        encoder_loss.backward()
        require_finite_gradients(f"{self.role}/critic", self.critic)
        require_finite_gradients(f"{self.role}/encoder", self.encoder)
        critic_grad = _clip_grad_norm(self.critic.parameters(), self.cfg.gradient_clip)
        encoder_grad = _clip_grad_norm(self.encoder.parameters(), self.cfg.gradient_clip)
        if not np.isfinite(critic_grad) or not np.isfinite(encoder_grad):
            raise FloatingPointError(
                f"{self.role} non-finite clipped gradient norm critic={critic_grad}, encoder={encoder_grad}"
            )

        transition_loss_value = 0.0
        transition_grad = 0.0
        if self.transition_model is not None and self.transition_optim is not None:
            with torch.no_grad():
                z_detached = self.encoder(obs).detach()
            target_delta = self._target_transition_delta(obs, next_obs)
            pred_delta = self.transition_model(z_detached, executed_actions)
            transition_loss = F.mse_loss(pred_delta, target_delta.detach())
            require_finite_tensor(f"{self.role}/transition_loss", transition_loss)
            self.transition_optim.zero_grad()
            transition_loss.backward()
            require_finite_gradients(f"{self.role}/transition_model", self.transition_model)
            transition_grad = _clip_grad_norm(self.transition_model.parameters(), self.cfg.gradient_clip)
            if not np.isfinite(transition_grad):
                raise FloatingPointError(
                    f"{self.role} non-finite transition gradient norm={transition_grad}"
                )
            transition_loss_value = float(transition_loss.detach().cpu())

        actor_loss_value = 0.0
        actor_grad = 0.0
        imitation_loss_value = 0.0
        projection_mask_ratio = 0.0
        behavior_match_ratio = 0.0
        imitation_mask_ratio = 0.0
        imitation_weight_value = projection_imitation_weight(self.cfg, self.total_updates)
        if self.role == "slow":
            imitation_weight_value *= float(self.cfg.slow_guard_imitation_multiplier)
        imitation_effective_weight = 0.0
        imitation_contribution_ratio = 0.0
        action_regularization_effective_weight = 0.0
        action_regularization_contribution_ratio = 0.0
        actor_saturation_ratio = 0.0
        actor_action_abs_mean = 0.0
        actor_reference_deviation_rms = 0.0
        fast_curtailment_action_mean = 0.0
        actor_update = (
            any(parameter.requires_grad for parameter in self.actor.parameters())
            and self.total_updates % self.cfg.policy_frequency == 0
        )
        if actor_update:
            actor_data = buffer.sample_uniform(batch_size)
            actor_raw_obs = actor_data["obs"].to(self.device)
            actor_obs = legacy.to_tensor(
                self.normalizer.normalize(actor_data["obs"].cpu().numpy()), self.device
            )
            actor_wg = legacy.worker_goal_tensor(actor_data["goals"], self.role)
            actor_historical_raw = actor_data["raw_actions"]
            actor_imitation_actions = actor_data.get("guarded_actions", actor_data["executed_actions"])
            z_pi = self.encoder(actor_obs).detach()
            self.actor_optim.zero_grad()
            with _frozen_parameters(self.critic):
                actor_raw_actions = self.actor(z_pi, actor_wg)
                actor_saturation_ratio = float(
                    (actor_raw_actions.abs() >= 0.98).float().mean().detach().cpu()
                )
                actor_action_abs_mean = float(actor_raw_actions.abs().mean().detach().cpu())
                policy_loss = -self.critic.q_min(z_pi, actor_wg, actor_raw_actions).mean()
                actor_loss = policy_loss
                action_reference = worker_action_regularization_reference(
                    actor_raw_obs, self.role, actor_raw_actions.shape[-1]
                )
                action_regularization = (actor_raw_actions - action_reference).pow(2).mean()
                actor_reference_deviation_rms = float(
                    torch.sqrt(action_regularization.detach().clamp_min(0.0)).cpu()
                )
                if self.role == "fast":
                    fast_curtailment_action_mean = float(
                        actor_raw_actions[:, actor_raw_actions.shape[-1] // 2:].mean().detach().cpu()
                    )
                action_regularization_effective_weight = adaptive_auxiliary_weight(
                    policy_loss, action_regularization, self.cfg.worker_action_l2_weight, self.cfg
                )
                action_regularization_contribution = (
                    action_regularization_effective_weight * action_regularization
                )
                actor_loss = actor_loss + action_regularization_contribution
                action_regularization_contribution_ratio = float(
                    action_regularization_contribution.detach().abs().div(
                        policy_loss.detach().abs().clamp_min(1e-8)
                    ).cpu()
                )
                projected_elements, behavior_match_mask, imitation_elements = (
                    projection_imitation_element_mask(
                        actor_raw_actions, actor_historical_raw, actor_imitation_actions,
                        self.cfg.projection_imitation_threshold,
                        self.cfg.projection_behavior_match_threshold,
                    )
                )
                projection_mask_ratio = float(
                    projected_elements.float().mean().detach().cpu()
                )
                behavior_match_ratio = float(behavior_match_mask.float().mean().detach().cpu())
                imitation_mask_ratio = float(
                    imitation_elements.float().mean().detach().cpu()
                )
                if bool(imitation_elements.any()) and imitation_weight_value > 0.0:
                    imitation_loss = F.mse_loss(
                        actor_raw_actions[imitation_elements],
                        actor_imitation_actions[imitation_elements],
                    )
                    imitation_effective_weight = adaptive_auxiliary_weight(
                        policy_loss, imitation_loss, imitation_weight_value, self.cfg
                    )
                    imitation_contribution = imitation_effective_weight * imitation_loss
                    actor_loss = actor_loss + imitation_contribution
                    imitation_contribution_ratio = float(
                        imitation_contribution.detach().abs().div(
                            policy_loss.detach().abs().clamp_min(1e-8)
                        ).cpu()
                    )
                    imitation_loss_value = float(imitation_loss.detach().cpu())
                require_finite_tensor(f"{self.role}/actor_loss", actor_loss)
                actor_loss.backward()
            require_finite_gradients(f"{self.role}/actor", self.actor)
            actor_grad = _clip_grad_norm(self.actor.parameters(), self.cfg.gradient_clip)
            if not np.isfinite(actor_grad):
                raise FloatingPointError(f"{self.role} non-finite actor gradient norm={actor_grad}")

        # Commit only after every active loss and gradient in this batch is finite.
        self.critic_optim.step()
        self.encoder_optim.step()
        if self.transition_model is not None and self.transition_optim is not None:
            self.transition_optim.step()
        if actor_update:
            self.actor_optim.step()
            actor_loss_value = float(actor_loss.detach().cpu())
            legacy.soft_update(self.target_actor, self.actor, self.cfg.tau)
            legacy.soft_update(self.target_critic, self.critic, self.cfg.tau)
            legacy.soft_update(self.target_encoder, self.encoder, self.cfg.tau)
            self.actor_updates += 1
        if hasattr(buffer, "update_priorities"):
            buffer.update_priorities(
                data["indices"].detach().cpu().numpy(), td_error.detach().cpu().numpy()
            )
        _clear_non_actor_gradients(self)

        self.critic_updates += 1
        self.total_updates += 1
        prefix = f"training_batch/{self.role}/"
        reward_clipping_ratio = float(reward_clipped.mean().detach().cpu())
        reward_group_denominator = (
            reward_global_safety.abs().mean() + reward_role_specific.abs().mean()
        ).clamp_min(1e-8)
        logs = {
            prefix + "critic_loss": float(critic_loss.detach().cpu()),
            prefix + "actor_loss": actor_loss_value,
            prefix + "projection_imitation_loss": imitation_loss_value,
            prefix + "projection_mask_ratio": projection_mask_ratio,
            prefix + "behavior_match_ratio": behavior_match_ratio,
            prefix + "projection_imitation_mask_ratio": imitation_mask_ratio,
            prefix + "projection_imitation_weight": imitation_weight_value,
            prefix + "projection_imitation_effective_weight": imitation_effective_weight,
            prefix + "projection_imitation_contribution_ratio": imitation_contribution_ratio,
            prefix + "action_regularization_effective_weight": action_regularization_effective_weight,
            prefix + "action_regularization_contribution_ratio": action_regularization_contribution_ratio,
            prefix + "sample_projection_mse": float((raw_actions - executed_actions).pow(2).mean().detach().cpu()),
            prefix + "raw_to_guarded_projection_mse": float(
                (raw_actions - imitation_actions).pow(2).mean().detach().cpu()
            ),
            prefix + "guarded_to_executed_projection_mse": float(
                (imitation_actions - executed_actions).pow(2).mean().detach().cpu()
            ),
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
            prefix + "target_q_clipping_ratio": target_q_clipping_ratio,
            prefix + "reward_external_mean": float(reward_external.mean().detach().cpu()),
            prefix + "reward_global_safety_mean": float(reward_global_safety.mean().detach().cpu()),
            prefix + "reward_global_safety_std": float(reward_global_safety.std(unbiased=False).detach().cpu()),
            prefix + "reward_role_specific_mean": float(reward_role_specific.mean().detach().cpu()),
            prefix + "reward_role_specific_std": float(reward_role_specific.std(unbiased=False).detach().cpu()),
            prefix + "reward_interaction_raw_mean": float(reward_interaction_raw.mean().detach().cpu()),
            prefix + "reward_interaction_raw_std": float(reward_interaction_raw.std(unbiased=False).detach().cpu()),
            prefix + "reward_latent_mean": float(reward_latent.mean().detach().cpu()),
            prefix + "reward_physical_mean": float(reward_physical.mean().detach().cpu()),
            prefix + "reward_projection_mean": float(reward_projection.mean().detach().cpu()),
            prefix + "reward_latent_duration_adjusted_mean": float(reward_latent_used.mean().detach().cpu()),
            prefix + "reward_physical_duration_adjusted_mean": float(reward_physical_used.mean().detach().cpu()),
            prefix + "reward_latent_contribution_mean": float(
                reward_latent_contribution.mean().detach().cpu()
            ),
            prefix + "reward_physical_contribution_mean": float(
                reward_physical_contribution.mean().detach().cpu()
            ),
            prefix + "shaping_reference_mean": float(shaping_reference.mean().detach().cpu()),
            prefix + "reward_projection_duration_adjusted_mean": float(reward_projection_used.mean().detach().cpu()),
            prefix + "shaping_duration_scale_mean": float(shaping_scale.mean().detach().cpu()),
            prefix + "batch_reward_mean": float(rewards.mean().detach().cpu()),
            prefix + "batch_reward_std": float(rewards.std(unbiased=False).detach().cpu()),
            prefix + "global_safety_abs_contribution_ratio": float(
                reward_global_safety.abs().mean().div(reward_raw.abs().mean().clamp_min(1e-8)).detach().cpu()
            ),
            prefix + "role_specific_abs_contribution_ratio": float(
                reward_role_specific.abs().mean().div(reward_raw.abs().mean().clamp_min(1e-8)).detach().cpu()
            ),
            prefix + "global_safety_group_share": float(
                reward_global_safety.abs().mean().div(reward_group_denominator).detach().cpu()
            ),
            prefix + "role_specific_group_share": float(
                reward_role_specific.abs().mean().div(reward_group_denominator).detach().cpu()
            ),
            prefix + "mean_duration_steps": float(durations.mean().detach().cpu()),
            prefix + "critic_raw_action_mean": float(raw_actions.mean().detach().cpu()),
            prefix + "executed_action_mean": float(executed_actions.mean().detach().cpu()),
            prefix + "priority_fallback_count": float(getattr(buffer, "priority_fallback_count", 0)),
        }
        if actor_update:
            logs.update({
                prefix + "actor_action_saturation_ratio": actor_saturation_ratio,
                prefix + "actor_action_abs_mean": actor_action_abs_mean,
                prefix + "actor_reference_deviation_rms": actor_reference_deviation_rms,
                prefix + "fast_curtailment_action_mean": fast_curtailment_action_mean,
            })
        logs.update({prefix + "replay/" + key: value
                     for key, value in getattr(buffer, "last_sampling_diagnostics", {}).items()})
        return logs

    def _update_once(self, buffer: Any, batch_size: int) -> Dict[str, float]:
        data_holder: Dict[str, Any] = {}
        original_sample = buffer.sample

        def tracked_sample(size: int) -> Dict[str, torch.Tensor]:
            sampled = original_sample(size)
            data_holder.update(sampled)
            return sampled

        buffer.sample = tracked_sample
        try:
            return self._update_once_impl(buffer, batch_size)
        except FloatingPointError as error:
            return _handle_nonfinite_update(self, self.role, data_holder, error)
        finally:
            buffer.sample = original_sample

    def update(self, buffer: Any, batch_size: int = 0, gamma: float = 0.0) -> Dict[str, float]:
        del batch_size, gamma
        role_batch, learning_starts, update_count = self._role_parameters()
        if len(buffer) < max(learning_starts, role_batch):
            return {}
        if self.role == "slow":
            insertion_id = int(getattr(buffer, "total_insertions", buffer.idx))
            if insertion_id == self._last_update_insertion_id:
                return {}
        logs = [self._update_once(buffer, role_batch) for _ in range(update_count)]
        if self.role == "slow":
            self._last_update_insertion_id = insertion_id
        return _mean_logs(logs)

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["nonfinite_batch_count"] = int(self.nonfinite_batch_count)
        state["last_update_insertion_id"] = int(self._last_update_insertion_id)
        state["critic_updates"] = int(self.critic_updates)
        state["actor_updates"] = int(self.actor_updates)
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        self._last_update_insertion_id = int(state.get("last_update_insertion_id", 0))
        self.nonfinite_batch_count = int(state.get("nonfinite_batch_count", 0))
        self.critic_updates = int(state.get("critic_updates", state.get("total_updates", 0)))
        self.actor_updates = int(state.get("actor_updates", self.critic_updates // self.cfg.policy_frequency))


# =============================================================================
# Agent construction, freezing, checkpoint and evaluation
# =============================================================================


def _set_agent_trainable(agent: Any, enabled: bool) -> None:
    for name in ("encoder", "actor", "critic", "transition_model"):
        module = getattr(agent, name, None)
        if module is not None:
            legacy.set_requires_grad(module, enabled)
            module.train(enabled)
    for name in ("target_encoder", "target_actor", "target_critic"):
        module = getattr(agent, name, None)
        if module is not None:
            legacy.set_requires_grad(module, False)
            module.eval()


def _apply_stage_trainability(agents: AgentBundle, cfg: TrainConfig,
                              reset_insertion_markers: bool = True) -> None:
    """统一应用参数、module mode、normalizer mode 和 Replay marker 的阶段状态。"""

    flags = legacy.stage_flags(cfg.training_stage)
    for name, agent in (("manager", agents.manager), ("slow", agents.slow), ("fast", agents.fast)):
        enabled = bool(flags[name])
        _set_agent_trainable(agent, enabled)
        agent.normalizer.train() if enabled else agent.normalizer.eval()
        if reset_insertion_markers and hasattr(agent, "_last_update_insertion_id"):
            agent._last_update_insertion_id = 0


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
        "run_training": ("cfg",),
        "safe_env_step": ("env", "action", "last_obs"),
        "evaluate_policy": ("agents", "cfg", "episodes", "max_steps", "seed"),
        "load_checkpoint": ("path", "agents", "map_location", "policy_only"),
        "load_agent_policy_state": ("agent", "state"),
        "load_agent_stage_transfer_state": ("agent", "state"),
        "metric_state_from_evaluation": ("stats", "cfg"),
        "is_better_evaluation": ("stats", "best_metric_state", "cfg"),
        "trusted_torch_load": ("path", "map_location"),
        "stage_flags": ("stage",),
        "worker_goal_tensor": ("goal", "role"),
        "expanded_goal_direction_tensor": ("goal", "role", "latent_dim"),
        "soft_update": ("target", "source", "tau"),
        "set_requires_grad": ("module", "enabled"),
        "capture_rng_state": (),
        "restore_rng_state": ("state",),
    }
    problems: List[str] = []
    actual_api_version = getattr(legacy, "LEGACY_ALGORITHM_API_VERSION", None)
    if actual_api_version != REQUIRED_LEGACY_ALGORITHM_API_VERSION:
        problems.append(
            f"LEGACY_ALGORITHM_API_VERSION={actual_api_version!r}, "
            f"required={REQUIRED_LEGACY_ALGORITHM_API_VERSION}"
        )
    for name, names in expected_parameters.items():
        function = getattr(legacy, name, None)
        if not callable(function):
            problems.append(f"missing callable legacy.{name}")
            continue
        actual = tuple(inspect.signature(function).parameters)
        if (names and actual[:len(names)] != names) or (not names and actual):
            problems.append(f"{name}{actual} expected prefix {names}")
    for class_name in ("FastReplayBuffer", "SlowReplayBuffer", "ManagerReplayBuffer"):
        replay_class = getattr(legacy, class_name, None)
        if replay_class is None:
            problems.append(f"missing legacy.{class_name}")
            continue
        for method_name, prefix in (("state_dict", ("self",)),
                                    ("load_state_dict", ("self", "state"))):
            method = getattr(replay_class, method_name, None)
            if not callable(method):
                problems.append(f"missing callable legacy.{class_name}.{method_name}")
                continue
            actual = tuple(inspect.signature(method).parameters)
            if actual[:len(prefix)] != prefix:
                problems.append(
                    f"{class_name}.{method_name}{actual} expected prefix {prefix}"
                )
    legacy_version = getattr(legacy, "ENV_MODEL_VERSION", None)
    if legacy_version != ENV_MODEL_VERSION:
        problems.append(f"environment version actual={legacy_version!r}, expected={ENV_MODEL_VERSION!r}")
    legacy_slow_schema = getattr(legacy, "SLOW_SAFETY_SCHEMA_VERSION", None)
    if legacy_slow_schema != REQUIRED_SLOW_SAFETY_SCHEMA_VERSION:
        problems.append(
            f"legacy SLOW_SAFETY_SCHEMA_VERSION={legacy_slow_schema!r}, "
            f"required={REQUIRED_SLOW_SAFETY_SCHEMA_VERSION}"
        )
    if SLOW_SAFETY_SCHEMA_VERSION != REQUIRED_SLOW_SAFETY_SCHEMA_VERSION:
        problems.append(
            f"environment SLOW_SAFETY_SCHEMA_VERSION={SLOW_SAFETY_SCHEMA_VERSION!r}, "
            f"required={REQUIRED_SLOW_SAFETY_SCHEMA_VERSION}"
        )
    if problems:
        raise RuntimeError("Incompatible legacy API: " + " / ".join(problems))


def validate_environment_contract(env: ElectricGasMultiScaleEnv, cfg: TrainConfig) -> None:
    validate_time_scale(env, cfg)
    action_shape = tuple(getattr(env.action_space, "shape", ()))
    observation_shape = tuple(getattr(env.observation_space, "shape", ()))
    problems: List[str] = []
    actual_schema = getattr(env, "slow_safety_schema_version", None)
    if actual_schema != REQUIRED_SLOW_SAFETY_SCHEMA_VERSION:
        problems.append(
            f"slow_safety_schema_version={actual_schema!r}, "
            f"expected={REQUIRED_SLOW_SAFETY_SCHEMA_VERSION}"
        )
    if SLOW_SAFETY_SCHEMA_VERSION != REQUIRED_SLOW_SAFETY_SCHEMA_VERSION:
        problems.append(
            f"module SLOW_SAFETY_SCHEMA_VERSION={SLOW_SAFETY_SCHEMA_VERSION!r}, "
            f"expected={REQUIRED_SLOW_SAFETY_SCHEMA_VERSION}"
        )
    if legacy.SLOW_OBSERVATION_LAYOUT.dimension != 49:
        problems.append(
            f"slow observation layout dimension={legacy.SLOW_OBSERVATION_LAYOUT.dimension}, expected=49"
        )
    if env.action_dim != env.slow_action_dim + env.fast_action_dim:
        problems.append(
            f"action_dim={env.action_dim}, slow+fast={env.slow_action_dim + env.fast_action_dim}"
        )
    if action_shape != (env.action_dim,):
        problems.append(f"action_space.shape={action_shape}, expected={(env.action_dim,)}")
    if observation_shape != (env.global_state_dim,):
        problems.append(f"observation_space.shape={observation_shape}, expected={(env.global_state_dim,)}")
    for name, actual, expected in (
        ("slow_action_dim", int(env.slow_action_dim), 10),
        ("fast_action_dim", int(env.fast_action_dim), 16),
        ("action_dim", int(env.action_dim), 26),
    ):
        if actual != expected:
            problems.append(f"{name}={actual}, expected={expected}")
    getter = getattr(env, "get_slow_safety_features", None)
    if not callable(getter):
        problems.append("missing callable get_slow_safety_features()")
    else:
        try:
            reset_obs, _ = env.reset(seed=cfg.seed)
            if np.asarray(reset_obs).reshape(-1).size != env.global_state_dim:
                problems.append(
                    f"reset observation size={np.asarray(reset_obs).size}, "
                    f"expected={env.global_state_dim}"
                )
            features = getter()
            expected_fields = tuple(name for name, _ in legacy.SLOW_OBSERVATION_LAYOUT.fields)
            actual_fields = tuple(features.keys())
            if set(actual_fields) != set(expected_fields):
                missing = sorted(set(expected_fields) - set(actual_fields))
                extra = sorted(set(actual_fields) - set(expected_fields))
                problems.append(
                    f"slow safety fields mismatch: missing={missing}, extra={extra}"
                )
            for field_name, expected_size in legacy.SLOW_OBSERVATION_LAYOUT.fields:
                if field_name not in features:
                    continue
                values = np.asarray(features[field_name]).reshape(-1)
                if values.size != expected_size:
                    problems.append(
                        f"slow safety field {field_name!r} size={values.size}, "
                        f"expected={expected_size}"
                    )
                if not np.all(np.isfinite(values)):
                    problems.append(f"slow safety field {field_name!r} contains NaN/Inf")
            if "held_slow_action" in features:
                held = np.asarray(features["held_slow_action"], dtype=np.float32).reshape(-1)
                if held.shape != (env.slow_action_dim,):
                    problems.append(
                        f"held_slow_action shape={held.shape}, "
                        f"expected={(env.slow_action_dim,)}"
                    )
                if not np.all(np.isfinite(held)):
                    problems.append("held_slow_action contains NaN/Inf")
                if np.any(held < -1.0 - 1e-6) or np.any(held > 1.0 + 1e-6):
                    problems.append(
                        "held_slow_action outside Slow Actor [-1,1] coordinates: "
                        f"min={float(np.min(held))}, max={float(np.max(held))}"
                    )
            try:
                flattened = legacy.SLOW_OBSERVATION_LAYOUT.flatten(features)
                if flattened.size != legacy.SLOW_OBSERVATION_LAYOUT.dimension:
                    problems.append(
                        f"flattened slow observation size={flattened.size}, "
                        f"expected={legacy.SLOW_OBSERVATION_LAYOUT.dimension}"
                    )
                if not np.all(np.isfinite(flattened)):
                    problems.append("flattened slow observation contains NaN/Inf")
                if flattened.size != 49:
                    problems.append(
                        f"flattened slow observation size={flattened.size}, expected safety schema v2 size=49"
                    )
            except (AssertionError, KeyError, ValueError) as exc:
                problems.append(f"cannot flatten slow safety fields: {exc}")
        except Exception as exc:
            problems.append(f"get_slow_safety_features contract failed: {type(exc).__name__}: {exc}")
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
    dimension_problems: List[str] = []
    expected_slow_obs_dim = legacy.SLOW_OBSERVATION_LAYOUT.dimension
    if agents.slow.obs_dim != expected_slow_obs_dim:
        dimension_problems.append(
            f"Slow Actor/Critic obs_dim={agents.slow.obs_dim}, expected={expected_slow_obs_dim}"
        )
    if tuple(agents.slow.normalizer.mean.shape) != (expected_slow_obs_dim,):
        dimension_problems.append(
            f"Slow normalizer shape={agents.slow.normalizer.mean.shape}, "
            f"expected={(expected_slow_obs_dim,)}"
        )
    if agents.slow.action_dim != env.slow_action_dim:
        dimension_problems.append(
            f"Slow Actor/Critic action_dim={agents.slow.action_dim}, "
            f"expected env.slow_action_dim={env.slow_action_dim}"
        )
    if dimension_problems:
        raise ValueError("Agent dimension contract is incompatible: " + " / ".join(dimension_problems))
    _apply_stage_trainability(agents, cfg)
    agents.environment_metadata = {
        "env_model_version": ENV_MODEL_VERSION,
        "slow_safety_schema_version": SLOW_SAFETY_SCHEMA_VERSION,
        "slow_observation_fields": tuple(
            name for name, _ in legacy.SLOW_OBSERVATION_LAYOUT.fields
        ),
        "global_state_dim": int(env.global_state_dim),
        "slow_observation_dim": int(slow_obs.size),
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
        "shaping_reference_floor", "reward_component_transform", "reward_scale_profile",
        "adaptive_auxiliary_loss_scaling", "auxiliary_loss_scale_max",
        "auxiliary_loss_coefficient_max", "worker_action_l2_weight",
        "slow_role_specific_reward_scale", "slow_guard_imitation_multiplier",
        "use_prioritized_replay", "priority_alpha", "constraint_priority_weight",
        "projection_priority_weight", "projection_behavior_match_threshold",
        "fast_priority_beta_anneal_updates", "slow_priority_beta_anneal_updates",
        "manager_priority_beta_anneal_updates", "prioritized_sample_fraction",
        "manager_use_prioritized_replay", "slow_shaping_duration_mode",
        "priority_component_normalization", "nonfinite_update_policy",
        "eval_seed_mode", "best_model_metric", "save_replay_in_checkpoint",
        "bootstrap_on_time_limit", "unexpected_env_exception_policy",
        "fast_random_warmup_steps", "slow_random_warmup_segments",
        "manager_random_warmup_segments", "warmup_blend_fraction",
        "goal_smoothing", "goal_change_penalty_weight", "worker_component_clip_abs",
        "fast_global_safety_weight", "fast_role_specific_weight",
        "slow_global_safety_weight", "slow_role_specific_weight",
        "joint_policy_frequency", "joint_freeze_manager_actor_episodes",
        "inherit_stage_best_on_joint_transfer",
        "joint_early_stop_patience_evaluations", "joint_early_stop_min_episodes",
    )
    return {name: getattr(cfg, name) for name in names}


def checkpoint_metadata(agents: AgentBundle, cfg: TrainConfig) -> Dict[str, Any]:
    env_meta = getattr(agents, "environment_metadata", {})
    return {
        "checkpoint_schema_version": 9,
        "algorithm_version": ALGORITHM_VERSION,
        "training_stage": cfg.training_stage,
        "env_model_version": ENV_MODEL_VERSION,
        "slow_safety_schema_version": SLOW_SAFETY_SCHEMA_VERSION,
        "slow_observation_fields": tuple(
            name for name, _ in legacy.SLOW_OBSERVATION_LAYOUT.fields
        ),
        "critic_action_semantics": "raw_request_action",
        "executed_action_semantics": (
            "environment_projected_action_for_transition_imitation_and_diagnostics"
        ),
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


_LIGHTWEIGHT_AGENT_KEYS = (
    "role", "encoder", "target_encoder", "actor", "target_actor", "critic",
    "target_critic", "transition_model", "normalizer", "nonfinite_batch_count",
)


def _agent_checkpoint_state(agent: Any, full: bool) -> Dict[str, Any]:
    state = agent.state_dict()
    if full:
        return state
    return {name: state[name] for name in _LIGHTWEIGHT_AGENT_KEYS if name in state}


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> Tuple[float, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    started = time.perf_counter()
    try:
        torch.save(dict(payload), str(temp_path))
        os.replace(str(temp_path), str(path))
    finally:
        if temp_path.exists():
            temp_path.unlink()
    elapsed = time.perf_counter() - started
    size = int(path.stat().st_size)
    LOGGER.info("Checkpoint saved path=%s kind=%s size_bytes=%s elapsed_seconds=%.3f",
                path, payload.get("checkpoint_kind", "unknown"), size, elapsed)
    return elapsed, size


def save_checkpoint(path: Path, cfg: TrainConfig, agents: AgentBundle, episode: int,
                    global_step: int, best_return: float,
                    best_metric_state: Optional[Mapping[str, Any]] = None,
                    best_evaluation_stats: Optional[Mapping[str, Any]] = None,
                    fast_replay: Optional[FastReplayBuffer] = None,
                    slow_replay: Optional[SlowReplayBuffer] = None,
                    manager_replay: Optional[ManagerReplayBuffer] = None,
                    next_episode: Optional[int] = None,
                    checkpoint_kind: str = "full_resume") -> None:
    if checkpoint_kind not in ("lightweight", "full_resume"):
        raise ValueError(f"checkpoint_kind={checkpoint_kind!r} must be lightweight or full_resume")
    full = checkpoint_kind == "full_resume"
    payload = {
        **checkpoint_metadata(agents, cfg),
        "checkpoint_kind": checkpoint_kind,
        "observation_dim": agents.manager.obs_dim,
        "config": asdict(cfg),
        "manager": _agent_checkpoint_state(agents.manager, full),
        "slow": _agent_checkpoint_state(agents.slow, full),
        "fast": _agent_checkpoint_state(agents.fast, full),
        "episode": episode,
        "global_step": global_step,
        "best_return": best_return,
        "best_metric_state": dict(best_metric_state or {}),
        "best_evaluation_stats": dict(best_evaluation_stats or {}),
    }
    if full:
        payload["next_episode"] = int(episode + 1 if next_episode is None else next_episode)
        payload["rng_state"] = legacy.capture_rng_state()
    if full and cfg.save_replay_in_checkpoint:
        if fast_replay is None or slow_replay is None or manager_replay is None:
            message = "full resume checkpoint requires all three Replay buffers"
            if cfg.strict_resume_required:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        else:
            payload.update({
                "fast_replay": fast_replay.state_dict(),
                "slow_replay": slow_replay.state_dict(),
                "manager_replay": manager_replay.state_dict(),
            })
    elif full and not cfg.save_replay_in_checkpoint:
        message = (
            "save_replay_in_checkpoint=False: full checkpoint cannot provide strict resume"
        )
        if cfg.strict_resume_required:
            raise ValueError(message)
        warnings.warn(
            message,
            RuntimeWarning, stacklevel=2,
        )
    _atomic_torch_save(payload, path)


def validate_checkpoint_compatibility(
    payload: Dict[str, Any], agents: AgentBundle, load_mode: str = "resume"
) -> None:
    """Validate architecture always and semantic schema strictly for full resume.

    Older checkpoints may be used only through an explicit ``stage_transfer`` or
    ``policy_only`` migration.  Those modes still require identical network
    dimensions, but deliberately start fresh optimizers/Replay or omit them.
    """

    if load_mode not in ("resume", "stage_transfer", "policy_only"):
        raise ValueError(
            f"checkpoint load mode={load_mode!r} must be resume, stage_transfer, or policy_only"
        )
    cfg = agents.fast.cfg
    expected = checkpoint_metadata(agents, cfg)
    architecture_required = (
        "manager_observation_dim", "slow_observation_dim",
        "fast_observation_dim", "slow_action_dim", "fast_action_dim", "total_action_dim", "goal_dim",
    )
    problems: List[str] = []
    for key in architecture_required:
        if key not in payload:
            problems.append(f"missing required metadata {key}")
        elif payload[key] != expected[key]:
            problems.append(f"{key}: checkpoint={payload[key]!r}, current={expected[key]!r}")

    semantic_required = (
        "checkpoint_schema_version",
        "algorithm_version",
        "env_model_version",
        "slow_safety_schema_version",
        "slow_observation_fields",
        "critic_action_semantics",
        "executed_action_semantics",
    )
    migration_notes: List[str] = []
    for key in semantic_required:
        actual = payload.get(key, None)
        expected_value = expected[key]
        if key == "slow_observation_fields" and actual is not None:
            actual = tuple(actual)
            expected_value = tuple(expected_value)
        if key not in payload:
            detail = f"missing {key} (current={expected_value!r})"
        elif actual != expected_value:
            detail = f"{key}: checkpoint={actual!r}, current={expected_value!r}"
        else:
            continue
        if load_mode == "resume":
            problems.append(detail)
        else:
            migration_notes.append(detail)

    old_config = payload.get("config", {}) if isinstance(payload.get("config", {}), Mapping) else {}
    for key in ("slow_interval", "manager_interval"):
        actual = payload.get(key, old_config.get(key))
        if actual is None:
            LOGGER.warning("Legacy checkpoint has no %s; using current value %s as explicit migration fallback.",
                           key, expected[key])
        elif int(actual) != int(expected[key]):
            problems.append(f"{key}: checkpoint={actual!r}, current={expected[key]!r}")
    if "global_state_dim" not in payload:
        migration_notes.append(
            "missing global_state_dim; network observation dimensions were still checked"
        )
    elif int(payload["global_state_dim"]) != int(expected["global_state_dim"]):
        problems.append(
            f"global_state_dim: checkpoint={payload['global_state_dim']!r}, current={expected['global_state_dim']!r}"
        )
    if "optimization_config" not in payload:
        if load_mode == "resume":
            problems.append("missing optimization_config required for strict resume")
        else:
            migration_notes.append(
                "missing optimization_config; current migration defaults will be used"
            )
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
        suffix = (
            ". Old or semantically different checkpoints are not eligible for strict resume; "
            "select checkpoint_load_mode='stage_transfer' or 'policy_only' explicitly to migrate."
            if load_mode == "resume" else ""
        )
        raise ValueError("Incompatible checkpoint: " + " / ".join(problems) + suffix)
    if migration_notes:
        warnings.warn(
            f"Explicit checkpoint {load_mode} migration accepted; strict resume is not "
            "available for this checkpoint: " + " / ".join(migration_notes),
            RuntimeWarning,
            stacklevel=2,
        )


def load_checkpoint(path: str, agents: AgentBundle, map_location: torch.device,
                    policy_only: bool = False, mode: Optional[str] = None,
                    fast_replay: Optional[FastReplayBuffer] = None,
                    slow_replay: Optional[SlowReplayBuffer] = None,
                    manager_replay: Optional[ManagerReplayBuffer] = None) -> Dict[str, Any]:
    load_mode = mode or ("policy_only" if policy_only else agents.fast.cfg.checkpoint_load_mode)
    payload = legacy.trusted_torch_load(path, map_location=map_location)
    validate_checkpoint_compatibility(payload, agents, load_mode=load_mode)
    saved_stage = payload.get("training_stage", payload.get("config", {}).get("training_stage"))
    if load_mode == "resume" and saved_stage is not None and saved_stage != agents.fast.cfg.training_stage:
        raise ValueError(
            f"Cannot resume checkpoint from training_stage={saved_stage!r} into "
            f"training_stage={agents.fast.cfg.training_stage!r}; use checkpoint_load_mode='stage_transfer'."
        )
    if load_mode == "policy_only":
        legacy.load_agent_policy_state(agents.manager, payload["manager"])
        legacy.load_agent_policy_state(agents.slow, payload["slow"])
        legacy.load_agent_policy_state(agents.fast, payload["fast"])
    elif load_mode == "stage_transfer":
        legacy.load_agent_stage_transfer_state(agents.manager, payload["manager"])
        legacy.load_agent_stage_transfer_state(agents.slow, payload["slow"])
        legacy.load_agent_stage_transfer_state(agents.fast, payload["fast"])
    elif load_mode == "resume":
        required = [
            "fast_replay", "slow_replay", "manager_replay", "rng_state", "next_episode",
            "global_step", "manager", "slow", "fast",
        ]
        optimizer_keys = ("encoder_optim", "actor_optim", "critic_optim")
        missing = [name for name in required if name not in payload]
        for replay_name, replay in (("fast_replay", fast_replay), ("slow_replay", slow_replay),
                                    ("manager_replay", manager_replay)):
            if replay is None:
                missing.append(replay_name + "_target_buffer")
        for role in ("manager", "slow", "fast"):
            role_state = payload.get(role, {})
            for optimizer_name in optimizer_keys:
                if optimizer_name not in role_state:
                    missing.append(f"{role}.{optimizer_name}")
        for role in ("manager", "slow"):
            if "last_update_insertion_id" not in payload.get(role, {}):
                missing.append(f"{role}.last_update_insertion_id")
        if "transition_model" in payload.get("slow", {}) and payload["slow"].get("transition_model") is not None:
            if "transition_optim" not in payload["slow"]:
                missing.append("slow.transition_optim")
        if "transition_model" in payload.get("fast", {}) and payload["fast"].get("transition_model") is not None:
            if "transition_optim" not in payload["fast"]:
                missing.append("fast.transition_optim")
        strict_restored = not missing
        payload["strict_resume_restored"] = strict_restored
        payload["resume_missing_components"] = sorted(set(missing))
        if missing and agents.fast.cfg.strict_resume_required:
            raise ValueError(
                "Strict resume checkpoint is missing core components: " + ", ".join(sorted(set(missing)))
            )
        if missing:
            warnings.warn(
                "Partial resume restored; missing components: " + ", ".join(sorted(set(missing))),
                RuntimeWarning, stacklevel=2,
            )
        for role, agent in (("manager", agents.manager), ("slow", agents.slow), ("fast", agents.fast)):
            role_state = payload[role]
            role_optimizer_keys = list(optimizer_keys)
            if role != "manager" and role_state.get("transition_model") is not None:
                role_optimizer_keys.append("transition_optim")
            if all(name in role_state for name in role_optimizer_keys):
                agent.load_state_dict(role_state)
            else:
                legacy.load_agent_stage_transfer_state(agent, role_state)
        replay_pairs = (
            ("fast_replay", fast_replay), ("slow_replay", slow_replay),
            ("manager_replay", manager_replay),
        )
        strict_replay = not missing
        for key, replay in replay_pairs:
            if replay is None or key not in payload:
                strict_replay = False
                continue
            replay.load_state_dict(payload[key])
        if not strict_replay:
            warnings.warn(
                "Resume checkpoint lacks complete Replay state or buffers were not supplied; continuation is not strict.",
                RuntimeWarning, stacklevel=2,
            )
        if "rng_state" in payload:
            legacy.restore_rng_state(payload["rng_state"])
        else:
            warnings.warn("Legacy checkpoint has no rng_state; stochastic continuation is not exact.",
                          RuntimeWarning, stacklevel=2)
    else:
        raise ValueError(f"checkpoint_load_mode={load_mode!r} must be resume, stage_transfer or policy_only")
    _apply_stage_learning_rates(agents, agents.fast.cfg)
    _apply_stage_trainability(agents, agents.fast.cfg, reset_insertion_markers=load_mode != "resume")
    if load_mode == "resume":
        for agent, replay in ((agents.manager, manager_replay), (agents.slow, slow_replay),
                              (agents.fast, fast_replay)):
            if hasattr(agent, "_last_update_insertion_id"):
                role = "manager" if agent is agents.manager else "slow" if agent is agents.slow else "fast"
                saved_marker = payload.get(role, {}).get("last_update_insertion_id")
                agent._last_update_insertion_id = int(
                    saved_marker if saved_marker is not None else
                    replay.total_insertions if replay is not None and strict_replay else 0
                )
    return payload


_STAGE_BEST_FILES = {
    "fast_pretrain": "best_fast.pt",
    "slow_pretrain": "best_slow.pt",
    "manager_train": "best_manager.pt",
    "joint_finetune": "best_joint.pt",
}


def save_best_files(root: Path, agents: AgentBundle, cfg: TrainConfig, episode: int,
                    global_step: int, best_return: float,
                    best_metric_state: Optional[Mapping[str, Any]] = None,
                    best_evaluation_stats: Optional[Mapping[str, Any]] = None,
                    fast_replay: Optional[FastReplayBuffer] = None,
                    slow_replay: Optional[SlowReplayBuffer] = None,
                    manager_replay: Optional[ManagerReplayBuffer] = None,
                    next_episode: Optional[int] = None) -> None:
    filename = _STAGE_BEST_FILES.get(cfg.training_stage)
    if filename is None:
        raise ValueError(f"Cannot choose best checkpoint filename for stage={cfg.training_stage!r}")
    save_checkpoint(root / "latest_policy.pt", cfg, agents, episode, global_step, best_return,
                    best_metric_state, best_evaluation_stats, fast_replay, slow_replay,
                    manager_replay, next_episode, checkpoint_kind="lightweight")
    save_checkpoint(root / filename, cfg, agents, episode, global_step, best_return,
                    best_metric_state, best_evaluation_stats, fast_replay, slow_replay,
                    manager_replay, next_episode, checkpoint_kind="lightweight")


def evaluation_control_spec(stage: str) -> Dict[str, str]:
    specs = {
        "fast_pretrain": {"manager": "fixed_goal", "slow": "rule", "fast": "deterministic_policy"},
        "slow_pretrain": {"manager": "fixed_goal", "slow": "deterministic_policy", "fast": "deterministic_policy"},
        "manager_train": {"manager": "deterministic_policy", "slow": "deterministic_policy", "fast": "deterministic_policy"},
        "joint_finetune": {"manager": "deterministic_policy", "slow": "deterministic_policy", "fast": "deterministic_policy"},
        "all": {"manager": "deterministic_policy", "slow": "deterministic_policy", "fast": "deterministic_policy"},
    }
    if stage not in specs:
        raise ValueError(f"Unknown evaluation stage: {stage}")
    return specs[stage]


def _resolve_eval_seeds(cfg: TrainConfig, episodes: int, seed: int) -> Tuple[int, ...]:
    if episodes <= 0:
        raise ValueError(f"evaluation episodes must be > 0, got {episodes}")
    configured = tuple(int(item) for item in cfg.eval_seeds)
    if cfg.eval_seed_mode == "fixed" and configured:
        resolved = list(configured[:episodes])
        next_seed = resolved[-1] + 1
        while len(resolved) < episodes:
            resolved.append(next_seed)
            next_seed += 1
        return tuple(resolved)
    if cfg.eval_seed_mode == "offset" and configured:
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
                    max_steps: int = EPISODE_STEPS, seed: int = 12345) -> Dict[str, Any]:
    evaluation_control_spec(cfg.training_stage)
    probe_env = ElectricGasMultiScaleEnv()
    validate_environment_contract(probe_env, cfg)
    seeds = _resolve_eval_seeds(cfg, int(episodes), int(seed))
    LOGGER.info("Evaluation stage=%s resolved_eval_seeds=%s", cfg.training_stage, list(seeds))
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
    if cfg.training_stage in ("slow_pretrain", "manager_train", "joint_finetune", "all"):
        missing_soc = [index for index, item in enumerate(results) if "soc_violation_rate" not in item]
        if missing_soc:
            raise ValueError(
                f"Evaluation episodes {missing_soc} are missing required soc_violation_rate; "
                "missing safety metrics cannot be interpreted as feasible."
            )
    aggregate: Dict[str, float] = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "solver_failures": float(sum(item["solver_failures"] for item in results)),
        "steps": total_steps,
        "episodes": float(len(results)),
        "resolved_eval_seeds": list(seeds),
        "power_success_rate": float(sum(item["power_success_rate"] * item["steps"] for item in results) / max(total_steps, 1.0)),
        "gas_success_rate": float(sum(item["gas_success_rate"] * item["steps"] for item in results) / max(total_steps, 1.0)),
        "power_solver_success_rate": float(sum(item["power_solver_success_rate"] * item["steps"] for item in results) / max(total_steps, 1.0)),
        "gas_solver_success_rate": float(sum(item["gas_solver_success_rate"] * item["steps"] for item in results) / max(total_steps, 1.0)),
        "soc_violation_rate": float(sum(item.get("soc_violation_rate", 0.0) * item["steps"]
                                          for item in results) / max(total_steps, 1.0)),
        "mean_voltage_rms_deviation_pu": float(np.mean([item["mean_voltage_rms_deviation_pu"] for item in results])),
        "mean_gas_pressure_rms_deviation_bar": float(np.mean([item["mean_gas_pressure_rms_deviation_bar"] for item in results])),
    }
    component_keys = set().union(*(item.keys() for item in results))
    for key in component_keys:
        if key.endswith("_cost_per_step"):
            aggregate[key] = float(
                sum(item.get(key, 0.0) * item["steps"] for item in results)
                / max(total_steps, 1.0)
            )
        elif key.endswith("_cost") or key.endswith("_cost_total"):
            aggregate[key] = float(sum(item.get(key, 0.0) for item in results))
        elif key.endswith("_rate"):
            aggregate[key] = float(sum(item.get(key, 0.0) * item["steps"] for item in results)
                                   / max(total_steps, 1.0))
        elif key.startswith("min_"):
            aggregate[key] = float(min(item[key] for item in results if key in item))
        elif key.startswith("max_"):
            aggregate[key] = float(max(item[key] for item in results if key in item))
    for key in list(aggregate):
        if key.endswith("_cost_total"):
            aggregate[key[:-len("_cost_total")] + "_cost_per_step"] = (
                aggregate[key] / max(total_steps, 1.0)
            )
    feasible, feasibility_reasons = legacy.is_feasible(aggregate, cfg)
    aggregate["constraint_feasibility"] = float(feasible)
    aggregate["feasibility_reasons"] = feasibility_reasons
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
    if cfg.training_stage == "joint_finetune" and cfg.policy_frequency != cfg.joint_policy_frequency:
        cfg = copy.deepcopy(cfg)
        cfg.policy_frequency = int(cfg.joint_policy_frequency)
        LOGGER.info(
            "Joint stabilization: policy_frequency adjusted to %s (critic updates unchanged)",
            cfg.policy_frequency,
        )
    sample_budget = validate_training_contract(cfg)
    _, resume_next_episode, initial_updates = _resume_replay_sizes(cfg)
    LOGGER.info("Replay sample budget: %s", sample_budget)
    LOGGER.info(
        "Loaded legacy module path=%s legacy API version=%s optimized file path=%s "
        "environment model version=%s",
        Path(legacy.__file__).resolve(), getattr(legacy, "LEGACY_ALGORITHM_API_VERSION", None),
        Path(__file__).resolve(), ENV_MODEL_VERSION,
    )
    LOGGER.info(
        "Discount diagnostics gamma_fast=%.6f slow_discount=%.8f manager_discount=%.8f "
        "episode_discount=%.8f",
        cfg.gamma_fast, cfg.gamma_fast ** cfg.slow_interval,
        cfg.gamma_fast ** cfg.manager_interval, cfg.gamma_fast ** cfg.episode_steps,
    )
    if cfg.gamma_fast >= 0.997 and (
            cfg.worker_reward_clip_abs <= 0.0 or cfg.target_q_clip_abs <= 0.0):
        LOGGER.warning(
            "High gamma_fast=%s is enabled while reward/Q clipping is disabled; monitor Q scale closely.",
            cfg.gamma_fast,
        )
    with runtime_overrides(cfg):
        result = _LEGACY_RUN_TRAINING(cfg)
    remaining_episodes = max(int(cfg.episodes) - int(resume_next_episode), 0)
    if remaining_episodes > 0:
        for role, active in legacy.stage_flags(cfg.training_stage).items():
            if not active:
                continue
            critic_delta = int(result[f"{role}_critic_update_count"]) - initial_updates[role + "_critic"]
            actor_delta = int(result[f"{role}_actor_update_count"]) - initial_updates[role + "_actor"]
            actor_deliberately_frozen = bool(
                cfg.training_stage == "joint_finetune"
                and role == "manager"
                and int(resume_next_episode) < int(cfg.joint_freeze_manager_actor_episodes)
                and int(cfg.episodes) <= int(cfg.joint_freeze_manager_actor_episodes)
            )
            if critic_delta <= 0 or (actor_delta <= 0 and not actor_deliberately_frozen):
                raise RuntimeError(
                    f"Invalid training run: trainable {role} made critic_delta={critic_delta}, "
                    f"actor_delta={actor_delta}, deliberately_frozen={actor_deliberately_frozen}; "
                    f"sample_budget={sample_budget[role]}"
                )
    result["sample_budget"] = sample_budget
    return result


def _resolve_stage_checkpoint_mode(path: str, current_stage: str, requested_mode: str,
                                   strict_resume_required: bool) -> str:
    if not path:
        return requested_mode
    metadata = legacy.trusted_torch_load(path, map_location=torch.device("cpu"))
    saved_stage = metadata.get("training_stage", metadata.get("config", {}).get("training_stage"))
    resolved = requested_mode
    if requested_mode == "resume" and saved_stage != current_stage:
        if strict_resume_required:
            raise ValueError(
                f"Strict resume requested across stages: saved stage={saved_stage!r}, "
                f"current stage={current_stage!r}; use checkpoint_load_mode='stage_transfer'"
            )
        resolved = "stage_transfer"
    LOGGER.info("Checkpoint stage resolution saved_stage=%s current_stage=%s requested=%s resolved=%s",
                saved_stage, current_stage, requested_mode, resolved)
    return resolved


def run_all_stages(cfg: TrainConfig) -> Dict[str, Any]:
    stages = (
        ("fast_pretrain", cfg.fast_pretrain_episodes),
        ("slow_pretrain", cfg.slow_pretrain_episodes),
        ("manager_train", cfg.manager_train_episodes),
        ("joint_finetune", cfg.joint_finetune_episodes),
    )
    root = Path(cfg.checkpoint_dir) / ("optimized_all_stages_" + time.strftime("%Y%m%d_%H%M%S"))
    previous_checkpoint = cfg.load_checkpoint
    results: Dict[str, Any] = {"stages": []}
    completed_stages = 0
    exploration_offset = int(cfg.exploration_episode_offset)
    for stage, count in stages:
        if count <= 0:
            continue
        stage_cfg = copy.deepcopy(cfg)
        stage_cfg.training_stage = stage
        stage_cfg.episodes = int(count)
        stage_cfg.exploration_episode_offset = exploration_offset
        stage_cfg.checkpoint_dir = str(root / stage)
        stage_cfg.load_checkpoint = previous_checkpoint
        if previous_checkpoint:
            if completed_stages == 0:
                stage_cfg.checkpoint_load_mode = _resolve_stage_checkpoint_mode(
                    previous_checkpoint, stage, cfg.checkpoint_load_mode,
                    cfg.strict_resume_required,
                )
            else:
                stage_cfg.checkpoint_load_mode = "stage_transfer"
        stage_cfg.load_policy_only = stage_cfg.checkpoint_load_mode == "policy_only"
        LOGGER.info("Starting optimized stage %s for %s episodes", stage, count)
        result = run_training(stage_cfg)
        result["stage"] = stage
        results["stages"].append(result)
        run_root = Path(result["run_root"])
        candidate = run_root / _STAGE_BEST_FILES[stage]
        previous_checkpoint = str(candidate if candidate.exists() else run_root / "latest_policy.pt")
        completed_stages += 1
        exploration_offset += int(count)
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
    durations = torch.tensor([[1.0], [5.0], [20.0], [40.0], [7.0]])
    rewards = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    dones = torch.zeros_like(rewards)
    q_next = torch.full_like(rewards, 10.0)
    target = build_smdp_target(rewards, dones, q_next, gamma, durations)
    expected = rewards + torch.pow(torch.tensor(gamma), durations) * 10.0
    assert torch.allclose(target, expected)
    terminal = build_smdp_target(rewards, torch.ones_like(dones), q_next, gamma, durations)
    assert torch.allclose(terminal, rewards), "terminated/truncated fragments must not bootstrap"


def _test_reward_scale_calibration() -> None:
    cfg = TrainConfig(run_mode="debug")
    assert set(legacy.GLOBAL_SAFETY_COMPONENTS).isdisjoint(legacy.FAST_COMPONENTS)
    assert set(legacy.GLOBAL_SAFETY_COMPONENTS).isdisjoint(legacy.SLOW_COMPONENTS)

    mild, mild_clipped = legacy.normalized_safety_component_cost(
        "voltage_violation", 100.0, cfg
    )
    severe, severe_clipped = legacy.normalized_safety_component_cost(
        "voltage_violation", 10_000.0, cfg
    )
    assert 0.0 < mild < severe <= cfg.worker_component_clip_abs
    assert not mild_clipped and not severe_clipped
    solver, _ = legacy.normalized_safety_component_cost("solver_failure", 5_000.0, cfg)
    deviation, _ = legacy.normalized_safety_component_cost("voltage_deviation", 25.0, cfg)
    assert solver > deviation, "solver failure must retain hard-safety priority"

    info = {"reward_components": {
        "voltage_deviation": 25.0, "voltage_violation": 100.0,
        "gas_pressure_deviation": 5.0, "soc_soft": 0.1,
        "renewable_curtailment": 5.0, "power_loss": 0.5,
        "compressor_energy": 0.5, "ess_action_change": 1.0,
    }}
    fast = legacy.worker_safety_reward_from_components(info, "fast", cfg)
    slow = legacy.worker_safety_reward_from_components(info, "slow", cfg)
    manager = legacy.manager_safety_reward_from_components(info, cfg)
    for result in (fast, slow, manager):
        values = [value for value in result.values() if np.isscalar(value)]
        assert np.all(np.isfinite(np.asarray(values, dtype=float)))
    assert fast["global_used"] < 0.0 and fast["role_used"] < 0.0
    assert slow["global_used"] < 0.0 and slow["role_used"] < 0.0
    assert slow["role_scale"] == cfg.slow_role_specific_reward_scale == 25.0
    assert math.isclose(
        slow["role_weighted"],
        cfg.slow_role_specific_weight * cfg.slow_role_specific_reward_scale
        * slow["role_used"],
        rel_tol=1e-7,
    )
    assert abs(manager["reward"]) < abs(manager["raw_reward"])

    shaped = legacy.build_worker_reward(-10.0, 1.0, 1.0, -0.5, cfg)
    expected = (
        -10.0 + cfg.beta_latent * 10.0 * math.tanh(1.0)
        + cfg.beta_physical * 10.0 * math.tanh(1.0) - 0.5
    )
    assert math.isclose(shaped, expected, rel_tol=1e-7)

    primary = torch.tensor(100.0)
    auxiliary = torch.tensor(2.0)
    coefficient = adaptive_auxiliary_weight(primary, auxiliary, 0.10, cfg)
    assert math.isclose(coefficient, 0.10, rel_tol=1e-6)
    adaptive_cfg = copy.deepcopy(cfg)
    adaptive_cfg.adaptive_auxiliary_loss_scaling = True
    adaptive_cfg.auxiliary_loss_coefficient_max = 0.05
    capped = adaptive_auxiliary_weight(
        torch.tensor(100.0), torch.tensor(1e-12), 0.10, adaptive_cfg
    )
    assert math.isclose(capped, 0.05, rel_tol=1e-6)


def _test_worker_action_regularization_reference() -> None:
    batch = 3
    fast_obs = torch.zeros((batch, 115), dtype=torch.float32)
    fast_reference = worker_action_regularization_reference(fast_obs, "fast", 16)
    assert fast_reference.shape == (batch, 16)
    assert torch.all(fast_reference[:, :8] == 0.0)
    assert torch.all(fast_reference[:, 8:] == -1.0)
    # A zero Fast action is not neutral because it requests half of maximum curtailment.
    assert math.isclose(
        float((torch.zeros_like(fast_reference) - fast_reference).pow(2).mean()),
        0.5,
        rel_tol=1e-6,
    )

    slow_obs = torch.zeros(
        (batch, legacy.SLOW_OBSERVATION_LAYOUT.dimension), dtype=torch.float32
    )
    held = torch.linspace(-0.8, 0.8, 10).repeat(batch, 1)
    slow_obs[:, legacy.SLOW_OBSERVATION_LAYOUT.slices["held_slow_action"]] = held
    slow_reference = worker_action_regularization_reference(slow_obs, "slow", 10)
    assert slow_reference.shape == (batch, 10)
    assert torch.allclose(slow_reference, held)
    held[0, 0] = 1.0
    assert slow_reference[0, 0] != held[0, 0], "reference must not alias Replay storage"


def _test_manager_goal_semantics(device: torch.device) -> None:
    cfg = TrainConfig(
        training_stage="manager_train", run_mode="debug", episodes=1, episode_steps=40,
        hidden_dim=16, manager_hidden_dim=16, critic_hidden_dim=16, manager_latent_dim=8,
        manager_batch_size=2, manager_learning_starts=2, manager_updates_per_boundary=1,
        manager_random_warmup_segments=0, target_noise=0.0, policy_frequency=1,
    )
    raw = np.linspace(-0.8, 0.8, GOAL_DIM, dtype=np.float32)
    previous = fixed_manager_goal()
    executed1 = execute_manager_goal_np(raw, previous, cfg.goal_smoothing)
    executed2 = execute_manager_goal_np(raw, previous, cfg.goal_smoothing)
    tensor_executed = execute_manager_goal_tensor(
        torch.as_tensor(raw[None, :]), torch.as_tensor(previous[None, :]), cfg.goal_smoothing
    ).numpy()[0]
    assert np.array_equal(executed1, executed2)
    assert np.allclose(executed1, tensor_executed, atol=1e-6)

    buffer = ManagerReplayBuffer(8, 6, GOAL_DIM, device, cfg)
    obs = np.zeros(6, dtype=np.float32)
    for index in range(4):
        buffer.add(obs + index, obs + index + 1, executed1, -1.0, False, 5,
                   raw_goal=raw, previous_executed_goal=previous)
    assert np.allclose(buffer.manager_goals[0], executed1)
    assert np.allclose(buffer.raw_goals[0], raw)
    assert np.allclose(buffer.previous_executed_goals[0], previous)

    manager = ManagerTD3(6, cfg, device)
    calls: List[Tuple[torch.Tensor, torch.Tensor]] = []
    original = legacy.execute_manager_goal_tensor

    def tracked(raw_goal: torch.Tensor, previous_goal: torch.Tensor, smoothing: float) -> torch.Tensor:
        calls.append((raw_goal.detach().clone(), previous_goal.detach().clone()))
        return original(raw_goal, previous_goal, smoothing)

    legacy.execute_manager_goal_tensor = tracked  # type: ignore[assignment]
    try:
        logs = manager.update(buffer)
    finally:
        legacy.execute_manager_goal_tensor = original  # type: ignore[assignment]
    assert calls and len(calls) >= 2, "target Actor and online Actor must both apply the goal transform"
    assert manager.critic_updates > 0 and manager.actor_updates > 0
    _assert_finite(logs)


def _test_pure_slow_action_inverse_mapping() -> None:
    """Verify physical-to-Actor coordinates without constructing either network."""

    env = object.__new__(ElectricGasMultiScaleEnv)
    env.n_ess = len(ESS_CONFIGS)
    env.n_gfg = len(GFG_CONFIGS)
    env.n_p2g = len(P2G_CONFIGS)
    env.n_renew = len(RENEWABLE_CONFIGS)
    env.n_controlled_comp = len(CONTROLLED_COMPRESSOR_INDICES)
    env.slow_action_dim = env.n_ess + env.n_gfg + env.n_p2g + env.n_controlled_comp
    env.fast_action_dim = 2 * env.n_renew
    env.action_dim = env.slow_action_dim + env.fast_action_dim
    env.profiles = object()  # Only the reset guard in _current_normalized_slow_action uses this.
    env.last_inverter_projection = []

    ess = np.asarray(
        [-ESS_CONFIGS[0].max_p_mw, 0.0, ESS_CONFIGS[2].max_p_mw], dtype=float
    )
    gfg = np.asarray(
        [0.0, 0.5 * GFG_CONFIGS[1].max_p_mw, GFG_CONFIGS[2].max_p_mw], dtype=float
    )
    p2g = np.asarray(
        [0.0, 0.5 * P2G_CONFIGS[1].max_p_mw, P2G_CONFIGS[2].max_p_mw], dtype=float
    )
    controlled_index = CONTROLLED_COMPRESSOR_INDICES[0]
    controlled_cfg = COMPRESSOR_CONFIGS[controlled_index]

    def physical_with_ratio(ratio: float, fixed_delta: float = 0.0) -> PhysicalActions:
        ratios = default_compressor_ratios()
        fixed_index = next(i for i, item in enumerate(COMPRESSOR_CONFIGS) if not item.controllable)
        ratios[fixed_index] += fixed_delta
        ratios[controlled_index] = ratio
        return PhysicalActions(
            ess_p_mw=ess.copy(),
            gfg_p_mw=gfg.copy(),
            p2g_p_mw=p2g.copy(),
            compressor_ratio=ratios,
            renewable_p_mw=np.zeros(env.n_renew, dtype=float),
            renewable_q_mvar=np.zeros(env.n_renew, dtype=float),
            renewable_curtailment=np.zeros(env.n_renew, dtype=float),
        )

    mid_ratio = 0.5 * (
        controlled_cfg.min_pressure_ratio + controlled_cfg.max_pressure_ratio
    )
    expected_devices = np.asarray([-1.0, 0.0, 1.0] * 3, dtype=np.float32)
    for ratio, expected_ratio_action in (
        (controlled_cfg.min_pressure_ratio, -1.0),
        (mid_ratio, 0.0),
        (controlled_cfg.max_pressure_ratio, 1.0),
    ):
        physical = physical_with_ratio(ratio)
        normalized = np.asarray(
            ElectricGasMultiScaleEnv._normalized_action_from_physical(env, physical),
            dtype=np.float32,
        )
        held = normalized[:env.slow_action_dim]
        assert held.shape == (10,)
        assert np.allclose(held[:9], expected_devices, atol=1e-6)
        assert np.isclose(held[-1], expected_ratio_action, atol=1e-6)
        assert np.all(held >= -1.0 - 1e-6) and np.all(held <= 1.0 + 1e-6)

    baseline_physical = physical_with_ratio(mid_ratio)
    env.last_physical_slow = {
        "ess_p_mw": baseline_physical.ess_p_mw.copy(),
        "gfg_p_mw": baseline_physical.gfg_p_mw.copy(),
        "p2g_p_mw": baseline_physical.p2g_p_mw.copy(),
        "compressor_ratio": baseline_physical.compressor_ratio.copy(),
    }
    baseline_held = ElectricGasMultiScaleEnv._current_normalized_slow_action(env)
    assert baseline_held.shape == (10,)
    assert np.allclose(baseline_held[:9], expected_devices, atol=1e-6)
    assert np.isclose(baseline_held[-1], 0.0, atol=1e-6)
    fixed_changed = physical_with_ratio(mid_ratio, fixed_delta=0.05)
    env.last_physical_slow["compressor_ratio"] = fixed_changed.compressor_ratio.copy()
    assert np.array_equal(
        ElectricGasMultiScaleEnv._current_normalized_slow_action(env), baseline_held
    ), "fixed compressor must not enter the 10-dimensional held RL action"


def _test_observation_layout() -> None:
    layout = legacy.SLOW_OBSERVATION_LAYOUT
    assert layout.dimension == sum(size for _, size in layout.fields)
    cursor = 0
    values: Dict[str, np.ndarray] = {}
    for name, size in layout.fields:
        slc = layout.slices[name]
        assert slc.start == cursor and slc.stop == cursor + size
        values[name] = np.arange(size, dtype=np.float32)
        cursor += size
    flattened = layout.flatten(values)
    assert flattened.shape == (layout.dimension,)
    required = {
        "soc", "soc_low_margin", "soc_high_margin", "voltage_summary", "line_summary",
        "power_balance", "power_loss", "power_forecast", "gas_pressure_summary", "pipe_summary",
        "source_utilization", "gas_forecast", "time", "held_slow_action",
    }
    assert required.issubset(layout.slices)
    assert layout.slices["held_slow_action"].stop - layout.slices["held_slow_action"].start == 10


def _test_formal_safety_config() -> None:
    default_cfg = TrainConfig()
    assert default_cfg.run_mode == "formal"
    assert default_cfg.best_model_metric == "feasible_then_return"
    TrainConfig(run_mode="formal", best_model_metric="feasible_then_return")
    TrainConfig(run_mode="debug", best_model_metric="return")

    invalid_cases = (
        ({"run_mode": "formal", "best_model_metric": "return"}, "best_model_metric"),
        ({"run_mode": "formal", "fast_global_safety_weight": 0.0}, "fast_global_safety_weight"),
        ({"run_mode": "formal", "slow_global_safety_weight": 0.0}, "slow_global_safety_weight"),
    )
    for overrides, expected_text in invalid_cases:
        try:
            TrainConfig(**overrides)
        except ValueError as exc:
            assert expected_text in str(exc)
        else:
            raise AssertionError(f"formal safety config must reject {overrides}")

    # Debug-only ablations may remove the global term, while the role-specific
    # term keeps the corresponding Worker reward non-degenerate.
    TrainConfig(run_mode="debug", fast_global_safety_weight=0.0)
    TrainConfig(run_mode="debug", slow_global_safety_weight=0.0)


def _test_real_slow_safety_contract(device: torch.device) -> None:
    cfg = TrainConfig(
        run_mode="debug", episode_steps=40, hidden_dim=16, manager_hidden_dim=16,
        critic_hidden_dim=16, manager_latent_dim=8, slow_latent_dim=8, fast_latent_dim=8,
    )
    env = ElectricGasMultiScaleEnv()
    global_obs, _ = env.reset(seed=20260714)
    validate_environment_contract(env, cfg)
    features = env.get_slow_safety_features()
    expected_fields = tuple(name for name, _ in legacy.SLOW_OBSERVATION_LAYOUT.fields)
    assert tuple(features) == expected_fields
    for name, size in legacy.SLOW_OBSERVATION_LAYOUT.fields:
        assert features[name].size == size, (name, features[name].size, size)
        assert np.all(np.isfinite(features[name])), name
    assert features["held_slow_action"].size == env.slow_action_dim == 10
    assert np.all(features["held_slow_action"] >= -1.0 - 1e-6)
    assert np.all(features["held_slow_action"] <= 1.0 + 1e-6)
    assert env.slow_safety_schema_version == REQUIRED_SLOW_SAFETY_SCHEMA_VERSION == 2
    assert env.fast_action_dim == 16 and env.action_dim == 26

    load_base_mw = float(sum(item[1] for item in IEEE33_LOAD_DATA))
    actual_loss_mw = float(np.nansum(np.asarray(env.power.net.res_line.pl_mw, dtype=float)))
    assert np.isclose(features["power_loss"][0], actual_loss_mw / load_base_mw, atol=1e-6)
    assert features["power_balance"].size == 5

    fixed_index = next(index for index, item in enumerate(COMPRESSOR_CONFIGS) if not item.controllable)
    controlled_index = CONTROLLED_COMPRESSOR_INDICES[0]
    baseline_held = features["held_slow_action"].copy()
    original_ratios = env.last_physical_slow["compressor_ratio"].copy()
    env.last_physical_slow["compressor_ratio"][fixed_index] += 0.05
    fixed_changed = env.get_slow_safety_features()["held_slow_action"]
    assert fixed_changed.size == 10 and np.array_equal(fixed_changed, baseline_held)
    env.last_physical_slow["compressor_ratio"] = original_ratios.copy()
    env.last_physical_slow["compressor_ratio"][controlled_index] = (
        COMPRESSOR_CONFIGS[controlled_index].max_pressure_ratio
    )
    controlled_changed = env.get_slow_safety_features()["held_slow_action"]
    assert np.isclose(controlled_changed[-1], 1.0)
    assert not np.isclose(controlled_changed[-1], baseline_held[-1])
    env.last_physical_slow["compressor_ratio"] = original_ratios

    zero_obs, zero_reward, _, zero_truncated, zero_info = env.step(
        np.zeros(env.action_dim, dtype=np.float32)
    )
    assert not zero_truncated and np.isfinite(zero_reward) and np.all(np.isfinite(zero_obs))
    assert bool(zero_info.get("power_converged")) and bool(zero_info.get("gas_converged"))

    global_obs, _ = env.reset(seed=20260715)
    random_action = np.random.default_rng(20260715).uniform(
        -0.25, 0.25, size=env.action_dim
    ).astype(np.float32)
    random_obs, random_reward, _, random_truncated, random_info = env.step(random_action)
    assert not random_truncated and np.isfinite(random_reward)
    assert np.all(np.isfinite(random_obs))
    assert bool(random_info.get("power_converged")) and bool(random_info.get("gas_converged"))

    global_obs, _ = env.reset(seed=20260716)
    action = np.zeros(env.action_dim, dtype=np.float32)
    action[env.slow_action_dim - 1] = 1.0
    next_obs, reward, _, truncated, info = env.step(action)
    assert not truncated and np.isfinite(reward)
    assert bool(info.get("power_converged")) and bool(info.get("gas_converged"))
    stepped_features = env.get_slow_safety_features()
    assert np.isclose(stepped_features["held_slow_action"][-1], 1.0)
    slow_obs = ObservationBuilder(env, cfg.manager_interval).slow_obs(next_obs)
    assert slow_obs.size == legacy.SLOW_OBSERVATION_LAYOUT.dimension
    assert np.all(np.isfinite(global_obs)) and np.all(np.isfinite(next_obs)) and np.all(np.isfinite(slow_obs))

    agents = build_agents(env, cfg, device)
    metadata = checkpoint_metadata(agents, cfg)
    assert metadata["slow_observation_dim"] == slow_obs.size == agents.slow.obs_dim
    assert metadata["slow_action_dim"] == agents.slow.action_dim == 10
    assert metadata["fast_action_dim"] == agents.fast.action_dim == 16
    assert metadata["total_action_dim"] == 26
    assert metadata["slow_safety_schema_version"] == 2
    assert metadata["slow_observation_fields"] == expected_fields
    assert metadata["critic_action_semantics"] == "raw_request_action"

    raw_goal = np.linspace(-1.0, 1.0, GOAL_DIM, dtype=np.float32)
    previous_goal = fixed_manager_goal()
    assert np.array_equal(
        legacy.execute_manager_goal_np(raw_goal, previous_goal, cfg.goal_smoothing),
        execute_manager_goal_np(raw_goal, previous_goal, cfg.goal_smoothing),
    )
    goal = fixed_manager_goal()
    changed_slow_obs = slow_obs.copy()
    changed_slow_obs[legacy.SLOW_OBSERVATION_LAYOUT.slices["soc"]] += 0.01
    assert math.isclose(
        legacy.slow_physical_progress(slow_obs, changed_slow_obs, goal),
        slow_physical_progress(slow_obs, changed_slow_obs, goal),
        rel_tol=1e-7, abs_tol=1e-7,
    )


def _test_feasibility_thresholds() -> None:
    cfg = TrainConfig()
    metrics: Dict[str, float] = {
        "solver_failures": 0.0,
        "power_solver_success_rate": 1.0,
        "gas_solver_success_rate": 1.0,
        "soc_violation_rate": 0.0,
        "voltage_violation_rate": 0.0,
        "gas_pressure_violation_rate": 0.0,
        "line_overload_rate": 0.0,
        "pipe_velocity_violation_rate": 0.0,
        "source_capacity_violation_rate": 0.0,
        "mean_voltage_rms_deviation_pu": 0.0,
        "mean_gas_pressure_rms_deviation_bar": 0.0,
    }
    assert legacy.is_feasible(metrics, cfg)[0]
    for name in tuple(metrics):
        broken = dict(metrics)
        if name.endswith("success_rate"):
            broken[name] = -1.0
        else:
            broken[name] = 1.0
        feasible, reasons = legacy.is_feasible(broken, cfg)
        assert not feasible and any(name in reason for reason in reasons), name


def _test_debug_terminal_soc_penalty() -> None:
    class FakeEnv:
        config = DEFAULT_CONFIG
        ess_soc = np.asarray([item.soc_initial - 0.1 for item in ESS_CONFIGS], dtype=float)

    cfg = TrainConfig(run_mode="debug")
    info: Dict[str, Any] = {"reward_components": {}}
    reward = legacy.apply_debug_terminal_soc_penalty(FakeEnv(), info, 0.0, cfg)
    assert reward < 0.0
    assert info["reward_components"]["terminal_soc"] > 0.0


def _test_slow_executed_action_summary() -> None:
    global _LAST_SLOW_ACTOR_RAW, _CURRENT_SLOW_PENDING
    raw = np.array([0.8, -0.8], dtype=np.float32)
    guarded = np.array([0.5, -0.5], dtype=np.float32)
    _LAST_SLOW_ACTOR_RAW = raw.copy()
    pending = PendingSlowSegment(np.zeros(3, dtype=np.float32), fixed_manager_goal(), guarded.copy())
    expected_sum = np.zeros(2, dtype=np.float64)
    expected_square = np.zeros(2, dtype=np.float64)
    weight_sum = 0.0
    for offset in range(20):
        executed = guarded + np.array([0.01 * offset, -0.005 * offset], dtype=np.float32)
        pending.record_executed_action(executed, 0.99)
        weight = 0.99 ** offset
        expected_sum += weight * executed
        expected_square += weight * np.square(executed)
        weight_sum += weight
        pending.duration_steps += 1
    discounted_mean, mean, variance = pending.executed_summary()
    executed_stack = np.stack([guarded + np.array([0.01 * step, -0.005 * step]) for step in range(20)])
    expected_discounted_mean = expected_sum / weight_sum
    expected_mean = np.mean(executed_stack, axis=0)
    expected_variance = np.var(executed_stack, axis=0)
    assert np.allclose(pending.raw_action, raw)
    assert np.allclose(pending.guarded_action, guarded)
    assert np.allclose(discounted_mean, expected_discounted_mean, atol=1e-6)
    assert np.allclose(mean, expected_mean, atol=1e-6)
    assert np.allclose(variance, expected_variance, atol=1e-6)
    assert np.allclose(pending.first_executed_action, guarded)
    assert np.allclose(pending.last_executed_action, guarded + np.array([0.19, -0.095]))
    assert pending.max_dynamic_projection > 0.0
    assert len(pending.executed_actions_by_step) == 20
    assert np.allclose(np.stack(pending.executed_actions_by_step), executed_stack)

    cfg = TrainConfig(run_mode="debug", slow_interval=20)
    buffer = SlowReplayBuffer(2, 3, 2, GOAL_DIM, torch.device("cpu"), cfg)
    buffer.add(
        np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32), raw,
        discounted_mean, -1.0, 0.0, 0.0, fixed_manager_goal(), fixed_manager_goal(),
        False, 20, executed_action_sequence=executed_stack,
    )
    assert buffer.executed_action_sequence_lengths[0, 0] == 20
    assert np.allclose(buffer.executed_action_sequences[0, :20], executed_stack)
    sampled = buffer.sample_uniform(1)
    assert sampled["executed_action_sequence_lengths"].item() == 20
    assert bool(sampled["executed_action_sequence_mask"].all())
    restored = SlowReplayBuffer(2, 3, 2, GOAL_DIM, torch.device("cpu"), cfg)
    restored.load_state_dict(buffer.state_dict())
    assert np.array_equal(restored.executed_action_sequences, buffer.executed_action_sequences)
    assert np.array_equal(
        restored.executed_action_sequence_lengths, buffer.executed_action_sequence_lengths
    )
    assert buffer.raw_to_guard_projection_rms[0, 0] > 0.0
    assert buffer.projection_scores[0] > 0.0
    assert np.array_equal(
        restored.raw_to_guard_projection_rms, buffer.raw_to_guard_projection_rms
    )
    _LAST_SLOW_ACTOR_RAW = None
    _CURRENT_SLOW_PENDING = None


def _test_projection_schedule_and_mask() -> None:
    cfg = TrainConfig()
    assert math.isclose(projection_imitation_weight(cfg, 0), 0.10)
    assert math.isclose(projection_imitation_weight(cfg, cfg.projection_imitation_decay_updates), 0.02)
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
    guarded = historical_raw.clone()
    guarded[0, 0] = 0.0
    guarded[1, 1] = 0.0
    projected_elements, supported, imitation_elements = projection_imitation_element_mask(
        current, historical_raw, guarded, 0.01, 0.05
    )
    assert projected_elements.tolist() == [[True, False], [False, True], [False, False]]
    assert supported.tolist() == [True, False, True]
    assert imitation_elements.tolist() == [[True, False], [False, False], [False, False]]


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

    partial = TrainConfig.from_mapping({
        "batch_size": 128, "fast_batch_size": 256,
        "learning_starts": 50, "slow_learning_starts": 75,
        "updates_per_step": 3, "manager_updates_per_boundary": 7,
    })
    assert (partial.fast_batch_size, partial.slow_batch_size, partial.manager_batch_size) == (256, 128, 128)
    assert (partial.fast_learning_starts, partial.slow_learning_starts,
            partial.manager_learning_starts) == (50, 75, 50)
    assert (partial.fast_updates_per_step, partial.slow_updates_per_boundary,
            partial.manager_updates_per_boundary) == (3, 3, 7)
    beta_migrated = TrainConfig.from_mapping({
        "priority_beta_anneal_steps": 1234,
        "fast_priority_beta_anneal_updates": 4321,
    })
    assert beta_migrated.fast_priority_beta_anneal_updates == 4321
    assert beta_migrated.slow_priority_beta_anneal_updates == 1234
    assert beta_migrated.manager_priority_beta_anneal_updates == 1234

    invalid_cases = (
        ("gamma_fast", 0.0), ("tau", 0.0), ("policy_frequency", 0),
        ("priority_alpha", -0.1), ("priority_epsilon", 0.0),
        ("constraint_priority_weight", -1.0), ("projection_priority_weight", -1.0),
        ("projection_imitation_initial_weight", -0.1),
        ("slow_guard_imitation_multiplier", -0.1),
        ("projection_imitation_decay_updates", 0), ("fast_buffer_size", 0),
        ("fast_batch_size", 0), ("eval_interval", 0), ("eval_episodes", 0),
        ("worker_reward_clip_abs", -1.0), ("target_q_clip_abs", -1.0),
        ("episode_steps", DEFAULT_CONFIG.time.steps_per_day + 1),
        ("episodes", -1), ("fast_pretrain_episodes", -1), ("fast_lr", 0.0),
        ("gradient_clip", 0.0), ("target_noise", -0.1),
        ("fast_exploration_noise", -0.1), ("noise_decay_episodes", 0),
        ("goal_smoothing", 1.1), ("prioritized_sample_fraction", 1.1),
        ("max_td_error_for_priority", float("inf")),
        ("manager_solver_priority_weight", -1.0),
        ("slow_role_specific_reward_scale", -1.0),
        ("joint_early_stop_patience_evaluations", 0),
        ("beta_latent", float("nan")),
    )
    for name, value in invalid_cases:
        try:
            TrainConfig(**{name: value})
        except ValueError as exc:
            assert name in str(exc) and repr(value) in str(exc)
        else:
            raise AssertionError(f"invalid {name}={value!r} must fail")


def _test_constraint_score_robustness() -> None:
    env = ElectricGasMultiScaleEnv()
    malformed = {
        "constraint_metrics": {
            "vm_min_pu": np.array([np.nan, 0.1]),
            "vm_max_pu": np.array([np.inf]),
            "gas_pressure_min_bar": np.array([]),
            "max_line_loading_percent": np.array([np.inf, 50.0]),
            "source_capacity_violation_kg_s": np.array([np.nan, np.inf, 1.0, 2.0, 3.0]),
            "soc_min": np.array([np.nan]),
        }
    }
    score = normalized_constraint_score(malformed, env)
    assert np.isfinite(score) and 0.0 <= score <= 1.0
    failed_score = normalized_constraint_score({"solver_failed": True, "constraint_metrics": {}}, env)
    assert failed_score == 1.0


def _test_per_initialization_and_sampling(device: torch.device) -> None:
    cfg = TrainConfig(use_prioritized_replay=True, priority_alpha=1.0,
                      prioritized_sample_fraction=0.75,
                      priority_component_normalization="none")
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
    probability_buffer.update_priorities(
        np.array([0, 1, 999, -1]), np.array([np.nan, np.inf, -np.inf, 1e30])
    )
    assert np.all(np.isfinite(probability_buffer.priorities[:2]))
    assert np.all(probability_buffer.priorities[:2] >= cfg.priority_epsilon)
    assert np.all(probability_buffer.priorities[:2] <= cfg.max_replay_priority)
    assert probability_buffer.priority_fallback_count >= 1

    probability_buffer.priorities[:2] = 0.0
    fallback_before = probability_buffer.priority_fallback_count
    fallback_indices, fallback_weights = probability_buffer._sample_indices(8, 2)
    assert fallback_indices.shape == (8,) and np.all(np.isfinite(fallback_weights))
    assert probability_buffer.priority_fallback_count == fallback_before + 1

    probability_buffer.priorities[:2] = np.array([1.0, 4.0], dtype=np.float32)
    calls_before = probability_buffer.sample_calls
    indices, weights = probability_buffer._sample_indices(16, 2)
    priority_probability = probability_buffer._priority_probability(2)
    mix_probability = 0.75 * priority_probability + 0.25 / 2.0
    beta_step = calls_before + 1
    beta = cfg.priority_beta_initial + min(
        beta_step / cfg.fast_priority_beta_anneal_updates, 1.0
    ) * (cfg.priority_beta_final - cfg.priority_beta_initial)
    expected_weights = np.power(2 * mix_probability[indices], -beta)
    expected_weights /= expected_weights.max()
    assert np.allclose(weights[:, 0], expected_weights.astype(np.float32), atol=1e-6)


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


def _test_mixed_per_monte_carlo(device: torch.device) -> None:
    cfg = TrainConfig(
        priority_alpha=1.0, prioritized_sample_fraction=0.37,
        fast_buffer_size=8, fast_batch_size=3, fast_learning_starts=8,
        priority_component_normalization="none",
    )
    buffer = FastReplayBuffer(8, 2, 1, GOAL_DIM, device, cfg)
    goal = fixed_manager_goal()
    for _ in range(3):
        buffer.add(np.zeros(2, np.float32), np.zeros(2, np.float32),
                   np.zeros(1, np.float32), np.zeros(1, np.float32),
                   0.0, 0.0, 0.0, goal, goal, False)
    buffer.priorities[:3] = np.array([1.0, 2.0, 7.0], dtype=np.float32)
    priority_probability = np.array([0.1, 0.2, 0.7])
    theoretical = 0.37 * priority_probability + 0.63 / 3.0
    counts = np.zeros(3, dtype=np.int64)
    for _ in range(6000):
        indices, weights = buffer._sample_indices(3, 3)
        np.add.at(counts, indices, 1)
        assert np.all(np.isfinite(weights)) and float(weights.max()) <= 1.0 + 1e-7
    empirical = counts / counts.sum()
    assert np.allclose(empirical, theoretical, atol=0.02), (empirical, theoretical)
    for fraction, expected in ((0.0, np.full(3, 1.0 / 3.0)),
                               (1.0, priority_probability)):
        cfg.prioritized_sample_fraction = fraction
        counts.fill(0)
        for _ in range(3000):
            indices, weights = buffer._sample_indices(1, 3)
            counts[indices[0]] += 1
            assert np.isfinite(weights[0, 0]) and weights[0, 0] == 1.0
        assert np.allclose(counts / counts.sum(), expected, atol=0.03)


def _test_raw_critic_executed_transition_and_per(device: torch.device) -> None:
    cfg = TrainConfig(
        hidden_dim=32, critic_hidden_dim=32, fast_latent_dim=8,
        fast_batch_size=2, fast_learning_starts=2, fast_updates_per_step=1,
        use_transition_model=True, policy_frequency=100,
    )
    buffer = FastReplayBuffer(32, 10, 2, GOAL_DIM, device, cfg)
    goal = fixed_manager_goal()
    _fill_worker_buffer(buffer, 10, 2, 8, goal)
    worker = WorkerTD3("fast", 10, 2, 8, 1e-3, cfg, device)
    critic_actions: List[torch.Tensor] = []
    transition_actions: List[torch.Tensor] = []
    latent_recompute_calls: List[int] = []
    target_delta_calls: List[int] = []
    original_critic_forward = worker.critic.forward
    original_transition_forward = worker.transition_model.forward  # type: ignore[union-attr]
    original_latent_reward = worker._target_latent_reward
    original_target_delta = worker._target_transition_delta
    actor_before = [parameter.detach().clone() for parameter in worker.actor.parameters()]

    def critic_forward(z: torch.Tensor, worker_goal: torch.Tensor, action: torch.Tensor):
        critic_actions.append(action.detach().cpu())
        return original_critic_forward(z, worker_goal, action)

    def transition_forward(z: torch.Tensor, action: torch.Tensor):
        transition_actions.append(action.detach().cpu())
        return original_transition_forward(z, action)

    def latent_reward(obs: torch.Tensor, next_obs: torch.Tensor, goals: torch.Tensor):
        latent_recompute_calls.append(obs.shape[0])
        return original_latent_reward(obs, next_obs, goals)

    def target_delta(obs: torch.Tensor, next_obs: torch.Tensor):
        target_delta_calls.append(obs.shape[0])
        return original_target_delta(obs, next_obs)

    worker.critic.forward = critic_forward  # type: ignore[method-assign]
    worker.transition_model.forward = transition_forward  # type: ignore[union-attr,method-assign]
    worker._target_latent_reward = latent_reward  # type: ignore[method-assign]
    worker._target_transition_delta = target_delta  # type: ignore[method-assign]
    logs = worker.update(buffer)
    _assert_finite(logs)
    assert torch.allclose(critic_actions[0], torch.full_like(critic_actions[0], 0.75))
    assert all(torch.allclose(action, torch.full_like(action, -0.25)) for action in transition_actions)
    assert len(target_delta_calls) == 2, "encoder auxiliary and transition losses must share target delta semantics"
    for target_module in (worker.target_encoder, worker.target_actor, worker.target_critic):
        assert all(parameter.grad is None for parameter in target_module.parameters())
    assert all(parameter.grad is None for parameter in worker.critic.parameters()), (
        "Critic must not accumulate gradients during Worker Actor backward"
    )
    assert any(parameter.grad is not None for parameter in worker.actor.parameters())
    assert any(not torch.allclose(before, after.detach())
               for before, after in zip(actor_before, worker.actor.parameters()))
    sampled = buffer.sample(4)
    assert "indices" in sampled and "is_weights" in sampled
    assert "rewards" not in sampled and "reward_intrinsic" not in sampled
    assert latent_recompute_calls, "sampled transitions must recompute latent reward with target encoder"
    for key in ("training_batch/fast/reward_external_mean", "training_batch/fast/reward_latent_mean",
                "training_batch/fast/reward_physical_mean", "training_batch/fast/reward_projection_mean",
                "training_batch/fast/batch_reward_mean"):
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
    assert math.isclose(logs["training_batch/slow/mean_duration_steps"], 7.0)
    assert worker.total_updates == 2
    assert worker.update(buffer) == {}, "没有新边界样本时不应重复更新 Slow"
    cfg.slow_shaping_duration_mode = "normalized"
    _fill_worker_buffer(buffer, 12, 2, 1, goal, slow=True)
    normalized_logs = worker.update(buffer)
    expected_scale = (1.0 - cfg.gamma_fast ** 7) / (1.0 - cfg.gamma_fast ** cfg.slow_interval)
    assert math.isclose(
        normalized_logs["training_batch/slow/shaping_duration_scale_mean"],
        expected_scale, rel_tol=1e-5,
    )
    assert math.isclose(
        normalized_logs["training_batch/slow/reward_projection_duration_adjusted_mean"],
        normalized_logs["training_batch/slow/reward_projection_mean"], rel_tol=1e-7,
    ), "interval-accumulated projection penalty must not receive a second duration scale"

    manager_buffer = ManagerReplayBuffer(8, 6, GOAL_DIM, device)
    manager = ManagerTD3(6, cfg, device)
    manager_actor_before = [parameter.detach().clone() for parameter in manager.actor.parameters()]
    for duration in (13, 40):
        manager_buffer.add(
            np.zeros(6, dtype=np.float32), np.ones(6, dtype=np.float32),
            goal, -1.0, False, duration,
        )
    manager_logs = manager.update(manager_buffer)
    _assert_finite(manager_logs)
    assert manager.total_updates == 4
    assert 13.0 <= manager_logs["training_batch/manager/mean_duration_steps"] <= 40.0
    assert all(parameter.grad is None for parameter in manager.critic.parameters())
    assert any(parameter.grad is not None for parameter in manager.actor.parameters())
    assert any(not torch.allclose(before, after.detach())
               for before, after in zip(manager_actor_before, manager.actor.parameters()))
    for target_module in (manager.target_encoder, manager.target_actor, manager.target_critic):
        assert all(parameter.grad is None for parameter in target_module.parameters())

    mean_score, max_score, final_score = aggregate_segment_constraint([0.0] * 19 + [0.2], False)
    assert math.isclose(mean_score, 0.01, rel_tol=1e-5)
    assert math.isclose(max_score, 0.2, rel_tol=1e-5)
    assert math.isclose(final_score, 0.105, rel_tol=1e-5)
    _, _, failed_score = aggregate_segment_constraint([0.0] * 20, True)
    assert failed_score == 1.0, "single solver failure must not be diluted by segment averaging"


def _test_nonfinite_update_transaction(device: torch.device) -> None:
    cfg = TrainConfig(
        hidden_dim=16, critic_hidden_dim=16, fast_latent_dim=8,
        fast_buffer_size=8, fast_batch_size=2, fast_learning_starts=2,
        nonfinite_update_policy="skip_batch", policy_frequency=1,
        priority_component_normalization="none",
    )
    goal = fixed_manager_goal()

    def module_snapshot(agent: WorkerTD3) -> List[torch.Tensor]:
        modules = (agent.encoder, agent.actor, agent.critic, agent.target_encoder,
                   agent.target_actor, agent.target_critic)
        return [parameter.detach().clone() for module in modules for parameter in module.parameters()]

    nan_buffer = FastReplayBuffer(8, 4, 2, GOAL_DIM, device, cfg)
    _fill_worker_buffer(nan_buffer, 4, 2, 2, goal)
    nan_buffer.reward_external[0, 0] = np.nan
    nan_buffer.reward_external[1, 0] = np.nan
    nan_worker = WorkerTD3("fast", 4, 2, 8, 1e-3, cfg, device)
    before = module_snapshot(nan_worker)
    priority_before = nan_buffer.priorities.copy()
    logs = nan_worker.update(nan_buffer)
    after = module_snapshot(nan_worker)
    assert logs["training_batch/fast/nonfinite_batch_skipped"] == 1.0
    assert all(torch.equal(left, right) for left, right in zip(before, after))
    assert np.array_equal(priority_before, nan_buffer.priorities)

    gradient_buffer = FastReplayBuffer(8, 4, 2, GOAL_DIM, device, cfg)
    _fill_worker_buffer(gradient_buffer, 4, 2, 2, goal)
    gradient_worker = WorkerTD3("fast", 4, 2, 8, 1e-3, cfg, device)
    before = module_snapshot(gradient_worker)
    first_parameter = next(gradient_worker.critic.parameters())
    handle = first_parameter.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
    try:
        logs = gradient_worker.update(gradient_buffer)
    finally:
        handle.remove()
    after = module_snapshot(gradient_worker)
    assert logs["training_batch/fast/nonfinite_batch_skipped"] == 1.0
    assert all(torch.equal(left, right) for left, right in zip(before, after))


def _test_replay_wrap_boundary_updates(device: torch.device) -> None:
    cfg = TrainConfig(
        hidden_dim=16, manager_hidden_dim=16, critic_hidden_dim=16,
        slow_latent_dim=8, manager_latent_dim=8,
        slow_batch_size=1, manager_batch_size=1,
        slow_learning_starts=1, manager_learning_starts=1,
        slow_updates_per_boundary=1, manager_updates_per_boundary=1,
        policy_frequency=2,
    )
    goal = fixed_manager_goal()
    slow_buffer = SlowReplayBuffer(2, 6, 2, GOAL_DIM, device, cfg)
    slow = WorkerTD3("slow", 6, 2, 8, 1e-3, cfg, device)
    for insertion in range(1, 6):
        obs = np.full(6, insertion * 0.01, dtype=np.float32)
        slow_buffer.add(obs, obs + 0.01, np.zeros(2, dtype=np.float32),
                        np.zeros(2, dtype=np.float32), -1.0, 0.0, 0.0,
                        goal, goal, False, 20)
        before = slow.total_updates
        logs = slow.update(slow_buffer)
        _assert_finite(logs)
        assert slow.total_updates == before + 1
        assert slow_buffer.total_insertions == insertion
        assert slow.update(slow_buffer) == {}, "same Slow insertion must not trigger twice"
    assert len(slow_buffer) == 2 and slow_buffer.total_insertions == 5

    manager_buffer = ManagerReplayBuffer(2, 5, GOAL_DIM, device)
    manager = ManagerTD3(5, cfg, device)
    for insertion in range(1, 6):
        obs = np.full(5, insertion * 0.01, dtype=np.float32)
        manager_buffer.add(obs, obs + 0.01, goal, -1.0, False, 40)
        before = manager.total_updates
        logs = manager.update(manager_buffer)
        _assert_finite(logs)
        assert manager.total_updates == before + 1
        assert manager_buffer.total_insertions == insertion
        assert manager.update(manager_buffer) == {}, "same Manager insertion must not trigger twice"
    assert len(manager_buffer) == 2 and manager_buffer.total_insertions == 5
    assert manager.state_dict()["last_update_insertion_id"] == manager._last_update_insertion_id


def _synthetic_evaluation_stats(mean_return: float = 0.0,
                                solver_failures: float = 0.0) -> Dict[str, float]:
    stats = {
        "mean_return": mean_return,
        "solver_failures": solver_failures,
        "power_success_rate": 1.0,
        "gas_success_rate": 1.0,
        "power_solver_success_rate": 1.0,
        "gas_solver_success_rate": 1.0,
        "mean_voltage_rms_deviation_pu": 0.0,
        "mean_gas_pressure_rms_deviation_bar": 0.0,
    }
    for name in (
        "voltage_violation_rate", "gas_pressure_violation_rate", "line_overload_rate",
        "pipe_velocity_violation_rate", "source_capacity_violation_rate", "soc_violation_rate",
        "max_voltage_violation", "max_pressure_violation", "max_line_overload",
        "max_pipe_velocity_violation", "max_source_capacity_violation",
        "voltage_violation_cost_per_step", "gas_pressure_violation_cost_per_step",
        "line_overload_cost_per_step", "pipe_velocity_violation_cost_per_step",
        "source_capacity_violation_cost_per_step",
    ):
        stats[name] = 0.0
    return stats


def _test_checkpoint_and_environment(device: torch.device) -> None:
    cfg = TrainConfig(
        training_stage="joint_finetune", hidden_dim=32, manager_hidden_dim=32, critic_hidden_dim=32,
        manager_latent_dim=8, slow_latent_dim=8, fast_latent_dim=8,
        fast_buffer_size=4, slow_buffer_size=1, manager_buffer_size=1,
        fast_batch_size=4, slow_batch_size=1, manager_batch_size=1,
        fast_learning_starts=1, slow_learning_starts=1, manager_learning_starts=1,
        eval_episodes=1, eval_seeds=(42,), no_tensorboard=True,
    )
    env = ElectricGasMultiScaleEnv()
    obs, _ = env.reset(seed=42)
    assert env.slow_action_dim == 10 and env.fast_action_dim == 16 and env.action_dim == 26
    next_obs, reward, _, truncated, info = env.step(np.zeros(env.action_dim, dtype=np.float32))
    assert next_obs.shape == obs.shape and np.isfinite(reward) and not truncated
    assert "reward_components" in info and "constraint_metrics" in info
    agents = build_agents(env, cfg, device)
    agents.fast.total_updates = 7
    agents.slow.total_updates = 5
    agents.manager.total_updates = 3
    root = Path("hierarchical_td3_test_runs")
    root.mkdir(exist_ok=True)
    path = root / "optimized_checkpoint_test.pt"
    synthetic_stats = _synthetic_evaluation_stats(999_999.0, 0.0)
    synthetic_metric = legacy.metric_state_from_evaluation(synthetic_stats, cfg)
    builder = ObservationBuilder(env, cfg.manager_interval)
    fast_buffer = FastReplayBuffer(
        cfg.fast_buffer_size, builder.fast_obs(0, obs).size, env.fast_action_dim, GOAL_DIM, device, cfg
    )
    slow_buffer = SlowReplayBuffer(
        cfg.slow_buffer_size, builder.slow_obs(obs).size, env.slow_action_dim, GOAL_DIM, device, cfg
    )
    manager_buffer = ManagerReplayBuffer(
        cfg.manager_buffer_size, builder.manager_obs(obs).size, GOAL_DIM, device, cfg
    )
    save_checkpoint(
        path, cfg, agents, 0, 123, 999_999.0, synthetic_metric, synthetic_stats,
        fast_buffer, slow_buffer, manager_buffer,
    )
    payload = legacy.trusted_torch_load(str(path), device)
    assert payload["checkpoint_kind"] == "full_resume"
    assert payload["fast_replay"]["valid_size"] == 0
    assert payload["fast_replay"]["obs"].shape[0] == 0
    for key in ("checkpoint_schema_version", "algorithm_version", "training_stage",
                "env_model_version", "slow_safety_schema_version", "slow_observation_fields",
                "critic_action_semantics", "executed_action_semantics",
                "slow_interval", "manager_interval",
                "global_state_dim", "slow_action_dim", "fast_action_dim", "total_action_dim",
                "goal_dim", "optimization_config", "best_metric_state", "best_evaluation_stats"):
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
    restored_fast = FastReplayBuffer(
        cfg.fast_buffer_size, builder.fast_obs(0, obs).size, env.fast_action_dim, GOAL_DIM, device, cfg
    )
    restored_slow = SlowReplayBuffer(
        cfg.slow_buffer_size, builder.slow_obs(obs).size, env.slow_action_dim, GOAL_DIM, device, cfg
    )
    restored_manager = ManagerReplayBuffer(
        cfg.manager_buffer_size, builder.manager_obs(obs).size, GOAL_DIM, device, cfg
    )
    load_checkpoint(
        str(path), restored_agents, device, mode="resume",
        fast_replay=restored_fast, slow_replay=restored_slow, manager_replay=restored_manager,
    )
    assert restored_agents.fast.total_updates == 7
    assert restored_agents.fast._last_update_insertion_id == 0
    goal = fixed_manager_goal()
    a = agents.fast.select_action(builder.fast_obs(0, obs), goal, 0.0, deterministic=True)
    b = restored_agents.fast.select_action(builder.fast_obs(0, obs), goal, 0.0, deterministic=True)
    assert np.allclose(a, b, atol=1e-5)

    transfer_cfg = copy.deepcopy(cfg)
    transfer_cfg.training_stage = "slow_pretrain"
    transfer_cfg.checkpoint_load_mode = "stage_transfer"
    transfer_agents = build_agents(env, transfer_cfg, device)
    load_checkpoint(str(path), transfer_agents, device, mode="stage_transfer")
    assert all(agent.total_updates == 0 for agent in
               (transfer_agents.manager, transfer_agents.slow, transfer_agents.fast))
    assert transfer_agents.slow.normalizer.training
    assert not transfer_agents.manager.normalizer.training and not transfer_agents.fast.normalizer.training
    assert all(agent._last_update_insertion_id == 0 for agent in
               (transfer_agents.manager, transfer_agents.slow, transfer_agents.fast))
    source_critic = next(agents.slow.critic.parameters()).detach()
    transferred_critic = next(transfer_agents.slow.critic.parameters()).detach()
    assert torch.allclose(source_critic, transferred_critic), "stage_transfer explicitly retains Critic weights"

    policy_cfg = copy.deepcopy(cfg)
    policy_cfg.training_stage = "fast_pretrain"
    policy_cfg.checkpoint_load_mode = "policy_only"
    policy_agents = build_agents(env, policy_cfg, device)
    fresh_critic = next(policy_agents.fast.critic.parameters()).detach().clone()
    load_checkpoint(str(path), policy_agents, device, mode="policy_only")
    assert torch.allclose(fresh_critic, next(policy_agents.fast.critic.parameters()).detach())
    assert policy_agents.fast.total_updates == 0
    assert policy_agents.fast.normalizer.training
    assert not policy_agents.slow.normalizer.training and not policy_agents.manager.normalizer.training

    legacy_payload = copy.deepcopy(payload)
    for key in ("checkpoint_schema_version", "training_stage", "slow_interval", "manager_interval",
                "global_state_dim", "optimization_config", "slow_safety_schema_version",
                "slow_observation_fields", "critic_action_semantics", "executed_action_semantics"):
        legacy_payload.pop(key, None)
    legacy_path = root / "optimized_legacy_checkpoint_test.pt"
    torch.save(legacy_payload, str(legacy_path))
    try:
        load_checkpoint(str(legacy_path), build_agents(env, cfg, device), device, mode="resume")
    except ValueError as exc:
        message = str(exc)
        assert "strict resume" in message and "slow_safety_schema_version" in message
    else:
        raise AssertionError("checkpoint without Slow safety/action semantics must not strict-resume")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_checkpoint(str(legacy_path), build_agents(env, cfg, device), device, mode="stage_transfer")
    assert any("Explicit checkpoint stage_transfer migration" in str(item.message) for item in caught)
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

    for stage, filename in _STAGE_BEST_FILES.items():
        stage_cfg = copy.deepcopy(cfg)
        stage_cfg.training_stage = stage
        stage_root = root / ("best_name_" + stage)
        save_best_files(stage_root, agents, stage_cfg, 0, 123, 1.0, synthetic_metric, synthetic_stats)
        assert (stage_root / filename).exists()
        best_payload = legacy.trusted_torch_load(str(stage_root / filename), device)
        assert best_payload["checkpoint_kind"] == "lightweight"
        assert "fast_replay" not in best_payload and "rng_state" not in best_payload
        assert "encoder_optim" not in best_payload["fast"]

    incomplete = copy.deepcopy(payload)
    incomplete.pop("fast_replay")
    incomplete_path = root / "incomplete_resume.pt"
    torch.save(incomplete, str(incomplete_path))
    strict_targets = build_agents(env, cfg, device)
    try:
        load_checkpoint(
            str(incomplete_path), strict_targets, device, mode="resume",
            fast_replay=restored_fast, slow_replay=restored_slow,
            manager_replay=restored_manager,
        )
    except ValueError as exc:
        assert "fast_replay" in str(exc)
    else:
        raise AssertionError("strict resume must reject a missing Replay component")

    non_strict_cfg = copy.deepcopy(cfg)
    non_strict_cfg.strict_resume_required = False
    non_strict_agents = build_agents(env, non_strict_cfg, device)
    partial_payload = load_checkpoint(
        str(incomplete_path), non_strict_agents, device, mode="resume",
        fast_replay=restored_fast, slow_replay=restored_slow,
        manager_replay=restored_manager,
    )
    assert not partial_payload["strict_resume_restored"]
    assert "fast_replay" in partial_payload["resume_missing_components"]

    atomic_path = root / "atomic_preserve_test.pt"
    atomic_path.write_bytes(b"previous-valid-checkpoint")
    original_torch_save = torch.save

    def failing_save(payload: Any, destination: str) -> None:
        Path(destination).write_bytes(b"partial")
        raise OSError("intentional atomic save failure")

    torch.save = failing_save  # type: ignore[assignment]
    try:
        try:
            _atomic_torch_save({"checkpoint_kind": "test"}, atomic_path)
        except OSError:
            pass
        else:
            raise AssertionError("atomic save failure must propagate")
    finally:
        torch.save = original_torch_save  # type: ignore[assignment]
    assert atomic_path.read_bytes() == b"previous-valid-checkpoint"


def _test_strict_resume_replay_rng_and_stage(device: torch.device) -> None:
    cfg = TrainConfig(
        training_stage="joint_finetune", run_mode="debug", hidden_dim=16, manager_hidden_dim=16,
        critic_hidden_dim=16, manager_latent_dim=8, slow_latent_dim=8, fast_latent_dim=8,
        fast_buffer_size=3, slow_buffer_size=2, manager_buffer_size=2,
        fast_batch_size=1, slow_batch_size=1, manager_batch_size=1,
        fast_learning_starts=1, slow_learning_starts=1, manager_learning_starts=1,
        slow_updates_per_boundary=1, manager_updates_per_boundary=1,
        manager_use_prioritized_replay=True, no_tensorboard=True,
        fast_random_warmup_steps=0, slow_random_warmup_segments=0,
        manager_random_warmup_segments=0,
    )
    env = ElectricGasMultiScaleEnv()
    obs, _ = env.reset(seed=cfg.seed)
    builder = ObservationBuilder(env, cfg.manager_interval)
    manager_dim = builder.manager_obs(obs).size
    slow_dim = builder.slow_obs(obs).size
    fast_dim = builder.fast_obs(0, obs).size
    agents = build_agents(env, cfg, device)
    fast = FastReplayBuffer(cfg.fast_buffer_size, fast_dim, env.fast_action_dim, GOAL_DIM, device, cfg)
    slow = SlowReplayBuffer(cfg.slow_buffer_size, slow_dim, env.slow_action_dim, GOAL_DIM, device, cfg)
    manager = ManagerReplayBuffer(cfg.manager_buffer_size, manager_dim, GOAL_DIM, device, cfg)
    goal = fixed_manager_goal()
    _fill_worker_buffer(fast, fast_dim, env.fast_action_dim, 5, goal)
    _fill_worker_buffer(slow, slow_dim, env.slow_action_dim, 4, goal, slow=True)
    for index in range(4):
        start = np.full(manager_dim, index * 0.01, dtype=np.float32)
        manager.add(start, start + 0.01, goal, -1.0, False, 40,
                    segment_constraint_mean=0.1 * index,
                    segment_constraint_max=0.2 * index,
                    solver_failure_seen=index == 3)
    fast.sample(1)
    slow.sample(1)
    manager.sample(1)
    fast.update_priorities(np.array([0, 1]), np.array([3.0, np.inf]))
    slow.update_priorities(np.array([0, 1]), np.array([2.0, 4.0]))
    manager.update_priorities(np.array([0, 1]), np.array([5.0, 6.0]))

    legacy.set_seed(9876)
    path = Path("hierarchical_td3_test_runs/strict_resume_checkpoint.pt")
    save_checkpoint(
        path, cfg, agents, episode=6, global_step=321, best_return=-5.0,
        fast_replay=fast, slow_replay=slow, manager_replay=manager, next_episode=7,
    )
    expected_random = random.random()
    expected_numpy = np.random.random(4)
    expected_torch = torch.rand(4)

    restored_agents = build_agents(env, cfg, device)
    restored_fast = FastReplayBuffer(3, fast_dim, env.fast_action_dim, GOAL_DIM, device, cfg)
    restored_slow = SlowReplayBuffer(2, slow_dim, env.slow_action_dim, GOAL_DIM, device, cfg)
    restored_manager = ManagerReplayBuffer(2, manager_dim, GOAL_DIM, device, cfg)
    payload = load_checkpoint(
        str(path), restored_agents, device, mode="resume",
        fast_replay=restored_fast, slow_replay=restored_slow, manager_replay=restored_manager,
    )
    assert payload["next_episode"] == 7 and payload["global_step"] == 321
    for original, restored in ((fast, restored_fast), (slow, restored_slow), (manager, restored_manager)):
        assert (restored.idx, restored.full, restored.total_insertions, restored.sample_calls) == (
            original.idx, original.full, original.total_insertions, original.sample_calls,
        )
        assert restored.priority_fallback_count == original.priority_fallback_count
        for name in ("priorities", "td_priorities", "constraint_scores", "projection_scores"):
            assert np.array_equal(getattr(restored, name), getattr(original, name))
    assert np.array_equal(restored_fast.obs, fast.obs)
    assert np.array_equal(restored_slow.duration_steps, slow.duration_steps)
    assert np.array_equal(restored_manager.solver_failure_seen, manager.solver_failure_seen)
    assert restored_agents.slow._last_update_insertion_id == agents.slow._last_update_insertion_id
    assert restored_agents.manager._last_update_insertion_id == agents.manager._last_update_insertion_id
    assert random.random() == expected_random
    assert np.array_equal(np.random.random(4), expected_numpy)
    assert torch.equal(torch.rand(4), expected_torch)
    rng_snapshot = legacy.capture_rng_state()
    original_batch = fast.sample(2)
    legacy.restore_rng_state(rng_snapshot)
    restored_batch = restored_fast.sample(2)
    assert torch.equal(original_batch["indices"], restored_batch["indices"])
    assert torch.allclose(original_batch["is_weights"], restored_batch["is_weights"])

    slow_obs = np.zeros(slow_dim, dtype=np.float32)
    restored_slow.add(slow_obs, slow_obs + 0.01, np.zeros(env.slow_action_dim, dtype=np.float32),
                      np.zeros(env.slow_action_dim, dtype=np.float32), -1.0, 0.0, 0.0,
                      goal, goal, False, 20)
    slow_updates_before = restored_agents.slow.total_updates
    _assert_finite(restored_agents.slow.update(restored_slow))
    assert restored_agents.slow.total_updates == slow_updates_before + 1
    manager_obs = np.zeros(manager_dim, dtype=np.float32)
    restored_manager.add(manager_obs, manager_obs + 0.01, goal, -1.0, False, 40)
    manager_updates_before = restored_agents.manager.total_updates
    _assert_finite(restored_agents.manager.update(restored_manager))
    assert restored_agents.manager.total_updates == manager_updates_before + 1

    wrong_stage_cfg = copy.deepcopy(cfg)
    wrong_stage_cfg.training_stage = "slow_pretrain"
    wrong_stage_agents = build_agents(env, wrong_stage_cfg, device)
    try:
        load_checkpoint(str(path), wrong_stage_agents, device, mode="resume")
    except ValueError as exc:
        assert "stage_transfer" in str(exc) and "training_stage" in str(exc)
    else:
        raise AssertionError("cross-stage resume must be rejected")
    load_checkpoint(str(path), wrong_stage_agents, device, mode="stage_transfer")
    assert wrong_stage_agents.slow.total_updates == 0

    def restored_copy() -> Tuple[AgentBundle, FastReplayBuffer]:
        copied_agents = build_agents(env, cfg, device)
        copied_fast = FastReplayBuffer(3, fast_dim, env.fast_action_dim, GOAL_DIM, device, cfg)
        copied_slow = SlowReplayBuffer(2, slow_dim, env.slow_action_dim, GOAL_DIM, device, cfg)
        copied_manager = ManagerReplayBuffer(2, manager_dim, GOAL_DIM, device, cfg)
        load_checkpoint(
            str(path), copied_agents, device, mode="resume",
            fast_replay=copied_fast, slow_replay=copied_slow, manager_replay=copied_manager,
        )
        return copied_agents, copied_fast

    copy_agents_a, copy_fast_a = restored_copy()
    copy_agents_b, copy_fast_b = restored_copy()
    deterministic_obs = np.zeros(fast_dim, dtype=np.float32)
    action_a = copy_agents_a.fast.select_action(deterministic_obs, goal, 0.0, deterministic=True)
    action_b = copy_agents_b.fast.select_action(deterministic_obs, goal, 0.0, deterministic=True)
    assert np.array_equal(action_a, action_b)
    update_rng = legacy.capture_rng_state()
    logs_a = copy_agents_a.fast._update_once(copy_fast_a, 1)
    legacy.restore_rng_state(update_rng)
    logs_b = copy_agents_b.fast._update_once(copy_fast_b, 1)
    assert logs_a.keys() == logs_b.keys()
    assert all(math.isclose(logs_a[key], logs_b[key], rel_tol=1e-6, abs_tol=1e-6)
               for key in logs_a)
    for module_a, module_b in ((copy_agents_a.fast.encoder, copy_agents_b.fast.encoder),
                               (copy_agents_a.fast.actor, copy_agents_b.fast.actor),
                               (copy_agents_a.fast.critic, copy_agents_b.fast.critic)):
        assert all(torch.allclose(a, b, atol=1e-7) for a, b in
                   zip(module_a.parameters(), module_b.parameters()))

    continuation_cfg = copy.deepcopy(cfg)
    continuation_cfg.episodes = 8
    continuation_cfg.episode_steps = 1
    continuation_cfg.device = "cpu"
    continuation_cfg.load_checkpoint = str(path)
    continuation_cfg.checkpoint_load_mode = "resume"
    continuation_cfg.eval_interval = 8
    continuation_cfg.eval_episodes = 1
    continuation_cfg.eval_seeds = (6543,)
    continuation_cfg.checkpoint_dir = "hierarchical_td3_test_runs/strict_resume_continuation"
    continuation = run_training(continuation_cfg)
    assert continuation["global_step"] == 322
    assert continuation["next_episode"] == 8


def _test_safe_best_model_metric() -> None:
    cfg = TrainConfig(training_stage="joint_finetune", best_model_metric="feasible_then_return")
    unsafe = _synthetic_evaluation_stats(1_000_000.0, 1.0)
    feasible = _synthetic_evaluation_stats(-10_000.0, 0.0)
    unsafe_state = legacy.metric_state_from_evaluation(unsafe, cfg)
    assert legacy.is_better_evaluation(feasible, unsafe_state, cfg)
    feasible_state = legacy.metric_state_from_evaluation(feasible, cfg)
    assert not legacy.is_better_evaluation(unsafe, feasible_state, cfg)
    missing = dict(feasible)
    missing.pop("soc_violation_rate")
    try:
        legacy.metric_state_from_evaluation(missing, cfg)
    except ValueError as exc:
        assert "soc_violation_rate" in str(exc)
    else:
        raise AssertionError("missing safety metrics must not be interpreted as safe")
    nearly_equal = dict(feasible)
    nearly_equal["max_voltage_violation"] += 1e-10
    nearly_equal["mean_return"] += 1.0
    assert legacy.is_better_evaluation(nearly_equal, feasible_state, cfg), (
        "metric noise inside tolerance must not hide a meaningful return improvement"
    )
    early_cfg = TrainConfig(
        run_mode="debug", training_stage="joint_finetune",
        joint_early_stop_min_episodes=20,
        joint_early_stop_patience_evaluations=3,
    )
    assert not legacy.should_early_stop_joint(early_cfg, 19, 3)
    assert not legacy.should_early_stop_joint(early_cfg, 20, 2)
    assert legacy.should_early_stop_joint(early_cfg, 20, 3)
    early_cfg.training_stage = "manager_train"
    assert not legacy.should_early_stop_joint(early_cfg, 50, 10)


def _test_stage_evaluation_and_runtime_restore(device: torch.device) -> None:
    expected_calls = {
        "fast_pretrain": (0, 0, 3),
        "slow_pretrain": (0, 1, 1),
        "manager_train": (1, 1, 1),
        "joint_finetune": (1, 1, 1),
    }
    expected_trainability = {
        "fast_pretrain": (False, False, True),
        "slow_pretrain": (False, True, False),
        "manager_train": (True, False, False),
        "joint_finetune": (True, True, True),
    }
    for stage, expected in expected_calls.items():
        cfg = TrainConfig(
            training_stage=stage, episode_steps=40, eval_seeds=(100,),
            hidden_dim=16, manager_hidden_dim=16, critic_hidden_dim=16,
            manager_latent_dim=8, slow_latent_dim=8, fast_latent_dim=8,
        )
        env = ElectricGasMultiScaleEnv()
        agents = build_agents(env, cfg, device)
        global_obs, _ = env.reset(seed=1)
        builder = ObservationBuilder(env, cfg.manager_interval)
        goal = fixed_manager_goal()
        agent_tuple = (agents.manager, agents.slow, agents.fast)
        enabled_tuple = expected_trainability[stage]
        for agent, enabled in zip(agent_tuple, enabled_tuple):
            assert any(parameter.requires_grad for parameter in agent.actor.parameters()) is enabled
            assert agent.normalizer.training is enabled
            for target_name in ("target_encoder", "target_actor", "target_critic"):
                target_module = getattr(agent, target_name)
                assert not target_module.training
                assert all(not parameter.requires_grad for parameter in target_module.parameters())
                assert all(parameter.grad is None for parameter in target_module.parameters())
        if stage == "fast_pretrain":
            target_before = [parameter.detach().clone() for parameter in agents.fast.target_actor.parameters()]
            with torch.no_grad():
                next(agents.fast.actor.parameters()).add_(0.01)
            legacy.soft_update(agents.fast.target_actor, agents.fast.actor, 0.5)
            assert any(not torch.allclose(before, after) for before, after in
                       zip(target_before, agents.fast.target_actor.parameters()))
            assert all(not parameter.requires_grad and parameter.grad is None
                       for parameter in agents.fast.target_actor.parameters())
        before_counts = tuple(agent.normalizer.count for agent in agent_tuple)
        agents.manager.select_goal(builder.manager_obs(global_obs), None, 0.5, deterministic=False)
        agents.slow.select_action(builder.slow_obs(global_obs), goal, 0.5, deterministic=False)
        agents.fast.select_action(builder.fast_obs(0, global_obs), goal, 0.5, deterministic=False)
        after_counts = tuple(agent.normalizer.count for agent in agent_tuple)
        for before, after, enabled in zip(before_counts, after_counts, enabled_tuple):
            assert (after > before) if enabled else (after == before)
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
        assert _resolve_eval_seeds(cfg, episodes, 777)[0] == 100
        assert tuple(stats["resolved_eval_seeds"]) == _resolve_eval_seeds(cfg, episodes, 777)
        assert normalizer_counts == tuple(
            agent.normalizer.count for agent in (agents.manager, agents.slow, agents.fast)
        )
        assert multiplier >= 1

    fixed_cfg = TrainConfig(eval_seed_mode="fixed", eval_seeds=(11, 22), eval_episodes=4)
    assert _resolve_eval_seeds(fixed_cfg, 4, 100) == (11, 22, 23, 24)
    assert _resolve_eval_seeds(fixed_cfg, 4, 999) == (11, 22, 23, 24)
    offset_cfg = TrainConfig(eval_seed_mode="offset", eval_seeds=(11, 22), eval_episodes=3)
    assert _resolve_eval_seeds(offset_cfg, 3, 100) == (100, 111, 101)
    assert _resolve_eval_seeds(offset_cfg, 3, 200) == (200, 211, 201)
    seed_env = ElectricGasMultiScaleEnv()
    seed_agents = build_agents(seed_env, fixed_cfg, device)
    fixed_a = evaluate_policy(seed_agents, fixed_cfg, episodes=2, max_steps=1, seed=100)
    fixed_b = evaluate_policy(seed_agents, fixed_cfg, episodes=2, max_steps=1, seed=999)
    assert fixed_a["resolved_eval_seeds"] == fixed_b["resolved_eval_seeds"] == [11, 22]
    offset_agents = build_agents(seed_env, offset_cfg, device)
    offset_a = evaluate_policy(offset_agents, offset_cfg, episodes=2, max_steps=1, seed=100)
    offset_b = evaluate_policy(offset_agents, offset_cfg, episodes=2, max_steps=1, seed=200)
    assert offset_a["resolved_eval_seeds"] == [100, 111]
    assert offset_b["resolved_eval_seeds"] == [200, 211]

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
    original_api_version = legacy.LEGACY_ALGORITHM_API_VERSION
    try:
        legacy.LEGACY_ALGORITHM_API_VERSION = original_api_version - 1
        try:
            validate_legacy_api()
        except RuntimeError as exc:
            assert "LEGACY_ALGORITHM_API_VERSION" in str(exc)
        else:
            raise AssertionError("legacy API version mismatch must fail at startup")
    finally:
        legacy.LEGACY_ALGORITHM_API_VERSION = original_api_version
    original_capture_rng = legacy.capture_rng_state
    try:
        legacy.capture_rng_state = None  # type: ignore[assignment]
        try:
            validate_legacy_api()
        except RuntimeError as exc:
            assert "capture_rng_state" in str(exc)
        else:
            raise AssertionError("missing RNG API must fail during startup validation")
    finally:
        legacy.capture_rng_state = original_capture_rng


def _test_short_training() -> None:
    common = dict(
        device="cpu", run_mode="debug", hidden_dim=32, manager_hidden_dim=32, critic_hidden_dim=32,
        manager_latent_dim=8, slow_latent_dim=8, fast_latent_dim=8,
        fast_batch_size=4, fast_learning_starts=4, fast_updates_per_step=1,
        slow_batch_size=1, slow_learning_starts=1, slow_updates_per_boundary=2,
        manager_batch_size=1, manager_learning_starts=1,
        eval_interval=1, eval_episodes=1, eval_seeds=(54_321,), no_tensorboard=True,
        fast_buffer_size=4, slow_buffer_size=1, manager_buffer_size=1,
        fast_random_warmup_steps=0, slow_random_warmup_segments=0,
        manager_random_warmup_segments=0,
    )
    fast_cfg = TrainConfig(
        episodes=1, episode_steps=20, training_stage="fast_pretrain", device="cpu",
        checkpoint_dir="hierarchical_td3_test_runs/optimized_fast_short",
        **{key: value for key, value in common.items() if key != "device"},
    )
    fast_result = run_training(fast_cfg)
    assert fast_result["global_step"] == 20 and np.isfinite(fast_result["best_eval_return"])
    assert fast_result["fast_buffer_size"] == 4
    assert fast_result["fast_parameter_change_l2"] > 0.0
    assert fast_result["slow_parameter_change_l2"] == 0.0
    assert fast_result["manager_parameter_change_l2"] == 0.0
    assert (Path(fast_result["run_root"]) / "best_fast.pt").exists()

    checkpoint_path = "hierarchical_td3_test_runs/optimized_checkpoint_test.pt"
    resume_cfg = TrainConfig(
        episodes=0, episode_steps=20, training_stage="joint_finetune",
        checkpoint_dir="hierarchical_td3_test_runs/optimized_resume_short",
        load_checkpoint=checkpoint_path, checkpoint_load_mode="resume", **common,
    )
    resume_result = run_training(resume_cfg)
    assert resume_result["best_eval_return"] == 999_999.0
    assert resume_result["global_step"] == 123

    slow_cfg = TrainConfig(
        episodes=1, episode_steps=40, training_stage="slow_pretrain",
        checkpoint_dir="hierarchical_td3_test_runs/optimized_slow_short",
        load_checkpoint=checkpoint_path, checkpoint_load_mode="stage_transfer", **common,
    )
    slow_result = run_training(slow_cfg)
    assert slow_result["global_step"] == 40 and np.isfinite(slow_result["best_eval_return"])
    assert slow_result["slow_buffer_size"] == 1
    assert slow_result["slow_parameter_change_l2"] > 0.0
    assert slow_result["fast_parameter_change_l2"] == 0.0
    assert slow_result["manager_parameter_change_l2"] == 0.0
    assert slow_result["best_eval_return"] != 999_999.0
    assert (Path(slow_result["run_root"]) / "best_slow.pt").exists()

    manager_cfg = TrainConfig(
        episodes=1, episode_steps=40, training_stage="manager_train",
        checkpoint_dir="hierarchical_td3_test_runs/optimized_manager_short",
        load_checkpoint=checkpoint_path, checkpoint_load_mode="stage_transfer", **common,
    )
    manager_result = run_training(manager_cfg)
    assert manager_result["manager_critic_update_count"] > 0
    assert manager_result["manager_actor_update_count"] > 0
    assert manager_result["manager_parameter_change_l2"] > 0.0
    assert manager_result["slow_parameter_change_l2"] == 0.0
    assert manager_result["fast_parameter_change_l2"] == 0.0
    assert (Path(manager_result["run_root"]) / "best_manager.pt").exists()

    joint_cfg = TrainConfig(
        episodes=1, episode_steps=40, training_stage="joint_finetune",
        checkpoint_dir="hierarchical_td3_test_runs/optimized_joint_short",
        load_checkpoint=checkpoint_path, checkpoint_load_mode="stage_transfer", **common,
    )
    # This test verifies that every trainable layer can update in one short
    # episode.  The production joint schedule deliberately freezes the Manager
    # actor at first, so disable that curriculum only for this update test.
    joint_cfg.joint_freeze_manager_actor_episodes = 0
    joint_result = run_training(joint_cfg)
    for role in ("fast", "slow", "manager"):
        assert joint_result[f"{role}_critic_update_count"] > 0
        assert joint_result[f"{role}_actor_update_count"] > 0
        assert joint_result[f"{role}_parameter_change_l2"] > 0.0
    assert _ACTIVE_CONFIG is None and _CURRENT_SLOW_PENDING is None


def _test_time_limit_fragment_finalization() -> None:
    global _LEGACY_EVALUATE_POLICY
    original_evaluate = _LEGACY_EVALUATE_POLICY

    def fast_evaluate(agents: AgentBundle, cfg: TrainConfig, episodes: int = 1,
                      max_steps: int = EPISODE_STEPS, seed: int = 0) -> Dict[str, Any]:
        del agents, cfg, episodes, seed
        stats: Dict[str, Any] = _synthetic_evaluation_stats(-1.0, 0.0)
        stats.update({"std_return": 0.0, "steps": float(max_steps)})
        return stats

    _LEGACY_EVALUATE_POLICY = fast_evaluate
    try:
        for steps in (7, 20, 33, 40, 47):
            cfg = TrainConfig(
                episodes=1, episode_steps=steps, training_stage="fast_pretrain", device="cpu",
                run_mode="debug",
                hidden_dim=16, manager_hidden_dim=16, critic_hidden_dim=16,
                manager_latent_dim=8, slow_latent_dim=8, fast_latent_dim=8,
                fast_buffer_size=64, slow_buffer_size=4, manager_buffer_size=2,
                fast_batch_size=1, slow_batch_size=1, manager_batch_size=1,
                fast_learning_starts=1, slow_learning_starts=1, manager_learning_starts=1,
                fast_random_warmup_steps=0, slow_random_warmup_segments=0,
                manager_random_warmup_segments=0,
                manager_use_prioritized_replay=True,
                eval_interval=1, eval_episodes=1, eval_seeds=(123,), no_tensorboard=True,
                checkpoint_dir=f"hierarchical_td3_test_runs/time_limit_{steps}",
            )
            result = run_training(cfg)
            expected_slow = steps % cfg.slow_interval or cfg.slow_interval
            expected_manager = steps % cfg.manager_interval or cfg.manager_interval
            assert result["fast_last_duration"] == 1.0 and result["fast_last_done"] == 1.0
            assert result["slow_last_duration"] == float(expected_slow)
            assert result["manager_last_duration"] == float(expected_manager)
            assert result["slow_last_done"] == 1.0 and result["manager_last_done"] == 1.0
            assert np.all(np.isfinite(result["slow_last_discounted_mean_executed_action"]))
            assert np.all(np.isfinite(result["slow_last_executed_action_variance"]))
            assert np.isfinite(result["manager_last_constraint_mean"])
            assert np.isfinite(result["manager_last_constraint_max"])
            assert result["manager_last_priority"] >= cfg.priority_epsilon
    finally:
        _LEGACY_EVALUATE_POLICY = original_evaluate


def run_minimum_tests() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("Running optimized SMDP-TD3 tests...")
    device = torch.device("cpu")
    legacy.set_seed(42)
    _test_smdp_discount()
    _test_reward_scale_calibration()
    _test_worker_action_regularization_reference()
    _test_manager_goal_semantics(device)
    _test_pure_slow_action_inverse_mapping()
    _test_observation_layout()
    _test_formal_safety_config()
    _test_real_slow_safety_contract(device)
    _test_feasibility_thresholds()
    _test_debug_terminal_soc_penalty()
    _test_slow_executed_action_summary()
    _test_projection_schedule_and_mask()
    _test_physical_features()
    _test_time_scale_validation()
    _test_cli_contract()
    _test_constraint_score_robustness()
    _test_per_initialization_and_sampling(device)
    _test_mixed_per_monte_carlo(device)
    _test_raw_critic_executed_transition_and_per(device)
    _test_independent_parameters_and_slow_duration(device)
    _test_nonfinite_update_transaction(device)
    _test_replay_wrap_boundary_updates(device)
    _test_checkpoint_and_environment(device)
    _test_strict_resume_replay_rng_and_stage(device)
    _test_safe_best_model_metric()
    _test_stage_evaluation_and_runtime_restore(device)
    _test_short_training()
    _test_time_limit_fragment_finalization()
    LOGGER.info("All optimized SMDP-TD3 tests passed.")


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv: Optional[Sequence[str]] = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description="Projection-aware physical hierarchical SMDP-TD3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=100,
                        help="Episodes for a single explicit stage; --training-stage all uses per-stage counts")
    parser.add_argument("--episode-steps", type=int, default=EPISODE_STEPS)
    parser.add_argument("--manager-interval", type=int, default=MANAGER_INTERVAL)
    parser.add_argument("--slow-interval", type=int, default=SLOW_INTERVAL)
    parser.add_argument("--training-stage", default="all",
                        choices=("fast_pretrain", "slow_pretrain", "manager_train", "joint_finetune", "all"))
    parser.add_argument("--run-mode", choices=("formal", "debug"), default="formal")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gamma-fast", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--target-noise", type=float, default=0.10)
    parser.add_argument("--target-noise-clip", type=float, default=0.30)
    parser.add_argument("--target-q-clip-abs", type=float, default=20_000.0)
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
    parser.add_argument("--joint-worker-lr", type=float, default=5e-5)
    parser.add_argument("--joint-manager-lr", type=float, default=1e-5)
    parser.add_argument("--joint-policy-frequency", type=int, default=6)
    parser.add_argument("--fast-buffer-size", type=int, default=200_000)
    parser.add_argument("--slow-buffer-size", type=int, default=50_000)
    parser.add_argument("--manager-buffer-size", type=int, default=10_000)
    parser.add_argument("--projection-imitation-initial-weight", type=float, default=0.10)
    parser.add_argument("--projection-imitation-final-weight", type=float, default=0.02)
    parser.add_argument("--projection-imitation-decay-updates", type=int, default=100_000)
    parser.add_argument("--projection-imitation-threshold", type=float, default=1e-3)
    parser.add_argument("--projection-behavior-match-threshold", type=float, default=0.05)
    parser.add_argument("--slow-guard-imitation-multiplier", type=float, default=5.0)
    parser.add_argument("--projection-imitation-weight", type=float)
    parser.add_argument("--disable-prioritized-replay", action="store_true")
    parser.add_argument("--priority-alpha", type=float, default=0.6)
    parser.add_argument("--priority-beta-initial", type=float, default=0.4)
    parser.add_argument("--priority-beta-final", type=float, default=1.0)
    parser.add_argument("--priority-beta-anneal-steps", type=int, default=200_000)
    parser.add_argument("--fast-priority-beta-anneal-updates", type=int, default=200_000)
    parser.add_argument("--slow-priority-beta-anneal-updates", type=int, default=50_000)
    parser.add_argument("--manager-priority-beta-anneal-updates", type=int, default=25_000)
    parser.add_argument("--constraint-priority-weight", type=float, default=1.0)
    parser.add_argument("--projection-priority-weight", type=float, default=1.0)
    parser.add_argument("--max-td-error-for-priority", type=float, default=1e6)
    parser.add_argument("--max-replay-priority", type=float, default=1e6)
    parser.add_argument("--prioritized-sample-fraction", type=float, default=0.75)
    parser.add_argument("--manager-use-prioritized-replay", action="store_true")
    parser.add_argument("--manager-constraint-priority-weight", type=float, default=1.0)
    parser.add_argument("--manager-solver-priority-weight", type=float, default=2.0)
    parser.add_argument("--priority-component-normalization",
                        choices=("none", "running_scale", "rank"), default="running_scale")
    parser.add_argument("--slow-shaping-duration-mode", choices=("terminal", "normalized"), default="terminal")
    parser.add_argument("--fast-pretrain-episodes", type=int, default=50)
    parser.add_argument("--slow-pretrain-episodes", type=int, default=50)
    parser.add_argument("--manager-train-episodes", type=int, default=50)
    parser.add_argument("--joint-finetune-episodes", type=int, default=30)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[20_026, 20_027, 20_028])
    parser.add_argument("--eval-seed-mode", choices=("fixed", "offset"), default="fixed")
    parser.add_argument("--checkpoint-dir", default="hierarchical_td3_optimized_runs")
    parser.add_argument("--load-checkpoint", default="")
    parser.add_argument("--checkpoint-load-mode", choices=("resume", "stage_transfer", "policy_only"),
                        default="resume")
    parser.add_argument("--load-policy-only", action="store_true")
    parser.add_argument("--best-model-metric", choices=("feasible_then_return", "return"),
                        default="feasible_then_return")
    parser.add_argument("--min-power-success-rate", type=float, default=0.999)
    parser.add_argument("--min-gas-success-rate", type=float, default=0.999)
    parser.add_argument("--max-soc-violation-rate", type=float, default=0.0)
    parser.add_argument("--max-voltage-rms-deviation-pu", type=float, default=0.05)
    parser.add_argument("--max-gas-pressure-rms-deviation-bar", type=float, default=0.5)
    parser.add_argument("--metric-comparison-tolerance", type=float, default=1e-6)
    parser.add_argument("--max-hard-constraint-violation-rate", type=float, default=0.0)
    parser.add_argument("--disable-replay-checkpoint", action="store_true")
    parser.add_argument("--full-resume-checkpoint-interval", type=int, default=5)
    parser.add_argument("--allow-partial-resume", action="store_true")
    parser.add_argument("--nonfinite-update-policy", choices=("raise", "skip_batch"), default="raise")
    parser.add_argument("--unexpected-env-exception-policy", choices=("raise", "truncate"), default="raise")
    parser.add_argument("--fast-random-warmup-steps", type=int, default=2_000)
    parser.add_argument("--slow-random-warmup-segments", type=int, default=128)
    parser.add_argument("--manager-random-warmup-segments", type=int, default=64)
    parser.add_argument("--warmup-blend-fraction", type=float, default=0.20)
    parser.add_argument("--strict-stage-sample-validation", action="store_true", default=True)
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
    parser.add_argument("--lambda-projection", type=float, default=1.0)
    parser.add_argument("--worker-reward-clip-abs", type=float, default=1_000.0)
    parser.add_argument("--manager-reward-clip-abs", type=float, default=2_000.0)
    parser.add_argument("--worker-action-l2-weight", type=float, default=0.002,
                        help="Reference-action regularization weight; zero is not a universal physical neutral action")
    parser.add_argument("--worker-component-clip-abs", type=float, default=50.0,
                        help="Per-component cap after reference normalization")
    parser.add_argument("--reward-component-transform", choices=("none", "log1p_reference"),
                        default="log1p_reference")
    parser.add_argument("--reward-scale-profile", default="safety_calibrated_20260716_v2")
    parser.add_argument("--slow-role-specific-reward-scale", type=float, default=25.0)
    parser.add_argument("--shaping-reference-floor", type=float, default=1.0)
    parser.add_argument("--auxiliary-loss-scale-max", type=float, default=1_000_000.0)
    parser.add_argument("--auxiliary-loss-coefficient-max", type=float, default=0.05)
    auxiliary_group = parser.add_mutually_exclusive_group()
    auxiliary_group.add_argument(
        "--enable-adaptive-auxiliary-loss-scaling",
        dest="adaptive_auxiliary_loss_scaling",
        action="store_true",
    )
    auxiliary_group.add_argument(
        "--disable-adaptive-auxiliary-loss-scaling",
        dest="adaptive_auxiliary_loss_scaling",
        action="store_false",
    )
    parser.set_defaults(adaptive_auxiliary_loss_scaling=False)
    parser.add_argument("--health-actor-collapse-warning-abs-mean", type=float, default=0.03)
    parser.add_argument("--joint-freeze-manager-actor-episodes", type=int, default=10)
    parser.add_argument("--joint-early-stop-patience-evaluations", type=int, default=3)
    parser.add_argument("--joint-early-stop-min-episodes", type=int, default=20)
    parser.add_argument("--disable-joint-stage-best-inheritance", action="store_true")
    parser.add_argument("--fast-global-safety-weight", type=float, default=0.50)
    parser.add_argument("--fast-role-specific-weight", type=float, default=0.50)
    parser.add_argument("--slow-global-safety-weight", type=float, default=0.50)
    parser.add_argument("--slow-role-specific-weight", type=float, default=0.50)
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
    if args.load_policy_only:
        warnings.warn("--load-policy-only is deprecated; using --checkpoint-load-mode policy_only.",
                      FutureWarning, stacklevel=2)
        args.checkpoint_load_mode = "policy_only"

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
        training_stage=args.training_stage, run_mode=args.run_mode, device=args.device, gamma_fast=args.gamma_fast,
        tau=args.tau, target_noise=args.target_noise, target_noise_clip=args.target_noise_clip,
        target_q_clip_abs=args.target_q_clip_abs, gradient_clip=args.gradient_clip,
        fast_batch_size=fast_batch, slow_batch_size=slow_batch, manager_batch_size=manager_batch,
        fast_learning_starts=fast_starts, slow_learning_starts=slow_starts,
        manager_learning_starts=manager_starts, fast_updates_per_step=fast_updates,
        slow_updates_per_boundary=slow_updates, manager_updates_per_boundary=manager_updates,
        fast_lr=args.fast_lr, slow_lr=args.slow_lr, manager_lr=args.manager_lr,
        joint_worker_lr=args.joint_worker_lr, joint_manager_lr=args.joint_manager_lr,
        joint_policy_frequency=args.joint_policy_frequency,
        fast_buffer_size=args.fast_buffer_size, slow_buffer_size=args.slow_buffer_size,
        manager_buffer_size=args.manager_buffer_size,
        projection_imitation_initial_weight=imitation_initial,
        projection_imitation_final_weight=imitation_final,
        projection_imitation_decay_updates=imitation_decay,
        projection_imitation_threshold=args.projection_imitation_threshold,
        projection_behavior_match_threshold=args.projection_behavior_match_threshold,
        slow_guard_imitation_multiplier=args.slow_guard_imitation_multiplier,
        use_prioritized_replay=not args.disable_prioritized_replay,
        priority_alpha=args.priority_alpha, priority_beta_initial=args.priority_beta_initial,
        priority_beta_final=args.priority_beta_final,
        priority_beta_anneal_steps=args.priority_beta_anneal_steps,
        fast_priority_beta_anneal_updates=args.fast_priority_beta_anneal_updates,
        slow_priority_beta_anneal_updates=args.slow_priority_beta_anneal_updates,
        manager_priority_beta_anneal_updates=args.manager_priority_beta_anneal_updates,
        constraint_priority_weight=args.constraint_priority_weight,
        projection_priority_weight=args.projection_priority_weight,
        max_td_error_for_priority=args.max_td_error_for_priority,
        max_replay_priority=args.max_replay_priority,
        prioritized_sample_fraction=args.prioritized_sample_fraction,
        manager_use_prioritized_replay=args.manager_use_prioritized_replay,
        manager_constraint_priority_weight=args.manager_constraint_priority_weight,
        manager_solver_priority_weight=args.manager_solver_priority_weight,
        priority_component_normalization=args.priority_component_normalization,
        slow_shaping_duration_mode=args.slow_shaping_duration_mode,
        fast_pretrain_episodes=args.fast_pretrain_episodes,
        slow_pretrain_episodes=args.slow_pretrain_episodes,
        manager_train_episodes=args.manager_train_episodes,
        joint_finetune_episodes=args.joint_finetune_episodes,
        eval_interval=args.eval_interval, eval_episodes=args.eval_episodes,
        eval_seeds=tuple(args.eval_seeds), eval_seed_mode=args.eval_seed_mode,
        checkpoint_dir=args.checkpoint_dir, load_checkpoint=args.load_checkpoint,
        checkpoint_load_mode=args.checkpoint_load_mode,
        load_policy_only=args.checkpoint_load_mode == "policy_only",
        best_model_metric=args.best_model_metric,
        min_power_success_rate=args.min_power_success_rate,
        min_gas_success_rate=args.min_gas_success_rate,
        max_soc_violation_rate=args.max_soc_violation_rate,
        max_voltage_rms_deviation_pu=args.max_voltage_rms_deviation_pu,
        max_gas_pressure_rms_deviation_bar=args.max_gas_pressure_rms_deviation_bar,
        metric_comparison_tolerance=args.metric_comparison_tolerance,
        max_hard_constraint_violation_rate=args.max_hard_constraint_violation_rate,
        save_replay_in_checkpoint=not args.disable_replay_checkpoint,
        full_resume_checkpoint_interval=args.full_resume_checkpoint_interval,
        strict_resume_required=not args.allow_partial_resume,
        nonfinite_update_policy=args.nonfinite_update_policy,
        unexpected_env_exception_policy=args.unexpected_env_exception_policy,
        fast_random_warmup_steps=args.fast_random_warmup_steps,
        slow_random_warmup_segments=args.slow_random_warmup_segments,
        manager_random_warmup_segments=args.manager_random_warmup_segments,
        warmup_blend_fraction=args.warmup_blend_fraction,
        strict_stage_sample_validation=args.strict_stage_sample_validation,
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
        worker_component_clip_abs=args.worker_component_clip_abs,
        reward_component_transform=args.reward_component_transform,
        reward_scale_profile=args.reward_scale_profile,
        slow_role_specific_reward_scale=args.slow_role_specific_reward_scale,
        shaping_reference_floor=args.shaping_reference_floor,
        adaptive_auxiliary_loss_scaling=args.adaptive_auxiliary_loss_scaling,
        auxiliary_loss_scale_max=args.auxiliary_loss_scale_max,
        auxiliary_loss_coefficient_max=args.auxiliary_loss_coefficient_max,
        fast_global_safety_weight=args.fast_global_safety_weight,
        fast_role_specific_weight=args.fast_role_specific_weight,
        slow_global_safety_weight=args.slow_global_safety_weight,
        slow_role_specific_weight=args.slow_role_specific_weight,
        health_actor_collapse_warning_abs_mean=args.health_actor_collapse_warning_abs_mean,
        joint_freeze_manager_actor_episodes=args.joint_freeze_manager_actor_episodes,
        joint_early_stop_patience_evaluations=args.joint_early_stop_patience_evaluations,
        joint_early_stop_min_episodes=args.joint_early_stop_min_episodes,
        inherit_stage_best_on_joint_transfer=not args.disable_joint_stage_best_inheritance,
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
