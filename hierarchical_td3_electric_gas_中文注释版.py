"""分层 TD3 的基础实现与兼容模块（中文注释版）。

【文件在项目中的作用】
本文件集中给出 Manager、Slow Worker、Fast Worker 的基础网络、Replay Buffer、
观测构造、奖励组合、checkpoint、评估以及多时间尺度训练主循环。它也是优化版
``hierarchical_td3_electric_gas_optimized.py`` 复用的兼容层；优化版通过受控的
runtime override 替换 PER Replay、物理目标、投影感知 transition 和严格恢复逻辑。

【与环境文件的关系】
``electric_gas_microgrid_single.py`` 定义电网/气网拓扑、设备参数、状态、动作映射、
安全投影和物理求解。本文件只消费环境接口，不复制或改变物理模型。一个快速步为
3 分钟，Slow 每 20 步决策一次，Manager 每 40 步给出一次 goal，正式 episode 为
480 步（24 小时）。

【正式训练入口】
本文件用于理解基础 TD3 和提供兼容 API；正式训练请运行
``hierarchical_td3_electric_gas_optimized.py``。基础文件的正式/调试训练入口按项目
契约被禁用，避免误用旧实现。

【初学者推荐阅读顺序】
TrainConfig → ObservationBuilder → Actor/Critic → Replay Buffer → ManagerTD3/
WorkerTD3 → Pending* → run_training → evaluate_policy/checkpoint。

【基础 TD3 与项目扩展】
Actor、双 Critic、target policy smoothing、delayed policy update、soft update 属于
TD3 基础；Manager goal、20/40 步 SMDP 片段、raw/guarded/executed 动作、安全投影、
终端 SOC 和四阶段训练属于本项目的多时间尺度安全控制扩展。

本注释版只增加中文说明和 docstring，不改变任何默认参数、公式、维度或控制流。
"""

from __future__ import annotations

# =============================================================================
# 中文详注版说明：仅增加注释，不修改任何可执行语句、变量名、默认参数或控制流。
# 阅读时优先关注时间尺度边界、raw/executed 动作、聚合折扣和目标网络更新。
# =============================================================================


import argparse
import copy
import csv
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - tensorboard is optional for minimal installs
    class SummaryWriter:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def add_scalar(self, *args: Any, **kwargs: Any) -> None:
            pass

        def add_text(self, *args: Any, **kwargs: Any) -> None:
            pass

        def close(self) -> None:
            pass


from electric_gas_microgrid_single import (
    COMPRESSOR_CONFIGS,
    CONTROLLED_COMPRESSOR_INDICES,
    ESS_CONFIGS,
    ENV_MODEL_VERSION,
    GAS_SUPPLIERS,
    GFG_CONFIGS,
    P2G_CONFIGS,
    SLOW_SAFETY_SCHEMA_VERSION,
    ElectricGasMultiScaleEnv,
)


LOGGER = logging.getLogger("hierarchical_td3_electric_gas")

FAST_INTERVAL = 1
SLOW_INTERVAL = 20
MANAGER_INTERVAL = 40
EPISODE_STEPS = 480

GOAL_SHARED_DIM = 8
GOAL_SLOW_DIM = 8
GOAL_FAST_DIM = 8
GOAL_PHYSICAL_DIM = 8
GOAL_DIM = GOAL_SHARED_DIM + GOAL_SLOW_DIM + GOAL_FAST_DIM + GOAL_PHYSICAL_DIM
LEGACY_ALGORITHM_API_VERSION = 5


# =============================================================================
# Configuration
# =============================================================================
#
# 【模块说明：配置】输入是实验与算法超参数，输出是统一的 TrainConfig。
# 它决定时间尺度、网络宽度、学习率、Replay 门槛、探索、奖励和 checkpoint 契约。
# 初学者先区分 gamma_fast 与由指数换算出的 gamma_slow/gamma_manager。
#


# 【中文导读】集中保存三层时间尺度、TD3、奖励塑形、探索、日志和阶段训练参数；gamma_slow/gamma_manager由快速步折扣指数换算。
@dataclass
class TrainConfig:
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：集中保存三层 SMDP-TD3 的全部训练契约。
    
    输入：构造时传入可覆盖的超参数。
    
    输出：不可变维度/训练设置的配置对象。
    
    核心步骤：校验时间尺度、学习率、Replay、探索、奖励、评估和恢复参数。
    
    强化学习含义：配置决定采样量是否足够产生梯度更新。
    
    【容易混淆】gamma_slow/manager 是 gamma_fast 的指数结果，不是片段奖励的第二次折扣。
    """

    # 实验基础设置：随机种子、训练轮数、每轮步数和运行设备。
    seed: int = 42
    episodes: int = 100
    episode_steps: int = EPISODE_STEPS
    manager_interval: int = MANAGER_INTERVAL
    slow_interval: int = SLOW_INTERVAL
    training_stage: str = "all"
    run_mode: str = "formal"
    exploration_episode_offset: int = 0
    device: str = "auto"

    # 神经网络大小。latent_dim 是 Encoder 压缩后的隐藏状态维度。
    manager_latent_dim: int = 64
    slow_latent_dim: int = 32
    fast_latent_dim: int = 32
    hidden_dim: int = 256
    manager_hidden_dim: int = 256
    critic_hidden_dim: int = 256

    # TD3 核心参数：折扣因子、软更新系数、批大小、目标策略平滑等。
    gamma_fast: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 1000
    policy_frequency: int = 2
    slow_update_interval_steps: int = 5
    manager_update_interval_steps: int = 20
    target_noise: float = 0.10
    target_noise_clip: float = 0.30
    target_q_clip_abs: float = 20_000.0
    gradient_clip: float = 1.0

    # 学习率。joint_finetune 阶段通常更保守，所以 Worker 可使用单独学习率。
    fast_lr: float = 3e-4
    slow_lr: float = 3e-4
    manager_lr: float = 1e-4
    joint_worker_lr: float = 5e-5

    # 三类 Replay Buffer 容量。时间尺度越慢，能产生的样本越少，所以容量也更小。
    fast_buffer_size: int = 200_000
    slow_buffer_size: int = 50_000
    manager_buffer_size: int = 10_000

    # 探索噪声。训练初期动作更随机，随后逐步衰减到最小噪声。
    fast_exploration_noise: float = 0.15
    slow_exploration_noise: float = 0.10
    manager_exploration_noise: float = 0.05
    min_fast_exploration_noise: float = 0.02
    min_slow_exploration_noise: float = 0.02
    min_manager_exploration_noise: float = 0.01
    noise_decay_episodes: int = 200
    fast_random_warmup_steps: int = 2_000
    slow_random_warmup_segments: int = 128
    manager_random_warmup_segments: int = 64
    warmup_blend_fraction: float = 0.20
    goal_smoothing: float = 0.20
    goal_change_penalty_weight: float = 0.05

    # Worker 奖励由外在奖励、隐空间方向奖励、物理进展和安全投影惩罚组成。
    alpha_external: float = 1.0
    # beta_* are relative shaping fractions after calibration against |external reward|.
    beta_latent: float = 0.05
    beta_physical: float = 0.10
    shaping_reference_floor: float = 1.0
    lambda_projection: float = 5.0
    worker_reward_clip_abs: float = 1_000.0
    manager_reward_clip_abs: float = 2_000.0
    worker_action_l2_weight: float = 0.02
    reward_component_transform: str = "log1p_reference"
    reward_scale_profile: str = "safety_calibrated_20260714_v1"
    # This cap is applied after reference normalization, never to raw physical cost.
    worker_component_clip_abs: float = 50.0
    projection_imitation_weight: float = 0.0
    use_ess_action_guard: bool = True
    delta_z_min: float = 1e-4

    # 可选 Transition Model：让 Worker 的隐空间更容易预测“动作导致的变化”。
    lambda_transition: float = 0.10
    lambda_latent_norm: float = 1e-4
    reachability_weight: float = 0.00
    use_transition_model: bool = False

    # 训练控制、评估和持久化。
    updates_per_step: int = 1
    eval_interval: int = 5
    eval_episodes: int = 1
    checkpoint_dir: str = "hierarchical_td3_runs"
    load_checkpoint: str = ""
    load_policy_only: bool = False
    checkpoint_load_mode: str = "resume"
    eval_seed_mode: str = "fixed"
    best_model_metric: str = "feasible_then_return"
    save_replay_in_checkpoint: bool = True
    bootstrap_on_time_limit: bool = False
    full_resume_checkpoint_interval: int = 5
    strict_resume_required: bool = True
    no_tensorboard: bool = False
    run_tests: bool = False

    @property
    def gamma_slow(self) -> float:
        return self.gamma_fast ** self.slow_interval

    @property
    def gamma_manager(self) -> float:
        return self.gamma_fast ** self.manager_interval


# =============================================================================
# Utilities
# =============================================================================
#
# 【模块说明：通用工具】提供随机数状态、设备选择、张量转换、目标网络更新、
# goal 归一化和观测归一化。soft_update 是 TD3 稳定训练的核心工具；RNG 保存则保证
# strict resume 后采样与探索能够确定性延续。
#


# 【中文导读】统一固定 Python、NumPy 和 PyTorch 随机源，便于复现实验。
def set_seed(seed: int) -> None:
    """固定随机种子，减少重复实验之间的随机差异。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    required = ("python", "numpy", "torch")
    missing = [name for name in required if name not in state]
    if missing:
        raise ValueError(f"rng_state is missing required entries: {missing}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8).cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        cuda_states = [torch.as_tensor(item, dtype=torch.uint8).cpu() for item in state["torch_cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)


# 【中文导读】把 auto/cpu/cuda 配置解析为 PyTorch 设备。
def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def trusted_torch_load(path: str, map_location: torch.device) -> Dict[str, Any]:
    # weights_only=False is safe here only for trusted checkpoints generated by this program.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# 【中文导读】完整复制在线网络到目标网络，通常仅在初始化或只加载策略时使用。
def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


# 【中文导读】执行 Polyak 平均，使目标网络缓慢跟随在线网络以稳定 TD 目标。
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


# 【中文导读】限制梯度范数，防止大尺度奖励造成梯度爆炸。
def clip_grad(parameters: Any, max_norm: float) -> None:
    nn.utils.clip_grad_norm_(list(parameters), max_norm)


# 【中文导读】临时冻结或解冻模块，控制辅助损失的梯度路径。
def set_requires_grad(module: Optional[nn.Module], enabled: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(enabled)


# 【中文导读】checkpoint 跨设备加载后迁移优化器动量状态。
def move_optimizer_state(optimizer: optim.Optimizer, device: torch.device) -> None:
    """加载checkpoint后把Adam等优化器内部状态移动到当前训练设备。"""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


# 【中文导读】统一把 NumPy float 数组放到训练设备。
def to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


# 【中文导读】将三段方向 goal 分别单位化，并将物理参考压到 [-1,1]。
def normalize_goal_tensor(goal: torch.Tensor) -> torch.Tensor:
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：把 goal 的各方向块归一化。
    
    输入：goal:[B,32]。
    
    输出：块结构保持不变的归一化 goal。
    
    核心步骤：分别处理共享/Slow/Fast 方向和物理目标。
    
    强化学习含义：限制高层动作尺度并保持方向语义。
    
    【容易混淆】这是 raw goal 的确定性变换，不是探索噪声。
    
    【张量形状】goal:[B,32]
    """
    shared = F.normalize(goal[..., 0:8], p=2, dim=-1, eps=1e-8)
    slow = F.normalize(goal[..., 8:16], p=2, dim=-1, eps=1e-8)
    fast = F.normalize(goal[..., 16:24], p=2, dim=-1, eps=1e-8)
    physical = torch.tanh(goal[..., 24:32])
    return torch.cat([shared, slow, fast, physical], dim=-1)


# 【中文导读】交互阶段的 NumPy goal 归一化，与网络端规则保持一致。
def normalize_goal_np(goal: np.ndarray) -> np.ndarray:
    """NumPy 版本的 goal 归一化，用于环境交互时处理单个 goal。"""

    g = np.asarray(goal, dtype=np.float32).copy()
    # 前 24 维分成 shared/slow/fast 三个方向向量，方向重要，长度不重要。
    for start in (0, 8, 16):
        norm = float(np.linalg.norm(g[start:start + 8]))
        if norm < 1e-8:
            g[start:start + 8] = 0.0
        else:
            g[start:start + 8] /= norm
    # 后 8 维表示物理参考量，用 tanh 限制到 [-1, 1]，避免 goal 无界。
    g[24:32] = np.tanh(g[24:32])
    return g.astype(np.float32)


def execute_manager_goal_np(raw_goal: np.ndarray, previous_executed_goal: np.ndarray,
                            smoothing: float) -> np.ndarray:
    """Apply the exact interaction-time Manager goal transform."""

    def canonical(value: np.ndarray) -> np.ndarray:
        result = np.asarray(value, dtype=np.float32).copy()
        for start in (0, 8, 16):
            norm = max(float(np.linalg.norm(result[start:start + 8])), 1e-8)
            result[start:start + 8] /= norm
        result[24:32] = np.clip(result[24:32], -1.0, 1.0)
        return result

    raw = canonical(raw_goal)
    previous = canonical(previous_executed_goal)
    return canonical((1.0 - float(smoothing)) * raw + float(smoothing) * previous)


def execute_manager_goal_tensor(raw_goal: torch.Tensor, previous_executed_goal: torch.Tensor,
                                smoothing: float) -> torch.Tensor:
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：执行与交互阶段一致的可微 goal 平滑变换。
    
    输入：raw_goal、previous_executed_goal、smoothing。
    
    输出：executed_goal。
    
    核心步骤：先归一化，再与上一 executed goal 平滑，最后保持合法范围。
    
    强化学习含义：Actor 更新和 target 更新通过同一变换保持动作语义一致。
    
    【容易混淆】Critic 学 executed goal，而非未执行的 raw goal。
    
    【张量形状】输入/输出:[B,32]
    """

    def canonical(value: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            F.normalize(value[..., 0:8], p=2, dim=-1, eps=1e-8),
            F.normalize(value[..., 8:16], p=2, dim=-1, eps=1e-8),
            F.normalize(value[..., 16:24], p=2, dim=-1, eps=1e-8),
            value[..., 24:32].clamp(-1.0, 1.0),
        ], dim=-1)

    raw = canonical(raw_goal)
    previous = canonical(previous_executed_goal)
    return canonical((1.0 - float(smoothing)) * raw + float(smoothing) * previous)


# 【中文导读】Slow/Fast Worker 从 32 维完整 goal 中提取 shared、private、physical 共 24 维条件。
def worker_goal_np(goal: np.ndarray, role: str) -> np.ndarray:
    """Worker receives g_shared, its private goal and g_physical."""
    g = np.asarray(goal, dtype=np.float32)
    if role == "slow":
        return np.concatenate([g[0:8], g[8:16], g[24:32]]).astype(np.float32)
    if role == "fast":
        return np.concatenate([g[0:8], g[16:24], g[24:32]]).astype(np.float32)
    raise ValueError(f"Unknown worker role: {role}")


# 【中文导读】批量张量版本的 Worker goal 切片。
def worker_goal_tensor(goal: torch.Tensor, role: str) -> torch.Tensor:
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：从 32 维 Manager goal 选择某 Worker 的 24 维子 goal。
    
    输入：goal 与 role。
    
    输出：shared 8 + role direction 8 + physical 8。
    
    核心步骤：Slow/Fast 选择各自方向块并共享物理块。
    
    强化学习含义：让两层共享全局意图又保留角色分工。
    
    【容易混淆】Fast/Slow 输出维度相同但中间 8 维来源不同。
    
    【张量形状】[B,32] -> [B,24]
    """
    if role == "slow":
        return torch.cat([goal[..., 0:8], goal[..., 8:16], goal[..., 24:32]], dim=-1)
    if role == "fast":
        return torch.cat([goal[..., 0:8], goal[..., 16:24], goal[..., 24:32]], dim=-1)
    raise ValueError(f"Unknown worker role: {role}")


# 【中文导读】把 8 维 shared/private 方向组合后平铺到 Worker latent 维度，用于余弦方向奖励。
def expanded_goal_direction_np(goal: np.ndarray, role: str, latent_dim: int) -> np.ndarray:
    """把8维方向目标平铺到Worker隐空间，避免维度不匹配。"""
    g = np.asarray(goal, dtype=np.float32)
    role_slice = g[8:16] if role == "slow" else g[16:24]
    direction = 0.5 * (g[0:8] + role_slice)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return np.zeros(latent_dim, dtype=np.float32)
    direction = direction / norm
    repeats = int(math.ceil(latent_dim / direction.size))
    tiled = np.tile(direction, repeats)[:latent_dim]
    tiled_norm = float(np.linalg.norm(tiled))
    return (tiled / max(tiled_norm, 1e-8)).astype(np.float32)


# 【中文导读】批量张量版本的 latent 目标方向构造。
def expanded_goal_direction_tensor(goal: torch.Tensor, role: str, latent_dim: int) -> torch.Tensor:
    role_part = goal[..., 8:16] if role == "slow" else goal[..., 16:24]
    direction = 0.5 * (goal[..., 0:8] + role_part)
    direction = F.normalize(direction, p=2, dim=-1, eps=1e-8)
    repeats = int(math.ceil(latent_dim / 8))
    tiled = direction.repeat(1, repeats)[..., :latent_dim]
    return F.normalize(tiled, p=2, dim=-1, eps=1e-8)


# 【中文导读】在线维护观测均值和方差；训练时更新，评估时冻结。
class RunningMeanStd:
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：在线估计 observation 均值和方差。
    
    输入：一批 observation。
    
    输出：可归一化新 observation 的统计状态。
    
    核心步骤：用稳定的并行方差公式更新 count/mean/var。
    
    强化学习含义：归一化可减小不同物理量量纲对网络优化的影响。
    
    【容易混淆】评估和冻结阶段不能继续更新统计，否则策略输入分布会漂移。
    """

    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)
        self.training = True

    def update(self, x: np.ndarray) -> None:
        if not self.training:
            return
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == len(self.mean.shape):
            arr = arr.reshape((1,) + self.mean.shape)
        # 使用 batch 均值/方差增量更新总体均值/方差，避免保存所有历史观测。
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = float(total_count)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        return np.clip((arr - self.mean.astype(np.float32)) / np.sqrt(self.var.astype(np.float32) + 1e-8),
                       -10.0, 10.0).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean, "var": self.var, "count": self.count, "training": self.training}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])
        self.training = bool(state.get("training", True))

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True


# 【中文导读】将 episode 级回报、约束、投影与求解器指标写入 CSV。
class EpisodeCSVLogger:
    """轻量训练日志；TensorBoard不可用时也能追踪每个episode。"""

    def __init__(self, path: Path):
        self.path = path
        self.fieldnames = [
            "stage", "episode", "global_step", "episode_return", "manager_return",
            "mean_step_reward", "min_step_reward", "max_step_reward",
            "eval_return", "best_eval_return", "eval_solver_failures",
            "eval_power_success_rate", "eval_gas_success_rate",
            "eval_mean_voltage_rms_deviation_pu",
            "eval_mean_gas_pressure_rms_deviation_bar",
            "fast_buffer_size", "slow_buffer_size", "manager_buffer_size",
            "solver_failures", "gas_solve_count", "mean_action_projection",
            "max_action_projection", "mean_slow_action_projection",
            "max_slow_action_projection", "mean_fast_action_projection",
            "max_fast_action_projection", "mean_ess_action_guard",
            "max_ess_action_guard", "mean_goal_change", "max_goal_change",
            "mean_voltage_rms_deviation_pu", "mean_gas_pressure_rms_deviation_bar",
            "voltage_deviation_cost", "gas_pressure_deviation_cost",
            "voltage_violation_cost", "gas_pressure_violation_cost",
            "pipe_velocity_violation_cost", "source_capacity_violation_cost",
            "gas_purchase_cost",
            "interaction_reward_estimated_clips", "manager_reward_clips",
            "fast_total_insertions", "slow_total_insertions", "manager_total_insertions",
            "fast_update_count", "slow_update_count", "manager_update_count",
            "fast_actor_update_count", "slow_actor_update_count", "manager_actor_update_count",
            "fast_critic_update_count", "slow_critic_update_count", "manager_critic_update_count",
            "nonfinite_batch_count", "fast_normalizer_count", "slow_normalizer_count",
            "manager_normalizer_count", "mean_target_q_clipping_ratio",
            "mean_reward_clipping_ratio", "mean_actor_action_saturation_ratio",
        ]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def write(self, row: Dict[str, Any]) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({key: row.get(key, "") for key in self.fieldnames})


# =============================================================================
# Observation builder
# =============================================================================
#
# 【模块说明：观测构造】把环境全局状态转换为三层各自的 observation。
# Manager 看全局和上一 executed goal；Slow 看紧凑电—气安全摘要；Fast 看快速电压、
# 线路和新能源信息。命名布局避免 observation 改动后仍使用魔法索引。
#


@dataclass(frozen=True)
class SlowObservationLayout:
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：集中定义 Slow observation 的字段、长度和切片。
    
    输入：字段布局以及环境安全特征字典。
    
    输出：49 维扁平 Slow observation。
    
    核心步骤：按 fields 顺序验证长度、有限性并拼接。
    
    强化学习含义：慢层同时看到 SOC、电侧安全、气侧安全、预测、时间和 held action。
    
    【容易混淆】held_slow_action 是 10 维 RL 动作，不包含固定压缩机。
    """

    fields: Tuple[Tuple[str, int], ...] = (
        ("soc", len(ESS_CONFIGS)),
        ("soc_low_margin", len(ESS_CONFIGS)),
        ("soc_high_margin", len(ESS_CONFIGS)),
        ("voltage_summary", 3),
        ("line_summary", 3),
        ("power_balance", 5),
        ("power_loss", 1),
        ("power_forecast", 4),
        ("gas_pressure_summary", 3),
        ("pipe_summary", 2),
        ("source_utilization", len(GAS_SUPPLIERS)),
        ("linepack", 1),
        ("gas_forecast", 2),
        ("time", 4),
        ("held_slow_action", len(ESS_CONFIGS) + len(GFG_CONFIGS)
         + len(P2G_CONFIGS) + len(CONTROLLED_COMPRESSOR_INDICES)),
    )

    @property
    def dimension(self) -> int:
        return int(sum(size for _, size in self.fields))

    @property
    def slices(self) -> Dict[str, slice]:
        result: Dict[str, slice] = {}
        cursor = 0
        for name, size in self.fields:
            result[name] = slice(cursor, cursor + size)
            cursor += size
        return result

    def flatten(self, values: Mapping[str, np.ndarray]) -> np.ndarray:
        pieces: List[np.ndarray] = []
        for name, size in self.fields:
            if name not in values:
                raise KeyError(f"Slow safety state is missing field {name!r}")
            piece = np.asarray(values[name], dtype=np.float32).reshape(-1)
            if piece.size != size:
                raise ValueError(
                    f"Slow safety field {name!r} has size={piece.size}, expected={size}"
                )
            pieces.append(piece)
        result = np.concatenate(pieces).astype(np.float32)
        if result.size != self.dimension:
            raise AssertionError(f"Slow observation size={result.size}, expected={self.dimension}")
        return np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)


SLOW_OBSERVATION_LAYOUT = SlowObservationLayout()


# 【中文导读】从环境全局状态构造 Manager、快 Worker、慢 Worker 的任务相关观测。
class ObservationBuilder:
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：从 165 维全局状态构造三层 observation。
    
    输入：环境对象、Manager 间隔、全局状态和 previous goal。
    
    输出：Manager 197、Slow 49、Fast 115 维 observation。
    
    核心步骤：按各层控制职责抽取信息并追加时间/历史动作。
    
    强化学习含义：分层策略不必共享完全相同的 observation。
    
    【容易混淆】state 是环境真实状态概念，observation 是提供给某层网络的向量。
    """

    def __init__(self, env: ElectricGasMultiScaleEnv, manager_interval: int):
        self.env = env
        self.manager_interval = manager_interval

    # 【中文导读】返回 Manager 使用的全局电—气状态。
    def manager_obs(self, fallback_global: Optional[np.ndarray] = None,
                    previous_executed_goal: Optional[np.ndarray] = None) -> np.ndarray:
        if hasattr(self.env, "get_manager_state"):
            base = np.asarray(self.env.get_manager_state(), dtype=np.float32)
        else:
            base = np.asarray(fallback_global, dtype=np.float32)
        previous = np.zeros(GOAL_DIM, dtype=np.float32) if previous_executed_goal is None else np.asarray(
            previous_executed_goal, dtype=np.float32
        ).reshape(-1)
        if previous.size != GOAL_DIM:
            raise ValueError(f"previous_executed_goal size={previous.size}, expected={GOAL_DIM}")
        return np.nan_to_num(np.concatenate([base.reshape(-1), previous]), nan=0.0).astype(np.float32)

    # 【中文导读】构造以电侧为主、补充时间、SOC、慢设备设定和 goal 年龄的快观测。
    def fast_obs(self, manager_age_steps: int, fallback_global: Optional[np.ndarray] = None) -> np.ndarray:
        global_obs = np.asarray(fallback_global if fallback_global is not None else self.env.get_global_state(),
                                dtype=np.float32)
        if hasattr(self.env, "get_fast_worker_state"):
            base = np.asarray(self.env.get_fast_worker_state(), dtype=np.float32)
        else:
            base = global_obs
        # 快 Worker 主要看电侧状态，但也需要知道时间、当前 Manager goal 年龄、
        # 以及慢速设备最近的设定值，否则它很难判断当前电压问题来自哪里。
        extras = [global_obs[-6:]]
        extras.append(np.asarray([manager_age_steps / max(self.manager_interval, 1)], dtype=np.float32))
        if hasattr(self.env, "ess_soc"):
            extras.append(np.asarray(self.env.ess_soc, dtype=np.float32))
        if hasattr(self.env, "last_physical_slow"):
            slow = self.env.last_physical_slow
            for key in ("ess_p_mw", "gfg_p_mw", "p2g_p_mw", "compressor_ratio"):
                extras.append(np.asarray(slow.get(key, []), dtype=np.float32).reshape(-1))
        return np.nan_to_num(np.concatenate([base] + extras).astype(np.float32), nan=0.0)

    # 【中文导读】构造以储能和气网为主、补充预测时间特征的慢观测。
    def slow_obs(self, fallback_global: Optional[np.ndarray] = None) -> np.ndarray:
        del fallback_global
        if not hasattr(self.env, "get_slow_safety_features"):
            raise RuntimeError("Environment must expose get_slow_safety_features()")
        return SLOW_OBSERVATION_LAYOUT.flatten(self.env.get_slow_safety_features())


# =============================================================================
# Replay buffers
# =============================================================================
#
# 【模块说明：经验回放】分别保存 Fast 单步、Slow 多步和 Manager 多步 transition。
# Replay 让 TD3 成为 off-policy 算法；duration_steps、done、raw/executed 动作决定 TD target
# 和安全信用是否正确。优化版会在这些基础结构上增加 PER 与投影诊断字段。
#


# 【中文导读】保存每个 3 分钟步的状态、raw/executed 动作、分项奖励、goal 与终止标志。
class FastReplayBuffer:
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：保存 Fast Worker 的单步 transition。
    
    输入：obs/next_obs、raw/executed action、奖励、goal、done。
    
    输出：供 Critic/Actor 采样的 tensor batch。
    
    核心步骤：环形写入并随机采样。
    
    强化学习含义：Fast transition 的 duration 固定为 1。
    
    【容易混淆】raw action 是 Critic 动作语义，executed action 描述真实转移。
    
    【张量形状】obs:[B,fast_obs_dim]；action:[B,16]；reward/done/duration:[B,1]
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, goal_dim: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.raw_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.executed_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward_external = np.zeros((capacity, 1), dtype=np.float32)
        self.reward_intrinsic = np.zeros((capacity, 1), dtype=np.float32)
        self.reward_total = np.zeros((capacity, 1), dtype=np.float32)
        self.goals = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.next_goals = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.goal_changed = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.duration_steps = np.ones((capacity, 1), dtype=np.float32)
        self.idx = 0
        self.full = False
        self.total_insertions = 0

    def add(self, obs: np.ndarray, next_obs: np.ndarray, raw_action: np.ndarray, executed_action: np.ndarray,
            reward_external: float, reward_intrinsic: float, reward_total: float, goal: np.ndarray,
            next_goal: np.ndarray, done: bool) -> None:
        i = self.idx % self.capacity
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.raw_actions[i] = raw_action
        self.executed_actions[i] = executed_action
        self.reward_external[i, 0] = reward_external
        self.reward_intrinsic[i, 0] = reward_intrinsic
        self.reward_total[i, 0] = reward_total
        self.goals[i] = goal
        self.next_goals[i] = next_goal
        self.goal_changed[i, 0] = float(np.linalg.norm(goal - next_goal) > 1e-6)
        self.dones[i, 0] = float(done)
        self.duration_steps[i, 0] = 1.0
        self.idx += 1
        self.full = self.full or self.idx >= self.capacity
        self.total_insertions += 1

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        max_idx = len(self)
        idx = np.random.randint(0, max_idx, size=batch_size)
        # 采样时直接转成 torch.Tensor，并放到当前训练设备，避免 update 中重复搬运。
        return {
            "obs": to_tensor(self.obs[idx], self.device),
            "next_obs": to_tensor(self.next_obs[idx], self.device),
            "raw_actions": to_tensor(self.raw_actions[idx], self.device),
            "executed_actions": to_tensor(self.executed_actions[idx], self.device),
            "rewards": to_tensor(self.reward_total[idx], self.device),
            "reward_external": to_tensor(self.reward_external[idx], self.device),
            "reward_intrinsic": to_tensor(self.reward_intrinsic[idx], self.device),
            "goals": to_tensor(self.goals[idx], self.device),
            "next_goals": to_tensor(self.next_goals[idx], self.device),
            "goal_changed": to_tensor(self.goal_changed[idx], self.device),
            "dones": to_tensor(self.dones[idx], self.device),
            "duration_steps": to_tensor(self.duration_steps[idx], self.device),
        }

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def state_dict(self) -> Dict[str, Any]:
        return _replay_state_dict(self, (
            "obs", "next_obs", "raw_actions", "executed_actions", "reward_external",
            "reward_intrinsic", "reward_total", "goals", "next_goals", "goal_changed", "dones",
            "duration_steps",
        ))

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        _load_replay_state_dict(self, state, (
            "obs", "next_obs", "raw_actions", "executed_actions", "reward_external",
            "reward_intrinsic", "reward_total", "goals", "next_goals", "goal_changed", "dones",
            "duration_steps",
        ))


# 【中文导读】保存跨多个快速步的慢时间尺度 SMDP 片段及 duration_steps。
class SlowReplayBuffer:
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：保存 Slow Worker 的多步 SMDP transition。
    
    输入：片段起止 observation、动作统计、片段奖励、duration。
    
    输出：Slow TD3 batch。
    
    核心步骤：一个样本概括通常 20 个快速步，尾段使用真实长度。
    
    强化学习含义：Slow 的一步是持续时间不固定的宏动作。
    
    【容易混淆】不能只保存区间第一步 executed action。
    
    【张量形状】obs:[B,49]；action:[B,10]；duration:[B,1]
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, goal_dim: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.obs_start = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.obs_end = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.raw_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.executed_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.discounted_reward = np.zeros((capacity, 1), dtype=np.float32)
        self.goals = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.next_goals = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.duration_steps = np.zeros((capacity, 1), dtype=np.float32)
        self.idx = 0
        self.full = False
        self.total_insertions = 0

    def add(self, obs_start: np.ndarray, obs_end: np.ndarray, raw_action: np.ndarray, executed_action: np.ndarray,
            discounted_reward: float, goal: np.ndarray, next_goal: np.ndarray, done: bool,
            duration_steps: int) -> None:
        i = self.idx % self.capacity
        self.obs_start[i] = obs_start
        self.obs_end[i] = obs_end
        self.raw_actions[i] = raw_action
        self.executed_actions[i] = executed_action
        self.discounted_reward[i, 0] = discounted_reward
        self.goals[i] = goal
        self.next_goals[i] = next_goal
        self.dones[i, 0] = float(done)
        self.duration_steps[i, 0] = float(duration_steps)
        self.idx += 1
        self.full = self.full or self.idx >= self.capacity
        self.total_insertions += 1

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        max_idx = len(self)
        idx = np.random.randint(0, max_idx, size=batch_size)
        return {
            "obs": to_tensor(self.obs_start[idx], self.device),
            "next_obs": to_tensor(self.obs_end[idx], self.device),
            "raw_actions": to_tensor(self.raw_actions[idx], self.device),
            "executed_actions": to_tensor(self.executed_actions[idx], self.device),
            "rewards": to_tensor(self.discounted_reward[idx], self.device),
            "goals": to_tensor(self.goals[idx], self.device),
            "next_goals": to_tensor(self.next_goals[idx], self.device),
            "dones": to_tensor(self.dones[idx], self.device),
            "duration_steps": to_tensor(self.duration_steps[idx], self.device),
        }

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def state_dict(self) -> Dict[str, Any]:
        return _replay_state_dict(self, (
            "obs_start", "obs_end", "raw_actions", "executed_actions", "discounted_reward",
            "goals", "next_goals", "dones", "duration_steps",
        ))

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        _load_replay_state_dict(self, state, (
            "obs_start", "obs_end", "raw_actions", "executed_actions", "discounted_reward",
            "goals", "next_goals", "dones", "duration_steps",
        ))


# 【中文导读】保存一个 Manager goal 持续区间的全局起止状态和折扣外在回报。
class ManagerReplayBuffer:
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：保存 Manager goal 片段。
    
    输入：当前/下一 Manager observation、raw/executed/previous goal、片段奖励与 duration。
    
    输出：Manager TD3 batch。
    
    核心步骤：每 40 个快速步或尾部结束时入库。
    
    强化学习含义：Manager 是 SMDP 高层策略。
    
    【容易混淆】goal smoothing 依赖 previous executed goal，因此它必须属于 Markov observation/Replay。
    
    【张量形状】obs:[B,197]；goal:[B,32]；reward/done/duration:[B,1]
    """

    def __init__(self, capacity: int, obs_dim: int, goal_dim: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.global_obs_start = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.global_obs_end = np.zeros((capacity, obs_dim), dtype=np.float32)
        # manager_goals remains the executed-goal array for old checkpoints.
        self.manager_goals = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.raw_goals = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.previous_executed_goals = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.discounted_external_reward = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.duration_steps = np.zeros((capacity, 1), dtype=np.float32)
        self.segment_constraint_mean = np.zeros((capacity, 1), dtype=np.float32)
        self.segment_constraint_max = np.zeros((capacity, 1), dtype=np.float32)
        self.solver_failure_seen = np.zeros((capacity, 1), dtype=np.float32)
        self.idx = 0
        self.full = False
        self.total_insertions = 0

    def add(self, global_obs_start: np.ndarray, global_obs_end: np.ndarray, manager_goal: np.ndarray,
            discounted_external_reward: float, done: bool, duration_steps: int,
            segment_constraint_mean: float = 0.0, segment_constraint_max: float = 0.0,
            solver_failure_seen: bool = False, raw_goal: Optional[np.ndarray] = None,
            previous_executed_goal: Optional[np.ndarray] = None) -> None:
        i = self.idx % self.capacity
        self.global_obs_start[i] = global_obs_start
        self.global_obs_end[i] = global_obs_end
        self.manager_goals[i] = manager_goal
        self.raw_goals[i] = manager_goal if raw_goal is None else raw_goal
        self.previous_executed_goals[i] = (
            np.zeros_like(manager_goal) if previous_executed_goal is None else previous_executed_goal
        )
        self.discounted_external_reward[i, 0] = discounted_external_reward
        self.dones[i, 0] = float(done)
        self.duration_steps[i, 0] = float(duration_steps)
        self.segment_constraint_mean[i, 0] = float(segment_constraint_mean)
        self.segment_constraint_max[i, 0] = float(segment_constraint_max)
        self.solver_failure_seen[i, 0] = float(bool(solver_failure_seen))
        self.idx += 1
        self.full = self.full or self.idx >= self.capacity
        self.total_insertions += 1

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        max_idx = len(self)
        idx = np.random.randint(0, max_idx, size=batch_size)
        return {
            "obs": to_tensor(self.global_obs_start[idx], self.device),
            "next_obs": to_tensor(self.global_obs_end[idx], self.device),
            "goals": to_tensor(self.manager_goals[idx], self.device),
            "executed_goals": to_tensor(self.manager_goals[idx], self.device),
            "raw_goals": to_tensor(self.raw_goals[idx], self.device),
            "previous_executed_goals": to_tensor(self.previous_executed_goals[idx], self.device),
            "rewards": to_tensor(self.discounted_external_reward[idx], self.device),
            "dones": to_tensor(self.dones[idx], self.device),
            "duration_steps": to_tensor(self.duration_steps[idx], self.device),
        }

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def state_dict(self) -> Dict[str, Any]:
        return _replay_state_dict(self, (
            "global_obs_start", "global_obs_end", "manager_goals", "raw_goals",
            "previous_executed_goals",
            "discounted_external_reward", "dones", "duration_steps",
            "segment_constraint_mean", "segment_constraint_max", "solver_failure_seen",
        ))

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        _load_replay_state_dict(self, state, (
            "global_obs_start", "global_obs_end", "manager_goals", "raw_goals",
            "previous_executed_goals",
            "discounted_external_reward", "dones", "duration_steps",
            "segment_constraint_mean", "segment_constraint_max", "solver_failure_seen",
        ))


def _replay_state_dict(buffer: Any, array_names: Sequence[str]) -> Dict[str, Any]:
    valid_size = int(len(buffer))
    stored_size = int(buffer.capacity if buffer.full else valid_size)
    if hasattr(buffer, "obs"):
        obs_dim = int(buffer.obs.shape[1])
    elif hasattr(buffer, "obs_start"):
        obs_dim = int(buffer.obs_start.shape[1])
    else:
        obs_dim = int(buffer.global_obs_start.shape[1])
    action_dim = 0
    if hasattr(buffer, "raw_actions"):
        action_dim = int(buffer.raw_actions.shape[1])
    goal_array = getattr(buffer, "goals", getattr(buffer, "manager_goals", None))
    goal_dim = int(goal_array.shape[1]) if goal_array is not None else 0
    state: Dict[str, Any] = {
        "replay_schema_version": 4,
        "replay_type": type(buffer).__name__,
        "capacity": int(buffer.capacity),
        "valid_size": valid_size,
        "idx": int(buffer.idx),
        "full": bool(buffer.full),
        "total_insertions": int(buffer.total_insertions),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "goal_dim": goal_dim,
        "dtype": "float32",
    }
    for name in array_names:
        if hasattr(buffer, name):
            state[name] = np.asarray(getattr(buffer, name))[:stored_size].copy()
    return state


def _load_replay_state_dict(buffer: Any, state: Mapping[str, Any], array_names: Sequence[str]) -> None:
    saved_capacity = int(state.get("capacity", -1))
    if saved_capacity != int(buffer.capacity):
        raise ValueError(
            f"Replay capacity mismatch: checkpoint={saved_capacity}, current={buffer.capacity}"
        )
    schema = int(state.get("replay_schema_version", 1))
    if schema not in (1, 2, 3, 4):
        raise ValueError(f"Unsupported replay_schema_version={schema}")
    if schema >= 2:
        expected_type = type(buffer).__name__
        if state.get("replay_type") != expected_type:
            raise ValueError(
                f"Replay type mismatch: checkpoint={state.get('replay_type')!r}, current={expected_type!r}"
            )
        expected_obs = (buffer.obs.shape[1] if hasattr(buffer, "obs") else
                        buffer.obs_start.shape[1] if hasattr(buffer, "obs_start") else
                        buffer.global_obs_start.shape[1])
        expected_action = buffer.raw_actions.shape[1] if hasattr(buffer, "raw_actions") else 0
        goal_array = getattr(buffer, "goals", getattr(buffer, "manager_goals", None))
        expected_goal = goal_array.shape[1] if goal_array is not None else 0
        for name, expected in (("obs_dim", expected_obs), ("action_dim", expected_action),
                               ("goal_dim", expected_goal)):
            if int(state.get(name, -1)) != int(expected):
                raise ValueError(
                    f"Replay {name} mismatch: checkpoint={state.get(name)!r}, current={expected}"
                )
        if state.get("dtype") != "float32":
            raise ValueError(f"Replay dtype mismatch: checkpoint={state.get('dtype')!r}, expected='float32'")
    full = bool(state.get("full", False))
    idx = int(state.get("idx", 0))
    valid_size = int(state.get("valid_size", saved_capacity if full else idx))
    if not 0 <= valid_size <= saved_capacity:
        raise ValueError(f"Replay valid_size={valid_size} must be in [0,{saved_capacity}]")
    if full and (valid_size != saved_capacity or idx < saved_capacity):
        raise ValueError(f"Invalid full Replay state: valid_size={valid_size}, idx={idx}, capacity={saved_capacity}")
    if not full and (idx != valid_size or idx >= saved_capacity):
        raise ValueError(f"Invalid partial Replay state: valid_size={valid_size}, idx={idx}, capacity={saved_capacity}")
    stored_size = saved_capacity if full else valid_size
    for name in array_names:
        if not hasattr(buffer, name):
            continue
        if name not in state:
            raise ValueError(f"Replay state is missing required array {name!r}")
        target = getattr(buffer, name)
        raw_value = np.asarray(state[name])
        if schema >= 2 and str(raw_value.dtype) != str(target.dtype):
            raise ValueError(
                f"Replay array {name} dtype mismatch: checkpoint={raw_value.dtype}, current={target.dtype}"
            )
        value = raw_value.astype(target.dtype, copy=False)
        # Schema 1 checkpoints predate compact serialization and always contain
        # the complete capacity, even when the Replay had not filled yet.
        array_stored_size = saved_capacity if schema == 1 else stored_size
        expected_shape = (array_stored_size,) + target.shape[1:]
        if value.shape != expected_shape:
            raise ValueError(
                f"Replay array {name} shape mismatch: checkpoint={value.shape}, expected={expected_shape}"
            )
        target[...] = 0
        target[:array_stored_size] = value
    buffer.idx = idx
    buffer.full = full
    buffer.total_insertions = int(state.get("total_insertions", buffer.idx))
    if buffer.idx < 0 or buffer.total_insertions < 0:
        raise ValueError(
            f"Invalid Replay counters: idx={buffer.idx}, total_insertions={buffer.total_insertions}"
        )


# =============================================================================
# Networks
# =============================================================================
#
# 【模块说明：神经网络】Encoder 把 observation 压缩为 latent state；Actor 产生 goal
# 或动作；双 Critic 分别估计 Q。两个 Critic 取较小值可降低过估计，是 TD3 相比 DDPG
# 最重要的稳定化设计之一。
#


# 【中文导读】构造 Actor/Critic/Encoder 共用的两层 MLP。
def make_mlp(input_dim: int, output_dim: int, hidden_dim: int, layer_norm: bool = True,
             output_activation: Optional[nn.Module] = None) -> nn.Sequential:
    """构造一个两层隐藏层 MLP，本文件所有 Actor/Critic/Encoder 共用。"""

    layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    if layer_norm:
        layers.append(nn.LayerNorm(hidden_dim))
    layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    if layer_norm:
        layers.append(nn.LayerNorm(hidden_dim))
    layers.append(nn.Linear(hidden_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


# 【中文导读】把高维物理观测编码为 latent state。
class MLPEncoder(nn.Module):
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：把 observation 编码为紧凑 latent state。
    
    输入：obs tensor。
    
    输出：latent tensor。
    
    核心步骤：多层感知机和非线性变换。
    
    强化学习含义：Actor/Critic 在归一化后的抽象状态上学习。
    
    【容易混淆】Encoder 不是环境状态转移模型。
    
    【张量形状】obs:[B,obs_dim] -> z:[B,latent_dim]
    """

    def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = make_mlp(obs_dim, latent_dim, hidden_dim, layer_norm=True)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# 【中文导读】根据 Manager latent state 生成 32 维组合式 goal。
class ManagerActor(nn.Module):
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：根据 Manager latent state 输出 raw goal。
    
    输入：z。
    
    输出：32 维 raw goal。
    
    核心步骤：MLP 输出后再由统一 goal transform 归一化/平滑。
    
    强化学习含义：Actor 是确定性高层策略。
    
    【容易混淆】raw goal 不能绕过 execute_manager_goal_* 直接当 executed goal。
    
    【张量形状】z:[B,manager_latent_dim] -> raw_goal:[B,32]
    """

    def __init__(self, latent_dim: int, goal_dim: int, hidden_dim: int):
        super().__init__()
        self.net = make_mlp(latent_dim, goal_dim, hidden_dim, layer_norm=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return normalize_goal_tensor(self.net(z))


# 【中文导读】双 Q 网络，估计全局 latent state 与 goal 的长期价值。
class ManagerCritic(nn.Module):
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：双 Q 网络评价 executed Manager goal。
    
    输入：z 与 32 维 goal。
    
    输出：Q1、Q2 标量。
    
    核心步骤：两个独立 MLP 并行估值。
    
    强化学习含义：取 min(Q1,Q2) 降低过估计。
    
    【容易混淆】优化版 Critic 动作语义是经过统一变换的 executed goal。
    
    【张量形状】z:[B,L]；goal:[B,32]；Q1/Q2:[B,1]
    """

    def __init__(self, latent_dim: int, goal_dim: int, hidden_dim: int):
        super().__init__()
        self.q1 = make_mlp(latent_dim + goal_dim, 1, hidden_dim, layer_norm=False)
        self.q2 = make_mlp(latent_dim + goal_dim, 1, hidden_dim, layer_norm=False)

    def forward(self, z: torch.Tensor, goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z, goal], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, z: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(z, goal)
        return torch.minimum(q1, q2)


# 【中文导读】根据 Worker latent state 和 24 维局部 goal 输出归一化控制动作。
class WorkerActor(nn.Module):
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：根据 Worker latent state 和子 goal 输出 raw action。
    
    输入：z 与 24 维 Worker goal。
    
    输出：归一化动作请求。
    
    核心步骤：拼接状态和 goal 后由 MLP 输出 [-1,1] 动作。
    
    强化学习含义：Worker Actor 学习满足 Manager 意图的设备控制。
    
    【容易混淆】输出还要经过 guard 和环境安全投影。
    
    【张量形状】slow action:[B,10]；fast action:[B,16]
    """

    def __init__(self, latent_dim: int, worker_goal_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = make_mlp(latent_dim + worker_goal_dim, action_dim, hidden_dim, layer_norm=True)

    def forward(self, z: torch.Tensor, worker_goal: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(torch.cat([z, worker_goal], dim=-1)))


# 【中文导读】双 Q 网络，估计状态、goal 与 Actor raw request 动作的价值。
class WorkerCritic(nn.Module):
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：双 Q 网络评价 raw request action。
    
    输入：z、Worker goal、raw action。
    
    输出：Q1、Q2。
    
    核心步骤：对相同输入用两个独立网络估值。
    
    强化学习含义：Critic 学习 Q(s,raw_request_action)，与 Actor 的输出空间一致。
    
    【容易混淆】executed action 用于真实转移/模仿，不应偷偷替换 Critic 动作。
    
    【张量形状】Q1/Q2:[B,1]
    """

    def __init__(self, latent_dim: int, worker_goal_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        in_dim = latent_dim + worker_goal_dim + action_dim
        self.q1 = make_mlp(in_dim, 1, hidden_dim, layer_norm=False)
        self.q2 = make_mlp(in_dim, 1, hidden_dim, layer_norm=False)

    def forward(self, z: torch.Tensor, worker_goal: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z, worker_goal, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, z: torch.Tensor, worker_goal: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(z, worker_goal, action)
        return torch.minimum(q1, q2)


# 【中文导读】可选隐空间动力学模型，预测动作导致的 latent 增量。
class TransitionModel(nn.Module):
    """可选的隐空间动力学模型，预测 action 导致的 latent delta。"""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = make_mlp(latent_dim + action_dim, latent_dim, hidden_dim, layer_norm=False)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, action], dim=-1))


# =============================================================================
# TD3 agents
# =============================================================================
#
# 【模块说明：TD3 智能体】ManagerTD3 学习高层 goal，WorkerTD3 学习设备动作。
# update 同时完成 Critic TD 回归、延迟 Actor 更新、target policy smoothing 和 target 网络
# 软更新。初学者应重点跟踪 online/target 网络与 Actor/Critic 的不同更新频率。
#


# 【中文导读】高层 TD3；动作是 goal，样本是约两小时的聚合片段。
class ManagerTD3:
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：封装 Manager 的 Actor、双 Critic、target 网络和优化器。
    
    输入：Manager observation、previous goal、Replay batch。
    
    输出：raw/executed goal或更新日志。
    
    核心步骤：选 goal 时统一变换；更新时构造 SMDP target、回归 Critic、延迟更新 Actor并软更新 target。
    
    强化学习含义：每 40 个快速步做一次高层决策。
    
    【容易混淆】Actor/target Actor 必须调用相同 goal transform。
    """

    # 【中文导读】创建在线/目标 Encoder、Actor、双 Critic，并完成硬同步和优化器初始化。
    def __init__(self, obs_dim: int, cfg: TrainConfig, device: torch.device):
        self.obs_dim = obs_dim
        self.cfg = cfg
        self.device = device
        self.encoder = MLPEncoder(obs_dim, cfg.manager_latent_dim, cfg.hidden_dim).to(device)
        self.target_encoder = MLPEncoder(obs_dim, cfg.manager_latent_dim, cfg.hidden_dim).to(device)
        self.actor = ManagerActor(cfg.manager_latent_dim, GOAL_DIM, cfg.manager_hidden_dim).to(device)
        self.target_actor = ManagerActor(cfg.manager_latent_dim, GOAL_DIM, cfg.manager_hidden_dim).to(device)
        self.critic = ManagerCritic(cfg.manager_latent_dim, GOAL_DIM, cfg.critic_hidden_dim).to(device)
        self.target_critic = ManagerCritic(cfg.manager_latent_dim, GOAL_DIM, cfg.critic_hidden_dim).to(device)
        hard_update(self.target_encoder, self.encoder)
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        self.encoder_optim = optim.Adam(self.encoder.parameters(), lr=cfg.manager_lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg.manager_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.manager_lr)
        self.normalizer = RunningMeanStd((obs_dim,))
        self.total_updates = 0

    def select_goal_pair(self, obs: np.ndarray, previous_goal: Optional[np.ndarray], noise_std: float,
                         deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Return raw and executed goals with the same transform used by training."""
        self.normalizer.update(obs)
        obs_n = self.normalizer.normalize(obs)
        with torch.no_grad():
            z = self.encoder(to_tensor(obs_n[None, :], self.device))
            raw_goal = self.actor(z).cpu().numpy()[0]
        if not deterministic and noise_std > 0.0:
            raw_goal += np.random.normal(0.0, noise_std, size=raw_goal.shape).astype(np.float32)
            raw_goal = normalize_goal_np(raw_goal)
        previous = np.zeros(GOAL_DIM, dtype=np.float32) if previous_goal is None else previous_goal
        executed_goal = execute_manager_goal_np(raw_goal, previous, self.cfg.goal_smoothing)
        return raw_goal.astype(np.float32), executed_goal.astype(np.float32)

    # Compatibility interface: Workers always receive the executed goal.
    def select_goal(self, obs: np.ndarray, previous_goal: Optional[np.ndarray], noise_std: float,
                    deterministic: bool = False) -> np.ndarray:
        return self.select_goal_pair(obs, previous_goal, noise_std, deterministic)[1]

    # 【中文导读】用 Manager 聚合样本执行双 Critic 回归、延迟 Actor 更新和目标网络软更新。
    def update(self, buffer: ManagerReplayBuffer, batch_size: int) -> Dict[str, float]:
        if len(buffer) < batch_size:
            return {}
        data = buffer.sample(batch_size)
        obs = to_tensor(self.normalizer.normalize(data["obs"].cpu().numpy()), self.device)
        next_obs = to_tensor(self.normalizer.normalize(data["next_obs"].cpu().numpy()), self.device)
        goals = data.get("executed_goals", data["goals"])
        previous_goals = data.get("previous_executed_goals", torch.zeros_like(goals))
        rewards = data["rewards"]
        dones = data["dones"]

        z = self.encoder(obs)
        with torch.no_grad():
            # TD3 目标：target actor 给 next_goal，target critic 给 next Q。
            # 加截断噪声是 TD3 的 target policy smoothing，可降低 Q 对尖锐动作的过拟合。
            next_z = self.target_encoder(next_obs)
            next_raw_goal = self.target_actor(next_z)
            noise = torch.randn_like(next_raw_goal) * self.cfg.target_noise
            next_raw_goal = normalize_goal_tensor(
                next_raw_goal + noise.clamp(-self.cfg.target_noise_clip, self.cfg.target_noise_clip)
            )
            next_goal = execute_manager_goal_tensor(next_raw_goal, goals, self.cfg.goal_smoothing)
            # 【TD3关键】两个目标 Critic 独立估值；后续取较小值，抑制函数逼近造成的 Q 过高估计。
            q1_next, q2_next = self.target_critic(next_z, next_goal)
            q_next = torch.minimum(q1_next, q2_next)
            target_q = rewards + (1.0 - dones) * torch.pow(
                torch.as_tensor(self.cfg.gamma_fast, dtype=data["duration_steps"].dtype,
                                device=self.device), data["duration_steps"]
            ) * q_next
            if self.cfg.target_q_clip_abs > 0.0:
                target_q = target_q.clamp(-self.cfg.target_q_clip_abs, self.cfg.target_q_clip_abs)

        q1, q2 = self.critic(z, goals)
        # 双 Critic 同时拟合同一个 target，后续取 min 减轻 Q 值高估。
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        latent_norm_loss = z.pow(2).mean()
        encoder_loss = critic_loss + self.cfg.lambda_latent_norm * latent_norm_loss
        self.critic_optim.zero_grad()
        self.encoder_optim.zero_grad()
        encoder_loss.backward()
        clip_grad(self.critic.parameters(), self.cfg.gradient_clip)
        clip_grad(self.encoder.parameters(), self.cfg.gradient_clip)
        self.critic_optim.step()
        self.encoder_optim.step()
        self.critic_optim.zero_grad(set_to_none=True)

        actor_loss_value = 0.0
        if self.total_updates % self.cfg.policy_frequency == 0:
            # Delayed policy update：Critic 多学几步后再更新 Actor，是 TD3 的稳定技巧。
            z_pi = self.encoder(obs).detach()
            self.actor_optim.zero_grad()
            set_requires_grad(self.critic, False)
            try:
                raw_goals_pi = self.actor(z_pi)
                goals_pi = execute_manager_goal_tensor(
                    raw_goals_pi, previous_goals, self.cfg.goal_smoothing
                )
                # 【TD3关键】Actor 通过最大化 Critic 评价来改进确定性策略；负号把最大化 Q 转为最小化 loss。
                actor_loss = -self.critic.q_min(z_pi, goals_pi).mean()
                actor_loss.backward()
            finally:
                set_requires_grad(self.critic, True)
            clip_grad(self.actor.parameters(), self.cfg.gradient_clip)
            self.actor_optim.step()
            actor_loss_value = float(actor_loss.detach().cpu())
            # 【TD3关键】target 网络缓慢追踪 online 网络，避免 bootstrap 目标随每次梯度更新剧烈移动。
            soft_update(self.target_actor, self.actor, self.cfg.tau)
            soft_update(self.target_critic, self.critic, self.cfg.tau)
            soft_update(self.target_encoder, self.encoder, self.cfg.tau)

        self.total_updates += 1
        return {
            "manager/critic_loss": float(critic_loss.detach().cpu()),
            "manager/actor_loss": actor_loss_value,
            "manager/q_value": float(torch.minimum(q1, q2).mean().detach().cpu()),
        }

    # 【中文导读】序列化完整高层训练状态。
    def state_dict(self) -> Dict[str, Any]:
        return {
            "encoder": self.encoder.state_dict(),
            "target_encoder": self.target_encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "encoder_optim": self.encoder_optim.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "normalizer": self.normalizer.state_dict(),
            "total_updates": self.total_updates,
        }

    # 【中文导读】恢复高层网络、优化器、归一化器和更新计数。
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.encoder.load_state_dict(state["encoder"])
        self.target_encoder.load_state_dict(state["target_encoder"])
        self.actor.load_state_dict(state["actor"])
        self.target_actor.load_state_dict(state["target_actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.encoder_optim.load_state_dict(state["encoder_optim"])
        self.actor_optim.load_state_dict(state["actor_optim"])
        self.critic_optim.load_state_dict(state["critic_optim"])
        move_optimizer_state(self.encoder_optim, self.device)
        move_optimizer_state(self.actor_optim, self.device)
        move_optimizer_state(self.critic_optim, self.device)
        self.normalizer.load_state_dict(state["normalizer"])
        self.total_updates = int(state.get("total_updates", 0))


# 【中文导读】Slow/Fast 共用 TD3；Critic 使用 raw request，Actor 生成归一化请求动作。
class WorkerTD3:
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：封装 Fast/Slow Worker 的 TD3 学习。
    
    输入：Worker observation、Manager goal、Replay。
    
    输出：raw action或损失/诊断。
    
    核心步骤：编码状态、计算 TD target、更新双 Critic，按 policy_frequency 延迟更新 Actor。
    
    强化学习含义：Fast 每步控制，Slow 每 20 步控制。
    
    【容易混淆】Slow/Fast goal 都是 24 维，但切片含义不同。
    """

    # 【中文导读】创建 Worker 在线/目标网络及可选隐空间动力学模型。
    def __init__(self, role: str, obs_dim: int, action_dim: int, latent_dim: int,
                 lr: float, cfg: TrainConfig, device: torch.device):
        self.role = role
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.cfg = cfg
        self.device = device
        self.encoder = MLPEncoder(obs_dim, latent_dim, cfg.hidden_dim).to(device)
        self.target_encoder = MLPEncoder(obs_dim, latent_dim, cfg.hidden_dim).to(device)
        self.actor = WorkerActor(latent_dim, 24, action_dim, cfg.hidden_dim).to(device)
        self.target_actor = WorkerActor(latent_dim, 24, action_dim, cfg.hidden_dim).to(device)
        self.critic = WorkerCritic(latent_dim, 24, action_dim, cfg.critic_hidden_dim).to(device)
        self.target_critic = WorkerCritic(latent_dim, 24, action_dim, cfg.critic_hidden_dim).to(device)
        self.transition_model = TransitionModel(latent_dim, action_dim, cfg.hidden_dim).to(device) if cfg.use_transition_model else None
        hard_update(self.target_encoder, self.encoder)
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        self.encoder_optim = optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)
        self.transition_optim = optim.Adam(self.transition_model.parameters(), lr=lr) if self.transition_model else None
        self.normalizer = RunningMeanStd((obs_dim,))
        self.total_updates = 0

    # 【中文导读】把局部观测与角色相关 goal 输入 Actor，并叠加探索噪声。
    def select_action(self, obs: np.ndarray, goal: np.ndarray, noise_std: float,
                      deterministic: bool = False) -> np.ndarray:
        """Worker 根据自己的局部观测和 Manager goal 输出 [-1, 1] 动作。"""

        self.normalizer.update(obs)
        obs_n = self.normalizer.normalize(obs)
        wg = worker_goal_np(goal, self.role)
        with torch.no_grad():
            z = self.encoder(to_tensor(obs_n[None, :], self.device))
            action = self.actor(z, to_tensor(wg[None, :], self.device)).cpu().numpy()[0]
        if not deterministic and noise_std > 0.0:
            action += np.random.normal(0.0, noise_std, size=action.shape).astype(np.float32)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    # 【中文导读】计算 latent 状态变化与 goal 方向的余弦相似度。
    def latent_goal_reward(self, obs: np.ndarray, next_obs: np.ndarray, goal: np.ndarray,
                           delta_z_min: float) -> float:
        """计算 FuN 风格的内在奖励：状态变化方向是否朝着 goal 指定方向前进。"""

        obs_n = self.normalizer.normalize(obs)
        next_obs_n = self.normalizer.normalize(next_obs)
        with torch.no_grad():
            z = self.target_encoder(to_tensor(obs_n[None, :], self.device)).cpu().numpy()[0]
            next_z = self.target_encoder(to_tensor(next_obs_n[None, :], self.device)).cpu().numpy()[0]
        delta_z = next_z - z
        delta_norm = float(np.linalg.norm(delta_z))
        if delta_norm < delta_z_min:
            return 0.0
        direction = expanded_goal_direction_np(goal, self.role, self.latent_dim)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-8:
            return 0.0
        return float(np.dot(delta_z, direction) / (delta_norm * direction_norm + 1e-8))

    # 【中文导读】执行 Worker TD3、Encoder 辅助损失、可选 transition model 和延迟策略更新。
    def update(self, buffer: Any, batch_size: int, gamma: float) -> Dict[str, float]:
        if len(buffer) < batch_size:
            return {}
        data = buffer.sample(batch_size)
        obs = to_tensor(self.normalizer.normalize(data["obs"].cpu().numpy()), self.device)
        next_obs = to_tensor(self.normalizer.normalize(data["next_obs"].cpu().numpy()), self.device)
        # raw_action 是 Actor 请求，也是 Critic 动作输入。executed_action 经环境
        # 逐步安全投影，只用于真实转移监督、投影模仿和诊断。
        raw_actions = data["raw_actions"]
        executed_actions = data["executed_actions"]
        rewards = data["rewards"]
        dones = data["dones"]
        goals = data["goals"]
        next_goals = data["next_goals"]
        wg = worker_goal_tensor(goals, self.role)
        next_wg = worker_goal_tensor(next_goals, self.role)

        z = self.encoder(obs)
        with torch.no_grad():
            # Worker 的 TD3 target：下一状态下的目标动作加平滑噪声，再用双 Critic 取较小值。
            next_z = self.target_encoder(next_obs)
            next_actions = self.target_actor(next_z, next_wg)
            noise = (torch.randn_like(next_actions) * self.cfg.target_noise).clamp(
                -self.cfg.target_noise_clip, self.cfg.target_noise_clip)
            next_actions = (next_actions + noise).clamp(-1.0, 1.0)
            # 【TD3关键】两个目标 Critic 独立估值；后续取较小值，抑制函数逼近造成的 Q 过高估计。
            q1_next, q2_next = self.target_critic(next_z, next_wg, next_actions)
            # 半马尔可夫 TD 目标。Fast 使用 gamma_fast；Slow 当前传入固定 gamma_slow。
            # 对提前结束的短片段，理论上更严谨的折扣应为 gamma_fast ** duration_steps。
            target_q = rewards + (1.0 - dones) * gamma * torch.minimum(q1_next, q2_next)
            if self.cfg.target_q_clip_abs > 0.0:
                target_q = target_q.clamp(-self.cfg.target_q_clip_abs, self.cfg.target_q_clip_abs)

        q1, q2 = self.critic(z, wg, raw_actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        latent_norm_loss = z.pow(2).mean()
        transition_encoder_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        if self.transition_model is not None:
            # 让Encoder的隐空间也服务于可预测的状态变化，但不在这一步更新TransitionModel参数。
            with torch.no_grad():
                target_z_for_encoder = self.target_encoder(obs)
                target_next_z_for_encoder = self.target_encoder(next_obs)
                target_delta_for_encoder = target_next_z_for_encoder - target_z_for_encoder
            set_requires_grad(self.transition_model, False)
            try:
                pred_delta_for_encoder = self.transition_model(z, executed_actions)
                transition_encoder_loss = F.mse_loss(pred_delta_for_encoder, target_delta_for_encoder)
            finally:
                set_requires_grad(self.transition_model, True)
        encoder_loss = (
            critic_loss
            + self.cfg.lambda_transition * transition_encoder_loss
            + self.cfg.lambda_latent_norm * latent_norm_loss
        )
        self.critic_optim.zero_grad()
        self.encoder_optim.zero_grad()
        encoder_loss.backward()
        clip_grad(self.critic.parameters(), self.cfg.gradient_clip)
        clip_grad(self.encoder.parameters(), self.cfg.gradient_clip)
        self.critic_optim.step()
        self.encoder_optim.step()
        self.critic_optim.zero_grad(set_to_none=True)

        transition_loss_value = 0.0
        if self.transition_model is not None and self.transition_optim is not None:
            with torch.no_grad():
                z_detached = self.encoder(obs).detach()
                z_target = self.target_encoder(obs)
                next_z_target = self.target_encoder(next_obs)
                target_delta = next_z_target - z_target
            pred_delta = self.transition_model(z_detached, executed_actions)
            transition_loss = F.mse_loss(pred_delta, target_delta.detach())
            self.transition_optim.zero_grad()
            transition_loss.backward()
            clip_grad(self.transition_model.parameters(), self.cfg.gradient_clip)
            self.transition_optim.step()
            transition_loss_value = float(transition_loss.detach().cpu())

        actor_loss_value = 0.0
        projection_imitation_loss_value = 0.0
        if self.total_updates % self.cfg.policy_frequency == 0:
            # Actor 的目标是最大化 Critic 认为好的动作；代码里写成最小化 -Q。
            z_pi = self.encoder(obs).detach()
            # Actor 输出仍是未经过环境安全投影的 raw action；因此需关注策略—执行动作偏差。
            self.actor_optim.zero_grad()
            set_requires_grad(self.critic, False)
            try:
                actions_pi = self.actor(z_pi, wg)
                # 【TD3关键】Actor 通过最大化 Critic 评价来改进确定性策略；负号把最大化 Q 转为最小化 loss。
                actor_loss = -self.critic.q_min(z_pi, wg, actions_pi).mean()
                if self.cfg.worker_action_l2_weight > 0.0:
                    actor_loss = actor_loss + self.cfg.worker_action_l2_weight * actions_pi.pow(2).mean()
                if self.cfg.projection_imitation_weight > 0.0:
                    projection_imitation_loss = F.mse_loss(actions_pi, executed_actions)
                    actor_loss = actor_loss + self.cfg.projection_imitation_weight * projection_imitation_loss
                    projection_imitation_loss_value = float(projection_imitation_loss.detach().cpu())
                if self.transition_model is not None and self.cfg.reachability_weight > 0.0:
                    predicted_delta = self.transition_model(z_pi, actions_pi)
                    direction = expanded_goal_direction_tensor(goals, self.role, self.latent_dim)
                    reachability = 1.0 - F.cosine_similarity(
                        predicted_delta, direction, dim=-1, eps=1e-8
                    ).mean()
                    actor_loss = actor_loss + self.cfg.reachability_weight * reachability
                actor_loss.backward()
            finally:
                set_requires_grad(self.critic, True)
            clip_grad(self.actor.parameters(), self.cfg.gradient_clip)
            self.actor_optim.step()
            actor_loss_value = float(actor_loss.detach().cpu())
            # 【TD3关键】target 网络缓慢追踪 online 网络，避免 bootstrap 目标随每次梯度更新剧烈移动。
            soft_update(self.target_actor, self.actor, self.cfg.tau)
            soft_update(self.target_critic, self.critic, self.cfg.tau)
            soft_update(self.target_encoder, self.encoder, self.cfg.tau)

        self.total_updates += 1
        prefix = f"{self.role}/"
        return {
            prefix + "critic_loss": float(critic_loss.detach().cpu()),
            prefix + "actor_loss": actor_loss_value,
            prefix + "projection_imitation_loss": projection_imitation_loss_value,
            prefix + "sample_projection_mse": float(
                F.mse_loss(raw_actions, executed_actions).detach().cpu()
            ),
            prefix + "transition_loss": transition_loss_value,
            prefix + "transition_encoder_loss": float(transition_encoder_loss.detach().cpu()),
            prefix + "q_value": float(torch.minimum(q1, q2).mean().detach().cpu()),
        }

    # 【中文导读】序列化 Worker 完整训练状态。
    def state_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "encoder": self.encoder.state_dict(),
            "target_encoder": self.target_encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "transition_model": None if self.transition_model is None else self.transition_model.state_dict(),
            "encoder_optim": self.encoder_optim.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "transition_optim": None if self.transition_optim is None else self.transition_optim.state_dict(),
            "normalizer": self.normalizer.state_dict(),
            "total_updates": self.total_updates,
        }

    # 【中文导读】恢复 Worker 网络、优化器、归一化器和更新计数。
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.encoder.load_state_dict(state["encoder"])
        self.target_encoder.load_state_dict(state["target_encoder"])
        self.actor.load_state_dict(state["actor"])
        self.target_actor.load_state_dict(state["target_actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        if self.transition_model is not None and state.get("transition_model") is not None:
            self.transition_model.load_state_dict(state["transition_model"])
        self.encoder_optim.load_state_dict(state["encoder_optim"])
        self.actor_optim.load_state_dict(state["actor_optim"])
        self.critic_optim.load_state_dict(state["critic_optim"])
        if self.transition_optim is not None and state.get("transition_optim") is not None:
            self.transition_optim.load_state_dict(state["transition_optim"])
        move_optimizer_state(self.encoder_optim, self.device)
        move_optimizer_state(self.actor_optim, self.device)
        move_optimizer_state(self.critic_optim, self.device)
        if self.transition_optim is not None:
            move_optimizer_state(self.transition_optim, self.device)
        self.normalizer.load_state_dict(state["normalizer"])
        self.total_updates = int(state.get("total_updates", 0))


# 【中文导读】把 Manager、Slow Worker、Fast Worker 作为一个整体传递和保存。
@dataclass
class AgentBundle:
    manager: ManagerTD3
    slow: WorkerTD3
    fast: WorkerTD3


# 【中文导读】按环境真实观测/动作维度构建三层智能体，并按阶段选择 Worker 学习率。
def build_agents(env: ElectricGasMultiScaleEnv, cfg: TrainConfig, device: torch.device) -> AgentBundle:
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：按真实 observation/action 维度创建三层智能体。
    
    输入：环境、TrainConfig、device。
    
    输出：AgentBundle。
    
    核心步骤：reset 环境、构造三层 observation、实例化 Manager/Slow/Fast。
    
    强化学习含义：网络输入维度必须与环境契约同步。
    
    【容易混淆】checkpoint schema 不匹配时不能靠填零继续。
    """

    obs, _ = env.reset(seed=cfg.seed)
    builder = ObservationBuilder(env, cfg.manager_interval)
    manager_obs = builder.manager_obs(obs)
    fast_obs = builder.fast_obs(0, obs)
    slow_obs = builder.slow_obs(obs)
    slow_lr = cfg.joint_worker_lr if cfg.training_stage == "joint_finetune" else cfg.slow_lr
    fast_lr = cfg.joint_worker_lr if cfg.training_stage == "joint_finetune" else cfg.fast_lr
    manager = ManagerTD3(manager_obs.size, cfg, device)
    slow = WorkerTD3("slow", slow_obs.size, env.slow_action_dim, cfg.slow_latent_dim, slow_lr, cfg, device)
    fast = WorkerTD3("fast", fast_obs.size, env.fast_action_dim, cfg.fast_latent_dim, fast_lr, cfg, device)
    return AgentBundle(manager=manager, slow=slow, fast=fast)


# =============================================================================
# Reward shaping
# =============================================================================
#
# 【模块说明：安全奖励】把环境分量组合为全局安全、角色特定、latent 方向、
# 物理目标进展、投影惩罚和动作正则。没有电价、气价或套利项。Slow Worker 虽控制慢设备，
# 仍必须获得电—气全局安全信用。
#


GLOBAL_SAFETY_COMPONENTS = (
    "voltage_deviation", "voltage_violation", "line_overload",
    "soc_soft", "terminal_soc", "gas_pressure_deviation", "gas_pressure_violation",
    "pipe_velocity_violation", "source_capacity_violation", "solver_failure",
)
# Role-specific groups are deliberately disjoint from GLOBAL_SAFETY_COMPONENTS.
# The earlier overlapping groups counted voltage terms twice for Fast and gas/SOC
# terms twice for Slow, so nominal 0.5/0.5 weights did not represent real credit.
FAST_COMPONENTS = ("power_loss", "renewable_curtailment")
SLOW_COMPONENTS = (
    "compressor_energy", "ess_action_change", "gfg_action_change", "p2g_action_change",
)
MANAGER_SAFETY_COMPONENTS = GLOBAL_SAFETY_COMPONENTS + FAST_COMPONENTS + SLOW_COMPONENTS


# Calibrated from the completed 2026-07-14 four-stage run. References are costs
# per fast step, not economic prices. The transform remains monotone, therefore
# a larger safety violation always produces a larger penalty without allowing a
# single raw 1e5-weight component to erase every other learning signal.
SAFETY_COMPONENT_REFERENCE_COSTS: Dict[str, float] = {
    "voltage_deviation": 25.0,
    "voltage_violation": 100.0,
    "line_overload": 10.0,
    "soc_soft": 0.10,
    "terminal_soc": 10.0,
    "gas_pressure_deviation": 5.0,
    "gas_pressure_violation": 25.0,
    "pipe_velocity_violation": 10.0,
    "source_capacity_violation": 25.0,
    "solver_failure": 5_000.0,
    "power_loss": 0.50,
    "renewable_curtailment": 5.0,
    "compressor_energy": 0.50,
    "ess_action_change": 1.0,
    "gfg_action_change": 1.0,
    "p2g_action_change": 1.0,
}

SAFETY_COMPONENT_IMPORTANCE: Dict[str, float] = {
    "voltage_deviation": 1.0,
    "voltage_violation": 10.0,
    "line_overload": 10.0,
    "soc_soft": 2.0,
    "terminal_soc": 5.0,
    "gas_pressure_deviation": 1.0,
    "gas_pressure_violation": 10.0,
    "pipe_velocity_violation": 10.0,
    "source_capacity_violation": 10.0,
    "solver_failure": 25.0,
    "power_loss": 1.0,
    "renewable_curtailment": 2.0,
    "compressor_energy": 3.0,
    "ess_action_change": 2.0,
    "gfg_action_change": 2.0,
    "p2g_action_change": 2.0,
}


def normalized_safety_component_cost(name: str, value: float, cfg: TrainConfig) -> Tuple[float, bool]:
    """Return a finite monotone dimensionless safety cost and cap diagnostic."""

    raw = float(value)
    if not np.isfinite(raw):
        raise FloatingPointError(f"Non-finite reward component {name}={raw}")
    mode = str(getattr(cfg, "reward_component_transform", "log1p_reference"))
    if mode == "none":
        transformed = raw
    elif mode == "log1p_reference":
        reference = float(SAFETY_COMPONENT_REFERENCE_COSTS.get(name, 1.0))
        importance = float(SAFETY_COMPONENT_IMPORTANCE.get(name, 1.0))
        transformed = math.copysign(
            importance * math.log1p(abs(raw) / max(reference, 1e-12)), raw
        )
    else:
        raise ValueError(f"Unknown reward_component_transform={mode!r}")
    cap = float(getattr(cfg, "worker_component_clip_abs", 0.0))
    used = float(np.clip(transformed, -cap, cap)) if cap > 0.0 else float(transformed)
    return used, bool(abs(used - transformed) > 1e-12)


# 【中文导读】从环境成本字典中选取本 Worker 负责的物理成本并取负作为外在奖励。
def external_reward_from_components(info: Dict[str, Any], keys: Tuple[str, ...]) -> float:
    """从环境 info 中抽取指定成本分量，并转成奖励符号。"""

    comps = info.get("reward_components", {})
    return -float(sum(float(comps.get(k, 0.0)) for k in keys))


def worker_safety_reward_from_components(info: Dict[str, Any], role: str,
                                         cfg: TrainConfig) -> Dict[str, Any]:
    """Build global and role-specific safety rewards without economic terms."""

    components = info.get("reward_components", {})
    role_keys = FAST_COMPONENTS if role == "fast" else SLOW_COMPONENTS
    normalized_components: Dict[str, float] = {}

    def group(keys: Tuple[str, ...]) -> Tuple[float, float, int]:
        raw_costs = np.asarray([float(components.get(name, 0.0)) for name in keys], dtype=float)
        used_costs: List[float] = []
        clipped_count = 0
        for name, raw_cost in zip(keys, raw_costs):
            used_cost, clipped = normalized_safety_component_cost(name, float(raw_cost), cfg)
            normalized_components[name] = used_cost
            used_costs.append(used_cost)
            clipped_count += int(clipped)
        return -float(np.sum(raw_costs)), -float(np.sum(used_costs)), clipped_count

    global_raw, global_used, global_clips = group(GLOBAL_SAFETY_COMPONENTS)
    role_raw, role_used, role_clips = group(role_keys)
    global_weight = float(getattr(cfg, f"{role}_global_safety_weight", 0.0))
    role_weight = float(getattr(cfg, f"{role}_role_specific_weight", 1.0))
    global_weighted = global_weight * global_used
    role_weighted = role_weight * role_used
    total = global_weighted + role_weighted
    raw_total = global_weight * global_raw + role_weight * role_raw
    return {
        "global_raw": global_raw,
        "global_used": global_used,
        "global_weighted": global_weighted,
        "role_raw": role_raw,
        "role_used": role_used,
        "role_weighted": role_weighted,
        "total": total,
        "raw_total": raw_total,
        "component_clipped": float(global_clips + role_clips > 0),
        "normalized_components": normalized_components,
    }


def manager_safety_reward_from_components(info: Dict[str, Any], cfg: TrainConfig) -> Dict[str, Any]:
    """Build the Manager's scale-calibrated global safety reward.

    Evaluation return remains the environment's raw safety return. Only the TD
    training reward is normalized, which prevents Manager SMDP segments from
    spending roughly half their samples at the replay clipping boundary.
    """

    components = info.get("reward_components", {})
    normalized: Dict[str, float] = {}
    raw_cost = 0.0
    clipped = 0
    for name in MANAGER_SAFETY_COMPONENTS:
        value = float(components.get(name, 0.0))
        raw_cost += value
        used, was_clipped = normalized_safety_component_cost(name, value, cfg)
        normalized[name] = used
        clipped += int(was_clipped)
    return {
        "reward": -float(sum(normalized.values())),
        "raw_reward": -raw_cost,
        "component_clipped": float(clipped > 0),
        "normalized_components": normalized,
    }


def apply_debug_terminal_soc_penalty(env: ElectricGasMultiScaleEnv, info: Dict[str, Any],
                                     reward: float, cfg: TrainConfig) -> float:
    """Add the environment's terminal-SOC safety cost to an early debug truncation."""

    if not bool(getattr(cfg, "debug_terminal_soc_penalty", True)):
        return float(reward)
    initial = np.asarray([item.soc_initial for item in ESS_CONFIGS], dtype=float)
    terminal_cost = float(env.config.reward.terminal_soc * np.sum(np.square(env.ess_soc - initial)))
    components = info.setdefault("reward_components", {})
    components["terminal_soc"] = float(components.get("terminal_soc", 0.0)) + terminal_cost
    info["debug_terminal_soc_cost"] = terminal_cost
    return float(reward) - terminal_cost


# 【中文导读】根据 raw 与 executed 动作的均方距离惩罚不可行动作请求。
def projection_penalty(raw_action: np.ndarray, executed_action: np.ndarray, scale: float) -> float:
    """安全投影越大，说明 Actor 越想做不可行动作，因此给负奖励。"""

    return -float(scale * np.mean(np.square(np.asarray(raw_action) - np.asarray(executed_action))))


# 【中文导读】比较相邻快观测的电压偏差和线路过载，奖励物理状态改善。
def fast_physical_progress(obs: np.ndarray, next_obs: np.ndarray, goal: np.ndarray) -> float:
    del goal
    if obs.size < 65 or next_obs.size < 65:
        return 0.0
    voltage_now = float(np.mean(np.abs(obs[:33])))
    voltage_next = float(np.mean(np.abs(next_obs[:33])))
    line_now = float(np.mean(np.maximum(obs[33:65] - 1.0, 0.0)))
    line_next = float(np.mean(np.maximum(next_obs[33:65] - 1.0, 0.0)))
    return (voltage_now + line_now) - (voltage_next + line_next)


# 【中文导读】比较 SOC 参考跟踪和气压越界程度，奖励慢设备带来的改善。
def slow_physical_goal_features(obs: np.ndarray) -> np.ndarray:
    """Map the centralized compact Slow observation to goal[24:32] space."""

    values = np.asarray(obs, dtype=np.float32).reshape(-1)
    features = np.zeros(GOAL_PHYSICAL_DIM, dtype=np.float32)
    if values.size != SLOW_OBSERVATION_LAYOUT.dimension:
        return features

    def unit_to_bipolar(value: float) -> float:
        return float(2.0 * np.clip(value, 0.0, 1.0) - 1.0)

    slices = SLOW_OBSERVATION_LAYOUT.slices
    mean_soc = float(np.mean(values[slices["soc"]]))
    pressure_rms_bar = float(values[slices["gas_pressure_summary"]][2])
    source_loading = values[slices["source_utilization"]]
    max_source_loading = float(np.max(source_loading)) if source_loading.size else 0.0
    linepack_scaled = float(values[slices["linepack"]][0])
    features[4] = unit_to_bipolar(mean_soc)
    features[5] = unit_to_bipolar(pressure_rms_bar / 2.5)
    features[6] = unit_to_bipolar(max_source_loading)
    features[7] = float(np.clip(linepack_scaled, -1.0, 1.0))
    return np.clip(features, -1.0, 1.0)


def slow_physical_progress(obs: np.ndarray, next_obs: np.ndarray, goal: np.ndarray) -> float:
    """Measure progress using named compact-state fields, never positional magic slices."""

    physical_goal = np.asarray(goal, dtype=np.float32)[24:32]
    before = slow_physical_goal_features(obs)[4:]
    after = slow_physical_goal_features(next_obs)[4:]
    target = physical_goal[4:]
    before_distance = float(np.sqrt(np.mean(np.square(before - target)) + 1e-12))
    after_distance = float(np.sqrt(np.mean(np.square(after - target)) + 1e-12))
    return before_distance - after_distance


# 【中文导读】按配置合并外在奖励、latent 方向奖励、物理进展和投影惩罚。
def build_worker_reward(external: float, latent: float, physical: float, proj: float,
                        cfg: TrainConfig) -> float:
    """把 Worker 的外在奖励、内在奖励和安全投影惩罚合成单个标量。"""

    reference = max(abs(float(external)), float(getattr(cfg, "shaping_reference_floor", 1.0)))
    return (cfg.alpha_external * external +
            cfg.beta_latent * reference * math.tanh(float(latent)) +
            cfg.beta_physical * reference * math.tanh(float(physical)) +
            proj)


# 【中文导读】裁剪极端奖励，限制少数求解失败样本对 Q 回归的支配。
def clip_reward_value(value: float, clip_abs: float) -> Tuple[float, bool]:
    """限制写入ReplayBuffer的奖励幅度，避免少数灾难回报主导Critic。"""
    if clip_abs <= 0.0:
        return float(value), False
    clipped = float(np.clip(value, -clip_abs, clip_abs))
    return clipped, bool(abs(clipped - float(value)) > 1e-9)


def agent_parameter_snapshot(agent: Any) -> List[torch.Tensor]:
    """Capture every online/target model parameter for stage-freeze auditing."""

    snapshots: List[torch.Tensor] = []
    for name in (
        "encoder", "target_encoder", "actor", "target_actor", "critic", "target_critic",
        "transition_model",
    ):
        module = getattr(agent, name, None)
        if module is not None:
            snapshots.extend(parameter.detach().cpu().clone() for parameter in module.parameters())
    return snapshots


def agent_parameter_change(agent: Any, before: Sequence[torch.Tensor]) -> Tuple[float, float]:
    """Return L2 and max-absolute parameter change since a snapshot."""

    current: List[torch.Tensor] = []
    for name in (
        "encoder", "target_encoder", "actor", "target_actor", "critic", "target_critic",
        "transition_model",
    ):
        module = getattr(agent, name, None)
        if module is not None:
            current.extend(parameter.detach().cpu() for parameter in module.parameters())
    if len(current) != len(before):
        raise RuntimeError(f"Agent parameter structure changed: before={len(before)}, after={len(current)}")
    squared_sum = 0.0
    maximum = 0.0
    for old, new in zip(before, current):
        delta = new.to(dtype=torch.float64) - old.to(dtype=torch.float64)
        squared_sum += float(torch.sum(delta * delta))
        maximum = max(maximum, float(torch.max(torch.abs(delta))))
    return math.sqrt(squared_sum), maximum


# 【中文导读】按 episode 线性衰减探索噪声。
def scheduled_noise(initial: float, minimum: float, episode: int, decay_episodes: int) -> float:
    """线性退火探索噪声；长训后期减少由噪声造成的动作尖峰。"""
    if decay_episodes <= 0:
        return float(initial)
    fraction = min(max(float(episode) / float(decay_episodes), 0.0), 1.0)
    return float(initial + fraction * (minimum - initial))


def warmup_actor_blend(count: int, warmup_count: int, blend_fraction: float) -> float:
    """Ramp Actor participation over the final part of safety warm-up."""

    if warmup_count <= 0 or count >= warmup_count:
        return 1.0
    blend_steps = max(int(math.ceil(warmup_count * float(blend_fraction))), 1)
    blend_start = max(warmup_count - blend_steps, 0)
    return float(np.clip((count - blend_start) / blend_steps, 0.0, 1.0))


# =============================================================================
# Action helpers and stage control
# =============================================================================
#
# 【模块说明：动作保护与阶段控制】规则动作和 ESS guard 只改变
# 送给环境的请求，不改变 raw Actor 动作的定义。stage_flags 决定四阶段中谁训练、谁冻结。
#


# 【中文导读】把物理压缩比反映射到 [-1,1] 动作空间。
def normalized_compressor_ratio(env: ElectricGasMultiScaleEnv, index: int, ratio: float) -> float:
    c = COMPRESSOR_CONFIGS[index]
    span = max(c.max_pressure_ratio - c.min_pressure_ratio, 1e-9)
    return float(np.clip(2.0 * (ratio - c.min_pressure_ratio) / span - 1.0, -1.0, 1.0))


# 【中文导读】计算慢动作保持期间仍满足 SOC 边界的 ESS 归一化功率上下限。
def ess_normalized_power_bounds(env: ElectricGasMultiScaleEnv, horizon_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return normalized ESS action bounds that remain feasible over a hold horizon."""
    from electric_gas_microgrid_single import ESS_CONFIGS

    horizon = max(int(horizon_steps), 1)
    dt_hours = float(env.config.time.dt_hours) * horizon
    lower: List[float] = []
    upper: List[float] = []
    for soc, ess in zip(np.asarray(env.ess_soc, dtype=float), ESS_CONFIGS):
        max_charge_by_soc = (ess.soc_max - float(soc)) * ess.capacity_mwh / (
            max(ess.eta_charge, 1e-9) * dt_hours
        )
        min_discharge_by_soc = (ess.soc_min - float(soc)) * ess.capacity_mwh * ess.eta_discharge / dt_hours
        lower_p = max(-ess.max_p_mw, min_discharge_by_soc)
        upper_p = min(ess.max_p_mw, max_charge_by_soc)
        if lower_p > upper_p:
            lower_p = upper_p = 0.0
        scale = max(ess.max_p_mw, 1e-9)
        lower.append(float(np.clip(lower_p / scale, -1.0, 1.0)))
        upper.append(float(np.clip(upper_p / scale, -1.0, 1.0)))
    return np.asarray(lower, dtype=np.float32), np.asarray(upper, dtype=np.float32)


# 【中文导读】在进入环境前对 ESS 动作做跨保持区间的安全保护。
def apply_ess_action_guard(env: ElectricGasMultiScaleEnv, slow_action: np.ndarray,
                           cfg: TrainConfig, horizon_steps: int) -> Tuple[np.ndarray, float]:
    """Clip ESS slow-action entries to SOC-feasible bounds before they reach the environment."""
    action = np.asarray(slow_action, dtype=np.float32).copy()
    if not cfg.use_ess_action_guard or env.n_ess <= 0:
        return np.clip(action, -1.0, 1.0), 0.0
    lower, upper = ess_normalized_power_bounds(env, horizon_steps)
    before = action[:env.n_ess].copy()
    action[:env.n_ess] = np.clip(before, lower, upper)
    action = np.clip(action, -1.0, 1.0)
    return action, float(np.linalg.norm(before - action[:env.n_ess]))


# 【中文导读】Fast Worker 预训练时提供固定慢设备基线控制。
def rule_slow_action(env: ElectricGasMultiScaleEnv) -> np.ndarray:
    action = np.zeros(env.slow_action_dim, dtype=np.float32)
    cursor = 0
    action[cursor:cursor + env.n_ess] = 0.0
    cursor += env.n_ess
    action[cursor:cursor + env.n_gfg] = -0.40
    cursor += env.n_gfg
    action[cursor:cursor + env.n_p2g] = -0.60
    cursor += env.n_p2g
    for action_pos, comp_idx in enumerate(CONTROLLED_COMPRESSOR_INDICES):
        comp = COMPRESSOR_CONFIGS[comp_idx]
        action[cursor + action_pos] = normalized_compressor_ratio(env, comp_idx, comp.initial_pressure_ratio)
    return action


# 【中文导读】Worker 预训练时提供固定且合法的组合式 goal。
def fixed_manager_goal() -> np.ndarray:
    goal = np.zeros(GOAL_DIM, dtype=np.float32)
    goal[0] = 1.0
    goal[8] = 1.0
    goal[16] = 1.0
    goal[24] = 0.0
    return normalize_goal_np(goal)


# 【中文导读】决定当前阶段哪些智能体执行参数更新。
def stage_flags(stage: str) -> Dict[str, bool]:
    if stage == "fast_pretrain":
        return {"manager": False, "slow": False, "fast": True}
    if stage == "slow_pretrain":
        return {"manager": False, "slow": True, "fast": False}
    if stage == "manager_train":
        return {"manager": True, "slow": False, "fast": False}
    if stage == "joint_finetune":
        return {"manager": True, "slow": True, "fast": True}
    if stage == "all":
        return {"manager": True, "slow": True, "fast": True}
    raise ValueError(f"Unknown training stage: {stage}")


# 【中文导读】把环境异常转换为带失败惩罚的 truncated transition，避免训练进程直接退出。
def safe_env_step(env: ElectricGasMultiScaleEnv, action: np.ndarray, last_obs: np.ndarray
                  ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
    try:
        return env.step(action)
    except Exception as exc:
        if getattr(env, "unexpected_env_exception_policy", "raise") == "raise":
            raise
        LOGGER.warning("Environment step failed and was converted to a failed transition: %r", exc)
        info = {
            "solver_failed": True,
            "converged": False,
            "exception": repr(exc),
            "raw_action": action.copy(),
            "applied_action": np.clip(action.copy(), -1.0, 1.0),
            "action_projection_magnitude": 0.0,
            "reward_components": {"solver_failure": 5000.0},
            "constraint_metrics": {},
            "slow_action_applied": False,
            "gas_solve_count": 0,
        }
        return last_obs.copy(), -5000.0, False, True, info


# =============================================================================
# Checkpointing
# =============================================================================
#
# 【模块说明：checkpoint 与安全评估】统一保存网络、优化器、Replay、normalizer、
# RNG、计数器和最佳评价状态；is_feasible 先检查安全阈值，再允许比较回报。旧 observation
# schema 与当前维度不一致时不能 strict resume。
#


# 【中文导读】保存三层在线/目标网络、优化器、归一化器与训练元数据。
def checkpoint_metadata(agents: AgentBundle) -> Dict[str, Any]:
    return {
        "env_model_version": ENV_MODEL_VERSION,
        "slow_safety_schema_version": SLOW_SAFETY_SCHEMA_VERSION,
        "slow_observation_fields": tuple(name for name, _ in SLOW_OBSERVATION_LAYOUT.fields),
        "critic_action_semantics": "raw_request_action",
        "executed_action_semantics": (
            "environment_projected_action_for_transition_imitation_and_diagnostics"
        ),
        "manager_observation_dim": agents.manager.obs_dim,
        "slow_observation_dim": agents.slow.obs_dim,
        "fast_observation_dim": agents.fast.obs_dim,
        "slow_action_dim": agents.slow.action_dim,
        "fast_action_dim": agents.fast.action_dim,
        "total_action_dim": agents.slow.action_dim + agents.fast.action_dim,
        "goal_dim": GOAL_DIM,
        "n_controlled_compressors": len(CONTROLLED_COMPRESSOR_INDICES),
        "n_total_compressors": len(COMPRESSOR_CONFIGS),
    }


def is_feasible(metrics: Mapping[str, Any], cfg: TrainConfig) -> Tuple[bool, List[str]]:
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：统一判断评估结果是否满足全部安全阈值。
    
    输入：metrics 与配置阈值。
    
    输出：可行布尔值及逐项失败原因。
    
    核心步骤：检查求解成功率、SOC/硬约束、电压/气压及其他配置阈值。
    
    强化学习含义：安全模型选择先比较可行性。
    
    【容易混淆】高 return 不能覆盖不可行约束。
    """

    checks = (
        ("solver_failures", "<=", 0.0),
        ("power_solver_success_rate", ">=", float(getattr(cfg, "min_power_success_rate", 0.999))),
        ("gas_solver_success_rate", ">=", float(getattr(cfg, "min_gas_success_rate", 0.999))),
        ("soc_violation_rate", "<=", float(getattr(cfg, "max_soc_violation_rate", 0.0))),
        ("voltage_violation_rate", "<=", float(getattr(cfg, "max_hard_constraint_violation_rate", 0.0))),
        ("gas_pressure_violation_rate", "<=", float(getattr(cfg, "max_hard_constraint_violation_rate", 0.0))),
        ("line_overload_rate", "<=", float(getattr(cfg, "max_hard_constraint_violation_rate", 0.0))),
        ("pipe_velocity_violation_rate", "<=", float(getattr(cfg, "max_hard_constraint_violation_rate", 0.0))),
        ("source_capacity_violation_rate", "<=", float(getattr(cfg, "max_hard_constraint_violation_rate", 0.0))),
        ("mean_voltage_rms_deviation_pu", "<=", float(getattr(cfg, "max_voltage_rms_deviation_pu", 0.05))),
        ("mean_gas_pressure_rms_deviation_bar", "<=", float(getattr(cfg, "max_gas_pressure_rms_deviation_bar", 0.5))),
    )
    reasons: List[str] = []
    for name, operator, threshold in checks:
        if name not in metrics:
            reasons.append(f"missing:{name}")
            continue
        value = float(metrics[name])
        if not np.isfinite(value):
            reasons.append(f"nonfinite:{name}={value!r}")
        elif operator == "<=" and value > threshold:
            reasons.append(f"{name}={value:.8g}>{threshold:.8g}")
        elif operator == ">=" and value < threshold:
            reasons.append(f"{name}={value:.8g}<{threshold:.8g}")
    return not reasons, reasons


def build_evaluation_metric_state(stats: Mapping[str, Any], cfg: TrainConfig) -> Dict[str, Any]:
    """验证评估统计并构造可解释、阶段感知的安全指标状态。"""

    if cfg.best_model_metric == "return":
        value = float(stats.get("mean_return", float("nan")))
        if not np.isfinite(value):
            raise ValueError(f"mean_return={value!r} must be present and finite")
        return {"metric": "return", "stage": cfg.training_stage, "feasible": True,
                "solver_failures": 0.0, "hard_violation_rate": 0.0,
                "maximum_violation": 0.0, "normalized_constraint_cost": 0.0,
                "constraint_score": 0.0, "mean_return": value, "raw_stats": dict(stats)}

    rate_names = (
        "voltage_violation_rate", "gas_pressure_violation_rate", "line_overload_rate",
        "pipe_velocity_violation_rate", "source_capacity_violation_rate", "soc_violation_rate",
    )
    maximum_names = (
        "max_voltage_violation", "max_pressure_violation", "max_line_overload",
        "max_pipe_velocity_violation", "max_source_capacity_violation",
    )
    cost_names = (
        "voltage_violation_cost_per_step", "gas_pressure_violation_cost_per_step",
        "line_overload_cost_per_step", "pipe_velocity_violation_cost_per_step",
        "source_capacity_violation_cost_per_step",
    )
    required = [
        "mean_return", "solver_failures", "power_solver_success_rate",
        "gas_solver_success_rate", *rate_names, *maximum_names, *cost_names,
    ]
    missing = [name for name in required if name not in stats]
    if missing:
        raise ValueError(
            f"Evaluation stats missing required safety fields for stage={cfg.training_stage!r}: {missing}"
        )
    values = {name: float(stats[name]) for name in required}
    invalid = {name: value for name, value in values.items() if not np.isfinite(value)}
    if invalid:
        raise ValueError(f"Evaluation safety metrics must be finite; invalid={invalid}")
    for name in ("solver_failures", *rate_names, *maximum_names, *cost_names):
        if name in values and values[name] < 0.0:
            raise ValueError(f"{name}={values[name]!r} must be non-negative")
    for name in ("power_solver_success_rate", "gas_solver_success_rate", *rate_names):
        if name in values and not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"{name}={values[name]!r} must be in [0,1]")

    power_shortfall = max(0.0, float(getattr(cfg, "min_power_success_rate", 0.999))
                              - values["power_solver_success_rate"])
    gas_shortfall = max(0.0, float(getattr(cfg, "min_gas_success_rate", 0.999))
                            - values["gas_solver_success_rate"])
    violation_threshold = float(getattr(cfg, "max_hard_constraint_violation_rate", 0.0))
    hard_violation_rate = max(values[name] for name in rate_names)
    maximum_violation = max(values[name] for name in maximum_names)
    normalized_constraint_cost = float(np.mean([values[name] for name in cost_names]))
    constraint_score = float(np.mean([
        power_shortfall, gas_shortfall, hard_violation_rate,
        maximum_violation, normalized_constraint_cost,
    ]))
    feasible, feasibility_reasons = is_feasible(stats, cfg)
    return {
        "metric": cfg.best_model_metric,
        "stage": cfg.training_stage,
        "feasible": feasible,
        "feasibility_reasons": feasibility_reasons,
        "solver_failures": values["solver_failures"],
        "hard_violation_rate": hard_violation_rate,
        "maximum_violation": maximum_violation,
        "normalized_constraint_cost": normalized_constraint_cost,
        "constraint_score": constraint_score,
        "mean_return": values["mean_return"],
        "raw_stats": dict(stats),
    }


def compare_evaluation_metric(current: Mapping[str, Any], previous: Mapping[str, Any],
                              tolerance: float = 1e-6) -> int:
    """返回 1/0/-1；容差内的安全浮点差异交给下一指标和回报决定。"""

    if bool(current["feasible"]) != bool(previous["feasible"]):
        return 1 if bool(current["feasible"]) else -1
    for name in ("solver_failures", "hard_violation_rate", "maximum_violation",
                 "normalized_constraint_cost", "constraint_score"):
        difference = float(current[name]) - float(previous[name])
        if abs(difference) > tolerance:
            return 1 if difference < 0.0 else -1
    return_difference = float(current["mean_return"]) - float(previous["mean_return"])
    if abs(return_difference) <= tolerance:
        return 0
    return 1 if return_difference > 0.0 else -1


def evaluation_metric_key(stats: Mapping[str, Any], cfg: TrainConfig) -> Tuple[float, ...]:
    state = build_evaluation_metric_state(stats, cfg)
    return (float(state["feasible"]), -float(state["solver_failures"]),
            -float(state["hard_violation_rate"]), -float(state["maximum_violation"]),
            -float(state["normalized_constraint_cost"]), -float(state["constraint_score"]),
            float(state["mean_return"]))


def metric_state_from_evaluation(stats: Mapping[str, Any], cfg: TrainConfig) -> Dict[str, Any]:
    return build_evaluation_metric_state(stats, cfg)


def is_better_evaluation(stats: Mapping[str, Any], best_metric_state: Optional[Mapping[str, Any]],
                         cfg: TrainConfig) -> bool:
    current = build_evaluation_metric_state(stats, cfg)
    if not best_metric_state:
        return True
    required = ("feasible", "solver_failures", "hard_violation_rate", "maximum_violation",
                "normalized_constraint_cost", "constraint_score", "mean_return")
    if any(name not in best_metric_state for name in required):
        warnings.warn(
            "Legacy best_metric_state has no explainable safety fields; the next valid evaluation replaces it.",
            RuntimeWarning, stacklevel=2,
        )
        return True
    tolerance = float(getattr(cfg, "metric_comparison_tolerance", 1e-6))
    return compare_evaluation_metric(current, best_metric_state, tolerance) > 0


def save_checkpoint(path: Path, cfg: TrainConfig, agents: AgentBundle, episode: int,
                    global_step: int, best_return: float,
                    best_metric_state: Optional[Mapping[str, Any]] = None,
                    best_evaluation_stats: Optional[Mapping[str, Any]] = None,
                    fast_replay: Optional[FastReplayBuffer] = None,
                    slow_replay: Optional[SlowReplayBuffer] = None,
                    manager_replay: Optional[ManagerReplayBuffer] = None,
                    next_episode: Optional[int] = None,
                    checkpoint_kind: str = "full_resume") -> None:
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：保存策略或完整训练状态。
    
    输入：路径、配置、智能体、Replay、计数器、最佳评价等。
    
    输出：磁盘 checkpoint。
    
    核心步骤：序列化网络、target、优化器、normalizer、Replay、RNG 和 metadata。
    
    强化学习含义：full resume 要求训练轨迹可严格延续。
    
    【容易混淆】policy checkpoint 与 full resume checkpoint 的用途不同。
    """
    if checkpoint_kind not in ("lightweight", "full_resume"):
        raise ValueError(f"checkpoint_kind={checkpoint_kind!r} must be lightweight or full_resume")
    path.parent.mkdir(parents=True, exist_ok=True)
    full = checkpoint_kind == "full_resume"
    lightweight_keys = {
        "role", "encoder", "target_encoder", "actor", "target_actor", "critic",
        "target_critic", "transition_model", "normalizer", "nonfinite_batch_count",
    }

    def agent_state(agent: Any) -> Dict[str, Any]:
        state = agent.state_dict()
        return state if full else {key: value for key, value in state.items() if key in lightweight_keys}

    payload = {
        **checkpoint_metadata(agents),
        "checkpoint_kind": checkpoint_kind,
        "observation_dim": agents.manager.obs_dim,
        "config": asdict(cfg),
        "manager": agent_state(agents.manager),
        "slow": agent_state(agents.slow),
        "fast": agent_state(agents.fast),
        "episode": episode,
        "global_step": global_step,
        "best_return": best_return,
        "best_metric_state": dict(best_metric_state or {}),
        "best_evaluation_stats": dict(best_evaluation_stats or {}),
    }
    if full:
        payload["next_episode"] = int(episode + 1 if next_episode is None else next_episode)
        payload["rng_state"] = capture_rng_state()
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
        message = "Replay checkpointing is disabled; strict resume is unavailable"
        if cfg.strict_resume_required:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    temp_path = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    started = time.perf_counter()
    try:
        torch.save(payload, str(temp_path))
        os.replace(str(temp_path), str(path))
    finally:
        if temp_path.exists():
            temp_path.unlink()
    LOGGER.info("Checkpoint saved path=%s kind=%s size_bytes=%s elapsed_seconds=%.3f",
                path, checkpoint_kind, path.stat().st_size, time.perf_counter() - started)


def validate_checkpoint_compatibility(
    payload: Dict[str, Any], agents: AgentBundle, load_mode: str = "resume"
) -> None:
    expected = checkpoint_metadata(agents)
    problems: List[str] = []
    migration_notes: List[str] = []
    semantic_keys = {
        "env_model_version", "slow_safety_schema_version", "slow_observation_fields",
        "critic_action_semantics", "executed_action_semantics",
    }
    for key, expected_value in expected.items():
        actual = payload.get(key)
        if key == "slow_observation_fields" and actual is not None:
            actual = tuple(actual)
        if key not in payload:
            detail = f"missing {key}"
            if load_mode != "resume" and key in semantic_keys:
                migration_notes.append(detail)
            else:
                problems.append(detail)
            continue
        if actual != expected_value:
            detail = f"{key} changed: expected {expected_value!r}, got {actual!r}"
            if load_mode != "resume" and key in semantic_keys:
                migration_notes.append(detail)
            else:
                problems.append(detail)
    if problems:
        raise ValueError(
            f"Checkpoint is incompatible with {ENV_MODEL_VERSION}: " + " / ".join(problems)
        )
    if migration_notes:
        warnings.warn(
            f"Explicit checkpoint {load_mode} migration accepted; strict resume is unavailable: "
            + " / ".join(migration_notes),
            RuntimeWarning,
            stacklevel=2,
        )


# 【中文导读】只加载 Encoder、Actor、归一化器等策略相关状态并重置目标网络。
def load_agent_policy_state(agent: Any, state: Dict[str, Any]) -> None:
    agent.encoder.load_state_dict(state["encoder"])
    agent.actor.load_state_dict(state["actor"])
    hard_update(agent.target_encoder, agent.encoder)
    hard_update(agent.target_actor, agent.actor)
    hard_update(agent.target_critic, agent.critic)
    if hasattr(agent, "normalizer") and "normalizer" in state:
        agent.normalizer.load_state_dict(state["normalizer"])
    agent.total_updates = 0
    if hasattr(agent, "critic_updates"):
        agent.critic_updates = 0
    if hasattr(agent, "actor_updates"):
        agent.actor_updates = 0
    if hasattr(agent, "_last_update_insertion_id"):
        agent._last_update_insertion_id = 0


def load_agent_stage_transfer_state(agent: Any, state: Dict[str, Any]) -> None:
    """阶段传递保留表示、策略和 Critic 权重，但使用当前阶段新优化器和新 Replay。"""

    for name in ("encoder", "target_encoder", "actor", "target_actor", "critic", "target_critic"):
        getattr(agent, name).load_state_dict(state[name])
    transition_state = state.get("transition_model")
    if getattr(agent, "transition_model", None) is not None and transition_state is not None:
        agent.transition_model.load_state_dict(transition_state)
    if "normalizer" in state:
        agent.normalizer.load_state_dict(state["normalizer"])
    agent.total_updates = 0
    if hasattr(agent, "critic_updates"):
        agent.critic_updates = 0
    if hasattr(agent, "actor_updates"):
        agent.actor_updates = 0
    if hasattr(agent, "_last_update_insertion_id"):
        agent._last_update_insertion_id = 0


# 【中文导读】按完整恢复或仅策略恢复两种模式载入 checkpoint。
def load_checkpoint(path: str, agents: AgentBundle, map_location: torch.device,
                    policy_only: bool = False, mode: Optional[str] = None,
                    fast_replay: Optional[FastReplayBuffer] = None,
                    slow_replay: Optional[SlowReplayBuffer] = None,
                    manager_replay: Optional[ManagerReplayBuffer] = None) -> Dict[str, Any]:
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：按 resume/stage_transfer/policy_only 语义加载 checkpoint。
    
    输入：路径、智能体、设备和加载模式。
    
    输出：恢复信息字典。
    
    核心步骤：先校验 schema/维度，再按模式恢复允许的状态。
    
    强化学习含义：三种模式对应继续训练、跨阶段迁移和仅策略评估。
    
    【容易混淆】缺 Replay/优化器/RNG 时不得静默 strict resume。
    """
    load_mode = mode or ("policy_only" if policy_only else "resume")
    payload = trusted_torch_load(path, map_location=map_location)
    validate_checkpoint_compatibility(payload, agents, load_mode=load_mode)
    cfg = agents.fast.cfg
    saved_stage = payload.get("training_stage", payload.get("config", {}).get("training_stage"))
    if load_mode == "resume" and saved_stage is not None and saved_stage != cfg.training_stage:
        raise ValueError(
            f"Cannot resume checkpoint from training_stage={saved_stage!r} into "
            f"training_stage={cfg.training_stage!r}; use checkpoint_load_mode='stage_transfer'."
        )
    if load_mode == "policy_only":
        load_agent_policy_state(agents.manager, payload["manager"])
        load_agent_policy_state(agents.slow, payload["slow"])
        load_agent_policy_state(agents.fast, payload["fast"])
    elif load_mode == "stage_transfer":
        load_agent_stage_transfer_state(agents.manager, payload["manager"])
        load_agent_stage_transfer_state(agents.slow, payload["slow"])
        load_agent_stage_transfer_state(agents.fast, payload["fast"])
    elif load_mode == "resume":
        required = ("fast_replay", "slow_replay", "manager_replay", "rng_state",
                    "next_episode", "global_step", "manager", "slow", "fast")
        missing = [name for name in required if name not in payload]
        for role in ("manager", "slow", "fast"):
            state = payload.get(role, {})
            for optimizer_name in ("encoder_optim", "actor_optim", "critic_optim"):
                if optimizer_name not in state:
                    missing.append(f"{role}.{optimizer_name}")
        payload["strict_resume_restored"] = not missing
        payload["resume_missing_components"] = sorted(set(missing))
        if missing and cfg.strict_resume_required:
            raise ValueError("Strict resume checkpoint is missing: " + ", ".join(sorted(set(missing))))
        if missing:
            warnings.warn("Partial resume missing: " + ", ".join(sorted(set(missing))),
                          RuntimeWarning, stacklevel=2)
        for role, agent in (("manager", agents.manager), ("slow", agents.slow), ("fast", agents.fast)):
            role_state = payload[role]
            optimizer_names = ["encoder_optim", "actor_optim", "critic_optim"]
            if role != "manager" and role_state.get("transition_model") is not None:
                optimizer_names.append("transition_optim")
            if all(name in role_state for name in optimizer_names):
                agent.load_state_dict(role_state)
            else:
                load_agent_stage_transfer_state(agent, role_state)
        replay_items = (
            ("fast_replay", fast_replay), ("slow_replay", slow_replay),
            ("manager_replay", manager_replay),
        )
        restored_all_replay = True
        for key, replay in replay_items:
            if replay is None or key not in payload:
                restored_all_replay = False
                continue
            replay.load_state_dict(payload[key])
        if not restored_all_replay:
            warnings.warn(
                "Checkpoint has no complete Replay state or buffers were not supplied; resume is not strict.",
                RuntimeWarning, stacklevel=2,
            )
        if "rng_state" in payload:
            restore_rng_state(payload["rng_state"])
        else:
            warnings.warn("Legacy checkpoint has no rng_state; stochastic continuation is not exact.",
                          RuntimeWarning, stacklevel=2)
        for agent, replay in ((agents.manager, manager_replay), (agents.slow, slow_replay),
                              (agents.fast, fast_replay)):
            if hasattr(agent, "_last_update_insertion_id"):
                agent._last_update_insertion_id = int(
                    replay.total_insertions if restored_all_replay and replay is not None else 0
                )
    else:
        raise ValueError(f"checkpoint_load_mode must be resume, stage_transfer or policy_only, got {load_mode!r}")
    return payload


# 【中文导读】在评估回报刷新时保存命名不同但内容完整的 checkpoint。
STAGE_BEST_FILES = {
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
    kwargs = {
        "best_metric_state": best_metric_state,
        "best_evaluation_stats": best_evaluation_stats,
        "fast_replay": fast_replay, "slow_replay": slow_replay,
        "manager_replay": manager_replay, "next_episode": next_episode,
    }
    filename = STAGE_BEST_FILES.get(cfg.training_stage)
    if filename is None:
        raise ValueError(f"Unknown training stage for best checkpoint: {cfg.training_stage!r}")
    kwargs["checkpoint_kind"] = "lightweight"
    save_checkpoint(root / "latest_policy.pt", cfg, agents, episode, global_step, best_return, **kwargs)
    save_checkpoint(root / filename, cfg, agents, episode, global_step, best_return, **kwargs)


# =============================================================================
# Training and evaluation
# =============================================================================
#
# 【模块说明：多时间尺度主循环】全局状态依次经过 Manager goal、Worker
# 动作、安全投影和环境求解；Fast 每步入库，Slow/Manager 到边界时聚合入库。这里是理解
# SMDP、done/truncated、终端 SOC 和四阶段交互的核心。
#


# 【中文导读】暂存刚执行的快动作，等待下一快观测后计算内在奖励并入库。
@dataclass
class PendingFastTransition:
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：暂存尚缺 next observation 的 Fast transition。
    
    输入：当前 obs、raw/executed action、奖励、goal、done。
    
    输出：下一步到来后可写入 Fast Replay 的记录。
    
    核心步骤：延迟一个快速步补齐 next_obs 与 shaping。
    
    强化学习含义：保证 transition 的时间对齐。
    
    【容易混淆】pending 不等于 Replay；只有 finalize 后才真正插入。
    """

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


# 【中文导读】累计一个慢动作保持区间内的折扣奖励、投影惩罚和持续步数。
@dataclass
class PendingSlowSegment:
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：聚合一个 Slow 保持区间。
    
    输入：片段起点、raw slow action、goal。
    
    输出：包含折扣奖励、动作序列统计和真实 duration 的 Slow transition。
    
    核心步骤：逐快速步累积 reward/projection，边界或 episode 结束时 finalize。
    
    强化学习含义：Slow 宏动作通常持续 20 步。
    
    【容易混淆】episode 尾部可能不足 20 步，不能伪造 duration。
    """

    obs_start: np.ndarray
    goal: np.ndarray
    raw_action: np.ndarray
    executed_action: Optional[np.ndarray] = None
    discounted_reward: float = 0.0
    discounted_global_safety_reward: float = 0.0
    discounted_role_specific_reward: float = 0.0
    discounted_raw_total_reward: float = 0.0
    component_clipped_steps: int = 0
    projection_penalty_sum: float = 0.0
    duration_steps: int = 0


# 【中文导读】累计一个 Manager goal 区间内的环境回报和持续步数。
@dataclass
class PendingManagerSegment:
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：聚合一个 Manager goal 区间。
    
    输入：Manager 起始 observation、raw/executed/previous goal。
    
    输出：通常持续 40 步的 Manager transition。
    
    核心步骤：逐步折扣累积全局安全奖励和约束统计。
    
    强化学习含义：高层动作的回报跨越多个底层控制步。
    
    【容易混淆】时间截断默认 done，因 reset 会切换外生场景，不能跨场景 bootstrap。
    """

    obs_start: np.ndarray
    goal: np.ndarray  # executed goal passed to both Workers
    raw_goal: Optional[np.ndarray] = None
    previous_executed_goal: Optional[np.ndarray] = None
    discounted_reward: float = 0.0
    duration_steps: int = 0
    constraint_scores: List[float] = None  # type: ignore[assignment]
    solver_failure_seen: bool = False

    def __post_init__(self) -> None:
        if self.raw_goal is None:
            self.raw_goal = self.goal.copy()
        if self.previous_executed_goal is None:
            self.previous_executed_goal = np.zeros_like(self.goal)
        if self.constraint_scores is None:
            self.constraint_scores = []


# 【中文导读】补齐快 transition 的 next_obs、next_goal 和内在奖励后写入经验池。
def finalize_fast_transition(pending: PendingFastTransition, next_obs: np.ndarray, next_goal: np.ndarray,
                             fast_agent: WorkerTD3, buffer: FastReplayBuffer, cfg: TrainConfig) -> Dict[str, float]:
    """补齐 next_obs/内在奖励后，把快 Worker transition 写入经验池。"""

    latent = fast_agent.latent_goal_reward(pending.obs, next_obs, pending.goal, cfg.delta_z_min)
    physical = fast_physical_progress(pending.obs, next_obs, pending.goal)
    total_raw = build_worker_reward(pending.reward_external, latent, physical, pending.projection, cfg)
    total, clipped = clip_reward_value(total_raw, cfg.worker_reward_clip_abs)
    buffer.add(
        pending.obs, next_obs, pending.raw_action, pending.executed_action,
        pending.reward_external, latent + physical, total, pending.goal, next_goal, pending.done,
    )
    return {"fast_external": pending.reward_external, "fast_latent": latent,
            "fast_physical": physical, "fast_total_raw": total_raw, "fast_total": total,
            "fast_reward_clipped": float(clipped), "projection": pending.projection}


# 【中文导读】在慢动作切换或 episode 结束时封装慢时间尺度聚合样本。
def finalize_slow_segment(pending: PendingSlowSegment, obs_end: np.ndarray, next_goal: np.ndarray,
                          slow_agent: WorkerTD3, buffer: SlowReplayBuffer, cfg: TrainConfig,
                          done: bool) -> Dict[str, float]:
    """结束一个慢速片段，并写入慢 Worker 经验池。"""

    executed = pending.executed_action
    if executed is None:
        executed = pending.raw_action.copy()
    latent = slow_agent.latent_goal_reward(pending.obs_start, obs_end, pending.goal, cfg.delta_z_min)
    physical = slow_physical_progress(pending.obs_start, obs_end, pending.goal)
    total_raw = build_worker_reward(
        pending.discounted_reward, latent, physical, pending.projection_penalty_sum, cfg
    )
    total, clipped = clip_reward_value(total_raw, cfg.worker_reward_clip_abs)
    buffer.add(
        pending.obs_start, obs_end, pending.raw_action, executed,
        total, pending.goal, next_goal, done, pending.duration_steps,
    )
    return {"slow_external": pending.discounted_reward, "slow_latent": latent,
            "slow_physical": physical, "slow_total_raw": total_raw, "slow_total": total,
            "slow_projection": pending.projection_penalty_sum,
            "slow_reward_clipped": float(clipped)}


# 【中文导读】三层时钟驱动的主训练循环，负责交互、聚合、更新、评估和保存。
def run_training(cfg: TrainConfig) -> Dict[str, Any]:
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：执行一个明确训练阶段的多时间尺度环境交互与更新。
    
    输入：TrainConfig。
    
    输出：运行目录、更新计数、最佳评价和参数变化等结果。
    
    核心步骤：reset→Manager goal→Slow/Fast action→安全投影→step→分层片段入库→按预算更新→评估/checkpoint。
    
    强化学习含义：这是三个时间尺度合流的核心。
    
    【容易混淆】环境 reward 用于评估；各层有效训练 reward 可有不同聚合。
    """

    set_seed(cfg.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = resolve_device(cfg.device)#读是不是使用cuda跑训练
    LOGGER.info("Using device: %s", device)

    # 1) 创建环境和三个智能体。环境动作维度来自电-气系统设备数量。
    env = ElectricGasMultiScaleEnv()
    env.unexpected_env_exception_policy = getattr(cfg, "unexpected_env_exception_policy", "raise")
    if env.slow_action_dim != 10 or env.fast_action_dim != 16 or env.action_dim != 26:
        LOGGER.warning("Expected slow=10 fast=16 action=26, got slow=%s fast=%s action=%s",
                       env.slow_action_dim, env.fast_action_dim, env.action_dim)
    agents = build_agents(env, cfg, device)
    loaded_best_return = -float("inf")
    loaded_best_metric_state: Dict[str, Any] = {}
    loaded_best_evaluation_stats: Dict[str, Any] = {}
    loaded_global_step = 0
    start_episode = 0
    strict_resume_restored = False
    resume_missing_components: List[str] = []
    load_mode = getattr(cfg, "checkpoint_load_mode", "policy_only" if cfg.load_policy_only else "resume")

    # 2) 根据环境实际返回的观测维度创建三类经验池。
    obs, _ = env.reset(seed=cfg.seed)
    builder = ObservationBuilder(env, cfg.manager_interval)
    manager_dim = builder.manager_obs(obs).size
    fast_dim = builder.fast_obs(0, obs).size
    slow_dim = builder.slow_obs(obs).size
    fast_buffer = FastReplayBuffer(cfg.fast_buffer_size, fast_dim, env.fast_action_dim, GOAL_DIM, device)
    slow_buffer = SlowReplayBuffer(cfg.slow_buffer_size, slow_dim, env.slow_action_dim, GOAL_DIM, device)
    manager_buffer = ManagerReplayBuffer(cfg.manager_buffer_size, manager_dim, GOAL_DIM, device)
    if cfg.load_checkpoint:
        LOGGER.info("Loading checkpoint from %s", cfg.load_checkpoint)
        payload = load_checkpoint(
            cfg.load_checkpoint, agents, device, mode=load_mode,
            fast_replay=fast_buffer, slow_replay=slow_buffer, manager_replay=manager_buffer,
        )
        if load_mode == "resume":
            strict_resume_restored = bool(payload.get("strict_resume_restored", False))
            resume_missing_components = list(payload.get("resume_missing_components", []))
            loaded_best = payload.get("best_return", -float("inf"))
            loaded_best_return = float(loaded_best) if loaded_best is not None else -float("inf")
            loaded_best_metric_state = dict(payload.get("best_metric_state", {}))
            loaded_best_evaluation_stats = dict(payload.get("best_evaluation_stats", {}))
            loaded_global_step = int(payload.get("global_step", 0))
            start_episode = int(payload.get("next_episode", int(payload.get("episode", -1)) + 1))
            LOGGER.info("Resumed complete training state at global_step=%s next_episode=%s.",
                        loaded_global_step, start_episode)
        elif load_mode == "stage_transfer":
            LOGGER.info("Stage transfer loaded network/normalizer weights and retained Critic weights; "
                        "optimizers, update counts, Replay, RNG, scheduling and stage best are reset.")
        elif load_mode == "policy_only":
            LOGGER.info("Loaded policy/normalizer only; critics, optimizers, targets, update counts, Replay, RNG and stage best are reset.")

    parameter_snapshots = {
        "fast": agent_parameter_snapshot(agents.fast),
        "slow": agent_parameter_snapshot(agents.slow),
        "manager": agent_parameter_snapshot(agents.manager),
    }

    # 3) 准备日志、TensorBoard 和 checkpoint 目录。
    run_root = Path(cfg.checkpoint_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    with open(run_root / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
    writer = SummaryWriter(str(run_root / "tb")) if not cfg.no_tensorboard else SummaryWriter()
    writer.add_text("config", json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    csv_logger = EpisodeCSVLogger(run_root / "episode_log.csv")

    train_flags = stage_flags(cfg.training_stage)
    best_eval_return = loaded_best_return if np.isfinite(loaded_best_return) else -float("inf")
    best_metric_state = loaded_best_metric_state
    best_evaluation_stats = loaded_best_evaluation_stats
    global_step = loaded_global_step
    consecutive_solver_failure_episodes = 0

    for episode in range(start_episode, cfg.episodes):
        # 4) 一个 episode 表示一天调度。每个快速步是 3 分钟，默认 480 步。
        global_obs, info = env.reset(seed=cfg.seed + episode)
        builder = ObservationBuilder(env, cfg.manager_interval)
        episode_return = 0.0
        manager_return = 0.0
        solver_failures = 0
        gas_solve_count = 0
        step_rewards: List[float] = []
        action_projections: List[float] = []
        slow_action_projections: List[float] = []
        fast_action_projections: List[float] = []
        ess_guard_adjustments: List[float] = []
        goal_changes: List[float] = []
        voltage_rms_deviations: List[float] = []
        gas_pressure_rms_deviations: List[float] = []
        component_totals: Dict[str, float] = {}
        interaction_reward_estimated_clips = 0
        manager_reward_clips = 0
        episode_training_metrics: Dict[str, List[float]] = {}
        episode_update_start = {
            "fast": int(agents.fast.total_updates),
            "slow": int(agents.slow.total_updates),
            "manager": int(agents.manager.total_updates),
        }
        exploration_episode = int(cfg.exploration_episode_offset) + episode
        fast_noise = scheduled_noise(
            cfg.fast_exploration_noise, cfg.min_fast_exploration_noise,
            exploration_episode, cfg.noise_decay_episodes)
        slow_noise = scheduled_noise(
            cfg.slow_exploration_noise, cfg.min_slow_exploration_noise,
            exploration_episode, cfg.noise_decay_episodes)
        manager_noise = scheduled_noise(
            cfg.manager_exploration_noise, cfg.min_manager_exploration_noise,
            exploration_episode, cfg.noise_decay_episodes)
        writer.add_scalar("exploration/fast_noise", fast_noise, episode)
        writer.add_scalar("exploration/slow_noise", slow_noise, episode)
        writer.add_scalar("exploration/manager_noise", manager_noise, episode)
        current_goal: Optional[np.ndarray] = None
        current_raw_goal: Optional[np.ndarray] = None
        previous_goal: Optional[np.ndarray] = None
        held_slow_action = rule_slow_action(env)
        held_slow_action, _ = apply_ess_action_guard(env, held_slow_action, cfg, cfg.slow_interval)
        pending_fast: Optional[PendingFastTransition] = None
        pending_slow: Optional[PendingSlowSegment] = None
        pending_manager: Optional[PendingManagerSegment] = None
        last_manager_step = 0
        done = False

        for t in range(cfg.episode_steps):
            # 构造三层各自的观测。Manager 看全局，fast/slow 看更偏任务的局部摘要。
            manager_age = t - last_manager_step
            manager_obs = builder.manager_obs(global_obs, current_goal)
            fast_obs = builder.fast_obs(manager_age, global_obs)
            slow_obs = builder.slow_obs(global_obs)

            if t % cfg.manager_interval == 0:
                # Manager 到点后结束上一个 goal 片段，并选择新的 32 维 goal。
                if pending_manager is not None:
                    manager_reward, clipped = clip_reward_value(
                        pending_manager.discounted_reward, cfg.manager_reward_clip_abs)
                    manager_reward_clips += int(clipped)
                    manager_buffer.add(pending_manager.obs_start, manager_obs, pending_manager.goal,
                                       manager_reward, False,
                                       pending_manager.duration_steps,
                                       float(np.mean(pending_manager.constraint_scores)) if pending_manager.constraint_scores else 0.0,
                                       float(np.max(pending_manager.constraint_scores)) if pending_manager.constraint_scores else 0.0,
                                       pending_manager.solver_failure_seen,
                                       raw_goal=pending_manager.raw_goal,
                                       previous_executed_goal=pending_manager.previous_executed_goal)
                previous_goal = current_goal
                if cfg.training_stage in ("fast_pretrain", "slow_pretrain"):
                    current_raw_goal = fixed_manager_goal()
                    current_goal = current_raw_goal.copy()
                elif manager_buffer.total_insertions < int(
                        getattr(cfg, "manager_random_warmup_segments", 0)):
                    random_goal = np.random.normal(0.0, 1.0, size=GOAL_DIM).astype(np.float32)
                    current_raw_goal = normalize_goal_np(random_goal)
                    actor_blend = warmup_actor_blend(
                        manager_buffer.total_insertions,
                        int(getattr(cfg, "manager_random_warmup_segments", 0)),
                        float(getattr(cfg, "warmup_blend_fraction", 0.0)),
                    )
                    if actor_blend > 0.0:
                        actor_raw_goal, _ = agents.manager.select_goal_pair(
                            manager_obs, previous_goal, manager_noise, deterministic=False
                        )
                        current_raw_goal = normalize_goal_np(
                            (1.0 - actor_blend) * current_raw_goal + actor_blend * actor_raw_goal
                        )
                    current_goal = execute_manager_goal_np(
                        current_raw_goal,
                        np.zeros(GOAL_DIM, dtype=np.float32) if previous_goal is None else previous_goal,
                        cfg.goal_smoothing,
                    )
                else:
                    current_raw_goal, current_goal = agents.manager.select_goal_pair(
                        manager_obs, previous_goal, manager_noise, deterministic=False)
                if previous_goal is not None:
                    goal_change = float(np.mean(np.square(current_goal - previous_goal)))
                    goal_changes.append(goal_change)
                    writer.add_scalar("manager/goal_change", goal_change, global_step)
                    writer.add_scalar("manager/goal_change_penalty",
                                      cfg.goal_change_penalty_weight * goal_change, global_step)
                writer.add_scalar("manager/goal_norm", float(np.linalg.norm(current_goal[:24])), global_step)
                initial_manager_reward = 0.0
                if previous_goal is not None:
                    initial_manager_reward = -cfg.goal_change_penalty_weight * float(
                        np.mean(np.square(current_goal - previous_goal))
                    )
                pending_manager = PendingManagerSegment(
                    manager_obs.copy(), current_goal.copy(),
                    raw_goal=current_raw_goal.copy(),
                    previous_executed_goal=(
                        np.zeros(GOAL_DIM, dtype=np.float32)
                        if previous_goal is None else previous_goal.copy()
                    ),
                    discounted_reward=initial_manager_reward,
                )
                last_manager_step = t
                manager_age = 0
                fast_obs = builder.fast_obs(manager_age, global_obs)

            assert current_goal is not None
            if pending_fast is not None:
                # 上一个快速步现在拥有 next_obs 了，可以入快 Worker buffer。
                logs = finalize_fast_transition(pending_fast, fast_obs, current_goal, agents.fast, fast_buffer, cfg)
                for k, v in logs.items():
                    writer.add_scalar(k if k.startswith("interaction/") else f"reward/{k}", v, global_step)
                pending_fast = None

            if t % cfg.slow_interval == 0:
                # 慢 Worker 到点后结束上一个慢片段，并产生新的慢动作。
                if pending_slow is not None:
                    logs = finalize_slow_segment(pending_slow, slow_obs, current_goal, agents.slow,
                                                 slow_buffer, cfg, False)
                    for k, v in logs.items():
                        writer.add_scalar(k if k.startswith("interaction/") else f"reward/{k}", v, global_step)
                if cfg.training_stage == "fast_pretrain":
                    raw_slow_action = rule_slow_action(env)
                elif slow_buffer.total_insertions < int(
                        getattr(cfg, "slow_random_warmup_segments", 0)):
                    warmup_slow_action = np.clip(
                        rule_slow_action(env)
                        + np.random.uniform(-0.05, 0.05, env.slow_action_dim).astype(np.float32),
                        -1.0, 1.0,
                    )
                    actor_blend = warmup_actor_blend(
                        slow_buffer.total_insertions,
                        int(getattr(cfg, "slow_random_warmup_segments", 0)),
                        float(getattr(cfg, "warmup_blend_fraction", 0.0)),
                    )
                    actor_slow_action = agents.slow.select_action(
                        slow_obs, current_goal, slow_noise, deterministic=False
                    )
                    raw_slow_action = np.clip(
                        (1.0 - actor_blend) * warmup_slow_action
                        + actor_blend * actor_slow_action, -1.0, 1.0
                    )
                else:
                    raw_slow_action = agents.slow.select_action(
                        slow_obs, current_goal, slow_noise, deterministic=False)
                horizon_steps = min(
                    max(1, cfg.slow_interval),
                    max(1, cfg.episode_steps - t),
                )
                held_slow_action, guard_adjustment = apply_ess_action_guard(
                    env, raw_slow_action, cfg, horizon_steps
                )
                ess_guard_adjustments.append(guard_adjustment)
                writer.add_scalar("action/ess_guard_adjustment", guard_adjustment, global_step)
                pending_slow = PendingSlowSegment(slow_obs.copy(), current_goal.copy(), held_slow_action.copy())

            if (train_flags["fast"] and global_step < int(
                    getattr(cfg, "fast_random_warmup_steps", 0))):
                random_fast_action = np.random.uniform(
                    -1.0, 1.0, env.fast_action_dim
                ).astype(np.float32)
                actor_blend = warmup_actor_blend(
                    global_step, int(getattr(cfg, "fast_random_warmup_steps", 0)),
                    float(getattr(cfg, "warmup_blend_fraction", 0.0)),
                )
                actor_fast_action = agents.fast.select_action(
                    fast_obs, current_goal, fast_noise, deterministic=False
                )
                fast_action = np.clip(
                    (1.0 - actor_blend) * random_fast_action
                    + actor_blend * actor_fast_action, -1.0, 1.0
                )
            elif cfg.training_stage in ("slow_pretrain", "manager_train"):
                # 预训练慢层/Manager 时，快层不加噪声，减少下层随机性对上层学习的干扰。
                fast_action = agents.fast.select_action(fast_obs, current_goal, 0.0, deterministic=True)
            else:
                fast_action = agents.fast.select_action(
                    fast_obs, current_goal, fast_noise, deterministic=False)

            # Environment action layout: Slow 10 = ESS(3)+GFG(3)+P2G(3)+controlled
            # compressor(1); Fast 16 = inverter Q(8)+renewable curtailment(8); total 26.
            # 实际 RL 布局：[ESS(3), GFG(3), P2G(3), controlled compressor(1),
            # inverter-Q(8), renewable-curtailment(8)]，共 10+16=26 维。
            # 【安全语义】环境动作严格为 Slow 10 维 + Fast 16 维；随后仍要经过环境物理安全投影。
            joint_action = np.concatenate([held_slow_action, fast_action]).astype(np.float32)
            next_global_obs, env_reward, terminated, truncated, info = safe_env_step(env, joint_action, global_obs)
            time_limit_truncated = bool(
                t + 1 >= cfg.episode_steps and not terminated and not truncated
            )
            if time_limit_truncated:
                truncated = True
                info["time_limit_truncated"] = True
                if (getattr(cfg, "run_mode", "debug") != "formal"
                        and t + 1 < env.config.time.steps_per_day):
                    env_reward = apply_debug_terminal_soc_penalty(env, info, env_reward, cfg)
            transition_done = bool(terminated or truncated)
            if pending_manager is not None:
                manager_constraint = float(info.get("_normalized_constraint_score", 0.0))
                pending_manager.constraint_scores.append(manager_constraint)
                pending_manager.solver_failure_seen = (
                    pending_manager.solver_failure_seen or bool(info.get("solver_failed", False))
                )
            applied = np.asarray(info.get("applied_action", np.clip(joint_action, -1.0, 1.0)), dtype=np.float32)
            raw = np.asarray(info.get("raw_action", joint_action), dtype=np.float32)
            projection_magnitude = float(info.get(
                "action_projection_magnitude", np.linalg.norm(raw - applied)))
            slow_projection_magnitude = float(np.linalg.norm(raw[:env.slow_action_dim] -
                                                             applied[:env.slow_action_dim]))
            fast_projection_magnitude = float(np.linalg.norm(raw[env.slow_action_dim:] -
                                                             applied[env.slow_action_dim:]))
            action_projections.append(projection_magnitude)
            slow_action_projections.append(slow_projection_magnitude)
            fast_action_projections.append(fast_projection_magnitude)
            fast_reward_parts = worker_safety_reward_from_components(info, "fast", cfg)
            slow_reward_parts = worker_safety_reward_from_components(info, "slow", cfg)
            fast_external = fast_reward_parts["total"]
            slow_external = slow_reward_parts["total"]
            slow_proj_pen = projection_penalty(raw[:env.slow_action_dim], applied[:env.slow_action_dim],
                                               cfg.lambda_projection)
            fast_proj_pen = projection_penalty(raw[env.slow_action_dim:], applied[env.slow_action_dim:],
                                               cfg.lambda_projection)
            pending_fast = PendingFastTransition(
                # 快 Worker 样本先 pending，到下一步拿到 next_fast_obs 后再写入 buffer。
                fast_obs.copy(), fast_action.copy(), applied[env.slow_action_dim:].copy(),
                fast_external, fast_proj_pen, current_goal.copy(), transition_done,
                reward_global_safety=fast_reward_parts["global_weighted"],
                reward_role_specific=fast_reward_parts["role_weighted"],
                reward_raw_total=fast_reward_parts["raw_total"],
                reward_component_clipped=fast_reward_parts["component_clipped"],
            )
            if pending_slow is not None:
                # 慢 Worker 的奖励跨多个快速步折扣累加，直到下一次慢动作或 episode 结束。
                if info.get("slow_action_applied", False):
                    pending_slow.executed_action = applied[:env.slow_action_dim].copy()
                # 片段内部按快速步折扣：R_seg = Σ gamma_fast^k * r_{t+k}。
                # 【SMDP关键】片段内部按快速步折扣累计；TD bootstrap 只再乘 gamma_fast**duration，不能二次折扣。
                pending_slow.discounted_reward += (cfg.gamma_fast ** pending_slow.duration_steps) * slow_external
                pending_slow.discounted_global_safety_reward += (
                    cfg.gamma_fast ** pending_slow.duration_steps
                ) * slow_reward_parts["global_weighted"]
                pending_slow.discounted_role_specific_reward += (
                    cfg.gamma_fast ** pending_slow.duration_steps
                ) * slow_reward_parts["role_weighted"]
                pending_slow.discounted_raw_total_reward += (
                    cfg.gamma_fast ** pending_slow.duration_steps
                ) * slow_reward_parts["raw_total"]
                pending_slow.component_clipped_steps += int(slow_reward_parts["component_clipped"])
                pending_slow.projection_penalty_sum += (
                    cfg.gamma_fast ** pending_slow.duration_steps
                ) * slow_proj_pen
                pending_slow.duration_steps += 1
            if pending_manager is not None:
                # Manager 直接看环境总奖励，学习 goal 对未来一段时间总回报的影响。
                # Manager 聚合完整环境回报；goal change penalty 已在片段初始化时一次性加入。
                manager_step_reward = manager_safety_reward_from_components(info, cfg)["reward"]
                # 【SMDP关键】Manager 奖励跨 goal 保持区间累计，duration 记录真实快速步数。
                pending_manager.discounted_reward += (
                    cfg.gamma_fast ** pending_manager.duration_steps
                ) * float(manager_step_reward)
                pending_manager.duration_steps += 1
                manager_return += float(manager_step_reward)

            comps = info.get("reward_components", {})
            metrics = info.get("constraint_metrics", {})
            for key, value in comps.items():
                component_totals[key] = component_totals.get(key, 0.0) + float(value)
            if "voltage_rms_deviation_pu" in metrics:
                voltage_rms_deviations.append(float(metrics["voltage_rms_deviation_pu"]))
            if "gas_pressure_rms_deviation_bar" in metrics:
                gas_pressure_rms_deviations.append(float(metrics["gas_pressure_rms_deviation_bar"]))
            episode_return += float(env_reward)
            step_rewards.append(float(env_reward))
            solver_failures += int(bool(info.get("solver_failed", False)))
            gas_solve_count = int(info.get("gas_solve_count", gas_solve_count))
            writer.add_scalar("reward/environment_step", float(env_reward), global_step)
            for key, value in comps.items():
                writer.add_scalar(f"components/{key}", float(value), global_step)
            for key, value in metrics.items():
                arr = np.asarray(value)
                if arr.size == 1 and np.issubdtype(arr.dtype, np.number):
                    writer.add_scalar(f"constraints/{key}", float(arr.reshape(-1)[0]), global_step)
            writer.add_scalar("solver/failure_count", solver_failures, global_step)
            writer.add_scalar("solver/gas_solve_count", gas_solve_count, global_step)

            global_obs = next_global_obs
            done = transition_done
            global_step += 1

            if global_step > cfg.learning_starts:
                # learning_starts 之前只收集经验；之后按配置频率更新各层 TD3。
                for _ in range(cfg.updates_per_step):
                    if (train_flags["fast"] and global_step >= int(
                            getattr(cfg, "fast_random_warmup_steps", 0))):
                        update_logs = agents.fast.update(fast_buffer, cfg.batch_size, cfg.gamma_fast)
                        for k, v in update_logs.items():
                            writer.add_scalar(k if k.startswith("training_batch/") else f"loss/{k}", v, global_step)
                            if np.isscalar(v) and np.isfinite(float(v)):
                                episode_training_metrics.setdefault(k, []).append(float(v))
                    if (train_flags["slow"]
                            and slow_buffer.total_insertions >= int(
                                getattr(cfg, "slow_random_warmup_segments", 0))
                            and global_step % max(1, cfg.slow_update_interval_steps) == 0):
                        update_logs = agents.slow.update(slow_buffer, cfg.batch_size, cfg.gamma_slow)
                        for k, v in update_logs.items():
                            writer.add_scalar(k if k.startswith("training_batch/") else f"loss/{k}", v, global_step)
                            if np.isscalar(v) and np.isfinite(float(v)):
                                episode_training_metrics.setdefault(k, []).append(float(v))
                    if (train_flags["manager"]
                            and manager_buffer.total_insertions >= int(
                                getattr(cfg, "manager_random_warmup_segments", 0))
                            and global_step % max(1, cfg.manager_update_interval_steps) == 0):
                        update_logs = agents.manager.update(manager_buffer, cfg.batch_size)
                        for k, v in update_logs.items():
                            writer.add_scalar(k if k.startswith("training_batch/") else f"loss/{k}", v, global_step)
                            if np.isscalar(v) and np.isfinite(float(v)):
                                episode_training_metrics.setdefault(k, []).append(float(v))

            if done:
                break

        # 5) episode 结束时，把还没入库的 pending 片段收尾。
        if current_goal is None:
            current_goal = fixed_manager_goal()
        final_manager_obs = builder.manager_obs(global_obs, current_goal)
        # 这里传入的是近似 manager_age；异常提前截断时可能与真实 t-last_manager_step 不一致。
        final_manager_age = max(0, min(cfg.manager_interval, (t + 1) - last_manager_step))
        final_fast_obs = builder.fast_obs(final_manager_age, global_obs)
        final_slow_obs = builder.slow_obs(global_obs)
        if pending_fast is not None:
            logs = finalize_fast_transition(pending_fast, final_fast_obs, current_goal, agents.fast, fast_buffer, cfg)
        if pending_slow is not None and pending_slow.duration_steps > 0:
            logs = finalize_slow_segment(pending_slow, final_slow_obs, current_goal, agents.slow, slow_buffer, cfg, done)
        if pending_manager is not None and pending_manager.duration_steps > 0:
            manager_reward, clipped = clip_reward_value(
                pending_manager.discounted_reward, cfg.manager_reward_clip_abs)
            manager_reward_clips += int(clipped)
            manager_buffer.add(pending_manager.obs_start, final_manager_obs, pending_manager.goal,
                               manager_reward, done, pending_manager.duration_steps,
                               float(np.mean(pending_manager.constraint_scores)) if pending_manager.constraint_scores else 0.0,
                               float(np.max(pending_manager.constraint_scores)) if pending_manager.constraint_scores else 0.0,
                               pending_manager.solver_failure_seen,
                               raw_goal=pending_manager.raw_goal,
                               previous_executed_goal=pending_manager.previous_executed_goal)

        # 尾片段是在最后一个环境步的常规更新门控之后才写入 Replay 的。
        # 因此必须在插入后再次触发各层 update；优化版 agent 通过 insertion id
        # 防止同一条样本重复更新，基础版则只在这里处理刚插入的尾样本。
        tail_updates: Tuple[Tuple[str, Any, Any, Tuple[Any, ...]], ...] = (
            ("fast", agents.fast, fast_buffer, (cfg.batch_size, cfg.gamma_fast)),
            ("slow", agents.slow, slow_buffer, (cfg.batch_size, cfg.gamma_slow)),
            ("manager", agents.manager, manager_buffer, (cfg.batch_size,)),
        )
        for role, agent, replay, update_args in tail_updates:
            if not train_flags[role]:
                continue
            update_logs = agent.update(replay, *update_args)
            for key, value in update_logs.items():
                writer.add_scalar(
                    key if key.startswith("training_batch/") else f"loss/{key}",
                    value,
                    global_step,
                )
                if np.isscalar(value) and np.isfinite(float(value)):
                    episode_training_metrics.setdefault(key, []).append(float(value))

        # 6) 汇总 episode 指标，用于日志、TensorBoard、CSV 和 checkpoint。
        mean_step_reward = float(np.mean(step_rewards)) if step_rewards else 0.0
        min_step_reward = float(np.min(step_rewards)) if step_rewards else 0.0
        max_step_reward = float(np.max(step_rewards)) if step_rewards else 0.0
        mean_projection = float(np.mean(action_projections)) if action_projections else 0.0
        max_projection = float(np.max(action_projections)) if action_projections else 0.0
        mean_slow_projection = float(np.mean(slow_action_projections)) if slow_action_projections else 0.0
        max_slow_projection = float(np.max(slow_action_projections)) if slow_action_projections else 0.0
        mean_fast_projection = float(np.mean(fast_action_projections)) if fast_action_projections else 0.0
        max_fast_projection = float(np.max(fast_action_projections)) if fast_action_projections else 0.0
        mean_ess_guard = float(np.mean(ess_guard_adjustments)) if ess_guard_adjustments else 0.0
        max_ess_guard = float(np.max(ess_guard_adjustments)) if ess_guard_adjustments else 0.0
        mean_goal_change = float(np.mean(goal_changes)) if goal_changes else 0.0
        max_goal_change = float(np.max(goal_changes)) if goal_changes else 0.0
        mean_voltage_rms = float(np.mean(voltage_rms_deviations)) if voltage_rms_deviations else 0.0
        mean_gas_pressure_rms = float(np.mean(gas_pressure_rms_deviations)) if gas_pressure_rms_deviations else 0.0

        def mean_training_metric(suffix: str) -> float:
            values = [
                value
                for name, items in episode_training_metrics.items()
                if name.endswith(suffix)
                for value in items
            ]
            return float(np.mean(values)) if values else 0.0

        update_deltas = {
            "fast": int(agents.fast.total_updates) - episode_update_start["fast"],
            "slow": int(agents.slow.total_updates) - episode_update_start["slow"],
            "manager": int(agents.manager.total_updates) - episode_update_start["manager"],
        }
        nonfinite_batch_count = int(sum(
            getattr(agent, "nonfinite_batch_count", 0)
            for agent in (agents.fast, agents.slow, agents.manager)
        ))
        mean_target_q_clipping = mean_training_metric("target_q_clipping_ratio")
        mean_reward_clipping = mean_training_metric("reward_clipping_ratio")
        mean_action_saturation = mean_training_metric("actor_action_saturation_ratio")
        mean_dynamic_projection_mse = mean_training_metric("dynamic_projection_mse")
        writer.add_scalar("health/nonfinite_batch_count", nonfinite_batch_count, episode)
        for role, count in update_deltas.items():
            writer.add_scalar(f"health/{role}_updates_this_episode", count, episode)
        for role, buffer in (("fast", fast_buffer), ("slow", slow_buffer), ("manager", manager_buffer)):
            writer.add_scalar(f"health/{role}_replay_insertions", buffer.total_insertions, episode)
        for role, agent in (("fast", agents.fast), ("slow", agents.slow), ("manager", agents.manager)):
            writer.add_scalar(f"health/{role}_critic_updates_total",
                              getattr(agent, "critic_updates", agent.total_updates), episode)
            writer.add_scalar(f"health/{role}_actor_updates_total",
                              getattr(agent, "actor_updates", 0), episode)
            writer.add_scalar(f"health/{role}_normalizer_count", agent.normalizer.count, episode)
            writer.add_scalar(f"health/{role}_normalizer_mean_abs",
                              float(np.mean(np.abs(agent.normalizer.mean))), episode)
            writer.add_scalar(f"health/{role}_normalizer_variance_mean",
                              float(np.mean(agent.normalizer.var)), episode)
        writer.add_scalar("health/mean_target_q_clipping_ratio", mean_target_q_clipping, episode)
        writer.add_scalar("health/mean_reward_clipping_ratio", mean_reward_clipping, episode)
        writer.add_scalar("health/mean_actor_action_saturation_ratio", mean_action_saturation, episode)
        writer.add_scalar("health/mean_dynamic_projection_rms",
                          math.sqrt(max(mean_dynamic_projection_mse, 0.0)), episode)

        patience = int(getattr(cfg, "health_warning_patience_episodes", 3))
        consecutive_solver_failure_episodes = (
            consecutive_solver_failure_episodes + 1 if solver_failures > 0 else 0
        )
        if consecutive_solver_failure_episodes >= patience:
            LOGGER.warning("Solver failures occurred for %s consecutive episodes.",
                           consecutive_solver_failure_episodes)
        role_buffers = {"fast": fast_buffer, "slow": slow_buffer, "manager": manager_buffer}
        role_starts = {
            "fast": int(getattr(cfg, "fast_learning_starts", 0)),
            "slow": int(getattr(cfg, "slow_learning_starts", 0)),
            "manager": int(getattr(cfg, "manager_learning_starts", 0)),
        }
        for role, active in train_flags.items():
            if active and len(role_buffers[role]) >= role_starts[role] and update_deltas[role] == 0:
                LOGGER.warning("%s is trainable with sufficient Replay data but made no updates in episode %s.",
                               role, episode)
        clip_limit = float(getattr(cfg, "health_clip_warning_ratio", 0.20))
        if max(mean_target_q_clipping, mean_reward_clipping) > clip_limit:
            LOGGER.warning("Training clipping ratio exceeded %.3f in episode %s (target_q=%.3f reward=%.3f).",
                           clip_limit, episode, mean_target_q_clipping, mean_reward_clipping)
        saturation_limit = float(getattr(cfg, "health_action_saturation_warning_ratio", 0.90))
        if mean_action_saturation > saturation_limit:
            LOGGER.warning("Actor action saturation ratio %.3f exceeded %.3f in episode %s.",
                           mean_action_saturation, saturation_limit, episode)
        projection_limit = float(getattr(cfg, "health_projection_warning_rms", 0.25))
        if math.sqrt(max(mean_dynamic_projection_mse, 0.0)) > projection_limit:
            LOGGER.warning("Dynamic projection RMS %.3f exceeded %.3f in episode %s.",
                           math.sqrt(max(mean_dynamic_projection_mse, 0.0)), projection_limit, episode)
        writer.add_scalar("episode/return", episode_return, episode)
        writer.add_scalar("episode/manager_return", manager_return, episode)
        writer.add_scalar("episode/fast_buffer_size", len(fast_buffer), episode)
        writer.add_scalar("episode/slow_buffer_size", len(slow_buffer), episode)
        writer.add_scalar("episode/manager_buffer_size", len(manager_buffer), episode)
        writer.add_scalar("episode/mean_action_projection", mean_projection, episode)
        writer.add_scalar("episode/max_action_projection", max_projection, episode)
        writer.add_scalar("episode/mean_slow_action_projection", mean_slow_projection, episode)
        writer.add_scalar("episode/max_slow_action_projection", max_slow_projection, episode)
        writer.add_scalar("episode/mean_fast_action_projection", mean_fast_projection, episode)
        writer.add_scalar("episode/max_fast_action_projection", max_fast_projection, episode)
        writer.add_scalar("episode/mean_ess_action_guard", mean_ess_guard, episode)
        writer.add_scalar("episode/max_ess_action_guard", max_ess_guard, episode)
        writer.add_scalar("episode/mean_voltage_rms_deviation_pu", mean_voltage_rms, episode)
        writer.add_scalar("episode/mean_gas_pressure_rms_deviation_bar", mean_gas_pressure_rms, episode)
        writer.add_scalar("episode/interaction_reward_estimated_clips",
                          interaction_reward_estimated_clips, episode)
        writer.add_scalar("episode/manager_reward_clips", manager_reward_clips, episode)
        LOGGER.info("Episode %s return %.3f | buffers fast=%s slow=%s manager=%s | failures=%s",
                    episode, episode_return, len(fast_buffer), len(slow_buffer), len(manager_buffer),
                    solver_failures)

        eval_return = ""
        eval_solver_failures: Any = ""
        eval_power_success_rate: Any = ""
        eval_gas_success_rate: Any = ""
        eval_mean_voltage_rms: Any = ""
        eval_mean_gas_pressure_rms: Any = ""
        if (episode + 1) % max(1, cfg.eval_interval) == 0 or episode == cfg.episodes - 1:
            eval_stats = evaluate_policy(agents, cfg, episodes=cfg.eval_episodes, max_steps=cfg.episode_steps,
                                         seed=cfg.seed + 10_000 + episode)
            eval_return = eval_stats["mean_return"]
            eval_solver_failures = eval_stats["solver_failures"]
            eval_power_success_rate = eval_stats["power_success_rate"]
            eval_gas_success_rate = eval_stats["gas_success_rate"]
            eval_mean_voltage_rms = eval_stats["mean_voltage_rms_deviation_pu"]
            eval_mean_gas_pressure_rms = eval_stats["mean_gas_pressure_rms_deviation_bar"]
            writer.add_scalar("eval/return", eval_stats["mean_return"], episode)
            writer.add_scalar("eval/solver_failures", eval_stats["solver_failures"], episode)
            writer.add_scalar("eval/power_success_rate", eval_stats["power_success_rate"], episode)
            writer.add_scalar("eval/gas_success_rate", eval_stats["gas_success_rate"], episode)
            writer.add_scalar("eval/mean_voltage_rms_deviation_pu",
                              eval_stats["mean_voltage_rms_deviation_pu"], episode)
            writer.add_scalar("eval/mean_gas_pressure_rms_deviation_bar",
                              eval_stats["mean_gas_pressure_rms_deviation_bar"], episode)
            if is_better_evaluation(eval_stats, best_metric_state, cfg):
                best_eval_return = eval_stats["mean_return"]
                best_metric_state = metric_state_from_evaluation(eval_stats, cfg)
                best_evaluation_stats = dict(eval_stats)
                save_best_files(
                    run_root, agents, cfg, episode, global_step, best_eval_return,
                    best_metric_state, best_evaluation_stats,
                    fast_buffer, slow_buffer, manager_buffer, episode + 1,
                )
        save_checkpoint(
            run_root / "latest_policy.pt", cfg, agents, episode, global_step, best_eval_return,
            best_metric_state, best_evaluation_stats,
            fast_buffer, slow_buffer, manager_buffer, episode + 1,
            checkpoint_kind="lightweight",
        )
        if ((episode + 1) % int(cfg.full_resume_checkpoint_interval) == 0
                or episode == cfg.episodes - 1):
            save_checkpoint(
                run_root / "resume_latest.pt", cfg, agents, episode, global_step, best_eval_return,
                best_metric_state, best_evaluation_stats,
                fast_buffer, slow_buffer, manager_buffer, episode + 1,
                checkpoint_kind="full_resume",
            )
        csv_logger.write({
            "stage": cfg.training_stage,
            "episode": episode,
            "global_step": global_step,
            "episode_return": episode_return,
            "manager_return": manager_return,
            "mean_step_reward": mean_step_reward,
            "min_step_reward": min_step_reward,
            "max_step_reward": max_step_reward,
            "eval_return": eval_return,
            "best_eval_return": best_eval_return,
            "eval_solver_failures": eval_solver_failures,
            "eval_power_success_rate": eval_power_success_rate,
            "eval_gas_success_rate": eval_gas_success_rate,
            "eval_mean_voltage_rms_deviation_pu": eval_mean_voltage_rms,
            "eval_mean_gas_pressure_rms_deviation_bar": eval_mean_gas_pressure_rms,
            "fast_buffer_size": len(fast_buffer),
            "slow_buffer_size": len(slow_buffer),
            "manager_buffer_size": len(manager_buffer),
            "solver_failures": solver_failures,
            "gas_solve_count": gas_solve_count,
            "mean_action_projection": mean_projection,
            "max_action_projection": max_projection,
            "mean_slow_action_projection": mean_slow_projection,
            "max_slow_action_projection": max_slow_projection,
            "mean_fast_action_projection": mean_fast_projection,
            "max_fast_action_projection": max_fast_projection,
            "mean_ess_action_guard": mean_ess_guard,
            "max_ess_action_guard": max_ess_guard,
            "mean_goal_change": mean_goal_change,
            "max_goal_change": max_goal_change,
            "mean_voltage_rms_deviation_pu": mean_voltage_rms,
            "mean_gas_pressure_rms_deviation_bar": mean_gas_pressure_rms,
            "voltage_deviation_cost": component_totals.get("voltage_deviation", 0.0),
            "gas_pressure_deviation_cost": component_totals.get("gas_pressure_deviation", 0.0),
            "voltage_violation_cost": component_totals.get("voltage_violation", 0.0),
            "gas_pressure_violation_cost": component_totals.get("gas_pressure_violation", 0.0),
            "pipe_velocity_violation_cost": component_totals.get("pipe_velocity_violation", 0.0),
            "source_capacity_violation_cost": component_totals.get("source_capacity_violation", 0.0),
            "gas_purchase_cost": component_totals.get("gas_purchase", 0.0),
            "interaction_reward_estimated_clips": interaction_reward_estimated_clips,
            "manager_reward_clips": manager_reward_clips,
            "fast_total_insertions": fast_buffer.total_insertions,
            "slow_total_insertions": slow_buffer.total_insertions,
            "manager_total_insertions": manager_buffer.total_insertions,
            "fast_update_count": agents.fast.total_updates,
            "slow_update_count": agents.slow.total_updates,
            "manager_update_count": agents.manager.total_updates,
            "fast_actor_update_count": getattr(agents.fast, "actor_updates", 0),
            "slow_actor_update_count": getattr(agents.slow, "actor_updates", 0),
            "manager_actor_update_count": getattr(agents.manager, "actor_updates", 0),
            "fast_critic_update_count": getattr(agents.fast, "critic_updates", agents.fast.total_updates),
            "slow_critic_update_count": getattr(agents.slow, "critic_updates", agents.slow.total_updates),
            "manager_critic_update_count": getattr(agents.manager, "critic_updates", agents.manager.total_updates),
            "nonfinite_batch_count": nonfinite_batch_count,
            "fast_normalizer_count": agents.fast.normalizer.count,
            "slow_normalizer_count": agents.slow.normalizer.count,
            "manager_normalizer_count": agents.manager.normalizer.count,
            "mean_target_q_clipping_ratio": mean_target_q_clipping,
            "mean_reward_clipping_ratio": mean_reward_clipping,
            "mean_actor_action_saturation_ratio": mean_action_saturation,
        })

    writer.close()
    LOGGER.info("Training complete. Checkpoints: %s", run_root)
    fast_last = (fast_buffer.idx - 1) % fast_buffer.capacity if len(fast_buffer) else None
    slow_last = (slow_buffer.idx - 1) % slow_buffer.capacity if len(slow_buffer) else None
    manager_last = (manager_buffer.idx - 1) % manager_buffer.capacity if len(manager_buffer) else None
    parameter_changes = {
        role: agent_parameter_change(getattr(agents, role), parameter_snapshots[role])
        for role in ("fast", "slow", "manager")
    }
    return {
        "run_root": str(run_root),
        "global_step": global_step,
        "next_episode": start_episode if cfg.episodes <= start_episode else cfg.episodes,
        "best_eval_return": best_eval_return,
        "best_metric_state": best_metric_state,
        "best_evaluation_stats": best_evaluation_stats,
        "fast_buffer_size": len(fast_buffer),
        "slow_buffer_size": len(slow_buffer),
        "manager_buffer_size": len(manager_buffer),
        "fast_total_insertions": int(fast_buffer.total_insertions),
        "slow_total_insertions": int(slow_buffer.total_insertions),
        "manager_total_insertions": int(manager_buffer.total_insertions),
        "fast_last_duration": float(fast_buffer.duration_steps[fast_last, 0]) if fast_last is not None else 0.0,
        "slow_last_duration": float(slow_buffer.duration_steps[slow_last, 0]) if slow_last is not None else 0.0,
        "manager_last_duration": float(manager_buffer.duration_steps[manager_last, 0]) if manager_last is not None else 0.0,
        "fast_last_done": float(fast_buffer.dones[fast_last, 0]) if fast_last is not None else 0.0,
        "slow_last_done": float(slow_buffer.dones[slow_last, 0]) if slow_last is not None else 0.0,
        "manager_last_done": float(manager_buffer.dones[manager_last, 0]) if manager_last is not None else 0.0,
        "strict_resume_restored": strict_resume_restored,
        "resume_missing_components": resume_missing_components,
        "fast_update_count": int(agents.fast.total_updates),
        "slow_update_count": int(agents.slow.total_updates),
        "manager_update_count": int(agents.manager.total_updates),
        "fast_actor_update_count": int(getattr(agents.fast, "actor_updates", 0)),
        "slow_actor_update_count": int(getattr(agents.slow, "actor_updates", 0)),
        "manager_actor_update_count": int(getattr(agents.manager, "actor_updates", 0)),
        "fast_critic_update_count": int(getattr(agents.fast, "critic_updates", agents.fast.total_updates)),
        "slow_critic_update_count": int(getattr(agents.slow, "critic_updates", agents.slow.total_updates)),
        "manager_critic_update_count": int(getattr(agents.manager, "critic_updates", agents.manager.total_updates)),
        "fast_parameter_change_l2": parameter_changes["fast"][0],
        "fast_parameter_change_max_abs": parameter_changes["fast"][1],
        "slow_parameter_change_l2": parameter_changes["slow"][0],
        "slow_parameter_change_max_abs": parameter_changes["slow"][1],
        "manager_parameter_change_l2": parameter_changes["manager"][0],
        "manager_parameter_change_max_abs": parameter_changes["manager"][1],
        "nonfinite_batch_count": int(sum(
            getattr(agent, "nonfinite_batch_count", 0)
            for agent in (agents.fast, agents.slow, agents.manager)
        )),
        "slow_last_guarded_action": (
            slow_buffer.guarded_actions[slow_last].copy()
            if slow_last is not None and hasattr(slow_buffer, "guarded_actions") else None
        ),
        "slow_last_discounted_mean_executed_action": (
            slow_buffer.executed_actions[slow_last].copy() if slow_last is not None else None
        ),
        "slow_last_executed_action_variance": (
            slow_buffer.executed_action_variance[slow_last].copy()
            if slow_last is not None and hasattr(slow_buffer, "executed_action_variance") else None
        ),
        "slow_last_max_dynamic_projection": (
            float(slow_buffer.max_dynamic_projection[slow_last, 0])
            if slow_last is not None and hasattr(slow_buffer, "max_dynamic_projection") else 0.0
        ),
        "manager_last_constraint_mean": (
            float(manager_buffer.segment_constraint_mean[manager_last, 0])
            if manager_last is not None and hasattr(manager_buffer, "segment_constraint_mean") else 0.0
        ),
        "manager_last_constraint_max": (
            float(manager_buffer.segment_constraint_max[manager_last, 0])
            if manager_last is not None and hasattr(manager_buffer, "segment_constraint_max") else 0.0
        ),
        "manager_last_solver_failure_seen": (
            bool(manager_buffer.solver_failure_seen[manager_last, 0])
            if manager_last is not None and hasattr(manager_buffer, "solver_failure_seen") else False
        ),
        "manager_last_priority": (
            float(manager_buffer.priorities[manager_last])
            if manager_last is not None and hasattr(manager_buffer, "priorities") else 0.0
        ),
    }


# 【中文导读】冻结观测统计并关闭探索噪声，评估回报和电气约束指标。
def evaluate_policy(agents: AgentBundle, cfg: TrainConfig, episodes: int = 1, max_steps: int = EPISODE_STEPS,
                    seed: int = 12345) -> Dict[str, float]:
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：在无探索噪声下评估当前策略。
    
    输入：三层智能体、配置、episode 数、步数和种子。
    
    输出：统一安全指标与 return。
    
    核心步骤：冻结 normalizer/网络模式，运行确定性控制并调用 is_feasible。
    
    强化学习含义：评估与最佳模型选择必须共享同一安全逻辑。
    
    【容易混淆】训练 return 改善不等于可行。
    """

    previous_modes = (
        agents.manager.normalizer.training,
        agents.slow.normalizer.training,
        agents.fast.normalizer.training,
    )
    agents.manager.normalizer.eval()
    agents.slow.normalizer.eval()
    agents.fast.normalizer.eval()
    env = ElectricGasMultiScaleEnv()
    env.unexpected_env_exception_policy = getattr(cfg, "unexpected_env_exception_policy", "raise")
    returns: List[float] = []
    solver_failures = 0
    power_ok = 0
    gas_ok = 0
    soc_violation_steps = 0
    step_count = 0
    component_totals: Dict[str, float] = {}
    component_violation_steps: Dict[str, int] = {}
    hard_violation_steps = {
        "voltage": 0, "gas_pressure": 0, "line_overload": 0,
        "pipe_velocity": 0, "source_capacity": 0, "soc": 0,
    }
    extrema = {
        "min_voltage_pu": float("inf"), "max_voltage_pu": -float("inf"),
        "max_line_loading_percent": 0.0, "min_gas_pressure_bar": float("inf"),
        "max_gas_pressure_bar": -float("inf"), "max_pipe_velocity_m_per_s": 0.0,
        "max_voltage_violation": 0.0, "max_pressure_violation": 0.0,
        "max_line_overload": 0.0, "max_pipe_velocity_violation": 0.0,
        "max_source_capacity_violation": 0.0,
    }
    voltage_rms_deviations: List[float] = []
    gas_pressure_rms_deviations: List[float] = []
    try:
        for ep in range(episodes):
            global_obs, _ = env.reset(seed=seed + ep)
            builder = ObservationBuilder(env, cfg.manager_interval)
            current_goal: Optional[np.ndarray] = None
            previous_goal: Optional[np.ndarray] = None
            held_slow_action = rule_slow_action(env)
            held_slow_action, _ = apply_ess_action_guard(env, held_slow_action, cfg, cfg.slow_interval)
            last_manager_step = 0
            ep_return = 0.0
            for t in range(max_steps):
                manager_age = t - last_manager_step
                manager_obs = builder.manager_obs(global_obs, current_goal)
                if t % cfg.manager_interval == 0:
                    if cfg.training_stage in ("fast_pretrain", "slow_pretrain"):
                        current_goal = fixed_manager_goal()
                    else:
                        current_goal = agents.manager.select_goal(
                            manager_obs, previous_goal, 0.0, deterministic=True
                        )
                    previous_goal = current_goal.copy()
                    last_manager_step = t
                    manager_age = 0
                if current_goal is None:
                    current_goal = fixed_manager_goal()
                fast_obs = builder.fast_obs(manager_age, global_obs)
                slow_obs = builder.slow_obs(global_obs)
                if t % cfg.slow_interval == 0:
                    if cfg.training_stage == "fast_pretrain":
                        raw_slow_action = rule_slow_action(env)
                    else:
                        raw_slow_action = agents.slow.select_action(
                            slow_obs, current_goal, 0.0, deterministic=True
                        )
                    horizon_steps = min(
                        max(1, cfg.slow_interval),
                        max(1, max_steps - t),
                    )
                    held_slow_action, _ = apply_ess_action_guard(env, raw_slow_action, cfg, horizon_steps)
                fast_action = agents.fast.select_action(fast_obs, current_goal, 0.0, deterministic=True)
                # 【安全语义】环境动作严格为 Slow 10 维 + Fast 16 维；随后仍要经过环境物理安全投影。
                joint_action = np.concatenate([held_slow_action, fast_action]).astype(np.float32)
                global_obs, reward, terminated, truncated, info = safe_env_step(env, joint_action, global_obs)
                if (t + 1 >= max_steps and not terminated and not truncated
                        and max_steps < env.config.time.steps_per_day):
                    truncated = True
                    info["time_limit_truncated"] = True
                    reward = apply_debug_terminal_soc_penalty(env, info, reward, cfg)
                comps = info.get("reward_components", {})
                metrics = info.get("constraint_metrics", {})
                for key, value in comps.items():
                    component_totals[key] = component_totals.get(key, 0.0) + float(value)
                    component_violation_steps[key] = component_violation_steps.get(key, 0) + int(
                        abs(float(value)) > 1e-12
                    )
                if "voltage_rms_deviation_pu" in metrics:
                    voltage_rms_deviations.append(float(metrics["voltage_rms_deviation_pu"]))
                if "gas_pressure_rms_deviation_bar" in metrics:
                    gas_pressure_rms_deviations.append(float(metrics["gas_pressure_rms_deviation_bar"]))
                soc_min = float(metrics.get("soc_min", 0.5))
                soc_max = float(metrics.get("soc_max", 0.5))
                soc_violation_steps += int(
                    soc_min < min(item.soc_min for item in ESS_CONFIGS)
                    or soc_max > max(item.soc_max for item in ESS_CONFIGS)
                )
                vm_min = float(metrics.get("vm_min_pu", 1.0))
                vm_max = float(metrics.get("vm_max_pu", 1.0))
                pressure_min = float(metrics.get("gas_pressure_min_bar", env.config.gas.network_pressure_target_bar))
                pressure_max = float(metrics.get("gas_pressure_max_bar", env.config.gas.network_pressure_target_bar))
                line_max = float(metrics.get("max_line_loading_percent", 0.0))
                pipe_max = float(metrics.get("max_pipe_velocity_m_per_s", 0.0))
                source_values = np.nan_to_num(np.asarray(
                    metrics.get("source_capacity_violation_kg_s", []), dtype=np.float64
                ).reshape(-1), nan=0.0, posinf=float("inf"), neginf=0.0)
                source_max = float(np.max(source_values)) if source_values.size else 0.0
                voltage_violation = max(env.config.power.voltage_min_pu - vm_min, 0.0,
                                        vm_max - env.config.power.voltage_max_pu)
                pressure_violation = max(env.config.gas.network_pressure_min_bar - pressure_min, 0.0,
                                         pressure_max - env.config.gas.network_pressure_max_bar)
                line_violation = max(line_max - env.config.power.max_line_loading_percent, 0.0)
                pipe_violation = max(pipe_max - env.config.gas.max_pipe_velocity_m_per_s, 0.0)
                hard_violation_steps["voltage"] += int(voltage_violation > 0.0)
                hard_violation_steps["gas_pressure"] += int(pressure_violation > 0.0)
                hard_violation_steps["line_overload"] += int(line_violation > 0.0)
                hard_violation_steps["pipe_velocity"] += int(pipe_violation > 0.0)
                hard_violation_steps["source_capacity"] += int(source_max > 0.0)
                hard_violation_steps["soc"] += int(
                    soc_min < min(item.soc_min for item in ESS_CONFIGS)
                    or soc_max > max(item.soc_max for item in ESS_CONFIGS)
                )
                extrema["min_voltage_pu"] = min(extrema["min_voltage_pu"], vm_min)
                extrema["max_voltage_pu"] = max(extrema["max_voltage_pu"], vm_max)
                extrema["max_line_loading_percent"] = max(extrema["max_line_loading_percent"], line_max)
                extrema["min_gas_pressure_bar"] = min(extrema["min_gas_pressure_bar"], pressure_min)
                extrema["max_gas_pressure_bar"] = max(extrema["max_gas_pressure_bar"], pressure_max)
                extrema["max_pipe_velocity_m_per_s"] = max(extrema["max_pipe_velocity_m_per_s"], pipe_max)
                extrema["max_voltage_violation"] = max(extrema["max_voltage_violation"], voltage_violation)
                extrema["max_pressure_violation"] = max(extrema["max_pressure_violation"], pressure_violation)
                extrema["max_line_overload"] = max(extrema["max_line_overload"], line_violation)
                extrema["max_pipe_velocity_violation"] = max(extrema["max_pipe_velocity_violation"], pipe_violation)
                extrema["max_source_capacity_violation"] = max(
                    extrema["max_source_capacity_violation"], source_max
                )
                ep_return += float(reward)
                solver_failures += int(bool(info.get("solver_failed", False)))
                power_ok += int(bool(info.get("power_converged", False)))
                gas_ok += int(bool(info.get("gas_converged", False)))
                step_count += 1
                if terminated or truncated:
                    break
            returns.append(ep_return)
    finally:
        if previous_modes[0]:
            agents.manager.normalizer.train()
        if previous_modes[1]:
            agents.slow.normalizer.train()
        if previous_modes[2]:
            agents.fast.normalizer.train()
    denom = max(step_count, 1)
    result = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "solver_failures": float(solver_failures),
        "power_success_rate": float(power_ok / denom),
        "gas_success_rate": float(gas_ok / denom),
        "power_solver_success_rate": float(power_ok / denom),
        "gas_solver_success_rate": float(gas_ok / denom),
        "soc_violation_rate": float(soc_violation_steps / denom),
        "steps": float(step_count),
        "mean_voltage_rms_deviation_pu": float(np.mean(voltage_rms_deviations)) if voltage_rms_deviations else 0.0,
        "mean_gas_pressure_rms_deviation_bar": float(np.mean(gas_pressure_rms_deviations)) if gas_pressure_rms_deviations else 0.0,
        "voltage_deviation_cost": float(component_totals.get("voltage_deviation", 0.0)),
        "gas_pressure_deviation_cost": float(component_totals.get("gas_pressure_deviation", 0.0)),
        "voltage_violation_cost": float(component_totals.get("voltage_violation", 0.0)),
        "gas_pressure_violation_cost": float(component_totals.get("gas_pressure_violation", 0.0)),
        "pipe_velocity_violation_cost": float(component_totals.get("pipe_velocity_violation", 0.0)),
        "source_capacity_violation_cost": float(component_totals.get("source_capacity_violation", 0.0)),
        "gas_purchase_cost": float(component_totals.get("gas_purchase", 0.0)),
    }
    for key, value in hard_violation_steps.items():
        result[key + "_violation_rate"] = float(value / denom)
    result["line_overload_rate"] = result.pop("line_overload_violation_rate")
    for key, value in extrema.items():
        result[key] = float(value if np.isfinite(value) else 0.0)
    for key, total in component_totals.items():
        result[key + "_cost_total"] = float(total)
        result[key + "_cost_per_step"] = float(total / denom)
        result[key + "_violation_rate"] = float(component_violation_steps.get(key, 0) / denom)
    feasible, feasibility_reasons = is_feasible(result, cfg)
    result["constraint_feasibility"] = float(feasible)
    result["feasibility_reasons"] = feasibility_reasons
    return result


# =============================================================================
# Minimum tests
# =============================================================================
#
# 【模块说明：基础测试】覆盖维度、SMDP、Replay、冻结参数、checkpoint 和短训练。
# 测试用于证明代码契约，不代表正式训练已经达到安全可行。
#


# 【中文导读】最小测试中检查训练日志是否为有限数。
def assert_no_nan_tensor_dict(logs: Dict[str, float]) -> None:
    for key, value in logs.items():
        assert np.isfinite(value), f"{key} is not finite"


# 【中文导读】执行维度、动作保护、经验池、更新、checkpoint 和短训练冒烟测试。
def run_minimum_tests() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("Running minimum tests...")
    cfg = TrainConfig(episodes=1, episode_steps=2, batch_size=4, learning_starts=0,
                      no_tensorboard=True, checkpoint_dir="hierarchical_td3_test_runs",
                      fast_buffer_size=128, slow_buffer_size=64, manager_buffer_size=32)
    set_seed(cfg.seed)
    device = torch.device("cpu")

    from collections import Counter as _Counter
    from electric_gas_microgrid_single import (
        COMPRESSOR_CONFIGS as _TEST_COMPRESSOR_CONFIGS,
        COMPRESSOR_POWER_BUSES as _TEST_COMPRESSOR_POWER_BUSES,
        EXPECTED_COMPRESSOR_ARCS as _TEST_EXPECTED_COMPRESSOR_ARCS,
        EXPECTED_PASSIVE_PIPE_EDGES as _TEST_EXPECTED_PASSIVE_PIPE_EDGES,
        GAS_BASE_LOADS_KG_S as _TEST_GAS_BASE_LOADS_KG_S,
        GFG_CONFIGS as _TEST_GFG_CONFIGS,
        GAS_PIPES as _TEST_GAS_PIPES,
        GAS_SOURCE_NODES as _TEST_GAS_SOURCE_NODES,
        GAS_SUPPLIERS as _TEST_GAS_SUPPLIERS,
        P2G_CONFIGS as _TEST_P2G_CONFIGS,
        build_gas_network as _TEST_BUILD_GAS_NETWORK,
        dispatch_gfg as _TEST_DISPATCH_GFG,
        dispatch_p2g as _TEST_DISPATCH_P2G,
        run_fixed_compressor_station_consistency_test as _TEST_RUN_FIXED_COMPRESSOR_STATION_TEST,
        run_compressor_ratio_sensitivity_test as _TEST_RUN_COMPRESSOR_RATIO_SENSITIVITY_TEST,
        run_forced_solver_failure_rollback_test as _TEST_RUN_FORCED_ROLLBACK_TEST,
        run_gas_calibration_tests as _TEST_RUN_GAS_CALIBRATION_TESTS,
        validate_belgian20_topology as _TEST_VALIDATE_BELGIAN20_TOPOLOGY,
    )

    topology_result = _TEST_VALIDATE_BELGIAN20_TOPOLOGY()
    assert topology_result.ok, topology_result.errors
    assert len(_TEST_GAS_PIPES) == 21
    assert len(_TEST_COMPRESSOR_CONFIGS) == 2
    assert len(CONTROLLED_COMPRESSOR_INDICES) == 1
    assert not _TEST_COMPRESSOR_CONFIGS[0].controllable
    assert _TEST_COMPRESSOR_CONFIGS[1].controllable
    assert _TEST_COMPRESSOR_POWER_BUSES == (7, 30)
    assert {s.supplier_node for s in _TEST_GAS_SUPPLIERS} == set(_TEST_GAS_SOURCE_NODES)
    assert {s.supplier_node for s in _TEST_GAS_SUPPLIERS} == {0, 7}
    assert next(s for s in _TEST_GAS_SUPPLIERS if s.supplier_node == 7).pressure_bar is None
    assert set(_TEST_GAS_BASE_LOADS_KG_S) == {2, 5, 6, 9, 11, 14, 15, 18, 19}
    assert abs(sum(_TEST_GAS_BASE_LOADS_KG_S.values()) - 0.18) < 1e-4
    assert _Counter(tuple(sorted((p.from_junction, p.to_junction))) for p in _TEST_GAS_PIPES) == _Counter(
        tuple(sorted(edge)) for edge in _TEST_EXPECTED_PASSIVE_PIPE_EDGES
    )
    assert _Counter((c.from_junction, c.to_junction) for c in _TEST_COMPRESSOR_CONFIGS) == _Counter(
        _TEST_EXPECTED_COMPRESSOR_ARCS
    )
    gas_artifacts = _TEST_BUILD_GAS_NETWORK()
    assert len(gas_artifacts.compressor_indices) == 2
    assert len(gas_artifacts.ext_grid_indices) == 1
    assert gas_artifacts.pressure_ext_grid_index in gas_artifacts.ext_grid_indices
    assert gas_artifacts.auxiliary_source_index in gas_artifacts.net.source.index
    assert len(gas_artifacts.p2g_source_indices) == 3
    assert len(gas_artifacts.gfg_sink_indices) == 3
    calibration_results = _TEST_RUN_GAS_CALIBRATION_TESTS()
    assert len(calibration_results) == 4
    assert all(r["converged"] for r in calibration_results)
    fixed_station = _TEST_RUN_FIXED_COMPRESSOR_STATION_TEST()
    assert fixed_station["fixed_pressure_ratio"] > 1.0
    sensitivity = _TEST_RUN_COMPRESSOR_RATIO_SENSITIVITY_TEST()
    assert sensitivity["controlled_compressor_index"] == 1
    assert max(sensitivity["max_downstream_pressure_delta_bar"],
               sensitivity["power_delta_mw"]) > 0.0
    gfg_max_mdot = np.array([
        _TEST_DISPATCH_GFG(cfg, cfg.max_p_mw, 50.0).gas_mdot_kg_s
        for cfg in _TEST_GFG_CONFIGS
    ])
    p2g_max_mdot = np.array([
        _TEST_DISPATCH_P2G(cfg, cfg.max_p_mw, 50.0).gas_mdot_kg_s
        for cfg in _TEST_P2G_CONFIGS
    ])
    assert np.allclose(gfg_max_mdot, [0.105263, 0.105263, 0.083333], atol=1e-5)
    assert np.isclose(float(np.sum(gfg_max_mdot)), 0.293860, atol=1e-5)
    assert np.allclose(p2g_max_mdot, [0.021, 0.021, 0.013], atol=1e-6)
    assert np.isclose(float(np.sum(p2g_max_mdot)), 0.055, atol=1e-6)

    env = ElectricGasMultiScaleEnv()
    obs, info = env.reset(seed=cfg.seed)
    assert isinstance(obs, np.ndarray) and obs.size > 0, "environment reset failed"
    assert env.slow_action_dim == 10 and env.fast_action_dim == 16 and env.action_dim == 26
    assert env.n_comp == 2 and env.n_controlled_comp == 1
    assert env.config.time.steps_per_day == 480
    step_obs, step_reward, _, step_truncated, step_info = env.step(np.zeros(env.action_dim, dtype=np.float32))
    assert isinstance(step_obs, np.ndarray) and step_obs.size == obs.size
    assert np.isfinite(step_reward)
    assert not step_truncated and not step_info.get("solver_failed", False)
    assert step_info["constraint_metrics"]["compressor_applied_ratio"].shape == (2,)
    assert env.last_physical_slow["compressor_ratio"].shape == (2,)
    obs, info = env.reset(seed=cfg.seed)

    agents = build_agents(env, cfg, device)
    builder = ObservationBuilder(env, cfg.manager_interval)
    goal = fixed_manager_goal()
    fast_obs = builder.fast_obs(0, obs)
    slow_obs = builder.slow_obs(obs)
    a1 = agents.fast.select_action(fast_obs, goal, 0.10, deterministic=False)
    a2 = agents.fast.select_action(fast_obs + 0.01, goal, 0.10, deterministic=False)
    assert a1.shape == (env.fast_action_dim,) and a2.shape == (env.fast_action_dim,)
    assert np.all(a1 <= 1.0) and np.all(a1 >= -1.0)
    slow_action = agents.slow.select_action(slow_obs, goal, 0.0, deterministic=True)
    assert slow_action.shape == (env.slow_action_dim,)
    guarded_action, guard_adjustment = apply_ess_action_guard(
        env, slow_action, cfg, env.config.time.slow_action_interval_steps
    )
    assert guarded_action.shape == (env.slow_action_dim,)
    assert guard_adjustment >= 0.0
    from electric_gas_microgrid_single import ESS_CONFIGS as _TEST_ESS_CONFIGS
    env.ess_soc = np.array([ess.soc_min for ess in _TEST_ESS_CONFIGS], dtype=float)
    unsafe_discharge = np.zeros(env.slow_action_dim, dtype=np.float32)
    unsafe_discharge[:env.n_ess] = -1.0
    guarded_discharge, discharge_adjustment = apply_ess_action_guard(
        env, unsafe_discharge, cfg, env.config.time.slow_action_interval_steps
    )
    assert discharge_adjustment > 0.0
    assert np.all(guarded_discharge[:env.n_ess] >= -1e-6)
    manager_goal = agents.manager.select_goal(builder.manager_obs(obs), None, 0.0, deterministic=True)
    assert manager_goal.shape == (GOAL_DIM,)
    assert np.all(slow_action <= 1.0) and np.all(slow_action >= -1.0)

    slow_updates = [t for t in range(41) if t % SLOW_INTERVAL == 0]
    manager_updates = [t for t in range(81) if t % MANAGER_INTERVAL == 0]
    assert slow_updates == [0, 20, 40]
    assert manager_updates == [0, 40, 80]

    fb = FastReplayBuffer(128, fast_obs.size, env.fast_action_dim, GOAL_DIM, device)
    sb = SlowReplayBuffer(64, slow_obs.size, env.slow_action_dim, GOAL_DIM, device)
    mb = ManagerReplayBuffer(32, obs.size, GOAL_DIM, device)
    next_goal = goal.copy()
    next_goal[16:24] *= -1.0
    for _ in range(40):
        fb.add(fast_obs, fast_obs + 0.01, a1, a1 * 0.5, -1.0, 0.0, -1.0, goal, next_goal, False)
    for _ in range(2):
        sb.add(slow_obs, slow_obs + 0.01, slow_action, slow_action * 0.5, -2.0, goal, next_goal, False, 20)
    mb.add(obs, obs, goal, -3.0, False, 40)
    assert len(fb) == 40 and len(sb) == 2 and len(mb) == 1
    assert fb.goal_changed[0, 0] == 1.0
    assert np.allclose(fb.executed_actions[0], a1 * 0.5)

    tiny_worker = WorkerTD3("fast", 10, 2, 8, 3e-4, cfg, device)
    tiny_buffer = FastReplayBuffer(64, 10, 2, GOAL_DIM, device)
    for _ in range(16):
        o = np.random.randn(10).astype(np.float32)
        no = o + 0.01 * np.random.randn(10).astype(np.float32)
        ra = np.random.uniform(-1, 1, 2).astype(np.float32)
        tiny_buffer.add(o, no, ra, np.clip(ra * 0.9, -1, 1), -1.0, 0.1, -0.9, goal, goal, False)
    before = [p.detach().clone() for p in tiny_worker.encoder.parameters()]
    logs = tiny_worker.update(tiny_buffer, 8, cfg.gamma_fast)
    assert_no_nan_tensor_dict(logs)
    after = list(tiny_worker.encoder.parameters())
    assert any(not torch.allclose(b, a.detach()) for b, a in zip(before, after)), "encoder did not update"
    transition_cfg = copy.deepcopy(cfg)
    transition_cfg.use_transition_model = True
    transition_worker = WorkerTD3("fast", 10, 2, 8, 3e-4, transition_cfg, device)
    transition_logs = transition_worker.update(tiny_buffer, 8, transition_cfg.gamma_fast)
    assert_no_nan_tensor_dict(transition_logs)
    assert transition_logs["fast/transition_encoder_loss"] >= 0.0

    source = MLPEncoder(4, 4, 8)
    target = MLPEncoder(4, 4, 8)
    for p in source.parameters():
        p.data.add_(1.0)
    target_before = [p.detach().clone() for p in target.parameters()]
    soft_update(target, source, 0.5)
    assert any(not torch.allclose(b, a.detach()) for b, a in zip(target_before, target.parameters()))

    test_root = Path("hierarchical_td3_test_runs")
    test_root.mkdir(exist_ok=True)
    ckpt_path = test_root / "checkpoint_test.pt"
    save_checkpoint(ckpt_path, cfg, agents, 0, 0, -1.0, checkpoint_kind="lightweight")
    ckpt_payload = trusted_torch_load(str(ckpt_path), map_location=device)
    for required_key in (
        "env_model_version", "manager_observation_dim", "slow_observation_dim",
        "fast_observation_dim", "slow_action_dim", "fast_action_dim",
        "total_action_dim", "goal_dim", "n_controlled_compressors", "n_total_compressors",
    ):
        assert required_key in ckpt_payload
    new_agents = build_agents(env, cfg, device)
    load_checkpoint(str(ckpt_path), new_agents, device, mode="policy_only")
    out1 = agents.fast.select_action(fast_obs, goal, 0.0, deterministic=True)
    out2 = new_agents.fast.select_action(fast_obs, goal, 0.0, deterministic=True)
    assert np.allclose(out1, out2, atol=1e-5), "checkpoint load changed deterministic output"
    old_ckpt_path = test_root / "checkpoint_old_model_test.pt"
    old_payload = trusted_torch_load(str(ckpt_path), map_location=device)
    old_payload["env_model_version"] = "legacy_model"
    old_payload["slow_action_dim"] = 999
    old_payload.pop("slow_observation_dim", None)
    torch.save(old_payload, str(old_ckpt_path))
    try:
        load_checkpoint(
            str(old_ckpt_path), build_agents(env, cfg, device), device, mode="policy_only"
        )
        raise AssertionError("old checkpoint was loaded silently")
    except ValueError as exc:
        message = str(exc)
        assert "Checkpoint is incompatible" in message
        assert "missing slow_observation_dim" in message

    rollback_result = _TEST_RUN_FORCED_ROLLBACK_TEST()
    assert rollback_result["rollback_ok"] and rollback_result["next_step_ok"]

    eval_stats = evaluate_policy(agents, cfg, episodes=1, max_steps=2, seed=cfg.seed + 99)
    assert np.isfinite(eval_stats["mean_return"])

    class FailingEnv:
        unexpected_env_exception_policy = "truncate"

        def step(self, action: np.ndarray) -> Any:
            raise RuntimeError("forced failure")

    last = np.zeros_like(obs)
    _, reward, _, truncated, failed_info = safe_env_step(FailingEnv(), np.zeros(env.action_dim, dtype=np.float32), last)  # type: ignore[arg-type]
    assert truncated and reward < 0 and failed_info["solver_failed"]

    short_cfg = copy.deepcopy(cfg)
    short_cfg.episodes = 1
    short_cfg.episode_steps = 2
    short_cfg.training_stage = "fast_pretrain"
    short_cfg.run_mode = "debug"
    short_cfg.batch_size = 2
    short_cfg.learning_starts = 0
    short_cfg.device = "cpu"
    short_result = run_training(short_cfg)
    assert short_result["global_step"] >= 1
    assert np.isfinite(short_result["best_eval_return"])
    LOGGER.info("All minimum tests passed.")


# =============================================================================
# CLI
# =============================================================================
#
# 【模块说明：命令行入口】把命令行参数映射为 TrainConfig。正式训练使用优化版入口；
# 基础版只保留兼容和测试用途。
#


# 【中文导读】把命令行参数映射为 TrainConfig。
def parse_args() -> TrainConfig:
    """把命令行参数转换成 TrainConfig。"""

    parser = argparse.ArgumentParser(description="Multi-scale hierarchical TD3 for electric-gas microgrid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--episode-steps", type=int, default=EPISODE_STEPS)
    parser.add_argument("--manager-interval", type=int, default=MANAGER_INTERVAL)
    parser.add_argument("--training-stage", type=str, default="all",
                        choices=["fast_pretrain", "slow_pretrain", "manager_train", "joint_finetune", "all"])
    parser.add_argument("--run-mode", choices=("formal", "debug"), default="formal")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--slow-update-interval-steps", type=int, default=5)
    parser.add_argument("--manager-update-interval-steps", type=int, default=20)
    parser.add_argument("--gamma-fast", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--target-noise", type=float, default=0.10)
    parser.add_argument("--target-noise-clip", type=float, default=0.30)
    parser.add_argument("--target-q-clip-abs", type=float, default=20_000.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--fast-lr", type=float, default=3e-4)
    parser.add_argument("--slow-lr", type=float, default=3e-4)
    parser.add_argument("--manager-lr", type=float, default=1e-4)
    parser.add_argument("--joint-worker-lr", type=float, default=5e-5)
    parser.add_argument("--fast-buffer-size", type=int, default=200_000)
    parser.add_argument("--slow-buffer-size", type=int, default=50_000)
    parser.add_argument("--manager-buffer-size", type=int, default=10_000)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=str, default="hierarchical_td3_runs")
    parser.add_argument("--load-checkpoint", type=str, default="")
    parser.add_argument("--checkpoint-load-mode", choices=("resume", "stage_transfer", "policy_only"),
                        default="resume")
    parser.add_argument("--load-policy-only", action="store_true")
    parser.add_argument("--best-model-metric", choices=("feasible_then_return", "return"),
                        default="feasible_then_return")
    parser.add_argument("--use-transition-model", action="store_true")
    parser.add_argument("--fast-exploration-noise", type=float, default=0.15)
    parser.add_argument("--slow-exploration-noise", type=float, default=0.10)
    parser.add_argument("--manager-exploration-noise", type=float, default=0.05)
    parser.add_argument("--min-fast-exploration-noise", type=float, default=0.02)
    parser.add_argument("--min-slow-exploration-noise", type=float, default=0.02)
    parser.add_argument("--min-manager-exploration-noise", type=float, default=0.01)
    parser.add_argument("--noise-decay-episodes", type=int, default=200)
    parser.add_argument("--fast-random-warmup-steps", type=int, default=2_000)
    parser.add_argument("--slow-random-warmup-segments", type=int, default=128)
    parser.add_argument("--manager-random-warmup-segments", type=int, default=64)
    parser.add_argument("--warmup-blend-fraction", type=float, default=0.20)
    parser.add_argument("--goal-smoothing", type=float, default=0.20)
    parser.add_argument("--goal-change-penalty-weight", type=float, default=0.05)
    parser.add_argument("--lambda-projection", type=float, default=5.0)
    parser.add_argument("--worker-reward-clip-abs", type=float, default=1_000.0)
    parser.add_argument("--manager-reward-clip-abs", type=float, default=2_000.0)
    parser.add_argument("--worker-action-l2-weight", type=float, default=0.02)
    parser.add_argument("--projection-imitation-weight", type=float, default=0.0)
    parser.add_argument("--disable-ess-action-guard", action="store_true")
    parser.add_argument("--reachability-weight", type=float, default=0.0)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    if args.load_policy_only:
        args.checkpoint_load_mode = "policy_only"
    cfg = TrainConfig(
        seed=args.seed,
        episodes=args.episodes,
        episode_steps=args.episode_steps,
        manager_interval=args.manager_interval,
        training_stage=args.training_stage,
        run_mode=args.run_mode,
        device=args.device,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        updates_per_step=args.updates_per_step,
        slow_update_interval_steps=args.slow_update_interval_steps,
        manager_update_interval_steps=args.manager_update_interval_steps,
        gamma_fast=args.gamma_fast,
        tau=args.tau,
        target_noise=args.target_noise,
        target_noise_clip=args.target_noise_clip,
        target_q_clip_abs=args.target_q_clip_abs,
        gradient_clip=args.gradient_clip,
        fast_lr=args.fast_lr,
        slow_lr=args.slow_lr,
        manager_lr=args.manager_lr,
        joint_worker_lr=args.joint_worker_lr,
        fast_buffer_size=args.fast_buffer_size,
        slow_buffer_size=args.slow_buffer_size,
        manager_buffer_size=args.manager_buffer_size,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        checkpoint_dir=args.checkpoint_dir,
        load_checkpoint=args.load_checkpoint,
        load_policy_only=args.load_policy_only,
        checkpoint_load_mode=args.checkpoint_load_mode,
        best_model_metric=args.best_model_metric,
        use_transition_model=args.use_transition_model,
        fast_exploration_noise=args.fast_exploration_noise,
        slow_exploration_noise=args.slow_exploration_noise,
        manager_exploration_noise=args.manager_exploration_noise,
        min_fast_exploration_noise=args.min_fast_exploration_noise,
        min_slow_exploration_noise=args.min_slow_exploration_noise,
        min_manager_exploration_noise=args.min_manager_exploration_noise,
        noise_decay_episodes=args.noise_decay_episodes,
        fast_random_warmup_steps=args.fast_random_warmup_steps,
        slow_random_warmup_segments=args.slow_random_warmup_segments,
        manager_random_warmup_segments=args.manager_random_warmup_segments,
        warmup_blend_fraction=args.warmup_blend_fraction,
        goal_smoothing=args.goal_smoothing,
        goal_change_penalty_weight=args.goal_change_penalty_weight,
        lambda_projection=args.lambda_projection,
        worker_reward_clip_abs=args.worker_reward_clip_abs,
        manager_reward_clip_abs=args.manager_reward_clip_abs,
        worker_action_l2_weight=args.worker_action_l2_weight,
        projection_imitation_weight=args.projection_imitation_weight,
        use_ess_action_guard=not args.disable_ess_action_guard,
        reachability_weight=args.reachability_weight,
        no_tensorboard=args.no_tensorboard,
        run_tests=args.run_tests,
    )
    return cfg


# 【中文导读】顺序执行 fast、slow、manager、joint 四个训练阶段并传递 checkpoint。
def run_all_stages(cfg: TrainConfig) -> Dict[str, Any]:
    """【初学者重点】下面按固定结构说明这段代码，建议先看功能和强化学习含义，再回到实现细节。
    
    功能：顺序执行四阶段训练。
    
    输入：总配置及各阶段 episode 数。
    
    输出：每阶段结果和最终 checkpoint。
    
    核心步骤：Fast→Slow→Manager→Joint，传递最佳网络与连续 exploration offset。
    
    强化学习含义：预训练降低联合学习的非平稳性。
    
    【容易混淆】stage_transfer 不恢复上一阶段 Replay/优化器。
    """
    stages = ["fast_pretrain", "slow_pretrain", "manager_train", "joint_finetune"]#分组
    total = max(int(cfg.episodes), 1)#总训练轮数
    base = total // len(stages)#每个阶段基础训练轮数
    remainder = total % len(stages)
    counts = [base + (1 if i < remainder else 0) for i in range(len(stages))]
    counts = [max(1, c) for c in counts]#储存了每个阶段的训练轮数
    all_root = Path(cfg.checkpoint_dir) / ("all_stages_" + time.strftime("%Y%m%d_%H%M%S"))
    previous_checkpoint = cfg.load_checkpoint#上一阶段checkpoint
    results: Dict[str, Any] = {"stages": []}
    completed_stages = 0
    for stage, count in zip(stages, counts):#zip把stages和counts对应起来，stages是训练状态，counts是训练轮数
        stage_cfg = copy.deepcopy(cfg)
        stage_cfg.training_stage = stage
        stage_cfg.episodes = count
        stage_cfg.checkpoint_dir = str(all_root / stage)#储存快照地址
        stage_cfg.load_checkpoint = previous_checkpoint
        if previous_checkpoint:
            stage_cfg.checkpoint_load_mode = (
                cfg.checkpoint_load_mode if completed_stages == 0 else "stage_transfer"
            )
        stage_cfg.load_policy_only = stage_cfg.checkpoint_load_mode == "policy_only"
        LOGGER.info("Starting stage %s for %s episode(s).", stage, count)
        result = run_training(stage_cfg)
        result["stage"] = stage
        results["stages"].append(result)
        run_root = Path(result["run_root"])
        best_checkpoint = run_root / STAGE_BEST_FILES[stage]
        previous_checkpoint = str(best_checkpoint if best_checkpoint.exists() else run_root / "latest_policy.pt")
        completed_stages += 1
    results["latest_checkpoint"] = previous_checkpoint
    return results


# 【中文导读】命令行入口。
def main() -> None:
    """命令行入口：可运行最小测试、四阶段训练或单阶段训练。"""

    cfg = parse_args()
    if cfg.run_tests:
        run_minimum_tests()
    else:
        raise RuntimeError(
            "hierarchical_td3_electric_gas.py is a legacy compatibility module and is not a "
            "formal training entry point. Run hierarchical_td3_electric_gas_optimized.py instead."
        )


if __name__ == "__main__":
    main()
