"""Hierarchical TD3 for the electric-gas multi-scale microgrid environment.

The simulator is imported from ``electric_gas_microgrid_single.py``.  This file
only implements the learning system: Manager, slow Worker, fast Worker,
asynchronous replay semantics, normalization, checkpointing and lightweight
tests.

初学者阅读路线：
1. 先看 ``TrainConfig``，理解一个 episode 有多少步、三个智能体多久决策一次。
2. 再看 ``ManagerTD3`` 和 ``WorkerTD3``，它们都是 TD3，只是动作含义不同：
   Manager 输出 goal，Worker 输出环境动作。
3. 接着看 ``Pending*`` 和 ``run_training``，这是多时间尺度最核心的部分：
   快 Worker 每步入库，慢 Worker 每小时聚合一次，Manager 每约两小时聚合一次。
4. 最后看 ``evaluate_policy`` 与 CLI，它们负责加载 checkpoint 和无探索评估。

Typical quick check:

    & 'D:\\anaconda\\anaconda\\envs\\python_3_8\\python.exe' hierarchical_td3_electric_gas.py --run-tests

Short debug training:

    & 'D:\\anaconda\\anaconda\\envs\\python_3_8\\python.exe' hierarchical_td3_electric_gas.py --episodes 1 --episode-steps 20 --batch-size 16 --learning-starts 8
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
from typing import Any, Dict, List, Optional, Tuple

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


from electric_gas_microgrid_single import ElectricGasMultiScaleEnv


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


# =============================================================================
# Configuration
# =============================================================================


# 【中文导读】集中保存三层时间尺度、TD3、奖励塑形、探索、日志和阶段训练参数；gamma_slow/gamma_manager由快速步折扣指数换算。
@dataclass
class TrainConfig:
    """训练超参数集合。

    强化学习代码通常把“算法参数”和“实验开关”都放在一个配置类里。
    这里的参数可以粗分为：时间尺度、网络大小、TD3 学习参数、探索噪声、
    奖励塑形、日志/评估/checkpoint。理解这些分组，比逐个背默认值更重要。
    """

    # 实验基础设置：随机种子、训练轮数、每轮步数和运行设备。
    seed: int = 42
    episodes: int = 3
    episode_steps: int = EPISODE_STEPS
    manager_interval: int = MANAGER_INTERVAL
    slow_interval: int = SLOW_INTERVAL
    training_stage: str = "joint_finetune"
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
    target_q_clip_abs: float = 200_000.0
    gradient_clip: float = 1.0

    # 学习率。joint_finetune 阶段通常更保守，所以 Worker 可使用单独学习率。
    fast_lr: float = 3e-4
    slow_lr: float = 3e-4
    manager_lr: float = 1e-4
    joint_worker_lr: float = 1e-4

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
    goal_smoothing: float = 0.20
    goal_change_penalty_weight: float = 0.05

    # Worker 奖励由外在奖励、隐空间方向奖励、物理进展和安全投影惩罚组成。
    alpha_external: float = 1.0
    beta_latent: float = 0.10
    beta_physical: float = 0.20
    lambda_projection: float = 0.10
    worker_reward_clip_abs: float = 5_000.0
    manager_reward_clip_abs: float = 25_000.0
    worker_action_l2_weight: float = 1e-3
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


# 【中文导读】统一固定 Python、NumPy 和 PyTorch 随机源，便于复现实验。
def set_seed(seed: int) -> None:
    """固定随机种子，减少重复实验之间的随机差异。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 【中文导读】把 auto/cpu/cuda 配置解析为 PyTorch 设备。
def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


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
    """Manager goal = three L2-normalized directions plus tanh physical references."""
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
    """在线观测归一化器；评估时可冻结统计量。"""

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
            "eval_mean_high_pressure_rms_deviation_bar",
            "eval_mean_prs_pressure_rms_deviation_bar",
            "fast_buffer_size", "slow_buffer_size", "manager_buffer_size",
            "solver_failures", "gas_solve_count", "mean_action_projection",
            "max_action_projection", "mean_slow_action_projection",
            "max_slow_action_projection", "mean_fast_action_projection",
            "max_fast_action_projection", "mean_ess_action_guard",
            "max_ess_action_guard", "mean_goal_change", "max_goal_change",
            "mean_voltage_rms_deviation_pu", "mean_high_pressure_rms_deviation_bar",
            "mean_prs_pressure_rms_deviation_bar",
            "voltage_deviation_cost", "high_pressure_deviation_cost",
            "prs_pressure_deviation_cost", "voltage_violation_cost",
            "high_pressure_violation_cost", "prs_pressure_violation_cost",
            "gas_purchase_cost",
            "worker_reward_clips", "manager_reward_clips",
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


# 【中文导读】从环境全局状态构造 Manager、快 Worker、慢 Worker 的任务相关观测。
class ObservationBuilder:
    """从环境读取多层状态，并补充训练所需的少量上下文。"""

    def __init__(self, env: ElectricGasMultiScaleEnv, manager_interval: int):
        self.env = env
        self.manager_interval = manager_interval

    # 【中文导读】返回 Manager 使用的全局电—气状态。
    def manager_obs(self, fallback_global: Optional[np.ndarray] = None) -> np.ndarray:
        if hasattr(self.env, "get_manager_state"):
            return np.asarray(self.env.get_manager_state(), dtype=np.float32)
        return np.asarray(fallback_global, dtype=np.float32)

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
        global_obs = np.asarray(fallback_global if fallback_global is not None else self.env.get_global_state(),
                                dtype=np.float32)
        if hasattr(self.env, "get_slow_worker_state"):
            base = np.asarray(self.env.get_slow_worker_state(), dtype=np.float32)
        else:
            base = global_obs
        # 慢 Worker 偏向 ESS/GFG/P2G/压缩机，但保留时间预测特征帮助跨小时调度。
        power_summary = global_obs[-6:]
        return np.nan_to_num(np.concatenate([base, power_summary]).astype(np.float32), nan=0.0)


# =============================================================================
# Replay buffers
# =============================================================================


# 【中文导读】保存每个 3 分钟步的状态、raw/executed 动作、分项奖励、goal 与终止标志。
class FastReplayBuffer:
    """快 Worker 的经验池。

    一条样本对应一个 3 分钟快速步。这里同时保存 raw_action 和
    executed_action：raw_action 是 Actor 想做的动作，executed_action 是经过
    环境安全投影后真正执行的动作。Critic 学真实执行动作，能减少“学到不可行动作”
    的风险。
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
        self.idx = 0
        self.full = False

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
        self.idx += 1
        self.full = self.full or self.idx >= self.capacity

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
        }

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx


# 【中文导读】保存跨多个快速步的慢时间尺度 SMDP 片段及 duration_steps。
class SlowReplayBuffer:
    """慢 Worker 的经验池。

    慢动作每 20 个快速步才更新一次，所以一条慢样本不是单步 transition，
    而是一个“片段”：obs_start -> obs_end，并累积片段内折扣奖励。
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


# 【中文导读】保存一个 Manager goal 持续区间的全局起止状态和折扣外在回报。
class ManagerReplayBuffer:
    """Manager 的经验池。

    Manager 的动作是 32 维 goal。它不直接控制电网/气网，而是学习“给 Worker
    什么方向目标，能让未来一段时间的总回报更好”。
    """

    def __init__(self, capacity: int, obs_dim: int, goal_dim: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.global_obs_start = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.global_obs_end = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.manager_goals = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.discounted_external_reward = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.duration_steps = np.zeros((capacity, 1), dtype=np.float32)
        self.idx = 0
        self.full = False

    def add(self, global_obs_start: np.ndarray, global_obs_end: np.ndarray, manager_goal: np.ndarray,
            discounted_external_reward: float, done: bool, duration_steps: int) -> None:
        i = self.idx % self.capacity
        self.global_obs_start[i] = global_obs_start
        self.global_obs_end[i] = global_obs_end
        self.manager_goals[i] = manager_goal
        self.discounted_external_reward[i, 0] = discounted_external_reward
        self.dones[i, 0] = float(done)
        self.duration_steps[i, 0] = float(duration_steps)
        self.idx += 1
        self.full = self.full or self.idx >= self.capacity

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        max_idx = len(self)
        idx = np.random.randint(0, max_idx, size=batch_size)
        return {
            "obs": to_tensor(self.global_obs_start[idx], self.device),
            "next_obs": to_tensor(self.global_obs_end[idx], self.device),
            "goals": to_tensor(self.manager_goals[idx], self.device),
            "rewards": to_tensor(self.discounted_external_reward[idx], self.device),
            "dones": to_tensor(self.dones[idx], self.device),
            "duration_steps": to_tensor(self.duration_steps[idx], self.device),
        }

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx


# =============================================================================
# Networks
# =============================================================================


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
    """把原始观测压缩成 latent state。

    TD3 本身不要求 Encoder，但这个环境观测维度较高，而且 Manager/Worker
    需要在隐空间里比较“状态变化方向”，所以这里显式加了 Encoder。
    """

    def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = make_mlp(obs_dim, latent_dim, hidden_dim, layer_norm=True)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# 【中文导读】根据 Manager latent state 生成 32 维组合式 goal。
class ManagerActor(nn.Module):
    """Manager Actor：输入 latent state，输出 32 维 goal。"""

    def __init__(self, latent_dim: int, goal_dim: int, hidden_dim: int):
        super().__init__()
        self.net = make_mlp(latent_dim, goal_dim, hidden_dim, layer_norm=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return normalize_goal_tensor(self.net(z))


# 【中文导读】双 Q 网络，估计全局 latent state 与 goal 的长期价值。
class ManagerCritic(nn.Module):
    """Manager Critic：估计在当前 latent state 下给出某个 goal 的价值。"""

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
    """Worker Actor：输入自己的 latent state 和 Manager goal，输出归一化动作。"""

    def __init__(self, latent_dim: int, worker_goal_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = make_mlp(latent_dim + worker_goal_dim, action_dim, hidden_dim, layer_norm=True)

    def forward(self, z: torch.Tensor, worker_goal: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(torch.cat([z, worker_goal], dim=-1)))


# 【中文导读】双 Q 网络，估计状态、goal、实际执行动作的价值。
class WorkerCritic(nn.Module):
    """Worker Critic：估计状态、goal 与实际执行动作的 Q 值。"""

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


# 【中文导读】高层 TD3；动作是 goal，样本是约两小时的聚合片段。
class ManagerTD3:
    """Manager 层 TD3。

    与普通 TD3 的差别：Actor 的“动作”不是环境动作，而是给 Worker 的 goal；
    Critic 学的是一个 Manager 时间段内累计出来的回报。
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

    # 【中文导读】归一化全局观测，生成 goal，叠加探索噪声，并与上一 goal 平滑。
    def select_goal(self, obs: np.ndarray, previous_goal: Optional[np.ndarray], noise_std: float,
                    deterministic: bool = False) -> np.ndarray:
        """根据 Manager 观测选择 goal；训练时加探索噪声，评估时 deterministic。"""

        self.normalizer.update(obs)
        obs_n = self.normalizer.normalize(obs)
        with torch.no_grad():
            z = self.encoder(to_tensor(obs_n[None, :], self.device))
            goal = self.actor(z).cpu().numpy()[0]
        if not deterministic and noise_std > 0.0:
            goal += np.random.normal(0.0, noise_std, size=goal.shape).astype(np.float32)
            goal = normalize_goal_np(goal)
        if previous_goal is not None and self.cfg.goal_smoothing > 0.0:
            # goal 平滑能避免高层目标剧烈跳变，降低下层 Worker 学习难度。
            smoothed = (1.0 - self.cfg.goal_smoothing) * goal + self.cfg.goal_smoothing * previous_goal
            goal = normalize_goal_np(smoothed)
        return goal.astype(np.float32)

    # 【中文导读】用 Manager 聚合样本执行双 Critic 回归、延迟 Actor 更新和目标网络软更新。
    def update(self, buffer: ManagerReplayBuffer, batch_size: int) -> Dict[str, float]:
        if len(buffer) < batch_size:
            return {}
        data = buffer.sample(batch_size)
        obs = to_tensor(self.normalizer.normalize(data["obs"].cpu().numpy()), self.device)
        next_obs = to_tensor(self.normalizer.normalize(data["next_obs"].cpu().numpy()), self.device)
        goals = data["goals"]
        rewards = data["rewards"]
        dones = data["dones"]

        z = self.encoder(obs)
        with torch.no_grad():
            # TD3 目标：target actor 给 next_goal，target critic 给 next Q。
            # 加截断噪声是 TD3 的 target policy smoothing，可降低 Q 对尖锐动作的过拟合。
            next_z = self.target_encoder(next_obs)
            next_goal = self.target_actor(next_z)
            noise = torch.randn_like(next_goal) * self.cfg.target_noise
            next_goal = normalize_goal_tensor(next_goal + noise.clamp(-self.cfg.target_noise_clip,
                                                                      self.cfg.target_noise_clip))
            q1_next, q2_next = self.target_critic(next_z, next_goal)
            q_next = torch.minimum(q1_next, q2_next)
            target_q = rewards + (1.0 - dones) * self.cfg.gamma_manager * q_next
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

        actor_loss_value = 0.0
        if self.total_updates % self.cfg.policy_frequency == 0:
            # Delayed policy update：Critic 多学几步后再更新 Actor，是 TD3 的稳定技巧。
            z_pi = self.encoder(obs).detach()
            goals_pi = self.actor(z_pi)
            actor_loss = -self.critic.q_min(z_pi, goals_pi).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            clip_grad(self.actor.parameters(), self.cfg.gradient_clip)
            self.actor_optim.step()
            actor_loss_value = float(actor_loss.detach().cpu())
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


# 【中文导读】Slow/Fast 共用 TD3；Critic 学执行动作，Actor 生成归一化请求动作。
class WorkerTD3:
    """快/慢 Worker 共用的 TD3 实现。

    role="fast" 时动作是新能源逆变器的无功和削减；role="slow" 时动作是
    ESS、GFG、P2G、压缩机设定。二者网络结构相同，观测维度、动作维度和折扣因子不同。
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
        # Critic 回归使用环境真实执行动作，而不是 Actor 原始请求动作。
        # actions: [batch_size, action_dim]；raw_actions 仅用于投影诊断/模仿损失。
        actions = data["executed_actions"]
        rewards = data["rewards"]
        dones = data["dones"]
        goals = data["goals"]
        next_goals = data["next_goals"]
        raw_actions = data["raw_actions"]
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
            q1_next, q2_next = self.target_critic(next_z, next_wg, next_actions)
            # 半马尔可夫 TD 目标。Fast 使用 gamma_fast；Slow 当前传入固定 gamma_slow。
            # 对提前结束的短片段，理论上更严谨的折扣应为 gamma_fast ** duration_steps。
            target_q = rewards + (1.0 - dones) * gamma * torch.minimum(q1_next, q2_next)
            if self.cfg.target_q_clip_abs > 0.0:
                target_q = target_q.clamp(-self.cfg.target_q_clip_abs, self.cfg.target_q_clip_abs)

        q1, q2 = self.critic(z, wg, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        latent_norm_loss = z.pow(2).mean()
        transition_encoder_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        if self.transition_model is not None:
            # 让Encoder的隐空间也服务于可预测的状态变化，但不在这一步更新TransitionModel参数。
            with torch.no_grad():
                target_delta_for_encoder = self.target_encoder(next_obs) - z.detach()
            set_requires_grad(self.transition_model, False)
            try:
                pred_delta_for_encoder = self.transition_model(z, actions)
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

        transition_loss_value = 0.0
        if self.transition_model is not None and self.transition_optim is not None:
            with torch.no_grad():
                z_detached = self.encoder(obs).detach()
                next_z_target = self.target_encoder(next_obs)
                target_delta = next_z_target - z_detached
            pred_delta = self.transition_model(z_detached, actions)
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
            actions_pi = self.actor(z_pi, wg)
            actor_loss = -self.critic.q_min(z_pi, wg, actions_pi).mean()
            if self.cfg.worker_action_l2_weight > 0.0:
                actor_loss = actor_loss + self.cfg.worker_action_l2_weight * actions_pi.pow(2).mean()
            if self.cfg.projection_imitation_weight > 0.0:
                projection_imitation_loss = F.mse_loss(actions_pi, actions)
                actor_loss = actor_loss + self.cfg.projection_imitation_weight * projection_imitation_loss
                projection_imitation_loss_value = float(projection_imitation_loss.detach().cpu())
            if self.transition_model is not None and self.cfg.reachability_weight > 0.0:
                predicted_delta = self.transition_model(z_pi, actions_pi)
                direction = expanded_goal_direction_tensor(goals, self.role, self.latent_dim)
                reachability = 1.0 - F.cosine_similarity(predicted_delta, direction, dim=-1, eps=1e-8).mean()
                actor_loss = actor_loss + self.cfg.reachability_weight * reachability
            self.actor_optim.zero_grad()
            actor_loss.backward()
            clip_grad(self.actor.parameters(), self.cfg.gradient_clip)
            self.actor_optim.step()
            actor_loss_value = float(actor_loss.detach().cpu())
            soft_update(self.target_actor, self.actor, self.cfg.tau)
            soft_update(self.target_critic, self.critic, self.cfg.tau)
            soft_update(self.target_encoder, self.encoder, self.cfg.tau)

        self.total_updates += 1
        prefix = f"{self.role}/"
        return {
            prefix + "critic_loss": float(critic_loss.detach().cpu()),
            prefix + "actor_loss": actor_loss_value,
            prefix + "projection_imitation_loss": projection_imitation_loss_value,
            prefix + "sample_projection_mse": float(F.mse_loss(raw_actions, actions).detach().cpu()),
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
    """按环境维度创建 Manager、慢 Worker、快 Worker。"""

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


FAST_COMPONENTS = (
    "voltage_deviation", "voltage_violation", "line_overload",
    "power_loss", "renewable_curtailment", "solver_failure",
)
SLOW_COMPONENTS = (
    "high_pressure_deviation", "high_pressure_violation",
    "prs_pressure_deviation", "prs_pressure_violation",
    "soc_soft", "terminal_soc", "compressor_energy", "ess_action_change",
    "gfg_action_change", "p2g_action_change", "solver_failure",
)


# 【中文导读】从环境成本字典中选取本 Worker 负责的物理成本并取负作为外在奖励。
def external_reward_from_components(info: Dict[str, Any], keys: Tuple[str, ...]) -> float:
    """从环境 info 中抽取指定成本分量，并转成奖励符号。"""

    comps = info.get("reward_components", {})
    return -float(sum(float(comps.get(k, 0.0)) for k in keys))


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
def slow_physical_progress(obs: np.ndarray, next_obs: np.ndarray, goal: np.ndarray) -> float:
    if obs.size < 29 or next_obs.size < 29:
        return 0.0
    physical = goal[24:32]
    soc_ref = float(np.clip(0.50 + 0.30 * physical[0], 0.20, 0.80))
    soc_now = float(np.mean(np.abs(obs[:3] - soc_ref)))
    soc_next = float(np.mean(np.abs(next_obs[:3] - soc_ref)))
    gas_now = float(np.mean(np.maximum(np.abs(obs[9:29]) - 1.0, 0.0)))
    gas_next = float(np.mean(np.maximum(np.abs(next_obs[9:29]) - 1.0, 0.0)))
    return (soc_now + gas_now) - (soc_next + gas_next)


# 【中文导读】按配置合并外在奖励、latent 方向奖励、物理进展和投影惩罚。
def build_worker_reward(external: float, latent: float, physical: float, proj: float,
                        cfg: TrainConfig) -> float:
    """把 Worker 的外在奖励、内在奖励和安全投影惩罚合成单个标量。"""

    return (cfg.alpha_external * external +
            cfg.beta_latent * latent +
            cfg.beta_physical * physical +
            proj)


# 【中文导读】裁剪极端奖励，限制少数求解失败样本对 Q 回归的支配。
def clip_reward_value(value: float, clip_abs: float) -> Tuple[float, bool]:
    """限制写入ReplayBuffer的奖励幅度，避免少数灾难回报主导Critic。"""
    if clip_abs <= 0.0:
        return float(value), False
    clipped = float(np.clip(value, -clip_abs, clip_abs))
    return clipped, bool(abs(clipped - float(value)) > 1e-9)


# 【中文导读】按 episode 线性衰减探索噪声。
def scheduled_noise(initial: float, minimum: float, episode: int, decay_episodes: int) -> float:
    """线性退火探索噪声；长训后期减少由噪声造成的动作尖峰。"""
    if decay_episodes <= 0:
        return float(initial)
    fraction = min(max(float(episode) / float(decay_episodes), 0.0), 1.0)
    return float(initial + fraction * (minimum - initial))


# =============================================================================
# Action helpers and stage control
# =============================================================================


# 【中文导读】把物理压缩比反映射到 [-1,1] 动作空间。
def normalized_compressor_ratio(env: ElectricGasMultiScaleEnv, index: int, ratio: float) -> float:
    from electric_gas_microgrid_single import COMPRESSOR_CONFIGS
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
    from electric_gas_microgrid_single import COMPRESSOR_CONFIGS
    action = np.zeros(env.slow_action_dim, dtype=np.float32)
    cursor = 0
    action[cursor:cursor + env.n_ess] = 0.0
    cursor += env.n_ess
    action[cursor:cursor + env.n_gfg] = -0.40
    cursor += env.n_gfg
    action[cursor:cursor + env.n_p2g] = -0.60
    cursor += env.n_p2g
    for i, comp in enumerate(COMPRESSOR_CONFIGS):
        action[cursor + i] = normalized_compressor_ratio(env, i, comp.initial_pressure_ratio)
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


# 【中文导读】保存三层在线/目标网络、优化器、归一化器与训练元数据。
def save_checkpoint(path: Path, cfg: TrainConfig, agents: AgentBundle, episode: int,
                    global_step: int, best_return: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(cfg),
        "manager": agents.manager.state_dict(),
        "slow": agents.slow.state_dict(),
        "fast": agents.fast.state_dict(),
        "episode": episode,
        "global_step": global_step,
        "best_return": best_return,
    }
    torch.save(payload, str(path))


# 【中文导读】只加载 Encoder、Actor、归一化器等策略相关状态并重置目标网络。
def load_agent_policy_state(agent: Any, state: Dict[str, Any]) -> None:
    agent.encoder.load_state_dict(state["encoder"])
    agent.actor.load_state_dict(state["actor"])
    hard_update(agent.target_encoder, agent.encoder)
    hard_update(agent.target_actor, agent.actor)
    if hasattr(agent, "normalizer") and "normalizer" in state:
        agent.normalizer.load_state_dict(state["normalizer"])
    transition_state = state.get("transition_model")
    if getattr(agent, "transition_model", None) is not None and transition_state is not None:
        agent.transition_model.load_state_dict(transition_state)
    agent.total_updates = 0


# 【中文导读】按完整恢复或仅策略恢复两种模式载入 checkpoint。
def load_checkpoint(path: str, agents: AgentBundle, map_location: torch.device,
                    policy_only: bool = False) -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    if policy_only:
        load_agent_policy_state(agents.manager, payload["manager"])
        load_agent_policy_state(agents.slow, payload["slow"])
        load_agent_policy_state(agents.fast, payload["fast"])
    else:
        agents.manager.load_state_dict(payload["manager"])
        agents.slow.load_state_dict(payload["slow"])
        agents.fast.load_state_dict(payload["fast"])
    return payload


# 【中文导读】在评估回报刷新时保存命名不同但内容完整的 checkpoint。
def save_best_files(root: Path, agents: AgentBundle, cfg: TrainConfig, episode: int,
                    global_step: int, best_return: float) -> None:
    save_checkpoint(root / "latest_checkpoint.pt", cfg, agents, episode, global_step, best_return)
    save_checkpoint(root / "best_manager.pt", cfg, agents, episode, global_step, best_return)
    save_checkpoint(root / "best_slow_worker.pt", cfg, agents, episode, global_step, best_return)
    save_checkpoint(root / "best_fast_worker.pt", cfg, agents, episode, global_step, best_return)


# =============================================================================
# Training and evaluation
# =============================================================================


# 【中文导读】暂存刚执行的快动作，等待下一快观测后计算内在奖励并入库。
@dataclass
class PendingFastTransition:
    """尚未入库的快步样本。

    之所以 pending，是因为执行动作后才能拿到 next_obs，并计算内在奖励。
    """

    obs: np.ndarray
    raw_action: np.ndarray
    executed_action: np.ndarray
    reward_external: float
    projection: float
    goal: np.ndarray
    done: bool


# 【中文导读】累计一个慢动作保持区间内的折扣奖励、投影惩罚和持续步数。
@dataclass
class PendingSlowSegment:
    """慢动作片段缓存。

    慢动作会保持多个快速步，所以先把片段奖励累加，等下一个慢动作时再入库。
    """

    obs_start: np.ndarray
    goal: np.ndarray
    raw_action: np.ndarray
    executed_action: Optional[np.ndarray] = None
    discounted_reward: float = 0.0
    projection_penalty_sum: float = 0.0
    duration_steps: int = 0


# 【中文导读】累计一个 Manager goal 区间内的环境回报和持续步数。
@dataclass
class PendingManagerSegment:
    """Manager 片段缓存，记录一个 goal 持续期间累计到的环境回报。"""

    obs_start: np.ndarray
    goal: np.ndarray
    discounted_reward: float = 0.0
    duration_steps: int = 0


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
    """主训练循环。

    这是全文件最重要的函数。读它时可以把循环想成三层时钟：
    - 每 1 个快速步：快 Worker 选动作，环境前进一步；
    - 每 20 个快速步：慢 Worker 重新给 ESS/GFG/P2G/压缩机设定；
    - 每 40 个快速步：Manager 重新给两个 Worker 一个高层 goal。
    """

    set_seed(cfg.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = resolve_device(cfg.device)
    LOGGER.info("Using device: %s", device)

    # 1) 创建环境和三个智能体。环境动作维度来自电-气系统设备数量。
    env = ElectricGasMultiScaleEnv()
    if env.slow_action_dim != 12 or env.fast_action_dim != 16:
        LOGGER.warning("Expected slow=12 fast=16, got slow=%s fast=%s",
                       env.slow_action_dim, env.fast_action_dim)
    agents = build_agents(env, cfg, device)
    loaded_best_return = -float("inf")
    if cfg.load_checkpoint:
        LOGGER.info("Loading checkpoint from %s", cfg.load_checkpoint)
        payload = load_checkpoint(cfg.load_checkpoint, agents, device, policy_only=cfg.load_policy_only)
        if cfg.load_policy_only:
            LOGGER.info("Loaded policy/normalizer only; critics and optimizers are reinitialized for the current reward.")
        else:
            loaded_best = payload.get("best_return", -float("inf"))
            loaded_best_return = float(loaded_best) if loaded_best is not None else -float("inf")

    # 2) 根据环境实际返回的观测维度创建三类经验池。
    obs, _ = env.reset(seed=cfg.seed)
    builder = ObservationBuilder(env, cfg.manager_interval)
    manager_dim = builder.manager_obs(obs).size
    fast_dim = builder.fast_obs(0, obs).size
    slow_dim = builder.slow_obs(obs).size
    fast_buffer = FastReplayBuffer(cfg.fast_buffer_size, fast_dim, env.fast_action_dim, GOAL_DIM, device)
    slow_buffer = SlowReplayBuffer(cfg.slow_buffer_size, slow_dim, env.slow_action_dim, GOAL_DIM, device)
    manager_buffer = ManagerReplayBuffer(cfg.manager_buffer_size, manager_dim, GOAL_DIM, device)

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
    global_step = 0
    if cfg.load_checkpoint and cfg.load_policy_only:
        initial_eval_stats = evaluate_policy(
            agents, cfg, episodes=cfg.eval_episodes,
            max_steps=min(cfg.episode_steps, EPISODE_STEPS), seed=cfg.seed + 10_000,
        )
        best_eval_return = initial_eval_stats["mean_return"]
        save_best_files(run_root, agents, cfg, -1, global_step, best_eval_return)
        with open(run_root / "initial_eval.json", "w", encoding="utf-8") as f:
            json.dump(initial_eval_stats, f, indent=2, ensure_ascii=False)
        LOGGER.info(
            "Initial policy-only checkpoint eval %.3f saved as baseline best.",
            best_eval_return,
        )
    elif cfg.load_checkpoint and np.isfinite(best_eval_return):
        save_best_files(run_root, agents, cfg, -1, global_step, best_eval_return)

    for episode in range(cfg.episodes):
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
        high_pressure_rms_deviations: List[float] = []
        prs_pressure_rms_deviations: List[float] = []
        component_totals: Dict[str, float] = {}
        worker_reward_clips = 0
        manager_reward_clips = 0
        fast_noise = scheduled_noise(
            cfg.fast_exploration_noise, cfg.min_fast_exploration_noise,
            episode, cfg.noise_decay_episodes)
        slow_noise = scheduled_noise(
            cfg.slow_exploration_noise, cfg.min_slow_exploration_noise,
            episode, cfg.noise_decay_episodes)
        manager_noise = scheduled_noise(
            cfg.manager_exploration_noise, cfg.min_manager_exploration_noise,
            episode, cfg.noise_decay_episodes)
        writer.add_scalar("exploration/fast_noise", fast_noise, episode)
        writer.add_scalar("exploration/slow_noise", slow_noise, episode)
        writer.add_scalar("exploration/manager_noise", manager_noise, episode)
        current_goal: Optional[np.ndarray] = None
        previous_goal: Optional[np.ndarray] = None
        held_slow_action = rule_slow_action(env)
        held_slow_action, _ = apply_ess_action_guard(env, held_slow_action, cfg, cfg.slow_interval)
        pending_fast: Optional[PendingFastTransition] = None
        pending_slow: Optional[PendingSlowSegment] = None
        pending_manager: Optional[PendingManagerSegment] = None
        last_manager_step = 0
        done = False

        for t in range(min(cfg.episode_steps, EPISODE_STEPS)):
            # 构造三层各自的观测。Manager 看全局，fast/slow 看更偏任务的局部摘要。
            manager_age = t - last_manager_step
            manager_obs = builder.manager_obs(global_obs)
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
                                       pending_manager.duration_steps)
                previous_goal = current_goal
                if cfg.training_stage in ("fast_pretrain", "slow_pretrain"):
                    current_goal = fixed_manager_goal()
                else:
                    current_goal = agents.manager.select_goal(
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
                    discounted_reward=initial_manager_reward,
                )
                last_manager_step = t
                manager_age = 0
                fast_obs = builder.fast_obs(manager_age, global_obs)

            assert current_goal is not None
            if pending_fast is not None:
                # 上一个快速步现在拥有 next_obs 了，可以入快 Worker buffer。
                logs = finalize_fast_transition(pending_fast, fast_obs, current_goal, agents.fast, fast_buffer, cfg)
                worker_reward_clips += int(logs.get("fast_reward_clipped", 0.0))
                for k, v in logs.items():
                    writer.add_scalar(f"reward/{k}", v, global_step)
                pending_fast = None

            if t % cfg.slow_interval == 0:
                # 慢 Worker 到点后结束上一个慢片段，并产生新的慢动作。
                if pending_slow is not None:
                    logs = finalize_slow_segment(pending_slow, slow_obs, current_goal, agents.slow,
                                                 slow_buffer, cfg, False)
                    worker_reward_clips += int(logs.get("slow_reward_clipped", 0.0))
                    for k, v in logs.items():
                        writer.add_scalar(f"reward/{k}", v, global_step)
                if cfg.training_stage == "fast_pretrain":
                    raw_slow_action = rule_slow_action(env)
                else:
                    raw_slow_action = agents.slow.select_action(
                        slow_obs, current_goal, slow_noise, deterministic=False)
                horizon_steps = min(
                    max(1, cfg.slow_interval),
                    max(1, min(cfg.episode_steps, EPISODE_STEPS) - t),
                )
                held_slow_action, guard_adjustment = apply_ess_action_guard(
                    env, raw_slow_action, cfg, horizon_steps
                )
                ess_guard_adjustments.append(guard_adjustment)
                writer.add_scalar("action/ess_guard_adjustment", guard_adjustment, global_step)
                pending_slow = PendingSlowSegment(slow_obs.copy(), current_goal.copy(), held_slow_action.copy())

            if cfg.training_stage in ("slow_pretrain", "manager_train"):
                # 预训练慢层/Manager 时，快层不加噪声，减少下层随机性对上层学习的干扰。
                fast_action = agents.fast.select_action(fast_obs, current_goal, 0.0, deterministic=True)
            else:
                fast_action = agents.fast.select_action(
                    fast_obs, current_goal, fast_noise, deterministic=False)

            # 环境只接收一个完整动作向量：前 12 维慢动作保持，后 16 维快动作每步更新。
            # 固定动作布局：[ESS(3), GFG(3), P2G(3), compressor(3), inverter-Q(8), curtailment(8)]。
            joint_action = np.concatenate([held_slow_action, fast_action]).astype(np.float32)
            next_global_obs, env_reward, terminated, truncated, info = safe_env_step(env, joint_action, global_obs)
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
            fast_external = external_reward_from_components(info, FAST_COMPONENTS)
            slow_external = external_reward_from_components(info, SLOW_COMPONENTS)
            slow_proj_pen = projection_penalty(raw[:env.slow_action_dim], applied[:env.slow_action_dim],
                                               cfg.lambda_projection)
            fast_proj_pen = projection_penalty(raw[env.slow_action_dim:], applied[env.slow_action_dim:],
                                               cfg.lambda_projection)
            pending_fast = PendingFastTransition(
                # 快 Worker 样本先 pending，到下一步拿到 next_fast_obs 后再写入 buffer。
                fast_obs.copy(), fast_action.copy(), applied[env.slow_action_dim:].copy(),
                fast_external, fast_proj_pen, current_goal.copy(), bool(terminated or truncated),
            )
            if pending_slow is not None:
                # 慢 Worker 的奖励跨多个快速步折扣累加，直到下一次慢动作或 episode 结束。
                if info.get("slow_action_applied", False):
                    pending_slow.executed_action = applied[:env.slow_action_dim].copy()
                # 片段内部按快速步折扣：R_seg = Σ gamma_fast^k * r_{t+k}。
                pending_slow.discounted_reward += (cfg.gamma_fast ** pending_slow.duration_steps) * slow_external
                pending_slow.projection_penalty_sum += (
                    cfg.gamma_fast ** pending_slow.duration_steps
                ) * slow_proj_pen
                pending_slow.duration_steps += 1
            if pending_manager is not None:
                # Manager 直接看环境总奖励，学习 goal 对未来一段时间总回报的影响。
                # Manager 聚合完整环境回报；goal change penalty 已在片段初始化时一次性加入。
                pending_manager.discounted_reward += (cfg.gamma_fast ** pending_manager.duration_steps) * float(env_reward)
                pending_manager.duration_steps += 1
                manager_return += float(env_reward)

            comps = info.get("reward_components", {})
            metrics = info.get("constraint_metrics", {})
            for key, value in comps.items():
                component_totals[key] = component_totals.get(key, 0.0) + float(value)
            if "voltage_rms_deviation_pu" in metrics:
                voltage_rms_deviations.append(float(metrics["voltage_rms_deviation_pu"]))
            if "high_pressure_rms_deviation_bar" in metrics:
                high_pressure_rms_deviations.append(float(metrics["high_pressure_rms_deviation_bar"]))
            if "prs_pressure_rms_deviation_bar" in metrics:
                prs_pressure_rms_deviations.append(float(metrics["prs_pressure_rms_deviation_bar"]))
            episode_return += float(env_reward)
            step_rewards.append(float(env_reward))
            solver_failures += int(bool(info.get("solver_failed", False)))
            gas_solve_count = int(info.get("gas_solve_count", gas_solve_count))
            writer.add_scalar("reward/environment_step", float(env_reward), global_step)
            for key, value in comps.items():
                writer.add_scalar(f"components/{key}", float(value), global_step)
            for key, value in metrics.items():
                writer.add_scalar(f"constraints/{key}", float(value), global_step)
            writer.add_scalar("solver/failure_count", solver_failures, global_step)
            writer.add_scalar("solver/gas_solve_count", gas_solve_count, global_step)

            global_obs = next_global_obs
            done = bool(terminated or truncated)
            global_step += 1

            if global_step > cfg.learning_starts:
                # learning_starts 之前只收集经验；之后按配置频率更新各层 TD3。
                for _ in range(cfg.updates_per_step):
                    if train_flags["fast"]:
                        for k, v in agents.fast.update(fast_buffer, cfg.batch_size, cfg.gamma_fast).items():
                            writer.add_scalar(f"loss/{k}", v, global_step)
                    if train_flags["slow"] and global_step % max(1, cfg.slow_update_interval_steps) == 0:
                        for k, v in agents.slow.update(slow_buffer, cfg.batch_size, cfg.gamma_slow).items():
                            writer.add_scalar(f"loss/{k}", v, global_step)
                    if train_flags["manager"] and global_step % max(1, cfg.manager_update_interval_steps) == 0:
                        for k, v in agents.manager.update(manager_buffer, cfg.batch_size).items():
                            writer.add_scalar(f"loss/{k}", v, global_step)

            if done:
                break

        # 5) episode 结束时，把还没入库的 pending 片段收尾。
        final_manager_obs = builder.manager_obs(global_obs)
        # 这里传入的是近似 manager_age；异常提前截断时可能与真实 t-last_manager_step 不一致。
        final_fast_obs = builder.fast_obs(max(0, min(cfg.manager_interval, cfg.episode_steps)), global_obs)
        final_slow_obs = builder.slow_obs(global_obs)
        if current_goal is None:
            current_goal = fixed_manager_goal()
        if pending_fast is not None:
            logs = finalize_fast_transition(pending_fast, final_fast_obs, current_goal, agents.fast, fast_buffer, cfg)
            worker_reward_clips += int(logs.get("fast_reward_clipped", 0.0))
        if pending_slow is not None and pending_slow.duration_steps > 0:
            logs = finalize_slow_segment(pending_slow, final_slow_obs, current_goal, agents.slow, slow_buffer, cfg, done)
            worker_reward_clips += int(logs.get("slow_reward_clipped", 0.0))
        if pending_manager is not None and pending_manager.duration_steps > 0:
            manager_reward, clipped = clip_reward_value(
                pending_manager.discounted_reward, cfg.manager_reward_clip_abs)
            manager_reward_clips += int(clipped)
            manager_buffer.add(pending_manager.obs_start, final_manager_obs, pending_manager.goal,
                               manager_reward, done, pending_manager.duration_steps)

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
        mean_high_pressure_rms = float(np.mean(high_pressure_rms_deviations)) if high_pressure_rms_deviations else 0.0
        mean_prs_pressure_rms = float(np.mean(prs_pressure_rms_deviations)) if prs_pressure_rms_deviations else 0.0
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
        writer.add_scalar("episode/mean_high_pressure_rms_deviation_bar", mean_high_pressure_rms, episode)
        writer.add_scalar("episode/mean_prs_pressure_rms_deviation_bar", mean_prs_pressure_rms, episode)
        writer.add_scalar("episode/worker_reward_clips", worker_reward_clips, episode)
        writer.add_scalar("episode/manager_reward_clips", manager_reward_clips, episode)
        LOGGER.info("Episode %s return %.3f | buffers fast=%s slow=%s manager=%s | failures=%s",
                    episode, episode_return, len(fast_buffer), len(slow_buffer), len(manager_buffer),
                    solver_failures)

        eval_return = ""
        eval_solver_failures: Any = ""
        eval_power_success_rate: Any = ""
        eval_gas_success_rate: Any = ""
        eval_mean_voltage_rms: Any = ""
        eval_mean_high_pressure_rms: Any = ""
        eval_mean_prs_pressure_rms: Any = ""
        if (episode + 1) % max(1, cfg.eval_interval) == 0 or episode == cfg.episodes - 1:
            eval_stats = evaluate_policy(agents, cfg, episodes=cfg.eval_episodes, max_steps=min(cfg.episode_steps, EPISODE_STEPS),
                                         seed=cfg.seed + 10_000 + episode)
            eval_return = eval_stats["mean_return"]
            eval_solver_failures = eval_stats["solver_failures"]
            eval_power_success_rate = eval_stats["power_success_rate"]
            eval_gas_success_rate = eval_stats["gas_success_rate"]
            eval_mean_voltage_rms = eval_stats["mean_voltage_rms_deviation_pu"]
            eval_mean_high_pressure_rms = eval_stats["mean_high_pressure_rms_deviation_bar"]
            eval_mean_prs_pressure_rms = eval_stats["mean_prs_pressure_rms_deviation_bar"]
            writer.add_scalar("eval/return", eval_stats["mean_return"], episode)
            writer.add_scalar("eval/solver_failures", eval_stats["solver_failures"], episode)
            writer.add_scalar("eval/power_success_rate", eval_stats["power_success_rate"], episode)
            writer.add_scalar("eval/gas_success_rate", eval_stats["gas_success_rate"], episode)
            writer.add_scalar("eval/mean_voltage_rms_deviation_pu",
                              eval_stats["mean_voltage_rms_deviation_pu"], episode)
            writer.add_scalar("eval/mean_high_pressure_rms_deviation_bar",
                              eval_stats["mean_high_pressure_rms_deviation_bar"], episode)
            writer.add_scalar("eval/mean_prs_pressure_rms_deviation_bar",
                              eval_stats["mean_prs_pressure_rms_deviation_bar"], episode)
            if eval_stats["mean_return"] > best_eval_return:
                best_eval_return = eval_stats["mean_return"]
                save_best_files(run_root, agents, cfg, episode, global_step, best_eval_return)
        save_checkpoint(run_root / "latest_checkpoint.pt", cfg, agents, episode, global_step, best_eval_return)
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
            "eval_mean_high_pressure_rms_deviation_bar": eval_mean_high_pressure_rms,
            "eval_mean_prs_pressure_rms_deviation_bar": eval_mean_prs_pressure_rms,
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
            "mean_high_pressure_rms_deviation_bar": mean_high_pressure_rms,
            "mean_prs_pressure_rms_deviation_bar": mean_prs_pressure_rms,
            "voltage_deviation_cost": component_totals.get("voltage_deviation", 0.0),
            "high_pressure_deviation_cost": component_totals.get("high_pressure_deviation", 0.0),
            "prs_pressure_deviation_cost": component_totals.get("prs_pressure_deviation", 0.0),
            "voltage_violation_cost": component_totals.get("voltage_violation", 0.0),
            "high_pressure_violation_cost": component_totals.get("high_pressure_violation", 0.0),
            "prs_pressure_violation_cost": component_totals.get("prs_pressure_violation", 0.0),
            "gas_purchase_cost": component_totals.get("gas_purchase", 0.0),
            "worker_reward_clips": worker_reward_clips,
            "manager_reward_clips": manager_reward_clips,
        })

    writer.close()
    LOGGER.info("Training complete. Checkpoints: %s", run_root)
    return {
        "run_root": str(run_root),
        "global_step": global_step,
        "best_eval_return": best_eval_return,
        "fast_buffer_size": len(fast_buffer),
        "slow_buffer_size": len(slow_buffer),
        "manager_buffer_size": len(manager_buffer),
    }


# 【中文导读】冻结观测统计并关闭探索噪声，评估回报和电气约束指标。
def evaluate_policy(agents: AgentBundle, cfg: TrainConfig, episodes: int = 1, max_steps: int = EPISODE_STEPS,
                    seed: int = 12345) -> Dict[str, float]:
    """无探索噪声评估策略。

    评估时冻结 RunningMeanStd，避免把评估轨迹混入训练统计；动作也不加噪声。
    返回的统计量既包括回报，也包括电压/气压稳定性，方便判断策略是否只是“奖励变好”。
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
    returns: List[float] = []
    solver_failures = 0
    power_ok = 0
    gas_ok = 0
    step_count = 0
    component_totals: Dict[str, float] = {}
    voltage_rms_deviations: List[float] = []
    high_pressure_rms_deviations: List[float] = []
    prs_pressure_rms_deviations: List[float] = []
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
                manager_obs = builder.manager_obs(global_obs)
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
                        max(1, min(max_steps, EPISODE_STEPS) - t),
                    )
                    held_slow_action, _ = apply_ess_action_guard(env, raw_slow_action, cfg, horizon_steps)
                fast_action = agents.fast.select_action(fast_obs, current_goal, 0.0, deterministic=True)
                joint_action = np.concatenate([held_slow_action, fast_action]).astype(np.float32)
                global_obs, reward, terminated, truncated, info = safe_env_step(env, joint_action, global_obs)
                comps = info.get("reward_components", {})
                metrics = info.get("constraint_metrics", {})
                for key, value in comps.items():
                    component_totals[key] = component_totals.get(key, 0.0) + float(value)
                if "voltage_rms_deviation_pu" in metrics:
                    voltage_rms_deviations.append(float(metrics["voltage_rms_deviation_pu"]))
                if "high_pressure_rms_deviation_bar" in metrics:
                    high_pressure_rms_deviations.append(float(metrics["high_pressure_rms_deviation_bar"]))
                if "prs_pressure_rms_deviation_bar" in metrics:
                    prs_pressure_rms_deviations.append(float(metrics["prs_pressure_rms_deviation_bar"]))
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
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "solver_failures": float(solver_failures),
        "power_success_rate": float(power_ok / denom),
        "gas_success_rate": float(gas_ok / denom),
        "steps": float(step_count),
        "mean_voltage_rms_deviation_pu": float(np.mean(voltage_rms_deviations)) if voltage_rms_deviations else 0.0,
        "mean_high_pressure_rms_deviation_bar": float(np.mean(high_pressure_rms_deviations)) if high_pressure_rms_deviations else 0.0,
        "mean_prs_pressure_rms_deviation_bar": float(np.mean(prs_pressure_rms_deviations)) if prs_pressure_rms_deviations else 0.0,
        "voltage_deviation_cost": float(component_totals.get("voltage_deviation", 0.0)),
        "high_pressure_deviation_cost": float(component_totals.get("high_pressure_deviation", 0.0)),
        "prs_pressure_deviation_cost": float(component_totals.get("prs_pressure_deviation", 0.0)),
        "voltage_violation_cost": float(component_totals.get("voltage_violation", 0.0)),
        "high_pressure_violation_cost": float(component_totals.get("high_pressure_violation", 0.0)),
        "prs_pressure_violation_cost": float(component_totals.get("prs_pressure_violation", 0.0)),
        "gas_purchase_cost": float(component_totals.get("gas_purchase", 0.0)),
    }


# =============================================================================
# Minimum tests
# =============================================================================


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

    env = ElectricGasMultiScaleEnv()
    obs, info = env.reset(seed=cfg.seed)
    assert isinstance(obs, np.ndarray) and obs.size > 0, "environment reset failed"
    assert env.slow_action_dim == 12 and env.fast_action_dim == 16 and env.action_dim == 28
    assert env.config.time.steps_per_day == 480

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
    save_checkpoint(ckpt_path, cfg, agents, 0, 0, -1.0)
    new_agents = build_agents(env, cfg, device)
    load_checkpoint(str(ckpt_path), new_agents, device)
    out1 = agents.fast.select_action(fast_obs, goal, 0.0, deterministic=True)
    out2 = new_agents.fast.select_action(fast_obs, goal, 0.0, deterministic=True)
    assert np.allclose(out1, out2, atol=1e-5), "checkpoint load changed deterministic output"

    eval_stats = evaluate_policy(agents, cfg, episodes=1, max_steps=2, seed=cfg.seed + 99)
    assert np.isfinite(eval_stats["mean_return"])

    class FailingEnv:
        def step(self, action: np.ndarray) -> Any:
            raise RuntimeError("forced failure")

    last = np.zeros_like(obs)
    _, reward, _, truncated, failed_info = safe_env_step(FailingEnv(), np.zeros(env.action_dim, dtype=np.float32), last)  # type: ignore[arg-type]
    assert truncated and reward < 0 and failed_info["solver_failed"]

    short_cfg = copy.deepcopy(cfg)
    short_cfg.episodes = 1
    short_cfg.episode_steps = 2
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


# 【中文导读】把命令行参数映射为 TrainConfig。
def parse_args() -> TrainConfig:
    """把命令行参数转换成 TrainConfig。"""

    parser = argparse.ArgumentParser(description="Multi-scale hierarchical TD3 for electric-gas microgrid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--episode-steps", type=int, default=EPISODE_STEPS)
    parser.add_argument("--manager-interval", type=int, default=MANAGER_INTERVAL)
    parser.add_argument("--training-stage", type=str, default="joint_finetune",
                        choices=["fast_pretrain", "slow_pretrain", "manager_train", "joint_finetune", "all"])
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
    parser.add_argument("--target-q-clip-abs", type=float, default=200_000.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--fast-lr", type=float, default=3e-4)
    parser.add_argument("--slow-lr", type=float, default=3e-4)
    parser.add_argument("--manager-lr", type=float, default=1e-4)
    parser.add_argument("--joint-worker-lr", type=float, default=1e-4)
    parser.add_argument("--fast-buffer-size", type=int, default=200_000)
    parser.add_argument("--slow-buffer-size", type=int, default=50_000)
    parser.add_argument("--manager-buffer-size", type=int, default=10_000)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=str, default="hierarchical_td3_runs")
    parser.add_argument("--load-checkpoint", type=str, default="")
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
    parser.add_argument("--projection-imitation-weight", type=float, default=0.0)
    parser.add_argument("--disable-ess-action-guard", action="store_true")
    parser.add_argument("--reachability-weight", type=float, default=0.0)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    cfg = TrainConfig(
        seed=args.seed,
        episodes=args.episodes,
        episode_steps=args.episode_steps,
        manager_interval=args.manager_interval,
        training_stage=args.training_stage,
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
        use_transition_model=args.use_transition_model,
        fast_exploration_noise=args.fast_exploration_noise,
        slow_exploration_noise=args.slow_exploration_noise,
        manager_exploration_noise=args.manager_exploration_noise,
        min_fast_exploration_noise=args.min_fast_exploration_noise,
        min_slow_exploration_noise=args.min_slow_exploration_noise,
        min_manager_exploration_noise=args.min_manager_exploration_noise,
        noise_decay_episodes=args.noise_decay_episodes,
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
    """按 fast -> slow -> manager -> joint 顺序训练，上一阶段checkpoint自动接到下一阶段。"""
    stages = ["fast_pretrain", "slow_pretrain", "manager_train", "joint_finetune"]
    total = max(int(cfg.episodes), 1)
    base = total // len(stages)
    remainder = total % len(stages)
    counts = [base + (1 if i < remainder else 0) for i in range(len(stages))]
    counts = [max(1, c) for c in counts]
    all_root = Path(cfg.checkpoint_dir) / ("all_stages_" + time.strftime("%Y%m%d_%H%M%S"))
    previous_checkpoint = cfg.load_checkpoint
    initial_checkpoint = cfg.load_checkpoint
    results: Dict[str, Any] = {"stages": []}
    for stage, count in zip(stages, counts):
        stage_cfg = copy.deepcopy(cfg)
        stage_cfg.training_stage = stage
        stage_cfg.episodes = count
        stage_cfg.checkpoint_dir = str(all_root / stage)
        stage_cfg.load_checkpoint = previous_checkpoint
        stage_cfg.load_policy_only = bool(
            cfg.load_policy_only and initial_checkpoint and previous_checkpoint == initial_checkpoint
        )
        LOGGER.info("Starting stage %s for %s episode(s).", stage, count)
        result = run_training(stage_cfg)
        result["stage"] = stage
        results["stages"].append(result)
        run_root = Path(result["run_root"])
        best_checkpoint = run_root / "best_manager.pt"
        previous_checkpoint = str(best_checkpoint if best_checkpoint.exists() else run_root / "latest_checkpoint.pt")
    results["latest_checkpoint"] = previous_checkpoint
    return results


# 【中文导读】命令行入口。
def main() -> None:
    """命令行入口：可运行最小测试、四阶段训练或单阶段训练。"""

    cfg = parse_args()
    if cfg.run_tests:
        run_minimum_tests()
    elif cfg.training_stage == "all":
        run_all_stages(cfg)
    else:
        run_training(cfg)


if __name__ == "__main__":
    main()
