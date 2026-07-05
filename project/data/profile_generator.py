"""日内负荷、可再生能源与气负荷曲线生成。

这些曲线属于“外生输入”：智能体不能控制它们，只能根据当前观测和下一小时预测
做调度。为了可复现实验，函数接受 seed，并用平滑噪声生成每天略有差异的曲线。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from project.config import TimeConfig
from project.data.ieee33_data import RENEWABLE_CONFIGS


@dataclass(frozen=True)
class DailyProfiles:
    """一天的外生曲线。

    所有数组长度均为 steps_per_day + slow_action_interval_steps + 1，便于
    下一小时预测读取时不越界。环境实际仿真仍只运行 steps_per_day 步。
    """

    load_multiplier: np.ndarray
    gas_multiplier: np.ndarray
    renewable_available_mw: np.ndarray
    next_hour_load_multiplier: np.ndarray
    next_hour_renewable_available_mw: np.ndarray


def _smooth_noise(rng: np.random.Generator, n: int, scale: float, window: int) -> np.ndarray:
    """生成平滑随机扰动，避免负荷/风光曲线每 3 分钟剧烈跳变。"""

    raw = rng.normal(0.0, scale, n + window)
    kernel = np.ones(window) / window
    return np.convolve(raw, kernel, mode="valid")[:n]


def generate_daily_profiles(
    time_config: TimeConfig,
    seed: int | None = None,
    extra_steps: int | None = None,
) -> DailyProfiles:
    """生成 3 分钟分辨率的日曲线。

    曲线 t 对应 step 中当前时刻；step 返回观测时会使用 t+1 的编码与预测。
    """

    rng = np.random.default_rng(seed)
    horizon = time_config.steps_per_day
    pad = extra_steps if extra_steps is not None else time_config.slow_action_interval_steps + 1
    n = horizon + pad
    dt_hours = time_config.dt_hours
    hours = (np.arange(n) * dt_hours) % 24.0

    # 电负荷采用早晚双峰形状，再叠加平滑随机扰动。
    morning = 0.28 * np.exp(-0.5 * ((hours - 8.5) / 2.6) ** 2)
    evening = 0.42 * np.exp(-0.5 * ((hours - 19.0) / 3.0) ** 2)
    base_load = 0.58 + morning + evening
    load_multiplier = np.clip(base_load + _smooth_noise(rng, n, 0.035, 7), 0.40, 1.25)

    # 气负荷也采用早晚峰，但幅值和噪声略不同。
    gas_morning = 0.30 * np.exp(-0.5 * ((hours - 7.0) / 2.4) ** 2)
    gas_evening = 0.34 * np.exp(-0.5 * ((hours - 18.5) / 3.0) ** 2)
    gas_multiplier = np.clip(0.55 + gas_morning + gas_evening + _smooth_noise(rng, n, 0.025, 9), 0.35, 1.15)

    renewable_available_mw = np.zeros((n, len(RENEWABLE_CONFIGS)), dtype=float)
    for i, cfg in enumerate(RENEWABLE_CONFIGS):
        if cfg.kind == "pv":
            # 光伏按日照曲线建模，中午最高，清晨和夜晚为 0。
            sunrise, sunset = 5.5, 19.5
            daylight = np.clip((hours - sunrise) / (sunset - sunrise), 0.0, 1.0)
            solar_shape = np.sin(np.pi * daylight)
            cloud = np.clip(_smooth_noise(rng, n, 0.18, 11), -0.45, 0.30)
            available = cfg.capacity_mw * np.clip(solar_shape * (1.0 + cloud), 0.0, 1.0)
        else:
            # 风电没有昼夜归零，因此用较慢的正弦变化加平滑噪声。
            wind_base = 0.48 + 0.16 * np.sin(2 * np.pi * (hours - 2.0) / 24.0)
            wind = np.clip(wind_base + _smooth_noise(rng, n, 0.18, 13), 0.08, 1.00)
            available = cfg.capacity_mw * wind
        renewable_available_mw[:, i] = available

    # 慢速设备每小时调度一次，因此额外提供“下一小时”预测特征。
    next_idx = np.minimum(np.arange(n) + time_config.slow_action_interval_steps, n - 1)
    next_hour_load = load_multiplier[next_idx]
    next_hour_renew = renewable_available_mw[next_idx]

    return DailyProfiles(
        load_multiplier=load_multiplier.astype(float),
        gas_multiplier=gas_multiplier.astype(float),
        renewable_available_mw=renewable_available_mw.astype(float),
        next_hour_load_multiplier=next_hour_load.astype(float),
        next_hour_renewable_available_mw=next_hour_renew.astype(float),
    )


def profile_at(profiles: DailyProfiles, time_index: int) -> Dict[str, np.ndarray | float]:
    """安全读取单个时刻的外生曲线。"""

    idx = min(max(time_index, 0), len(profiles.load_multiplier) - 1)
    return {
        "load_multiplier": float(profiles.load_multiplier[idx]),
        "gas_multiplier": float(profiles.gas_multiplier[idx]),
        "renewable_available_mw": profiles.renewable_available_mw[idx].copy(),
        "next_hour_load_multiplier": float(profiles.next_hour_load_multiplier[idx]),
        "next_hour_renewable_available_mw": profiles.next_hour_renewable_available_mw[idx].copy(),
    }
