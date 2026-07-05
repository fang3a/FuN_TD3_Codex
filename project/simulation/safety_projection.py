"""动作安全投影。

强化学习 Actor 输出的是连续动作，但这些动作不一定满足设备物理边界。
本文件负责把请求动作投影到可行集合，并记录投影幅度。投影幅度越大，说明
Actor 越常提出不可执行动作，训练脚本会用它作为惩罚信号。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from project.config import ESSConfig, RenewableConfig


@dataclass(frozen=True)
class ActionProjectionResult:
    """单个动作投影结果。"""

    raw_action: float
    applied_action: float
    projection_magnitude: float
    hit_boundary: bool


@dataclass(frozen=True)
class ESSProjectionBatch:
    """ESS 投影批结果。"""

    raw_p_mw: np.ndarray
    applied_p_mw: np.ndarray
    projection_magnitude_mw: np.ndarray
    hit_soc_boundary: np.ndarray


@dataclass(frozen=True)
class InverterProjection:
    """逆变器 P/Q 投影结果。"""

    p_actual_mw: float
    q_actual_mvar: float
    curtailment: float
    q_limit_mvar: float
    apparent_power_mva: float
    q_was_clipped: bool
    curtailment_was_clipped: bool


def project_ess_power(
    requested_p_mw: float,
    soc: float,
    ess: ESSConfig,
    dt_hours: float,
) -> ActionProjectionResult:
    """投影 ESS 功率，遵循 pandapower storage 符号。

    p_mw > 0：充电；p_mw < 0：放电。投影边界由当前 SOC、容量、效率、
    功率上限和当前步长共同决定。
    """

    if dt_hours <= 0.0:
        raise ValueError("dt_hours 必须为正")

    # 充电上界：当前 SOC 到 soc_max 还剩多少能量，再折算成当前步允许的 MW。
    max_charge_by_soc = (ess.soc_max - soc) * ess.capacity_mwh / (ess.eta_charge * dt_hours)
    # 放电下界是负数：当前 SOC 到 soc_min 可释放多少能量，再折算成功率下界。
    min_discharge_by_soc = (ess.soc_min - soc) * ess.capacity_mwh * ess.eta_discharge / dt_hours
    lower_bound = max(-ess.max_p_mw, min_discharge_by_soc)
    upper_bound = min(ess.max_p_mw, max_charge_by_soc)
    if lower_bound > upper_bound:
        lower_bound = upper_bound = 0.0

    # np.clip 就是投影到一维闭区间 [lower_bound, upper_bound]。
    applied = float(np.clip(requested_p_mw, lower_bound, upper_bound))
    projection = abs(applied - requested_p_mw)
    hit_boundary = projection > 1e-9 or applied <= lower_bound + 1e-9 or applied >= upper_bound - 1e-9
    return ActionProjectionResult(
        raw_action=float(requested_p_mw),
        applied_action=applied,
        projection_magnitude=float(projection),
        hit_boundary=bool(hit_boundary),
    )


def project_ess_batch(
    requested_p_mw: Sequence[float],
    soc: Sequence[float],
    ess_configs: Sequence[ESSConfig],
    dt_hours: float,
) -> ESSProjectionBatch:
    """批量投影 ESS 功率。"""

    results = [
        project_ess_power(float(p), float(s), ess, dt_hours)
        for p, s, ess in zip(requested_p_mw, soc, ess_configs)
    ]
    return ESSProjectionBatch(
        raw_p_mw=np.array([r.raw_action for r in results], dtype=float),
        applied_p_mw=np.array([r.applied_action for r in results], dtype=float),
        projection_magnitude_mw=np.array([r.projection_magnitude for r in results], dtype=float),
        hit_soc_boundary=np.array([r.hit_boundary for r in results], dtype=bool),
    )


def update_ess_soc(
    soc: float,
    p_mw: float,
    ess: ESSConfig,
    dt_hours: float,
) -> float:
    """按执行功率更新 SOC，不额外裁剪隐藏越限。"""

    if p_mw >= 0.0:
        # 充电时只有 eta_charge 比例的电能真正进入电池。
        delta_e_mwh = ess.eta_charge * p_mw * dt_hours
    else:
        # 放电时电池内部减少的能量大于对外放出的电能。
        delta_e_mwh = p_mw * dt_hours / ess.eta_discharge
    return float(soc + delta_e_mwh / ess.capacity_mwh)


def project_inverter_action(
    renewable: RenewableConfig,
    p_available_mw: float,
    requested_q_mvar: float,
    requested_curtailment: float,
) -> InverterProjection:
    """投影逆变器有功削减和无功，使 P²+Q²<=S²。"""

    # 先裁剪有功削减比例，再计算剩余有功 P。
    curtailment = float(np.clip(requested_curtailment, 0.0, renewable.max_curtailment))
    p_actual = max(0.0, p_available_mw) * (1.0 - curtailment)
    # 逆变器视在功率限制：S^2 = P^2 + Q^2，因此 Q 的可用范围由当前 P 决定。
    q_limit = float(np.sqrt(max(renewable.s_rated_mva**2 - p_actual**2, 0.0)))
    q_actual = float(np.clip(requested_q_mvar, -q_limit, q_limit))
    s_actual = float(np.sqrt(p_actual**2 + q_actual**2))
    return InverterProjection(
        p_actual_mw=float(p_actual),
        q_actual_mvar=q_actual,
        curtailment=curtailment,
        q_limit_mvar=q_limit,
        apparent_power_mva=s_actual,
        q_was_clipped=abs(q_actual - requested_q_mvar) > 1e-9,
        curtailment_was_clipped=abs(curtailment - requested_curtailment) > 1e-9,
    )
