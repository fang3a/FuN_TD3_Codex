"""P2G 电转气模型。"""

from __future__ import annotations

from dataclasses import dataclass

from project.config import P2GConfig
from project.coupling.energy_conversion import p2g_power_to_gas_mdot_kg_s


@dataclass(frozen=True)
class P2GDispatch:
    """P2G 调度结果。"""

    electric_power_mw: float
    gas_mdot_kg_s: float


def dispatch_p2g(config: P2GConfig, requested_power_mw: float, hhv_mj_per_kg: float) -> P2GDispatch:
    """根据电输入功率决定高压气网注气量。"""

    # P2G 是可调电负荷，先把请求功率裁剪到设备容量范围。
    p_mw = min(max(requested_power_mw, 0.0), config.max_p_mw)
    # 裁剪后的电功率再换算成气网 source 的质量流量。
    mdot = p2g_power_to_gas_mdot_kg_s(p_mw, config.efficiency, hhv_mj_per_kg)
    return P2GDispatch(electric_power_mw=p_mw, gas_mdot_kg_s=mdot)
