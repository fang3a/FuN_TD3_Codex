"""燃气轮机 GFG 模型。"""

from __future__ import annotations

from dataclasses import dataclass

from project.config import GFGConfig
from project.coupling.energy_conversion import gfg_power_to_gas_mdot_kg_s


@dataclass(frozen=True)
class GFGDispatch:
    """GFG 调度结果。"""

    electric_power_mw: float
    gas_mdot_kg_s: float


def dispatch_gfg(config: GFGConfig, requested_power_mw: float, hhv_mj_per_kg: float) -> GFGDispatch:
    """根据发电功率决定耗气，不使用经验比例系数。"""

    # 智能体给的是期望出力，设备模型先裁剪到 [0, max_p_mw] 的物理范围。
    p_mw = min(max(requested_power_mw, 0.0), config.max_p_mw)
    # 裁剪后的发电功率再换算成气网 sink 的质量流量。
    mdot = gfg_power_to_gas_mdot_kg_s(p_mw, config.efficiency, hhv_mj_per_kg)
    return GFGDispatch(electric_power_mw=p_mw, gas_mdot_kg_s=mdot)
