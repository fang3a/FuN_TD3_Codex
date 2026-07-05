"""电-气能量换算工具。

核心单位关系：1 MW = 1 MJ/s。天然气的 HHV 单位是 MJ/kg，因此
``MW / (MJ/kg)`` 会得到 kg/s。P2G 是电能变气体，GFG 是气体变电能。
"""

from __future__ import annotations


def p2g_power_to_gas_mdot_kg_s(electric_power_mw: float, efficiency: float, hhv_mj_per_kg: float) -> float:
    """P2G 电功率转天然气质量流量。

    公式：mdot = eta * P_MW / HHV_MJ_per_kg。因为 1 MW = 1 MJ/s。
    """

    # P2G：输入电功率 * 效率 = 注入气网的化学能流率。
    if hhv_mj_per_kg <= 0.0:
        raise ValueError("hhv_mj_per_kg 必须为正")
    return max(0.0, efficiency * max(0.0, electric_power_mw) / hhv_mj_per_kg)


def gfg_power_to_gas_mdot_kg_s(electric_power_mw: float, efficiency: float, hhv_mj_per_kg: float) -> float:
    """GFG 发电功率转天然气消耗质量流量。"""

    # GFG：输出电功率 / 效率 = 需要消耗的天然气化学能流率。
    if hhv_mj_per_kg <= 0.0:
        raise ValueError("hhv_mj_per_kg 必须为正")
    if efficiency <= 0.0:
        raise ValueError("efficiency 必须为正")
    return max(0.0, max(0.0, electric_power_mw) / (efficiency * hhv_mj_per_kg))


def gas_mdot_to_gfg_power_mw(gas_mdot_kg_s: float, efficiency: float, hhv_mj_per_kg: float) -> float:
    """GFG 天然气消耗质量流量转可发电功率。"""

    if hhv_mj_per_kg <= 0.0:
        raise ValueError("hhv_mj_per_kg 必须为正")
    return max(0.0, gas_mdot_kg_s) * efficiency * hhv_mj_per_kg
