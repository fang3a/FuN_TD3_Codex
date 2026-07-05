"""电驱压缩机简化模型。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from project.config import CompressorConfig


@dataclass(frozen=True)
class CompressorDispatch:
    """压缩机执行结果。"""

    pressure_ratio: float
    mdot_kg_s: float
    electric_power_mw: float
    clipped_by_ratio: bool
    clipped_by_power: bool


def estimate_compressor_power_mw(
    mdot_kg_s: float,
    pressure_ratio: float,
    isentropic_efficiency: float,
    inlet_temperature_k: float = 293.15,
    cp_j_per_kg_k: float = 2_200.0,
    heat_capacity_ratio: float = 1.30,
) -> float:
    """估算电驱压缩机功率。

    简化假设：天然气视为定比热理想气体，忽略机械损失与变工况效率曲线。
    P = mdot * cp * T / eta * (r^((gamma-1)/gamma)-1)。
    """

    if pressure_ratio <= 1.0 or mdot_kg_s <= 0.0:
        return 0.0
    if isentropic_efficiency <= 0.0:
        raise ValueError("isentropic_efficiency 必须为正")
    exponent = (heat_capacity_ratio - 1.0) / heat_capacity_ratio
    watts = mdot_kg_s * cp_j_per_kg_k * inlet_temperature_k
    watts *= np.power(pressure_ratio, exponent) - 1.0
    watts /= isentropic_efficiency
    return float(max(watts / 1_000_000.0, 0.0))


def dispatch_compressor(
    config: CompressorConfig,
    requested_ratio: float,
    mdot_estimate_kg_s: float | None = None,
    inlet_pressure_bar: float | None = None,
    outlet_pressure_bar: float | None = None,
) -> CompressorDispatch:
    """计算压缩机实际压力比和电功率。

    压缩机只消耗电功率并调节压力，不创建 source 或天然气质量。
    """

    # 压缩机的控制量是压力比；先裁剪到设备允许范围。
    ratio = min(max(requested_ratio, config.min_pressure_ratio), config.max_pressure_ratio)
    clipped_by_ratio = abs(ratio - requested_ratio) > 1e-9
    if inlet_pressure_bar is not None and inlet_pressure_bar < config.inlet_min_bar:
        ratio = 1.0
        clipped_by_ratio = True
    if outlet_pressure_bar is not None and outlet_pressure_bar > config.outlet_max_bar:
        ratio = 1.0
        clipped_by_ratio = True

    # 功率估算需要流量；若 pandapipes 上一步没有可用估计，就使用名义流量。
    mdot = max(0.0, config.nominal_flow_kg_s if mdot_estimate_kg_s is None else mdot_estimate_kg_s)
    power_mw = estimate_compressor_power_mw(mdot, ratio, config.isentropic_efficiency)
    clipped_by_power = power_mw > config.max_power_mw
    if clipped_by_power:
        # 达到功率上限时，当前简化模型只裁剪电功率，不反求新的压力比。
        power_mw = config.max_power_mw

    return CompressorDispatch(
        pressure_ratio=float(ratio),
        mdot_kg_s=float(mdot),
        electric_power_mw=float(power_mw),
        clipped_by_ratio=clipped_by_ratio,
        clipped_by_power=clipped_by_power,
    )
