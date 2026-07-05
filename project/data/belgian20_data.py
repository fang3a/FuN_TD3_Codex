"""Belgian 20 节点高压气网参考数据。

数据来源：
- Excel 表 `Belgian 20-node gas system`：节点负荷、压力上下限、供应商容量、
  管道端点、Wmn/Kmn 稳态流量系数。
- Excel 表 `Coupling`：燃气机组参考耦合节点。

注意：
Excel 中的 Wmn/Kmn 不是 pandapipes 可直接使用的长度、直径或粗糙度。本文件
将这些系数保留为 reference 字段，并用明确标注的暂定等效管道参数构建第一版
准稳态 pipeflow 模型。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from project.config import CompressorConfig


@dataclass(frozen=True)
class GasNodeData:
    """气网节点负荷与压力约束。

    demand_mm3_per_day 的单位是 million m3/day，后续会近似换算成 kg/s。
    """

    node: int
    demand_mm3_per_day: float
    p_min_bar: float
    p_max_bar: float


@dataclass(frozen=True)
class GasSupplierData:
    """供应商参考数据。supplier_node 为待校准映射。"""

    name: str
    supplier_node: int
    capacity_mm3_per_day: float
    marginal_cost_musd_per_mm3_day: float
    hourly_ramping_mm3_per_day: float
    needs_calibration: bool = True


@dataclass(frozen=True)
class GasPipeData:
    """气网管道数据。

    Wmn/Kmn 为文献稳态模型系数，只保留用于校准追踪。length_km、
    diameter_m、roughness_mm 是第一版 pandapipes 暂定等效参数。
    """

    name: str
    from_junction: int
    to_junction: int
    wmn_reference: float
    kmn_reference: float
    length_km: float
    diameter_m: float
    roughness_mm: float
    allow_parallel: bool = False
    needs_calibration: bool = True


@dataclass(frozen=True)
class CouplingReference:
    """Excel Coupling 表参考。IEEE 39 电网 bus 不混入 IEEE 33。"""

    unit_id: int
    ieee39_bus: int
    gas_node: int
    efficient_mm3_per_day_per_mw: float


STANDARD_GAS_DENSITY_KG_PER_M3 = 0.80
# 这里使用标准状态密度做一阶换算，真实工程应按气质、温度、压力重新校准。


def mm3_per_day_to_kg_per_s(value_mm3_per_day: float) -> float:
    """将 million m3/day 近似换算为 kg/s。

    该换算使用 STANDARD_GAS_DENSITY_KG_PER_M3，属于待校准气质参数。
    """

    return value_mm3_per_day * 1_000_000.0 * STANDARD_GAS_DENSITY_KG_PER_M3 / 86_400.0


GAS_NODES: Tuple[GasNodeData, ...] = (
    GasNodeData(0, 0.000, 30.0, 70.0),
    GasNodeData(1, 0.000, 30.0, 70.0),
    GasNodeData(2, 0.000, 30.0, 70.0),
    GasNodeData(3, 0.000, 30.0, 70.0),
    GasNodeData(4, 0.000, 30.0, 70.0),
    GasNodeData(5, 4.034, 30.0, 70.0),
    GasNodeData(6, 5.256, 30.0, 70.0),
    GasNodeData(7, 0.000, 30.0, 70.0),
    GasNodeData(8, 0.000, 30.0, 70.0),
    GasNodeData(9, 6.365, 30.0, 70.0),
    GasNodeData(10, 0.000, 30.0, 70.0),
    GasNodeData(11, 2.120, 30.0, 70.0),
    GasNodeData(12, 1.200, 30.0, 70.0),
    GasNodeData(13, 0.960, 30.0, 70.0),
    GasNodeData(14, 6.848, 30.0, 70.0),
    GasNodeData(15, 15.616, 30.0, 70.0),
    GasNodeData(16, 0.000, 30.0, 70.0),
    GasNodeData(17, 0.000, 30.0, 70.0),
    GasNodeData(18, 0.222, 30.0, 70.0),
    GasNodeData(19, 1.919, 30.0, 70.0),
)


GAS_SUPPLIERS: Tuple[GasSupplierData, ...] = (
    GasSupplierData("Sup_1", 0, 31.2, 0.03600, 3.12),
    GasSupplierData("Sup_2", 1, 24.0, 0.04320, 2.40),
    GasSupplierData("Sup_3", 2, 8.4, 0.03420, 4.20),
    GasSupplierData("Sup_4", 3, 4.8, 0.03240, 2.40),
    GasSupplierData("Sup_5", 4, 2.4, 0.04104, 1.20),
    GasSupplierData("Sup_6", 5, 2.4, 0.03888, 1.20),
)


TOTAL_GAS_DEMAND_PROFILE_MM3_PER_H: Tuple[float, ...] = (
    1.581,
    1.650,
    1.721,
    1.818,
    1.901,
    1.992,
    2.123,
    2.232,
    2.239,
    2.268,
    2.274,
    2.284,
)


def _temporary_pipe_parameters(wmn_reference: float) -> tuple[float, float, float]:
    """返回暂定等效管道参数。

    该函数不把 Wmn/Kmn 当作物理参数，只按通道强弱粗分等级，便于第一版
    pandapipes 模型连通并可求解。后续应整体替换为文献或工程管道参数。
    """

    # pandapipes 0.10.0 下，直接使用较保守的长度/直径会使 Excel 负荷水平
    # 下的高压网不收敛。这里采用“等效高通量”暂定参数：长度取基准的 0.5，
    # 管径取基准的 1.5。该处理仍是待校准参数，不代表文献真实管道。
    if wmn_reference >= 2.0:
        base_length_km, base_diameter_m = 40.0, 1.000
    elif wmn_reference >= 0.8:
        base_length_km, base_diameter_m = 55.0, 0.800
    elif wmn_reference >= 0.25:
        base_length_km, base_diameter_m = 35.0, 0.650
    else:
        base_length_km, base_diameter_m = 25.0, 0.500
    return 0.5 * base_length_km, 1.5 * base_diameter_m, 0.05


_PIPE_ROWS = (
    ("Pipe_1", 1, 2, 3.011689, 0.002752, True),
    ("Pipe_2", 1, 2, 3.011689, 0.002752, True),
    ("Pipe_3", 2, 3, 2.459034, 0.004128, True),
    ("Pipe_4", 2, 3, 2.459034, 0.004128, True),
    ("Pipe_5", 3, 4, 1.181283, 0.017890, False),
    ("Pipe_6", 5, 6, 0.316632, 0.013007, False),
    ("Pipe_7", 6, 7, 0.385558, 0.008772, False),
    ("Pipe_8", 7, 4, 0.476335, 0.005747, False),
    ("Pipe_9", 4, 14, 0.812192, 0.037844, False),
    ("Pipe_10", 9, 10, 1.346867, 0.013762, True),
    ("Pipe_11", 9, 10, 0.164342, 0.002711, True),
    ("Pipe_12", 10, 11, 1.204674, 0.017202, True),
    ("Pipe_13", 10, 11, 0.146992, 0.003388, True),
    ("Pipe_14", 11, 12, 0.929428, 0.028899, False),
    ("Pipe_15", 12, 13, 0.952380, 0.027523, False),
    ("Pipe_16", 13, 14, 2.693737, 0.003440, False),
    ("Pipe_17", 14, 15, 1.904760, 0.006881, False),
    ("Pipe_18", 15, 16, 1.204674, 0.017202, False),
    ("Pipe_19", 11, 17, 0.226814, 0.001427, False),
    ("Pipe_20", 18, 19, 0.041271, 0.006917, False),
    ("Pipe_21", 19, 20, 0.166790, 0.000519, False),
    ("Pipe_22", 9, 7, 0.385558, 0.008772, False),
    ("Pipe_23", 10, 14, 0.929428, 0.028899, False),
)


GAS_PIPES: Tuple[GasPipeData, ...] = tuple(
    # 这里把 Excel 中 1-based 的 gas node 转成 0-based 的 pandapipes junction。
    GasPipeData(
        name=name,
        from_junction=from_gnode - 1,
        to_junction=to_gnode - 1,
        wmn_reference=wmn,
        kmn_reference=kmn,
        length_km=_temporary_pipe_parameters(wmn)[0],
        diameter_m=_temporary_pipe_parameters(wmn)[1],
        roughness_mm=_temporary_pipe_parameters(wmn)[2],
        allow_parallel=allow_parallel,
    )
    for name, from_gnode, to_gnode, wmn, kmn, allow_parallel in _PIPE_ROWS
)


# Excel 文件没有在可解析表格中给出 compressor 物理参数。这里沿用旧脚本中
# Belgian 20 相关 compressor 拓扑，作为待校准的连通性补充。
COMPRESSOR_CONFIGS: Tuple[CompressorConfig, ...] = (
    CompressorConfig("COMP_8_to_18", from_junction=7, to_junction=17, initial_pressure_ratio=1.15, max_power_mw=0.25),
    CompressorConfig("COMP_14_to_15", from_junction=13, to_junction=14, initial_pressure_ratio=1.10, max_power_mw=0.25),
    CompressorConfig("COMP_17_to_18", from_junction=16, to_junction=17, initial_pressure_ratio=1.15, max_power_mw=0.25),
)


COUPLING_REFERENCES: Tuple[CouplingReference, ...] = (
    CouplingReference(1, ieee39_bus=30, gas_node=4, efficient_mm3_per_day_per_mw=0.006),
    CouplingReference(2, ieee39_bus=37, gas_node=5, efficient_mm3_per_day_per_mw=0.006),
    CouplingReference(3, ieee39_bus=36, gas_node=18, efficient_mm3_per_day_per_mw=0.006),
)


N_GAS_JUNCTIONS = 20
