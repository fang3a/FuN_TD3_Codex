"""Standalone IEEE33 + Belgian-20-derived electric-gas coupled microgrid simulator.

This file is a single-file integration of the modular project under `project/`.
It contains:
- IEEE 33-bus distribution network data and pandapower builder
- Belgian-20-derived medium-pressure micro gas distribution network data and pandapipes builder
- GFG, P2G, electric compressor and ESS/inverter safety models
- Event-driven coupled solver
- Gymnasium-like multiscale environment
- Random-policy runner and topology/dashboard plotting

给强化学习初学者的定位：
这个文件把数据、物理模型、环境和随机策略全部放在一起，适合单步调试和
快速查看“环境到底做了什么”。如果想系统学习项目结构，优先阅读 `project/`
下的模块化版本；如果想确认训练脚本实际导入的环境，就看这里的
`ElectricGasMultiScaleEnv`。

Run with the tested conda environment:

    & 'D:\\anaconda\\anaconda\\envs\\python_3_8\\python.exe' electric_gas_microgrid_single.py --mode both
"""

from __future__ import annotations

# =============================================================================
# 中文详注版说明：仅增加注释，不修改任何可执行语句、变量名、默认参数或控制流。
# 阅读时优先关注时间尺度边界、raw/executed 动作、聚合折扣和目标网络更新。
# =============================================================================


import argparse
import copy
from collections import Counter
import csv
import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


# =============================================================================
# Configuration
# =============================================================================


ENV_MODEL_VERSION = "belgian20_derived_mp_v2"
GAS_MODEL_NAME = "Belgian-20-derived medium-pressure micro gas distribution network"


# 【中文导读】定义 3 分钟快速步、一天 480 步和 20 步慢动作周期。
@dataclass(frozen=True)#@dataclass是用来包装函数的，会自动执行 TimeConfig=dataclass(TimeConfig)的操作，其中包含初始化__Init__等
class TimeConfig:
    """时间尺度配置：3 分钟一步，一天 480 步，慢动作每 20 步更新。"""

    dt_minutes: int = 3
    steps_per_day: int = 480
    slow_action_interval_steps: int = 20

    @property
    def dt_hours(self) -> float:
        return self.dt_minutes / 60.0


# 【中文导读】定义 IEEE 33 节点配电网的基准量和安全边界。
@dataclass(frozen=True)
class PowerConfig:
    """IEEE 33 节点配电网的电压边界和基准容量。"""

    base_kv: float = 12.66
    base_mva: float = 10.0  #基准容量 Mvar是无功功率的单位，基准容量是有功功率与无功功率的共同标幺值，MVA是视在功率的单位
    slack_bus: int = 0  #基准节点
    slack_vm_pu: float = 1.0  #基准电压
    voltage_target_pu: float = 1.0  #目标电压
    voltage_min_pu: float = 0.95  #最小电压，0.95  #最小电压
    voltage_max_pu: float = 1.05  #最大电压，1.05  #最大电压
    max_line_loading_percent: float = 100.0  #最大线路加载率，100.0  


# 【中文导读】定义 Belgian-20-derived 中压微型配气网的压力边界、气体参数和换算常数。
@dataclass(frozen=True)
class GasConfig:
    """Belgian-20-derived medium-pressure micro gas distribution network parameters."""

    fluid_name: str = "lgas"
    network_pressure_min_bar: float = 2.5
    network_pressure_max_bar: float = 5.0
    network_pressure_target_bar: float = 4.0
    junction_initial_pressure_bar: float = 4.0
    gas_temperature_k: float = 293.15
    max_pipe_velocity_m_per_s: float = 12.0
    gas_compressibility_z: float = 0.95
    gas_specific_gas_constant_j_per_kg_k: float = 518.28
    hhv_mj_per_kg: float = 50.0


# 【中文导读】描述储能位置、功率、容量、效率与 SOC 边界。
@dataclass(frozen=True)
class ESSConfig:
    name: str
    bus: int
    max_p_mw: float
    capacity_mwh: float
    eta_charge: float
    eta_discharge: float
    soc_min: float = 0.10
    soc_max: float = 0.95
    soc_initial: float = 0.50#初始SOC


# 【中文导读】描述风光设备容量、逆变器额定容量和最大弃电率。
@dataclass(frozen=True)
class RenewableConfig:
    name: str
    bus: int
    kind: str
    capacity_mw: float
    s_rated_mva: float
    max_curtailment: float = 0.50


# 【中文导读】描述燃气发电机的电/气耦合位置、功率和效率。
@dataclass(frozen=True)
class GFGConfig:
    name: str
    power_bus: int
    gas_junction: int
    max_p_mw: float
    efficiency: float


# 【中文导读】描述电转气设备的电/气耦合位置、功率和效率。
@dataclass(frozen=True)
class P2GConfig:
    name: str
    power_bus: int
    gas_junction: int
    max_p_mw: float
    efficiency: float


# 【中文导读】描述压缩机节点、压比范围、效率和功率限制。
@dataclass(frozen=True)
class CompressorConfig:
    name: str
    from_junction: int
    to_junction: int
    min_pressure_ratio: float = 1.0
    max_pressure_ratio: float = 1.5
    initial_pressure_ratio: float = 1.2  #初始压比
    isentropic_efficiency: float = 0.75
    max_power_mw: float = 0.030
    nominal_flow_kg_s: float = 0.20
    inlet_min_bar: float = 2.5
    outlet_max_bar: float = 5.0
    equivalent_units: int = 1
    controllable: bool = True
    fixed_pressure_ratio: float | None = None
    needs_calibration: bool = False


# 【中文导读】定义触发气网重新求解的设备变化阈值。
@dataclass(frozen=True)
class EventConfig:
    gfg_mdot_threshold_kg_s: float = 0.002
    p2g_mdot_threshold_kg_s: float = 0.001
    compressor_ratio_threshold: float = 0.005
    gas_load_relative_threshold: float = 0.02


# 【中文导读】定义各物理成本权重；环境最终返回成本和的相反数。
@dataclass(frozen=True)
class RewardConfig:
    """奖励权重。代码中先计算成本，再用 reward = -sum(cost)。"""

    voltage_deviation: float = 25.0 #电压偏差权重
    voltage_violation: float = 500.0 #电压违规权重
    gas_pressure_deviation: float = 25.0
    gas_pressure_violation: float = 500.0
    pipe_velocity_violation: float = 50.0
    source_capacity_violation: float = 500.0
    line_overload: float = 10.0 #线路过载权重
    power_loss: float = 20.0 #功率损失权重
    grid_energy_price: float = 0.0 #电网电价权重
    gas_price: float = 0.0 #燃气电价权重
    renewable_curtailment: float = 40.0
    ess_action_change: float = 0.5 #储能动作变化权重
    gfg_action_change: float = 1.0 #燃气发电机动作变化权重
    p2g_action_change: float = 1.0 #电转气设备动作变化权重
    compressor_energy: float = 60.0 #压缩机能量权重
    soc_soft: float = 20.0 #SOC软边界权重
    solver_failure: float = 5000.0 #求解失败权重，5000.0  #求解失败截断权重
    terminal_soc: float = 200.0 #终端SOC权重，200.0  #终端SOC权重


# 【中文导读】定义求解失败截断、SOC 软边界和耦合迭代次数。
@dataclass(frozen=True)
class SafetyConfig:
    max_consecutive_solver_failures: int = 5
    soc_soft_low: float = 0.20
    soc_soft_high: float = 0.90
    solver_iterations: int = 2


# 【中文导读】聚合环境所有配置。
@dataclass(frozen=True)
class ProjectConfig:
    time: TimeConfig = field(default_factory=TimeConfig)#field使得所有实体类具有独立的属性
    power: PowerConfig = field(default_factory=PowerConfig)
    gas: GasConfig = field(default_factory=GasConfig)
    event: EventConfig = field(default_factory=EventConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    random_seed: int = 42


DEFAULT_CONFIG = ProjectConfig() #默认配置


# =============================================================================
# Data
# =============================================================================


IEEE33_LINE_DATA: Tuple[Tuple[int, int, float, float], ...] = (
    (0, 1, 0.0922, 0.0470), (1, 2, 0.4930, 0.2511),
    (2, 3, 0.3660, 0.1864), (3, 4, 0.3811, 0.1941),
    (4, 5, 0.8190, 0.7070), (5, 6, 0.1872, 0.6188),
    (6, 7, 0.7114, 0.2351), (7, 8, 1.0300, 0.7400),
    (8, 9, 1.0440, 0.7400), (9, 10, 0.1966, 0.0650),
    (10, 11, 0.3744, 0.1238), (11, 12, 1.4680, 1.1550),
    (12, 13, 0.5416, 0.7129), (13, 14, 0.5910, 0.5260),
    (14, 15, 0.7463, 0.5450), (15, 16, 1.2890, 1.7210),
    (16, 17, 0.7320, 0.5740), (1, 18, 0.1640, 0.1565),
    (18, 19, 1.5042, 1.3554), (19, 20, 0.4095, 0.4784),
    (20, 21, 0.7089, 0.9373), (2, 22, 0.4512, 0.3083),
    (22, 23, 0.8980, 0.7091), (23, 24, 0.8960, 0.7011),
    (5, 25, 0.2030, 0.1034), (25, 26, 0.2842, 0.1447),
    (26, 27, 1.0590, 0.9337), (27, 28, 0.8042, 0.7006),
    (28, 29, 0.5075, 0.2585), (29, 30, 0.9744, 0.9630),
    (30, 31, 0.3105, 0.3619), (31, 32, 0.3410, 0.5302),
)#在创建线路时，调用为(u,v,r,x)元组，u是起始节点，v是终点节点，r是电阻每千米，x是电感


IEEE33_LOAD_DATA: Tuple[Tuple[int, float, float], ...] = (
    (1, 0.100, 0.060), (2, 0.090, 0.040), (3, 0.120, 0.080),
    (4, 0.060, 0.030), (5, 0.060, 0.020), (6, 0.200, 0.100),
    (7, 0.200, 0.100), (8, 0.060, 0.020), (9, 0.060, 0.020),
    (10, 0.045, 0.030), (11, 0.060, 0.035), (12, 0.060, 0.035),
    (13, 0.120, 0.080), (14, 0.060, 0.010), (15, 0.060, 0.020),
    (16, 0.060, 0.020), (17, 0.090, 0.040), (18, 0.090, 0.040),
    (19, 0.090, 0.040), (20, 0.090, 0.040), (21, 0.090, 0.040),
    (22, 0.090, 0.050), (23, 0.420, 0.200), (24, 0.420, 0.200),
    (25, 0.060, 0.025), (26, 0.060, 0.025), (27, 0.060, 0.020),
    (28, 0.120, 0.070), (29, 0.200, 0.600), (30, 0.150, 0.070),
    (31, 0.210, 0.100), (32, 0.060, 0.040),
)


ESS_CONFIGS: Tuple[ESSConfig, ...] = (
    ESSConfig("ESS_0", bus=6, max_p_mw=1.0, capacity_mwh=4.0, eta_charge=0.92, eta_discharge=0.92),
    ESSConfig("ESS_1", bus=15, max_p_mw=1.0, capacity_mwh=4.0, eta_charge=0.92, eta_discharge=0.92),
    ESSConfig("ESS_2", bus=29, max_p_mw=0.5, capacity_mwh=2.0, eta_charge=0.90, eta_discharge=0.90),
)


RENEWABLE_CONFIGS: Tuple[RenewableConfig, ...] = (
    RenewableConfig("PV_0", 9, "pv", 1.0, 1.08),
    RenewableConfig("PV_1", 13, "pv", 1.0, 1.08),
    RenewableConfig("WT_0", 17, "wind", 1.5, 1.60),
    RenewableConfig("WT_1", 20, "wind", 1.5, 1.60),
    RenewableConfig("PV_2", 23, "pv", 1.0, 1.08),
    RenewableConfig("PV_3", 24, "pv", 1.0, 1.08),
    RenewableConfig("WT_2", 31, "wind", 1.0, 1.05),
    RenewableConfig("PV_4", 32, "pv", 1.0, 1.08),
)


GFG_CONFIGS: Tuple[GFGConfig, ...] = (
    GFGConfig("GFG_0", power_bus=18, gas_junction=4, max_p_mw=2.0, efficiency=0.38),
    GFGConfig("GFG_1", power_bus=22, gas_junction=5, max_p_mw=2.0, efficiency=0.38),
    GFGConfig("GFG_2", power_bus=32, gas_junction=18, max_p_mw=1.5, efficiency=0.36),
)


P2G_CONFIGS: Tuple[P2GConfig, ...] = (
    P2GConfig("P2G_0", power_bus=7, gas_junction=7, max_p_mw=1.5, efficiency=0.70),
    P2GConfig("P2G_1", power_bus=24, gas_junction=14, max_p_mw=1.5, efficiency=0.70),
    P2GConfig("P2G_2", power_bus=30, gas_junction=19, max_p_mw=1.0, efficiency=0.65),
)


N_POWER_BUSES = 33
N_GAS_JUNCTIONS = 20
COMPRESSOR_POWER_BUSES = (7, 30)
STANDARD_GAS_DENSITY_KG_PER_M3 = 0.80 #标准气体密度，单位：kg/m³


# 【中文导读】Belgian-20-derived 中压微型配气网节点负荷与压力边界数据。
@dataclass(frozen=True)
class GasNodeData:
    node: int
    name: str
    base_mdot_kg_s: float
    p_min_bar: float
    p_max_bar: float


# 【中文导读】气源容量、边际成本与爬坡数据。
@dataclass(frozen=True)
class GasSupplierData:
    name: str
    supplier_node: int
    pressure_bar: float | None
    max_mdot_kg_s: float
    marginal_cost: float
    aux_source_share: float = 0.0
    needs_calibration: bool = True


# 【中文导读】气管拓扑和暂定等效物理参数。
@dataclass(frozen=True)
class GasPipeData:
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


# 【中文导读】把百万立方米/日换算为 kg/s。
def mm3_per_day_to_kg_per_s(value_mm3_per_day: float) -> float:
    """Report-only legacy conversion helper; not used by v2 hydraulic solves."""

    return value_mm3_per_day * 1_000_000.0 * STANDARD_GAS_DENSITY_KG_PER_M3 / 86_400.0


GAS_NODE_NAMES: Tuple[str, ...] = (
    "Zeebrugge", "Dudzele", "Brugge", "Zomergem", "Loenhout",
    "Antwerpen", "Gent", "Voeren", "Berneau", "Liege",
    "Warnand", "Namur", "Anderlues", "Peronnes", "Mons",
    "Blaregnies", "Wanze", "Sinsin", "Arlon", "Petange",
)

GAS_SOURCE_NODES = frozenset({0, 7})
GAS_BASE_LOADS_KG_S: Dict[int, float] = {
    2: 0.01522,
    5: 0.01568,
    6: 0.02043,
    9: 0.02481,
    11: 0.00824,
    14: 0.02661,
    15: 0.06069,
    18: 0.00086,
    19: 0.00745,
}
GAS_TRANSIT_NODES = frozenset(set(range(N_GAS_JUNCTIONS)) - set(GAS_SOURCE_NODES) - set(GAS_BASE_LOADS_KG_S))

GAS_NODES: Tuple[GasNodeData, ...] = tuple(
    GasNodeData(
        node=i,
        name=GAS_NODE_NAMES[i],
        base_mdot_kg_s=GAS_BASE_LOADS_KG_S.get(i, 0.0),
        p_min_bar=DEFAULT_CONFIG.gas.network_pressure_min_bar,
        p_max_bar=DEFAULT_CONFIG.gas.network_pressure_max_bar,
    )
    for i in range(N_GAS_JUNCTIONS)
)


GAS_SUPPLIERS: Tuple[GasSupplierData, ...] = (
    GasSupplierData(
        name="MAIN_CITY_GATE",
        supplier_node=0,
        pressure_bar=4.5,
        max_mdot_kg_s=0.55,
        marginal_cost=0.0,
        needs_calibration=True,
    ),
    GasSupplierData(
        name="AUXILIARY_GATE_AT_VOEREN",
        supplier_node=7,
        pressure_bar=None,
        max_mdot_kg_s=0.20,
        marginal_cost=0.0,
        aux_source_share=0.30,
        needs_calibration=True,
    ),
)


EXPECTED_PASSIVE_PIPE_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, 1), (1, 2), (1, 2), (2, 3),
    (4, 5), (5, 6), (6, 3), (3, 13),
    (8, 9), (8, 9), (9, 10), (9, 10), (10, 11),
    (11, 12), (12, 13), (13, 14), (14, 15),
    (10, 16), (17, 18), (18, 19),
)
EXPECTED_COMPRESSOR_ARCS: Tuple[Tuple[int, int], ...] = ((7, 8), (16, 17))
FORBIDDEN_PASSIVE_PIPE_EDGES = frozenset({tuple(sorted((7, 9))), tuple(sorted((10, 14)))})
FORBIDDEN_COMPRESSOR_ARCS = frozenset({(7, 17), (13, 14)})
GAS_SUPPLIER_CAPACITY_KG_S: Dict[int, float] = {0: 0.55, 7: 0.20}

_PIPE_ROUGHNESS_MM = 0.05
_PIPE_ROWS = (
    ("Pipe_01_0_1_A", 0, 1, 0.120, 0.180, True),
    ("Pipe_02_0_1_B", 0, 1, 0.120, 0.180, True),
    ("Pipe_03_1_2_A", 1, 2, 0.180, 0.180, True),
    ("Pipe_04_1_2_B", 1, 2, 0.180, 0.180, True),
    ("Pipe_05_2_3", 2, 3, 0.780, 0.180, False),
    ("Pipe_06_4_5", 4, 5, 1.290, 0.140, False),
    ("Pipe_07_5_6", 5, 6, 0.870, 0.140, False),
    ("Pipe_08_6_3", 6, 3, 0.570, 0.140, False),
    ("Pipe_09_3_13", 3, 13, 1.650, 0.180, False),
    ("Pipe_10_8_9_A", 8, 9, 0.600, 0.180, True),
    ("Pipe_11_8_9_B", 8, 9, 0.600, 0.100, True),
    ("Pipe_12_9_10_A", 9, 10, 0.750, 0.180, True),
    ("Pipe_13_9_10_B", 9, 10, 0.750, 0.100, True),
    ("Pipe_14_10_11", 10, 11, 1.260, 0.180, False),
    ("Pipe_15_11_12", 11, 12, 1.200, 0.180, False),
    ("Pipe_16_12_13", 12, 13, 0.150, 0.180, False),
    ("Pipe_17_13_14", 13, 14, 0.300, 0.180, False),
    ("Pipe_18_14_15", 14, 15, 0.750, 0.180, False),
    ("Pipe_19_10_16", 10, 16, 0.315, 0.100, False),
    ("Pipe_20_17_18", 17, 18, 2.940, 0.080, False),
    ("Pipe_21_18_19", 18, 19, 0.180, 0.080, False),
)


GAS_PIPES: Tuple[GasPipeData, ...] = tuple(
    GasPipeData(
        name=name,
        from_junction=from_junction,
        to_junction=to_junction,
        wmn_reference=0.0,
        kmn_reference=0.0,
        length_km=length_km,
        diameter_m=diameter_m,
        roughness_mm=_PIPE_ROUGHNESS_MM,
        allow_parallel=allow_parallel,
    )
    for name, from_junction, to_junction, length_km, diameter_m, allow_parallel in _PIPE_ROWS
)


COMPRESSOR_CONFIGS: Tuple[CompressorConfig, ...] = (
    CompressorConfig(
        name="COMP_STATION_8_TO_9_EQ",
        from_junction=7,
        to_junction=8,
        min_pressure_ratio=1.00,
        max_pressure_ratio=1.18,
        initial_pressure_ratio=1.12,
        isentropic_efficiency=0.75,
        max_power_mw=0.030,
        nominal_flow_kg_s=0.20,
        inlet_min_bar=2.5,
        outlet_max_bar=5.0,
        equivalent_units=2,
        controllable=False,
        fixed_pressure_ratio=1.12,
        needs_calibration=False,
    ),
    CompressorConfig(
        name="COMP_17_TO_18",
        from_junction=16,
        to_junction=17,
        min_pressure_ratio=1.00,
        max_pressure_ratio=1.20,
        initial_pressure_ratio=1.08,
        isentropic_efficiency=0.75,
        max_power_mw=0.015,
        nominal_flow_kg_s=0.10,
        inlet_min_bar=2.5,
        outlet_max_bar=5.0,
        equivalent_units=1,
        controllable=True,
        needs_calibration=False,
    ),
)

CONTROLLED_COMPRESSOR_INDICES: Tuple[int, ...] = tuple(
    i for i, comp in enumerate(COMPRESSOR_CONFIGS) if comp.controllable
)


def compressor_engineering_ratio(config: CompressorConfig) -> float:
    return float(
        config.initial_pressure_ratio
        if config.fixed_pressure_ratio is None
        else config.fixed_pressure_ratio
    )


def default_compressor_ratios() -> np.ndarray:
    return np.asarray([compressor_engineering_ratio(c) for c in COMPRESSOR_CONFIGS], dtype=float)


def full_compressor_ratios_from_action(values: Sequence[float], enforce_fixed: bool = True) -> np.ndarray:
    raw = np.asarray(values, dtype=float).reshape(-1)
    n_total = len(COMPRESSOR_CONFIGS)
    n_controlled = len(CONTROLLED_COMPRESSOR_INDICES)
    if raw.size == n_total:
        ratios = raw.copy()
    elif raw.size == n_controlled:
        ratios = default_compressor_ratios()
        for action_pos, comp_idx in enumerate(CONTROLLED_COMPRESSOR_INDICES):
            ratios[comp_idx] = raw[action_pos]
    else:
        raise ValueError(
            f"Compressor ratio vector must have {n_controlled} controlled values "
            f"or {n_total} total values, got {raw.size}"
        )
    for i, cfg in enumerate(COMPRESSOR_CONFIGS):
        if enforce_fixed and not cfg.controllable:
            ratios[i] = compressor_engineering_ratio(cfg)
        ratios[i] = float(np.clip(ratios[i], cfg.min_pressure_ratio, cfg.max_pressure_ratio))
    return ratios


# 【中文导读】集中给出气网参数尚需标定的警告。
def calibration_warning_messages() -> Tuple[str, ...]:
    return (
        f"{GAS_MODEL_NAME} ({ENV_MODEL_VERSION}).",
        "Node names are Belgian-20 topology-source labels only; pipe length, diameter, pressure, load, source and compressor data are medium-pressure research calibrations.",
        "The 7->8 station is a fixed-ratio hydraulic compressor with equivalent_units=2; flow and power values already represent the aggregated station.",
        "Only COMP_17_TO_18 is exposed to the RL slow action in belgian20_derived_mp_v2.",
        "P2G includes gas conditioning and injection boosting in the aggregate efficiency; no extra P2G compressors are modeled.",
        "Supplier marginal costs, pipe roughness, gas composition and all device ratings remain research assumptions unless project data are supplied.",
    )


# =============================================================================
# Profiles
# =============================================================================


# 【中文导读】保存电负荷、气负荷、风光可用功率及下一小时预测。
@dataclass(frozen=True)
class DailyProfiles:
    """一天内的外生曲线，长度比一天多一个慢动作间隔，方便读取下一小时预测。"""

    load_multiplier: np.ndarray
    gas_multiplier: np.ndarray
    renewable_available_mw: np.ndarray
    next_hour_load_multiplier: np.ndarray
    next_hour_renewable_available_mw: np.ndarray


# 【中文导读】生成低频平滑随机扰动，避免 3 分钟曲线不现实地跳变。
def _smooth_noise(rng: np.random.Generator, n: int, scale: float, window: int) -> np.ndarray:
    """生成平滑噪声，避免负荷和风光每 3 分钟剧烈跳变。"""

    raw = rng.normal(0.0, scale, n + window)
    kernel = np.ones(window) / window
    return np.convolve(raw, kernel, mode="valid")[:n]


# 【中文导读】生成一个 episode 使用的日内外生曲线。
def generate_daily_profiles(time_config: TimeConfig, seed: int | None = None) -> DailyProfiles:
    """生成负荷、气负荷和新能源可用功率曲线。

    这些曲线是外生扰动，智能体不能控制，只能根据观测和预测做调度。
    """

    rng = np.random.default_rng(seed)
    horizon = time_config.steps_per_day
    n = horizon + time_config.slow_action_interval_steps + 1
    hours = (np.arange(n) * time_config.dt_hours) % 24.0

    # 电负荷采用早晚双峰形状，再叠加平滑随机扰动。
    load = 0.58
    load += 0.28 * np.exp(-0.5 * ((hours - 8.5) / 2.6) ** 2)
    load += 0.42 * np.exp(-0.5 * ((hours - 19.0) / 3.0) ** 2)
    load_multiplier = np.clip(load + _smooth_noise(rng, n, 0.035, 7), 0.40, 1.25)

    # 气负荷也采用早晚峰，但峰值时间和幅度略不同。
    gas = 0.55
    gas += 0.30 * np.exp(-0.5 * ((hours - 7.0) / 2.4) ** 2)
    gas += 0.34 * np.exp(-0.5 * ((hours - 18.5) / 3.0) ** 2)
    gas_multiplier = np.clip(gas + _smooth_noise(rng, n, 0.025, 9), 0.35, 1.15)

    renewable_available_mw = np.zeros((n, len(RENEWABLE_CONFIGS)), dtype=float)
    for i, cfg in enumerate(RENEWABLE_CONFIGS):
        if cfg.kind == "pv":
            # 光伏按日照曲线建模，中午最高，夜晚为 0。
            daylight = np.clip((hours - 5.5) / (19.5 - 5.5), 0.0, 1.0)
            solar = np.sin(np.pi * daylight)
            cloud = np.clip(_smooth_noise(rng, n, 0.18, 11), -0.45, 0.30)
            available = cfg.capacity_mw * np.clip(solar * (1.0 + cloud), 0.0, 1.0)
        else:
            # 风电不按日照归零，使用较慢的日内波动加随机扰动。
            wind = 0.48 + 0.16 * np.sin(2 * np.pi * (hours - 2.0) / 24.0)
            available = cfg.capacity_mw * np.clip(wind + _smooth_noise(rng, n, 0.18, 13), 0.08, 1.0)
        renewable_available_mw[:, i] = available

    next_idx = np.minimum(np.arange(n) + time_config.slow_action_interval_steps, n - 1)
    return DailyProfiles(
        load_multiplier=load_multiplier,
        gas_multiplier=gas_multiplier,
        renewable_available_mw=renewable_available_mw,
        next_hour_load_multiplier=load_multiplier[next_idx],
        next_hour_renewable_available_mw=renewable_available_mw[next_idx],
    )


# 【中文导读】安全读取指定时间索引的外生曲线。
def profile_at(profiles: DailyProfiles, time_index: int) -> Dict[str, np.ndarray | float]:
    """安全读取某个时间步的外生曲线，索引越界时自动夹到合法范围。"""

    idx = min(max(time_index, 0), len(profiles.load_multiplier) - 1)
    return {
        "load_multiplier": float(profiles.load_multiplier[idx]),
        "gas_multiplier": float(profiles.gas_multiplier[idx]),
        "renewable_available_mw": profiles.renewable_available_mw[idx].copy(),
        "next_hour_load_multiplier": float(profiles.next_hour_load_multiplier[idx]),
        "next_hour_renewable_available_mw": profiles.next_hour_renewable_available_mw[idx].copy(),
    }


# =============================================================================
# Network builders and topology validation
# =============================================================================


# 【中文导读】保存气网拓扑检查结果并在错误时抛出异常。
@dataclass
class TopologyValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    source_component_nodes: Set[int] = field(default_factory=set)

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ValueError("Gas topology validation failed: " + "; ".join(self.errors))


# 【中文导读】检查重复管道、非法参数和关键节点可达性。
def validate_belgian20_topology() -> TopologyValidationResult:
    errors: List[str] = []
    warnings: List[str] = []

    node_ids = {node.node for node in GAS_NODES}
    expected_nodes = set(range(N_GAS_JUNCTIONS))
    if len(GAS_NODES) != N_GAS_JUNCTIONS or node_ids != expected_nodes:
        errors.append(f"Expected gas nodes {sorted(expected_nodes)}, got {sorted(node_ids)}")
    if tuple(node.name for node in sorted(GAS_NODES, key=lambda item: item.node)) != GAS_NODE_NAMES:
        errors.append("Gas node names do not match the Belgian-20 source-label ordering")

    sources = {supplier.supplier_node for supplier in GAS_SUPPLIERS}
    if sources != set(GAS_SOURCE_NODES):
        errors.append(f"Gas supplier nodes should be {sorted(GAS_SOURCE_NODES)}, got {sorted(sources)}")
    pressure_source_nodes = {supplier.supplier_node for supplier in GAS_SUPPLIERS if supplier.pressure_bar is not None}
    auxiliary_source_nodes = {supplier.supplier_node for supplier in GAS_SUPPLIERS if supplier.pressure_bar is None}
    if pressure_source_nodes != {0}:
        errors.append(f"Exactly one pressure ext_grid supplier should be node 0, got {sorted(pressure_source_nodes)}")
    if auxiliary_source_nodes != {7}:
        errors.append(f"Exactly one auxiliary source supplier should be node 7, got {sorted(auxiliary_source_nodes)}")
    supplier_capacity = {supplier.supplier_node: supplier.max_mdot_kg_s for supplier in GAS_SUPPLIERS}
    for node, expected in GAS_SUPPLIER_CAPACITY_KG_S.items():
        actual = supplier_capacity.get(node)
        if actual is None or abs(actual - expected) > 1e-9:
            errors.append(f"Supplier capacity at node {node} should be {expected}, got {actual}")
    for supplier in GAS_SUPPLIERS:
        if supplier.supplier_node == 0:
            if supplier.pressure_bar is None:
                errors.append("MAIN_CITY_GATE pressure_bar must be set because it is the pressure ext_grid")
            elif not (DEFAULT_CONFIG.gas.network_pressure_min_bar <= supplier.pressure_bar <= DEFAULT_CONFIG.gas.network_pressure_max_bar):
                errors.append(f"{supplier.name} pressure is outside the medium-pressure range")
        if supplier.supplier_node == 7 and supplier.pressure_bar is not None:
            errors.append("AUXILIARY_GATE_AT_VOEREN is a bounded source and must have pressure_bar is None")

    load_nodes = {node.node for node in GAS_NODES if node.base_mdot_kg_s > 0.0}
    if load_nodes != set(GAS_BASE_LOADS_KG_S):
        errors.append(f"Gas load nodes should be {sorted(GAS_BASE_LOADS_KG_S)}, got {sorted(load_nodes)}")
    for node in GAS_NODES:
        expected_demand = GAS_BASE_LOADS_KG_S.get(node.node, 0.0)
        if abs(node.base_mdot_kg_s - expected_demand) > 1e-9:
            errors.append(f"Gas node {node.node} base load should be {expected_demand}, got {node.base_mdot_kg_s}")
        if (abs(node.p_min_bar - DEFAULT_CONFIG.gas.network_pressure_min_bar) > 1e-9 or
                abs(node.p_max_bar - DEFAULT_CONFIG.gas.network_pressure_max_bar) > 1e-9):
            errors.append(f"Gas node {node.node} pressure boundary should be 2.5..5.0 bar")
    if abs(sum(GAS_BASE_LOADS_KG_S.values()) - 0.18) >= 1e-4:
        errors.append(f"Total base gas load should be about 0.18 kg/s, got {sum(GAS_BASE_LOADS_KG_S.values()):.6f}")
    zero_injection_nodes = set(range(N_GAS_JUNCTIONS)) - set(GAS_BASE_LOADS_KG_S)
    bad_zero = sorted(node.node for node in GAS_NODES
                      if node.node in zero_injection_nodes and abs(node.base_mdot_kg_s) > 1e-12)
    if bad_zero:
        errors.append(f"Source/transit nodes must have zero demand, got {bad_zero}")

    if len(GAS_PIPES) != 21:
        errors.append(f"Expected 21 passive pipes, got {len(GAS_PIPES)}")
    if len(COMPRESSOR_CONFIGS) != 2:
        errors.append(f"Expected 2 hydraulic compressor stations, got {len(COMPRESSOR_CONFIGS)}")
    if COMPRESSOR_POWER_BUSES != (7, 30):
        errors.append(f"COMPRESSOR_POWER_BUSES should be (7, 30), got {COMPRESSOR_POWER_BUSES}")
    if len(COMPRESSOR_POWER_BUSES) != len(COMPRESSOR_CONFIGS):
        errors.append("COMPRESSOR_POWER_BUSES length must equal COMPRESSOR_CONFIGS length")

    pipe_multiset = Counter(tuple(sorted((pipe.from_junction, pipe.to_junction))) for pipe in GAS_PIPES)
    expected_pipe_multiset = Counter(tuple(sorted(edge)) for edge in EXPECTED_PASSIVE_PIPE_EDGES)
    if pipe_multiset != expected_pipe_multiset:
        errors.append(f"Passive pipe endpoints differ from the retained Belgian-20 topology: {pipe_multiset} != {expected_pipe_multiset}")

    compressor_multiset = Counter((comp.from_junction, comp.to_junction) for comp in COMPRESSOR_CONFIGS)
    expected_compressor_multiset = Counter(EXPECTED_COMPRESSOR_ARCS)
    if compressor_multiset != expected_compressor_multiset:
        errors.append(f"Compressor arcs should be {expected_compressor_multiset}, got {compressor_multiset}")
    if len(COMPRESSOR_CONFIGS) >= 2:
        if COMPRESSOR_CONFIGS[0].equivalent_units != 2:
            errors.append("First compressor station must have equivalent_units == 2")
        if COMPRESSOR_CONFIGS[1].equivalent_units != 1:
            errors.append("Second compressor station must have equivalent_units == 1")
        if COMPRESSOR_CONFIGS[0].controllable:
            errors.append("First 7->8 compressor station must be fixed, not RL-controllable")
        if not COMPRESSOR_CONFIGS[1].controllable:
            errors.append("Second 16->17 compressor station must be the only RL-controllable compressor")
        fixed_ratio = COMPRESSOR_CONFIGS[0].fixed_pressure_ratio
        if fixed_ratio is None or abs(fixed_ratio - COMPRESSOR_CONFIGS[0].initial_pressure_ratio) > 1e-12:
            errors.append("First compressor fixed_pressure_ratio must equal its initial_pressure_ratio")
    if CONTROLLED_COMPRESSOR_INDICES != (1,):
        errors.append(f"Controlled compressor indices should be (1,), got {CONTROLLED_COMPRESSOR_INDICES}")
    if compressor_multiset.get((7, 8), 0) != 1:
        errors.append("There must be exactly one 7->8 hydraulic compressor station")

    forbidden_pipes = sorted(edge for edge in pipe_multiset if edge in FORBIDDEN_PASSIVE_PIPE_EDGES)
    if forbidden_pipes:
        errors.append(f"Forbidden passive gas pipes are present: {forbidden_pipes}")
    if tuple(sorted((7, 8))) in pipe_multiset:
        errors.append("7-8 must not be modeled as a passive pipe")
    if tuple(sorted((16, 17))) in pipe_multiset:
        errors.append("16-17 must not be modeled as a passive pipe")
    forbidden_compressors = sorted(arc for arc in compressor_multiset if arc in FORBIDDEN_COMPRESSOR_ARCS)
    if forbidden_compressors:
        errors.append(f"Forbidden compressor arcs are present: {forbidden_compressors}")

    edges: List[Tuple[int, int]] = []
    seen: Dict[Tuple[int, int], str] = {}
    for pipe in GAS_PIPES:
        if pipe.from_junction not in expected_nodes or pipe.to_junction not in expected_nodes:
            errors.append(f"{pipe.name} endpoint is outside 0..{N_GAS_JUNCTIONS - 1}")
        if pipe.from_junction == pipe.to_junction:
            errors.append(f"{pipe.name} has identical endpoints")
        if pipe.length_km <= 0 or pipe.diameter_m <= 0 or pipe.roughness_mm <= 0:
            errors.append(f"{pipe.name} has non-positive physical parameter")
        if not (0.12 - 1e-9 <= pipe.length_km <= 2.94 + 1e-9):
            errors.append(f"{pipe.name} length {pipe.length_km} km is outside 0.12..2.94 km")
        if not (0.08 - 1e-9 <= pipe.diameter_m <= 0.18 + 1e-9):
            errors.append(f"{pipe.name} diameter {pipe.diameter_m} m is outside 0.08..0.18 m")
        if abs(pipe.roughness_mm - _PIPE_ROUGHNESS_MM) > 1e-12:
            errors.append(f"{pipe.name} roughness should be {_PIPE_ROUGHNESS_MM} mm")
        key = tuple(sorted((pipe.from_junction, pipe.to_junction)))
        if key in seen and not pipe.allow_parallel:
            errors.append(f"{pipe.name} duplicates {seen[key]}")
        if key in seen and pipe.allow_parallel:
            warnings.append(f"{pipe.name} is an allowed parallel pipe")
        seen.setdefault(key, pipe.name)
        edges.append((pipe.from_junction, pipe.to_junction))
    for comp in COMPRESSOR_CONFIGS:
        if comp.from_junction not in expected_nodes or comp.to_junction not in expected_nodes:
            errors.append(f"{comp.name} endpoint is outside 0..{N_GAS_JUNCTIONS - 1}")
        if comp.from_junction == comp.to_junction:
            errors.append(f"{comp.name} has identical endpoints")
        if comp.inlet_min_bar != DEFAULT_CONFIG.gas.network_pressure_min_bar:
            errors.append(f"{comp.name} inlet_min_bar should be {DEFAULT_CONFIG.gas.network_pressure_min_bar}")
        if comp.outlet_max_bar != DEFAULT_CONFIG.gas.network_pressure_max_bar:
            errors.append(f"{comp.name} outlet_max_bar should be {DEFAULT_CONFIG.gas.network_pressure_max_bar}")
        if comp.name.startswith("COMP_8_TO_9_"):
            errors.append(f"{comp.name} is a removed unaggregated parallel compressor")
        edges.append((comp.from_junction, comp.to_junction))

    adjacency = {i: set() for i in range(N_GAS_JUNCTIONS)}
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    covered_nodes = {node for edge in edges for node in edge}
    if covered_nodes != expected_nodes:
        errors.append(f"Pipe/compressor graph should cover all nodes, missing {sorted(expected_nodes - covered_nodes)}")

    visited: Set[int] = set()
    stack = list(sources)
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(sorted(adjacency[node] - visited))

    required = {n.node for n in GAS_NODES if n.base_mdot_kg_s > 0}
    required |= {p.gas_junction for p in P2G_CONFIGS}
    required |= {g.gas_junction for g in GFG_CONFIGS}
    required |= sources
    missing = sorted(required - visited)
    if missing:
        errors.append(f"Gas source component cannot reach nodes {missing}")
    return TopologyValidationResult(not errors, errors, warnings, visited)


# 【中文导读】保存 pandapower 网络及训练 step 需要修改的元件索引。
@dataclass
class PowerNetworkArtifacts:
    net: object
    load_indices: List[int]
    renewable_sgen_indices: List[int]
    ess_storage_indices: List[int]
    gfg_sgen_indices: List[int]
    p2g_load_indices: List[int]
    compressor_load_indices: List[int]
    ext_grid_index: int


# 【中文导读】保存 pandapipes 网络及 source/sink/压缩机等索引。
@dataclass
class GasNetworkArtifacts:
    net: object
    base_sink_indices_by_node: Dict[int, int]
    p2g_source_indices: List[int]
    gfg_sink_indices: List[int]
    compressor_indices: List[int]
    pressure_ext_grid_index: int
    auxiliary_source_index: int
    ext_grid_indices: List[int] = field(default_factory=list)


# 【中文导读】构建 IEEE33 配电网并挂接负荷、新能源、ESS 和耦合设备。
def build_power_network(config: ProjectConfig | None = None) -> PowerNetworkArtifacts:
    """构建 pandapower IEEE33 电网，并保存后续 step 需要修改的元件索引。"""

    cfg = config or DEFAULT_CONFIG
    import pandapower as pp

    net = pp.create_empty_network(sn_mva=cfg.power.base_mva)#加载默认配置的电网参数
    # 母线和线路构成基础配电网；负荷/新能源/储能/耦合设备随后挂接上去。
    for bus in range(N_POWER_BUSES):
        pp.create_bus(net, vn_kv=cfg.power.base_kv, name=f"Bus_{bus}",
                      min_vm_pu=cfg.power.voltage_min_pu, max_vm_pu=cfg.power.voltage_max_pu)#创建母线节点
    ext_grid = pp.create_ext_grid(net, bus=cfg.power.slack_bus, vm_pu=cfg.power.slack_vm_pu,
                                  va_degree=0.0, name="Utility_Grid")
    for i, (u, v, r, x) in enumerate(IEEE33_LINE_DATA):
        pp.create_line_from_parameters(net, u, v, length_km=1.0, r_ohm_per_km=r,
                                       x_ohm_per_km=x, c_nf_per_km=0.0,
                                       max_i_ka=0.50, name=f"Line_{i}_{u}_{v}")#u是起始节点，v是终点节点，r是电阻每千米，x是电感，c是电容

    load_indices = [int(pp.create_load(net, bus, p, q, name=f"BaseLoad_bus_{bus}"))
                    for bus, p, q in IEEE33_LOAD_DATA]#创建负荷节点
    renewable_indices = [
        int(pp.create_sgen(net, r.bus, p_mw=0.0, q_mvar=0.0, name=r.name,
                           max_p_mw=r.capacity_mw, min_p_mw=0.0,
                           max_q_mvar=r.s_rated_mva, min_q_mvar=-r.s_rated_mva))
        for r in RENEWABLE_CONFIGS
    ]#遍历RENEWABLE_CONFIGS，创建新能源节点，q_mvar为0
    ess_indices = [
        int(pp.create_storage(net, ess.bus, p_mw=0.0, q_mvar=0.0, max_e_mwh=ess.capacity_mwh,
                              soc_percent=100.0 * ess.soc_initial, name=ess.name))
        for ess in ESS_CONFIGS
    ]#遍历ESS_CONFIGS，创建储能节点，q_mvar为0
    gfg_indices = [
        int(pp.create_sgen(net, g.power_bus, p_mw=0.0, q_mvar=0.0, name=g.name,
                           max_p_mw=g.max_p_mw, min_p_mw=0.0))
        for g in GFG_CONFIGS
    ]
    p2g_indices = [
        int(pp.create_load(net, p.power_bus, p_mw=0.0, q_mvar=0.0, name=p.name))
        for p in P2G_CONFIGS
    ]
    comp_indices = [
        int(pp.create_load(net, bus, p_mw=0.0, q_mvar=0.0, name=f"{c.name}_electric_load"))
        for c, bus in zip(COMPRESSOR_CONFIGS, COMPRESSOR_POWER_BUSES)
    ]
    return PowerNetworkArtifacts(net, load_indices, renewable_indices, ess_indices,
                                 gfg_indices, p2g_indices, comp_indices, int(ext_grid))


# 【中文导读】构建 Belgian-20-derived 中压微型配气网并挂接气源、负荷、GFG、P2G 和压缩机站。
def build_gas_network(config: ProjectConfig | None = None) -> GasNetworkArtifacts:
    """Build the Belgian-20-derived medium-pressure micro gas distribution network."""

    cfg = config or DEFAULT_CONFIG
    import pandapipes as pp

    logger = logging.getLogger(__name__)
    for msg in calibration_warning_messages():
        logger.warning(msg)
    try:
        net = pp.create_empty_network(fluid=cfg.gas.fluid_name)
    except TypeError:
        net = pp.create_empty_network()
        pp.create_fluid_from_lib(net, cfg.gas.fluid_name, overwrite=True)

    # Node names are Belgian-20 topology-source labels; pn_bar is only a pipeflow initial value.
    for node in GAS_NODES:
        pp.create_junction(net, pn_bar=cfg.gas.junction_initial_pressure_bar, tfluid_k=cfg.gas.gas_temperature_k,
                           name=f"Gnode_{node.node}_{node.name}")
    pressure_supplier = next(s for s in GAS_SUPPLIERS if s.supplier_node == 0)
    auxiliary_supplier = next(s for s in GAS_SUPPLIERS if s.supplier_node == 7)
    if pressure_supplier.pressure_bar is None:
        raise ValueError("MAIN_CITY_GATE must define pressure_bar for the unique ext_grid")
    pressure_ext_grid_index = int(
        pp.create_ext_grid(
            net,
            junction=pressure_supplier.supplier_node,
            p_bar=pressure_supplier.pressure_bar,
            t_k=cfg.gas.gas_temperature_k,
            name=f"{pressure_supplier.name}_pressure_balance",
        )
    )
    auxiliary_source_index = int(
        pp.create_source(
            net,
            auxiliary_supplier.supplier_node,
            mdot_kg_per_s=0.0,
            name=f"{auxiliary_supplier.name}_bounded_source",
        )
    )
    for pipe in GAS_PIPES:
        sections = max(1, int(np.ceil(pipe.length_km / 0.5)))
        try:
            pp.create_pipe_from_parameters(
                net, pipe.from_junction, pipe.to_junction,
                length_km=pipe.length_km, inner_diameter_mm=pipe.diameter_m * 1000.0,
                k_mm=pipe.roughness_mm, sections=sections, name=pipe.name,
            )
        except TypeError:
            pp.create_pipe_from_parameters(
                net, pipe.from_junction, pipe.to_junction,
                length_km=pipe.length_km, diameter_m=pipe.diameter_m,
                k_mm=pipe.roughness_mm, sections=sections, name=pipe.name,
            )
    compressor_indices = [
        int(pp.create_compressor(net, c.from_junction, c.to_junction,
                                 pressure_ratio=compressor_engineering_ratio(c), name=c.name,
                                 in_service=True))
        for c in COMPRESSOR_CONFIGS
    ]
    base_sink_indices: Dict[int, int] = {}
    for node in GAS_NODES:
        if node.base_mdot_kg_s <= 0.0:
            continue
        base_sink_indices[node.node] = int(
            pp.create_sink(net, node.node, mdot_kg_per_s=node.base_mdot_kg_s,
                           name=f"BaseGasDemand_Gnode_{node.node}_{node.name}")
        )
    p2g_source_indices = [
        int(pp.create_source(net, p.gas_junction, mdot_kg_per_s=0.0, name=f"{p.name}_gas_source"))
        for p in P2G_CONFIGS
    ]
    gfg_sink_indices = [
        int(pp.create_sink(net, g.gas_junction, mdot_kg_per_s=0.0, name=f"{g.name}_gas_consumption"))
        for g in GFG_CONFIGS
    ]
    return GasNetworkArtifacts(
        net=net,
        base_sink_indices_by_node=base_sink_indices,
        p2g_source_indices=p2g_source_indices,
        gfg_sink_indices=gfg_sink_indices,
        compressor_indices=compressor_indices,
        pressure_ext_grid_index=pressure_ext_grid_index,
        auxiliary_source_index=auxiliary_source_index,
        ext_grid_indices=[pressure_ext_grid_index],
    )


# =============================================================================
# Coupling and safety models
# =============================================================================


# 【中文导读】按效率和高位热值把 P2G 电功率换算为产气流量。
def p2g_power_to_gas_mdot_kg_s(power_mw: float, efficiency: float, hhv_mj_per_kg: float) -> float:
    return max(0.0, efficiency * max(0.0, power_mw) / hhv_mj_per_kg)


# 【中文导读】按效率和高位热值把 GFG 电功率换算为耗气流量。
def gfg_power_to_gas_mdot_kg_s(power_mw: float, efficiency: float, hhv_mj_per_kg: float) -> float:
    if efficiency <= 0.0:
        raise ValueError("GFG efficiency must be positive")
    return max(0.0, max(0.0, power_mw) / (efficiency * hhv_mj_per_kg))


# 【中文导读】保存 GFG 实际发电功率和耗气流量。
@dataclass(frozen=True)
class GFGDispatch:
    electric_power_mw: float
    gas_mdot_kg_s: float


# 【中文导读】对 GFG 请求功率限幅并计算耗气。
def dispatch_gfg(config: GFGConfig, requested_power_mw: float, hhv_mj_per_kg: float) -> GFGDispatch:
    p_mw = min(max(requested_power_mw, 0.0), config.max_p_mw)
    return GFGDispatch(p_mw, gfg_power_to_gas_mdot_kg_s(p_mw, config.efficiency, hhv_mj_per_kg))


# 【中文导读】保存 P2G 实际耗电功率和产气流量。
@dataclass(frozen=True)
class P2GDispatch:
    electric_power_mw: float
    gas_mdot_kg_s: float


# 【中文导读】对 P2G 请求功率限幅并计算产气。
def dispatch_p2g(config: P2GConfig, requested_power_mw: float, hhv_mj_per_kg: float) -> P2GDispatch:
    p_mw = min(max(requested_power_mw, 0.0), config.max_p_mw)
    return P2GDispatch(p_mw, p2g_power_to_gas_mdot_kg_s(p_mw, config.efficiency, hhv_mj_per_kg))


# 【中文导读】保存压缩机压比、估计功率和限幅信息。
@dataclass(frozen=True)
class CompressorDispatch:
    requested_pressure_ratio: float
    applied_pressure_ratio: float
    signed_mdot_kg_s: float
    effective_mdot_kg_s: float
    electric_power_mw: float
    clipped_by_ratio: bool
    clipped_by_power: bool
    bypassed: bool
    reverse_flow: bool
    power_limited: bool
    ratio_projection_magnitude: float

    @property
    def pressure_ratio(self) -> float:
        return self.applied_pressure_ratio

    @property
    def mdot_kg_s(self) -> float:
        return self.effective_mdot_kg_s


# 【中文导读】根据压比、流量和热力学参数估算压缩机电功率。
def estimate_compressor_power_mw(mdot_kg_s: float, ratio: float, efficiency: float) -> float:
    if ratio <= 1.0 or mdot_kg_s <= 0.0:
        return 0.0
    cp_j_per_kg_k = 2200.0
    inlet_temperature_k = 293.15
    gamma = 1.30
    exponent = (gamma - 1.0) / gamma
    watts = mdot_kg_s * cp_j_per_kg_k * inlet_temperature_k
    watts *= np.power(ratio, exponent) - 1.0
    watts /= efficiency
    return float(max(watts / 1_000_000.0, 0.0))


def _compressor_dispatch_from_applied(config: CompressorConfig, requested_ratio: float,
                                      applied_ratio: float, signed_mdot_kg_s: float,
                                      clipped_by_ratio: bool,
                                      power_limited: bool) -> CompressorDispatch:
    signed_mdot = float(signed_mdot_kg_s)
    effective_mdot = max(signed_mdot, 0.0)
    reverse_flow = signed_mdot <= 0.0
    bypassed = reverse_flow
    power = 0.0 if bypassed else estimate_compressor_power_mw(
        effective_mdot, applied_ratio, config.isentropic_efficiency
    )
    requested_clipped = min(max(float(requested_ratio), config.min_pressure_ratio), config.max_pressure_ratio)
    return CompressorDispatch(
        requested_pressure_ratio=float(requested_ratio),
        applied_pressure_ratio=float(applied_ratio),
        signed_mdot_kg_s=signed_mdot,
        effective_mdot_kg_s=float(effective_mdot),
        electric_power_mw=float(power),
        clipped_by_ratio=bool(clipped_by_ratio),
        clipped_by_power=bool(power_limited),
        bypassed=bool(bypassed),
        reverse_flow=bool(reverse_flow),
        power_limited=bool(power_limited),
        ratio_projection_magnitude=float(max(requested_clipped - applied_ratio, 0.0)),
    )


# 【中文导读】投影压比并限制压缩机功率。
def dispatch_compressor(config: CompressorConfig, requested_ratio: float, mdot_estimate_kg_s: float | None) -> CompressorDispatch:
    requested = float(requested_ratio)
    ratio = min(max(requested, config.min_pressure_ratio), config.max_pressure_ratio)
    clipped_by_ratio = abs(ratio - requested) > 1e-9
    signed_mdot = float(config.nominal_flow_kg_s if mdot_estimate_kg_s is None else mdot_estimate_kg_s)
    effective_mdot = max(signed_mdot, 0.0)
    if signed_mdot <= 0.0 or ratio <= 1.0:
        return _compressor_dispatch_from_applied(
            config, requested, ratio, signed_mdot, clipped_by_ratio, False
        )

    requested_power = estimate_compressor_power_mw(effective_mdot, ratio, config.isentropic_efficiency)
    power_limited = requested_power > config.max_power_mw + 1e-12
    applied_ratio = ratio
    if power_limited:
        lo, hi = 1.0, ratio
        target_power = max(config.max_power_mw * (1.0 - 1e-6), 0.0)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            mid_power = estimate_compressor_power_mw(effective_mdot, mid, config.isentropic_efficiency)
            if mid_power <= target_power:
                lo = mid
            else:
                hi = mid
        applied_ratio = max(config.min_pressure_ratio, min(lo, ratio))
    return _compressor_dispatch_from_applied(
        config, requested, applied_ratio, signed_mdot, clipped_by_ratio, power_limited
    )


def _table_column_values(net: Any, table_name: str, column: str, length: int, default: float) -> np.ndarray:
    if not hasattr(net, table_name):
        return np.full(length, default, dtype=float)
    table = getattr(net, table_name)
    if column not in table:
        return np.full(length, default, dtype=float)
    values = np.asarray(table[column].values, dtype=float)
    if values.size < length:
        values = np.pad(values, (0, length - values.size), constant_values=default)
    return values[:length]


def _table_index_value(net: Any, table_name: str, column: str, index: int, default: float) -> float:
    if not hasattr(net, table_name):
        return float(default)
    table = getattr(net, table_name)
    if column not in table or index not in table.index:
        return float(default)
    value = table.at[index, column]
    return float(value) if np.isfinite(value) else float(default)


def infer_ext_grid_supply_mdot_kg_s(net: Any, expected_count: int) -> Tuple[np.ndarray, str]:
    """Infer which res_ext_grid sign means supply into the gas network.

    The returned array is signed in a supply-positive convention. Negative
    values mean that the pressure boundary absorbs gas from the network.
    """

    raw = _table_column_values(net, "res_ext_grid", "mdot_kg_per_s", expected_count, 0.0)
    sink = float(np.nansum(getattr(net, "sink")["mdot_kg_per_s"].values)) if hasattr(net, "sink") else 0.0
    source = float(np.nansum(getattr(net, "source")["mdot_kg_per_s"].values)) if hasattr(net, "source") else 0.0
    net_external_supply = sink - source
    positive_residual = abs(float(np.nansum(raw)) - net_external_supply)
    negative_residual = abs(float(np.nansum(-raw)) - net_external_supply)
    if positive_residual <= negative_residual:
        return raw, "positive"
    return -raw, "negative"


def read_gas_source_mdot_kg_s(net: Any, gas: GasNetworkArtifacts) -> Tuple[np.ndarray, str]:
    """Return [main ext_grid, auxiliary source] in a supply-positive convention."""

    main_supply, sign = infer_ext_grid_supply_mdot_kg_s(net, 1)
    aux_setpoint = _table_index_value(
        net, "source", "mdot_kg_per_s", gas.auxiliary_source_index, 0.0
    )
    aux_supply = _table_index_value(
        net, "res_source", "mdot_kg_per_s", gas.auxiliary_source_index, aux_setpoint
    )
    return np.asarray([float(main_supply[0]), float(aux_supply)], dtype=float), sign


def compute_auxiliary_source_mdot_kg_s(gas_multiplier: float, gfg_mdot: Sequence[float],
                                       p2g_mdot: Sequence[float]) -> float:
    auxiliary = next(s for s in GAS_SUPPLIERS if s.supplier_node == 7)
    base_load = sum(node.base_mdot_kg_s for node in GAS_NODES) * float(gas_multiplier)
    net_demand = base_load + float(np.sum(gfg_mdot)) - float(np.sum(p2g_mdot))
    mdot = auxiliary.aux_source_share * max(net_demand, 0.0)
    return float(np.clip(mdot, 0.0, auxiliary.max_mdot_kg_s))


def read_pipe_velocity_m_per_s(net: Any, expected_count: int) -> np.ndarray:
    if not hasattr(net, "res_pipe"):
        return np.zeros(expected_count, dtype=float)
    table = net.res_pipe
    if "v_mean_m_per_s" in table:
        return _table_column_values(net, "res_pipe", "v_mean_m_per_s", expected_count, 0.0)
    if "v_from_m_per_s" in table and "v_to_m_per_s" in table:
        v_from = _table_column_values(net, "res_pipe", "v_from_m_per_s", expected_count, 0.0)
        v_to = _table_column_values(net, "res_pipe", "v_to_m_per_s", expected_count, 0.0)
        return np.where(np.abs(v_from) >= np.abs(v_to), v_from, v_to)
    for col in ("v_from_m_per_s", "v_to_m_per_s", "velocity_m_per_s"):
        if col in table:
            return _table_column_values(net, "res_pipe", col, expected_count, 0.0)
    return np.zeros(expected_count, dtype=float)


def read_gfg_inlet_pressures_bar(net: Any, default_pressure_bar: float) -> np.ndarray:
    p = _table_column_values(net, "res_junction", "p_bar", N_GAS_JUNCTIONS, default_pressure_bar)
    return np.asarray([p[g.gas_junction] for g in GFG_CONFIGS], dtype=float)


def read_compressor_mdot_estimates(net: Any, compressor_indices: Sequence[int],
                                   previous: Sequence[float] | None = None) -> np.ndarray:
    defaults = np.asarray(previous if previous is not None else [c.nominal_flow_kg_s for c in COMPRESSOR_CONFIGS],
                          dtype=float)
    if not hasattr(net, "res_compressor") or len(net.res_compressor) == 0:
        return defaults.copy()
    table = net.res_compressor
    cols = (
        ("mdot_from_kg_per_s", 1.0),
        ("mf_from_kg_per_s", 1.0),
        ("mdot_kg_per_s", 1.0),
        ("mdot_to_kg_per_s", -1.0),
        ("mf_to_kg_per_s", -1.0),
    )
    mdots = []
    for pos, (idx, cfg) in enumerate(zip(compressor_indices, COMPRESSOR_CONFIGS)):
        value = float(defaults[pos] if pos < defaults.size else cfg.nominal_flow_kg_s)
        for col, sign in cols:
            if col in table.columns and idx in table.index:
                raw = table.at[idx, col]
                if np.isfinite(raw):
                    value = sign * float(raw)
                    break
        mdots.append(value)
    return np.asarray(mdots, dtype=float)


# 【中文导读】通用标量动作投影结果。
@dataclass(frozen=True)
class ActionProjectionResult:
    raw_action: float
    applied_action: float
    projection_magnitude: float
    hit_boundary: bool


# 【中文导读】保存多个 ESS 的请求功率、执行功率和边界命中情况。
@dataclass(frozen=True)
class ESSProjectionBatch:
    raw_p_mw: np.ndarray
    applied_p_mw: np.ndarray
    projection_magnitude_mw: np.ndarray
    hit_soc_boundary: np.ndarray


# 【中文导读】依据当前 SOC 和一步能量变化投影 ESS 功率。
def project_ess_power(requested_p_mw: float, soc: float, ess: ESSConfig, dt_hours: float) -> ActionProjectionResult:
    """把 ESS 请求功率投影到当前 SOC 可行区间。"""

    # p_mw>0 为充电，p_mw<0 为放电；SOC 决定当前步还能充/放多少。
    max_charge_by_soc = (ess.soc_max - soc) * ess.capacity_mwh / (ess.eta_charge * dt_hours)
    min_discharge_by_soc = (ess.soc_min - soc) * ess.capacity_mwh * ess.eta_discharge / dt_hours
    lower = max(-ess.max_p_mw, min_discharge_by_soc)
    upper = min(ess.max_p_mw, max_charge_by_soc)
    if lower > upper:
        lower = upper = 0.0
    applied = float(np.clip(requested_p_mw, lower, upper))
    projection = abs(applied - requested_p_mw)
    hit = projection > 1e-9 or applied <= lower + 1e-9 or applied >= upper - 1e-9
    return ActionProjectionResult(float(requested_p_mw), applied, float(projection), bool(hit))


# 【中文导读】批量执行 ESS 安全投影。
def project_ess_batch(requested_p_mw: Sequence[float], soc: Sequence[float], dt_hours: float) -> ESSProjectionBatch:
    """对所有 ESS 批量执行安全投影。"""

    results = [project_ess_power(float(p), float(s), ess, dt_hours)
               for p, s, ess in zip(requested_p_mw, soc, ESS_CONFIGS)]
    return ESSProjectionBatch(
        raw_p_mw=np.array([r.raw_action for r in results]),
        applied_p_mw=np.array([r.applied_action for r in results]),
        projection_magnitude_mw=np.array([r.projection_magnitude for r in results]),
        hit_soc_boundary=np.array([r.hit_boundary for r in results], dtype=bool),
    )


# 【中文导读】用实际执行功率和充放电效率推进 SOC。
def update_ess_soc(soc: float, p_mw: float, ess: ESSConfig, dt_hours: float) -> float:
    """根据实际执行功率更新 SOC。"""

    if p_mw >= 0.0:
        delta_e_mwh = ess.eta_charge * p_mw * dt_hours
    else:
        delta_e_mwh = p_mw * dt_hours / ess.eta_discharge
    return float(soc + delta_e_mwh / ess.capacity_mwh)


# 【中文导读】保存逆变器有功、无功、弃电和容量限幅信息。
@dataclass(frozen=True)
class InverterProjection:
    p_actual_mw: float
    q_actual_mvar: float
    curtailment: float
    q_limit_mvar: float
    apparent_power_mva: float
    q_was_clipped: bool
    curtailment_was_clipped: bool


# 【中文导读】投影弃电率和无功，使 P²+Q² 不超过逆变器额定容量。
def project_inverter_action(cfg: RenewableConfig, p_available_mw: float, requested_q_mvar: float, requested_curtailment: float) -> InverterProjection:
    """投影新能源逆变器动作，满足削减边界和 P²+Q²<=S²。"""

    curtailment = float(np.clip(requested_curtailment, 0.0, cfg.max_curtailment))
    p_actual = max(0.0, p_available_mw) * (1.0 - curtailment)
    q_limit = float(np.sqrt(max(cfg.s_rated_mva ** 2 - p_actual ** 2, 0.0)))
    q_actual = float(np.clip(requested_q_mvar, -q_limit, q_limit))
    s_actual = float(np.sqrt(p_actual ** 2 + q_actual ** 2))
    return InverterProjection(p_actual, q_actual, curtailment, q_limit, s_actual,
                              abs(q_actual - requested_q_mvar) > 1e-9,
                              abs(curtailment - requested_curtailment) > 1e-9)


# =============================================================================
# Event-driven coupled solver
# =============================================================================


# 【中文导读】保存最近一次气网求解时的触发变量。
@dataclass
class GasEventState:
    gfg_mdot_kg_s: np.ndarray
    p2g_mdot_kg_s: np.ndarray
    compressor_ratio: np.ndarray
    gas_load_multiplier: float


# 【中文导读】表示当前快速步是否重算气网及原因。
@dataclass(frozen=True)
class GasSolveDecision:
    should_solve: bool
    reason: str


# 【中文导读】根据慢时钟和设备变化阈值决定是否运行 pipeflow。
class EventScheduler:
    """决定当前快速步是否需要重新运行气网 pipeflow。

    气网求解较慢，因此不是每 3 分钟都强制计算；只有首次、整点慢动作、
    或 GFG/P2G/压缩机/气负荷变化超过阈值时才刷新。
    """

    def __init__(self, time_config: TimeConfig, event_config: EventConfig):
        self.time_config = time_config
        self.event_config = event_config
        self.last_state: GasEventState | None = None

    def reset(self) -> None:
        self.last_state = None

    # 【中文导读】根据首次、整点和变化阈值决定是否刷新气网状态。
    def decide(self, time_index: int, gfg_mdot: np.ndarray, p2g_mdot: np.ndarray,
               comp_ratio: np.ndarray, gas_load_multiplier: float) -> GasSolveDecision:
        """返回是否需要求解气网，以及触发原因字符串。"""

        if self.last_state is None:
            return GasSolveDecision(True, "initial")
        if time_index % self.time_config.slow_action_interval_steps == 0:
            return GasSolveDecision(True, "hourly")
        if np.max(np.abs(gfg_mdot - self.last_state.gfg_mdot_kg_s)) > self.event_config.gfg_mdot_threshold_kg_s:
            return GasSolveDecision(True, "gfg_change")
        if np.max(np.abs(p2g_mdot - self.last_state.p2g_mdot_kg_s)) > self.event_config.p2g_mdot_threshold_kg_s:
            return GasSolveDecision(True, "p2g_change")
        if np.max(np.abs(comp_ratio - self.last_state.compressor_ratio)) > self.event_config.compressor_ratio_threshold:
            return GasSolveDecision(True, "compressor_change")
        previous = max(abs(self.last_state.gas_load_multiplier), 1e-9)
        if abs(gas_load_multiplier - self.last_state.gas_load_multiplier) / previous > self.event_config.gas_load_relative_threshold:
            return GasSolveDecision(True, "gas_load_change")
        return GasSolveDecision(False, "hold_previous_gas_state")

    def mark_solved(self, gfg_mdot: np.ndarray, p2g_mdot: np.ndarray, comp_ratio: np.ndarray, gas_load_multiplier: float) -> None:
        """记录最近一次 pipeflow 使用的驱动量。"""

        self.last_state = GasEventState(gfg_mdot.copy(), p2g_mdot.copy(), comp_ratio.copy(), float(gas_load_multiplier))


# 【中文导读】环境实际送入耦合求解器的物理动作集合。
@dataclass(frozen=True)
class PhysicalActions:
    ess_p_mw: np.ndarray
    gfg_p_mw: np.ndarray
    p2g_p_mw: np.ndarray
    compressor_ratio: np.ndarray
    renewable_p_mw: np.ndarray
    renewable_q_mvar: np.ndarray
    renewable_curtailment: np.ndarray


# 【中文导读】保存一次电—气耦合求解的收敛、事件和物理结果。
@dataclass
class CoupledSolveResult:
    power_converged: bool
    gas_converged: bool
    gas_solved_this_step: bool
    gas_solve_reason: str
    gas_state_age: int
    gfg_mdot_kg_s: np.ndarray
    p2g_mdot_kg_s: np.ndarray
    compressor_dispatches: List[CompressorDispatch] = field(default_factory=list)
    equivalent_linepack_indicator: float = 0.0

    @property
    def converged(self) -> bool:
        return self.power_converged and self.gas_converged


# 【中文导读】写入设备动作，事件驱动求气网，并在每个快速步求电网。
class CoupledSolver:
    """显式手工电-气耦合求解器。

    它把慢/快动作写入 pandapower 和 pandapipes 网络表，然后按事件策略运行
    气网 pipeflow，并每个快速步运行电网 powerflow。
    """

    def __init__(self, power: PowerNetworkArtifacts, gas: GasNetworkArtifacts, config: ProjectConfig):
        self.power = power
        self.gas = gas
        self.config = config
        self.scheduler = EventScheduler(config.time, config.event)  #判断气网计算逻辑
        self.gas_state_age = 0
        self.gas_solve_count = 0
        self.last_gas_converged = False
        self.last_compressor_mdot_kg_s = np.array([c.nominal_flow_kg_s for c in COMPRESSOR_CONFIGS], dtype=float)

    def reset(self) -> None:
        """重置气网事件状态和上一轮压缩机流量估计。"""

        self.scheduler.reset()
        self.gas_state_age = 0
        self.gas_solve_count = 0
        self.last_gas_converged = False
        self.last_compressor_mdot_kg_s = np.array([c.nominal_flow_kg_s for c in COMPRESSOR_CONFIGS], dtype=float)

    def solve_step(self, time_index: int, profile: Dict[str, np.ndarray | float],
                   actions: PhysicalActions, force_gas: bool = False) -> CoupledSolveResult:
        compressor_ratio = full_compressor_ratios_from_action(actions.compressor_ratio)
        if not np.allclose(compressor_ratio, np.asarray(actions.compressor_ratio, dtype=float).reshape(-1), atol=0.0, rtol=0.0):
            actions = PhysicalActions(
                ess_p_mw=actions.ess_p_mw,
                gfg_p_mw=actions.gfg_p_mw,
                p2g_p_mw=actions.p2g_p_mw,
                compressor_ratio=compressor_ratio,
                renewable_p_mw=actions.renewable_p_mw,
                renewable_q_mvar=actions.renewable_q_mvar,
                renewable_curtailment=actions.renewable_curtailment,
            )
        self._write_power_profile_and_actions(profile, actions)
        gfg_mdot = self._write_gfg(actions.gfg_p_mw)
        p2g_mdot = self._write_p2g(actions.p2g_p_mw)
        gas_multiplier = float(profile["gas_multiplier"])
        self._write_gas_loads(gas_multiplier)
        self._write_auxiliary_source(gas_multiplier, gfg_mdot, p2g_mdot)
        self._write_compressor_ratios(compressor_ratio)

        decision = self.scheduler.decide(
            time_index, gfg_mdot, p2g_mdot, compressor_ratio, gas_multiplier
        )
        gas_solved = force_gas or decision.should_solve
        gas_reason = "forced" if force_gas and not decision.should_solve else decision.reason
        gas_converged = self.last_gas_converged
        if gas_solved:
            self._run_pipeflow()
            gas_converged = True
            self.last_gas_converged = True
            self.last_compressor_mdot_kg_s = self._read_compressor_mdot_estimates()
            comp_dispatches = self._write_compressor_power_loads(
                compressor_ratio, self.last_compressor_mdot_kg_s, project_power=True
            )
            if any(d.power_limited and d.ratio_projection_magnitude > 1e-9 for d in comp_dispatches):
                self._run_pipeflow()
                self.last_compressor_mdot_kg_s = self._read_compressor_mdot_estimates()
                comp_dispatches = self._write_compressor_power_loads(
                    self._current_compressor_ratios(),
                    self.last_compressor_mdot_kg_s,
                    project_power=False,
                    original_requested=compressor_ratio,
                )
            applied_ratio = np.asarray([d.applied_pressure_ratio for d in comp_dispatches], dtype=float)
            self.scheduler.mark_solved(gfg_mdot, p2g_mdot, applied_ratio, gas_multiplier)
            self.gas_state_age = 0
            self.gas_solve_count += 1
        else:
            comp_dispatches = self._write_compressor_power_loads(
                compressor_ratio, self.last_compressor_mdot_kg_s, project_power=True
            )
            self.gas_state_age += 1

        self._run_powerflow()
        return CoupledSolveResult(
            True, gas_converged, gas_solved, gas_reason,
            self.gas_state_age, gfg_mdot, p2g_mdot,
            comp_dispatches, self.compute_equivalent_linepack_indicator(),
        )

    def _write_power_profile_and_actions(self, profile: Dict[str, np.ndarray | float], actions: PhysicalActions) -> None:
        """把电负荷、新能源和 ESS 动作写入 pandapower 网络。"""

        net = self.power.net
        load_mult = float(profile["load_multiplier"])
        for i, (_, p_base, q_base) in enumerate(IEEE33_LOAD_DATA):
            idx = self.power.load_indices[i]
            net.load.at[idx, "p_mw"] = p_base * load_mult
            net.load.at[idx, "q_mvar"] = q_base * load_mult
        for i, idx in enumerate(self.power.renewable_sgen_indices):
            net.sgen.at[idx, "p_mw"] = float(actions.renewable_p_mw[i])
            net.sgen.at[idx, "q_mvar"] = float(actions.renewable_q_mvar[i])
        for i, idx in enumerate(self.power.ess_storage_indices):
            net.storage.at[idx, "p_mw"] = float(actions.ess_p_mw[i])
            net.storage.at[idx, "q_mvar"] = 0.0

    def _write_gfg(self, requested_p_mw: Sequence[float]) -> np.ndarray:
        """写入 GFG：电侧是 sgen，气侧是 sink。"""

        mdot = []
        for i, cfg in enumerate(GFG_CONFIGS):
            disp = dispatch_gfg(cfg, float(requested_p_mw[i]), self.config.gas.hhv_mj_per_kg)
            self.power.net.sgen.at[self.power.gfg_sgen_indices[i], "p_mw"] = disp.electric_power_mw
            self.power.net.sgen.at[self.power.gfg_sgen_indices[i], "q_mvar"] = 0.0
            self.gas.net.sink.at[self.gas.gfg_sink_indices[i], "mdot_kg_per_s"] = disp.gas_mdot_kg_s
            mdot.append(disp.gas_mdot_kg_s)
        return np.asarray(mdot, dtype=float)

    def _write_p2g(self, requested_p_mw: Sequence[float]) -> np.ndarray:
        """写入 P2G：电侧是 load，气侧是 source。"""

        mdot = []
        for i, cfg in enumerate(P2G_CONFIGS):
            disp = dispatch_p2g(cfg, float(requested_p_mw[i]), self.config.gas.hhv_mj_per_kg)
            self.power.net.load.at[self.power.p2g_load_indices[i], "p_mw"] = disp.electric_power_mw
            self.power.net.load.at[self.power.p2g_load_indices[i], "q_mvar"] = 0.0
            self.gas.net.source.at[self.gas.p2g_source_indices[i], "mdot_kg_per_s"] = disp.gas_mdot_kg_s
            mdot.append(disp.gas_mdot_kg_s)
        return np.asarray(mdot, dtype=float)

    def _write_gas_loads(self, gas_multiplier: float) -> None:
        for node in GAS_NODES:
            idx = self.gas.base_sink_indices_by_node.get(node.node)
            if idx is None:
                continue
            self.gas.net.sink.at[idx, "mdot_kg_per_s"] = node.base_mdot_kg_s * gas_multiplier

    def _write_auxiliary_source(self, gas_multiplier: float, gfg_mdot: Sequence[float],
                                p2g_mdot: Sequence[float]) -> float:
        mdot = compute_auxiliary_source_mdot_kg_s(gas_multiplier, gfg_mdot, p2g_mdot)
        self.gas.net.source.at[self.gas.auxiliary_source_index, "mdot_kg_per_s"] = mdot
        return mdot

    def _write_compressor_ratios(self, requested_ratio: Sequence[float]) -> None:
        ratios = full_compressor_ratios_from_action(requested_ratio)
        for i, idx in enumerate(self.gas.compressor_indices):
            self.gas.net.compressor.at[idx, "pressure_ratio"] = float(ratios[i])

    def _current_compressor_ratios(self) -> np.ndarray:
        return np.asarray([
            float(self.gas.net.compressor.at[idx, "pressure_ratio"])
            for idx in self.gas.compressor_indices
        ], dtype=float)

    def _write_compressor_power_loads(self, requested_ratio: Sequence[float],
                                      signed_mdot: Sequence[float] | None = None,
                                      project_power: bool = True,
                                      original_requested: Sequence[float] | None = None) -> List[CompressorDispatch]:
        dispatches = []
        mdot = np.asarray(signed_mdot if signed_mdot is not None else self.last_compressor_mdot_kg_s, dtype=float)
        requested = full_compressor_ratios_from_action(requested_ratio, enforce_fixed=project_power)
        original = full_compressor_ratios_from_action(
            original_requested if original_requested is not None else requested_ratio
        )
        for i, cfg in enumerate(COMPRESSOR_CONFIGS):
            if project_power:
                disp = dispatch_compressor(cfg, float(requested[i]), float(mdot[i]))
            else:
                applied_ratio = float(np.clip(requested[i], cfg.min_pressure_ratio, cfg.max_pressure_ratio))
                raw_requested = float(original[i])
                clipped_by_ratio = abs(np.clip(raw_requested, cfg.min_pressure_ratio, cfg.max_pressure_ratio) -
                                       raw_requested) > 1e-9
                power_limited = max(
                    float(np.clip(raw_requested, cfg.min_pressure_ratio, cfg.max_pressure_ratio)) - applied_ratio,
                    0.0,
                ) > 1e-9
                disp = _compressor_dispatch_from_applied(
                    cfg, raw_requested, applied_ratio, float(mdot[i]), clipped_by_ratio, power_limited
                )
            self.gas.net.compressor.at[self.gas.compressor_indices[i], "pressure_ratio"] = disp.applied_pressure_ratio
            self.power.net.load.at[self.power.compressor_load_indices[i], "p_mw"] = disp.electric_power_mw
            self.power.net.load.at[self.power.compressor_load_indices[i], "q_mvar"] = 0.0
            dispatches.append(disp)
        return dispatches

    def _run_pipeflow(self) -> None:
        import pandapipes as pp
        pp.pipeflow(self.gas.net, max_iter_hyd=50, tol_p=1e-5, use_numba=False)

    def _run_powerflow(self) -> None:
        import pandapower as pp
        attempts = (
            {"algorithm": "bfsw", "init": "flat", "max_iteration": 100, "tolerance_mva": 1e-7},
            {"algorithm": "nr", "init": "results", "max_iteration": 50, "tolerance_mva": 1e-7},
            {"algorithm": "nr", "init": "flat", "max_iteration": 50, "tolerance_mva": 1e-7},
        )
        last_error: Exception | None = None
        for kwargs in attempts:
            try:
                pp.runpp(self.power.net, **kwargs)
                return
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error

    def _read_compressor_mdot_estimates(self) -> np.ndarray:
        return read_compressor_mdot_estimates(
            self.gas.net,
            self.gas.compressor_indices,
            self.last_compressor_mdot_kg_s,
        )

    def compute_equivalent_linepack_indicator(self) -> float:
        net = self.gas.net
        if not hasattr(net, "res_junction") or "p_bar" not in net.res_junction:
            return 0.0
        p_bar = net.res_junction["p_bar"]
        total = 0.0
        for pipe in GAS_PIPES:
            if pipe.from_junction not in p_bar.index or pipe.to_junction not in p_bar.index:
                continue
            p_avg_pa = 0.5 * (float(p_bar.at[pipe.from_junction]) + float(p_bar.at[pipe.to_junction])) * 1e5
            volume = np.pi * (pipe.diameter_m / 2.0) ** 2 * pipe.length_km * 1000.0
            rho = p_avg_pa / (self.config.gas.gas_specific_gas_constant_j_per_kg_k *
                              self.config.gas.gas_temperature_k *
                              self.config.gas.gas_compressibility_z)
            total += rho * volume
        return float(total)


# =============================================================================
# Environment
# =============================================================================


# 【中文导读】在 Gymnasium 不可用时提供最小动作/观测空间接口。
class SimpleBox:
    def __init__(self, low: float, high: float, shape: Tuple[int, ...], dtype=np.float32):
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high).astype(self.dtype)


# 【中文导读】保存环境回滚所需的设备状态和网络表。
@dataclass
class Snapshot:
    current_step: int
    ess_soc: np.ndarray
    last_slow_action: np.ndarray
    last_raw_action: np.ndarray
    last_applied_action: np.ndarray
    last_physical_slow: Dict[str, np.ndarray]
    previous_device_actions: Dict[str, np.ndarray]
    consecutive_solver_failures: int
    last_ess_projection: ESSProjectionBatch | None
    last_inverter_projection: List[InverterProjection]
    last_solve_result: CoupledSolveResult | None
    solver_gas_state_age: int
    solver_gas_solve_count: int
    solver_last_gas_converged: bool
    solver_last_compressor_mdot_kg_s: np.ndarray
    scheduler_last_state: GasEventState | None
    power_tables: Dict[str, Any]
    gas_tables: Dict[str, Any]


# 【中文导读】Gymnasium 风格环境，完成动作映射、安全投影、求解、状态和奖励。
class ElectricGasMultiScaleEnv:
    """Gymnasium 风格的电-气耦合环境。

    环境负责把智能体的 [-1, 1] 动作翻译成工程量，调用求解器推进 3 分钟，
    然后返回下一步观测、奖励和 info。训练脚本中的 TD3 并不直接接触 pandapower
    或 pandapipes，它只通过这个类学习控制。
    """

    # 【中文导读】构建两个物理网络、耦合求解器、动作空间和设备记忆。
    def __init__(self, config: ProjectConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        # 先检查气网拓扑，尽早暴露节点越界、孤立节点等建模错误。
        validate_belgian20_topology().raise_if_invalid()
        # 构建电网、气网和耦合求解器；后续 step 都会复用这些网络对象。
        self.power = build_power_network(self.config)
        self.gas = build_gas_network(self.config)
        self.solver = CoupledSolver(self.power, self.gas, self.config)
        # 动作维度由设备数量决定：慢动作控制 ESS/GFG/P2G/压缩机，快动作控制逆变器。
        self.n_ess = len(ESS_CONFIGS)
        self.n_gfg = len(GFG_CONFIGS)
        self.n_p2g = len(P2G_CONFIGS)
        self.n_comp = len(COMPRESSOR_CONFIGS)
        self.n_controlled_comp = len(CONTROLLED_COMPRESSOR_INDICES)
        self.n_renew = len(RENEWABLE_CONFIGS)
        self.slow_action_dim = self.n_ess + self.n_gfg + self.n_p2g + self.n_controlled_comp
        self.fast_action_dim = 2 * self.n_renew  #逆变器动作维度是2维，1维是无功功率，第2维是弃电率
        self.action_dim = self.slow_action_dim + self.fast_action_dim
        self.action_space = self._make_action_space()#对动作维度返回.Box()的连续动作空间
        self.observation_space = SimpleBox(-10.0, 10.0, shape=(self.global_state_dim,), dtype=np.float32)
        self.current_step = 0
        self.profiles: DailyProfiles | None = None
        self.ess_soc = np.array([e.soc_initial for e in ESS_CONFIGS], dtype=float)
        self.last_slow_action = np.zeros(self.slow_action_dim, dtype=float)
        self.last_physical_slow: Dict[str, np.ndarray] = {}  
        self.previous_device_actions: Dict[str, np.ndarray] = {}
        self.consecutive_solver_failures = 0 
        self.last_solve_result: CoupledSolveResult | None = None
        self.last_ess_projection: ESSProjectionBatch | None = None
        self.last_inverter_projection: List[InverterProjection] = []
        self.last_raw_action = np.zeros(self.action_dim, dtype=float)
        self.last_applied_action = np.zeros(self.action_dim, dtype=float)
        self._reset_device_memory()

    @property
    def global_state_dim(self) -> int:
        """全局观测向量长度，用于创建 observation_space 和神经网络输入层。"""

        power_dim = 33 + len(IEEE33_LINE_DATA) + 5 + 3 * self.n_renew  
        ess_dim = 3 * self.n_ess
        gas_dim = N_GAS_JUNCTIONS + len(GFG_CONFIGS) + len(GAS_SUPPLIERS)
        gas_dim += len(COMPRESSOR_CONFIGS) + self.n_gfg + self.n_p2g + 2 + len(GAS_PIPES)
        return power_dim + ess_dim + gas_dim + 6

    def _make_action_space(self):
        try:
            from gymnasium import spaces
            return spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        except Exception:
            return SimpleBox(-1.0, 1.0, shape=(self.action_dim,), dtype=np.float32)

    def _reset_device_memory(self) -> None:
        """清空慢速设备记忆，保证新 episode 不继承上一天的动作。"""

        self.last_physical_slow = {
            "ess_p_mw": np.zeros(self.n_ess),
            "gfg_p_mw": np.zeros(self.n_gfg),
            "p2g_p_mw": np.zeros(self.n_p2g),
            "compressor_ratio": default_compressor_ratios(),
        }
        self.previous_device_actions = {k: v.copy() for k, v in self.last_physical_slow.items()}

    # 【中文导读】重置一天曲线、SOC、设备动作和求解器，并计算有效初始潮流。
    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """开始新 episode。

        reset 会重新生成一天的外生曲线，并强制求解一次初始电/气状态，
        因此返回的初始观测已经包含有效潮流结果。
        """

        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.profiles = generate_daily_profiles(self.config.time, seed=seed or self.config.random_seed)
        #生成501（480+20+1）步的负荷曲线、气体负荷曲线、可再生能源曲线、下一时刻的负荷曲线、下一时刻的可再生能源曲线（20步到500步）
        self.ess_soc = np.array([e.soc_initial for e in ESS_CONFIGS], dtype=float)
        self.last_slow_action = np.zeros(self.slow_action_dim, dtype=float)
        self.last_raw_action = np.zeros(self.action_dim, dtype=float)
        self.last_applied_action = np.zeros(self.action_dim, dtype=float)
        self._reset_device_memory()
        self.consecutive_solver_failures = 0
        self.solver.reset()
        profile = profile_at(self.profiles, 0)
        fast = self._project_fast_actions(np.zeros(self.fast_action_dim), profile)
        actions = PhysicalActions(
            ess_p_mw=self.last_physical_slow["ess_p_mw"].copy(),
            gfg_p_mw=self.last_physical_slow["gfg_p_mw"].copy(),
            p2g_p_mw=self.last_physical_slow["p2g_p_mw"].copy(),
            compressor_ratio=self.last_physical_slow["compressor_ratio"].copy(),
            renewable_p_mw=fast["renewable_p_mw"],
            renewable_q_mvar=fast["renewable_q_mvar"],
            renewable_curtailment=fast["renewable_curtailment"],
        )
        self.last_solve_result = self.solver.solve_step(0, profile, actions, force_gas=True)
        return self.get_global_state(), self._build_info({}, {}, False, True)

    # 【中文导读】执行一个 3 分钟步；成功时推进物理状态，失败时回滚并返回惩罚。
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一个 3 分钟环境步。

        返回值遵循 Gymnasium：obs, reward, terminated, truncated, info。
        terminated 表示一天自然结束；truncated 表示连续求解失败等异常截断。
        """

        if self.profiles is None:
            raise RuntimeError("Call reset before step")
        if self.current_step >= self.config.time.steps_per_day:
            return self.get_global_state(), 0.0, True, False, {"already_done": True}

        # 求解器可能因为动作或数值问题失败，所以先保存快照，失败时回滚。
        snapshot = self._make_snapshot()
        time_index = self.current_step
        profile = profile_at(self.profiles, time_index)
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.size != self.action_dim:
            raise ValueError(f"Action dimension should be {self.action_dim}, got {action.size}")
        self.last_raw_action = action.copy()
        attempted_applied_action = np.clip(action.copy(), -1.0, 1.0)
        # 慢动作每 20 个快速步才真正更新，其余步沿用 last_physical_slow。
        # 环境自身以 20 个快速步为慢动作生效时钟；训练脚本的 slow_interval 必须与此一致。
        slow_action_applied = time_index % self.config.time.slow_action_interval_steps == 0
        try:
            # 归一化动作 -> 安全投影后的物理动作 -> 电/气网络求解。
            physical = self._map_and_project_action(action, profile, slow_action_applied)
            attempted_applied_action = self.last_applied_action.copy()
            result = self.solver.solve_step(time_index, profile, physical)
            self.last_solve_result = result
            if result.compressor_dispatches:
                applied_ratios = np.asarray(
                    [d.applied_pressure_ratio for d in result.compressor_dispatches], dtype=float
                )
                self.last_physical_slow["compressor_ratio"] = applied_ratios.copy()
                physical = PhysicalActions(
                    ess_p_mw=physical.ess_p_mw,
                    gfg_p_mw=physical.gfg_p_mw,
                    p2g_p_mw=physical.p2g_p_mw,
                    compressor_ratio=applied_ratios,
                    renewable_p_mw=physical.renewable_p_mw,
                    renewable_q_mvar=physical.renewable_q_mvar,
                    renewable_curtailment=physical.renewable_curtailment,
                )
                self.last_applied_action = self._normalized_action_from_physical(physical)
            # SOC 更新必须使用实际执行的 ESS 功率，而不是 Actor 原始请求。
            for i, ess in enumerate(ESS_CONFIGS):
                self.ess_soc[i] = update_ess_soc(self.ess_soc[i], physical.ess_p_mw[i], ess, self.config.time.dt_hours)
            self.consecutive_solver_failures = 0
            self.current_step += 1
            reward, components, metrics = self._compute_reward(False)
            terminated = self.current_step >= self.config.time.steps_per_day
            truncated = False
        except Exception as exc:
            # 回滚后仍推进一步并给失败惩罚，避免训练在同一个坏状态无限循环。
            self._restore_snapshot(snapshot)
            self.last_raw_action = action.copy()
            self.last_applied_action = attempted_applied_action.copy()
            self.consecutive_solver_failures += 1
            self.current_step = min(snapshot.current_step + 1, self.config.time.steps_per_day)
            reward, components, metrics = self._compute_reward(True)
            terminated = self.current_step >= self.config.time.steps_per_day
            truncated = self.consecutive_solver_failures >= self.config.safety.max_consecutive_solver_failures
            info = self._build_info(components, metrics, True, slow_action_applied)
            info["exception"] = repr(exc)
            return self.get_global_state(), reward, terminated, truncated, info
        return self.get_global_state(), reward, terminated, truncated, self._build_info(components, metrics, False, slow_action_applied)

    # 【中文导读】按固定动作顺序映射慢/快动作并执行 ESS、逆变器安全投影。
    def _map_and_project_action(self, action: np.ndarray, profile: Dict[str, np.ndarray | float], slow_action_applied: bool) -> PhysicalActions:
        """把 [-1, 1] 动作映射为 MW、Mvar、压力比等物理量。"""

        slow = np.clip(action[: self.slow_action_dim], -1.0, 1.0)
        fast = np.clip(action[self.slow_action_dim:], -1.0, 1.0)
        if slow_action_applied:
            # ESS 既有功率上限又有 SOC 上下限，所以先投影到当前步可行区间。
            self.last_slow_action = slow.copy()
            cursor = 0
            requested_ess = np.array([slow[cursor + i] * ESS_CONFIGS[i].max_p_mw for i in range(self.n_ess)])
            cursor += self.n_ess
            self.last_ess_projection = project_ess_batch(requested_ess, self.ess_soc, self.config.time.dt_hours)
            self.last_physical_slow["ess_p_mw"] = self.last_ess_projection.applied_p_mw.copy()
            self.last_physical_slow["gfg_p_mw"] = np.array([
                0.5 * (slow[cursor + i] + 1.0) * GFG_CONFIGS[i].max_p_mw for i in range(self.n_gfg)
            ])
            cursor += self.n_gfg
            self.last_physical_slow["p2g_p_mw"] = np.array([
                0.5 * (slow[cursor + i] + 1.0) * P2G_CONFIGS[i].max_p_mw for i in range(self.n_p2g)
            ])
            cursor += self.n_p2g
            ratios = default_compressor_ratios()
            for action_pos, comp_idx in enumerate(CONTROLLED_COMPRESSOR_INDICES):
                comp = COMPRESSOR_CONFIGS[comp_idx]
                ratios[comp_idx] = comp.min_pressure_ratio + 0.5 * (slow[cursor + action_pos] + 1.0) * (
                    comp.max_pressure_ratio - comp.min_pressure_ratio
                )
            self.last_physical_slow["compressor_ratio"] = ratios
        else:
            # 慢动作保持期间 SOC 会变化，因此 ESS 功率仍需重新检查可行性。
            self.last_ess_projection = project_ess_batch(
                self.last_physical_slow["ess_p_mw"], self.ess_soc, self.config.time.dt_hours
            )
            self.last_physical_slow["ess_p_mw"] = self.last_ess_projection.applied_p_mw.copy()

        fast_projected = self._project_fast_actions(fast, profile)
        physical = PhysicalActions(
            ess_p_mw=self.last_physical_slow["ess_p_mw"].copy(),
            gfg_p_mw=self.last_physical_slow["gfg_p_mw"].copy(),
            p2g_p_mw=self.last_physical_slow["p2g_p_mw"].copy(),
            compressor_ratio=self.last_physical_slow["compressor_ratio"].copy(),
            renewable_p_mw=fast_projected["renewable_p_mw"],
            renewable_q_mvar=fast_projected["renewable_q_mvar"],
            renewable_curtailment=fast_projected["renewable_curtailment"],
        )
        # executed_action 反映所有 SOC/容量约束后的真实动作，训练 Critic 时应使用这一版本。
        self.last_applied_action = self._normalized_action_from_physical(physical)
        return physical

    # 【中文导读】把实际物理动作反算为 replay 使用的 executed_action。
    def _normalized_action_from_physical(self, physical: PhysicalActions) -> np.ndarray:
        """将安全投影后的物理动作反算回 Actor 使用的 [-1, 1] 归一化动作。

        训练脚本会把 raw_action 和 applied_action 一起写入 replay buffer，
        让 Critic 学“实际执行动作”的价值。
        """
        values: List[float] = []
        values.extend([
            float(np.clip(physical.ess_p_mw[i] / max(ESS_CONFIGS[i].max_p_mw, 1e-9), -1.0, 1.0))
            for i in range(self.n_ess)
        ])
        values.extend([
            float(np.clip(2.0 * physical.gfg_p_mw[i] / max(GFG_CONFIGS[i].max_p_mw, 1e-9) - 1.0, -1.0, 1.0))
            for i in range(self.n_gfg)
        ])
        values.extend([
            float(np.clip(2.0 * physical.p2g_p_mw[i] / max(P2G_CONFIGS[i].max_p_mw, 1e-9) - 1.0, -1.0, 1.0))
            for i in range(self.n_p2g)
        ])
        for i in CONTROLLED_COMPRESSOR_INDICES:
            cfg = COMPRESSOR_CONFIGS[i]
            span = max(cfg.max_pressure_ratio - cfg.min_pressure_ratio, 1e-9)
            values.append(float(np.clip(2.0 * (physical.compressor_ratio[i] - cfg.min_pressure_ratio) / span - 1.0, -1.0, 1.0)))
        values.extend([
            float(np.clip(physical.renewable_q_mvar[i] / max(RENEWABLE_CONFIGS[i].s_rated_mva, 1e-9), -1.0, 1.0))
            for i in range(self.n_renew)
        ])
        values.extend([
            float(np.clip(2.0 * physical.renewable_curtailment[i] / max(RENEWABLE_CONFIGS[i].max_curtailment, 1e-9) - 1.0, -1.0, 1.0))
            for i in range(self.n_renew)
        ])
        return np.asarray(values, dtype=float)

    # 【中文导读】将 8 个无功请求和 8 个弃电请求投影到逆变器能力圆。
    def _project_fast_actions(self, fast: np.ndarray, profile: Dict[str, np.ndarray | float]) -> Dict[str, np.ndarray]:
        """投影快动作，使逆变器满足削减边界和 P²+Q²<=S²。"""

        q_norm = np.asarray(fast[: self.n_renew], dtype=float)
        c_norm = np.asarray(fast[self.n_renew:], dtype=float)
        available = np.asarray(profile["renewable_available_mw"], dtype=float)
        projections = []
        for i, cfg in enumerate(RENEWABLE_CONFIGS):
            q_request = q_norm[i] * cfg.s_rated_mva
            c_request = 0.5 * (c_norm[i] + 1.0) * cfg.max_curtailment
            projections.append(project_inverter_action(cfg, available[i], q_request, c_request))
        self.last_inverter_projection = projections
        return {
            "renewable_p_mw": np.array([p.p_actual_mw for p in projections]),
            "renewable_q_mvar": np.array([p.q_actual_mvar for p in projections]),
            "renewable_curtailment": np.array([p.curtailment for p in projections]),
        }

    # 【中文导读】按固定段顺序拼接并归一化动态维度的全局状态。
    def get_global_state(self) -> np.ndarray:
        """返回归一化后的全局观测。

        观测向量把电压、线路负载、SOC、压力、管道流量和时间特征拼接在一起。
        多数物理量都被缩放到神经网络更容易学习的范围。
        """

        if self.profiles is None:
            return np.zeros(self.global_state_dim, dtype=np.float32)
        net_p = self.power.net
        net_g = self.gas.net
        t = min(self.current_step, self.config.time.steps_per_day)
        profile = profile_at(self.profiles, t)
        vm = self._series_values(net_p, "res_bus", "vm_pu", 33, 1.0)
        loading = self._series_values(net_p, "res_line", "loading_percent", len(IEEE33_LINE_DATA), 0.0)
        ext_p = self._series_values(net_p, "res_ext_grid", "p_mw", 1, 0.0)
        p_loss = np.array([np.nansum(self._series_values(net_p, "res_line", "pl_mw", len(IEEE33_LINE_DATA), 0.0))])
        renewable_avail = np.asarray(profile["renewable_available_mw"], dtype=float)
        renewable_actual = np.array([p.p_actual_mw for p in self.last_inverter_projection] or np.zeros(self.n_renew))
        renewable_q = np.array([p.q_actual_mvar for p in self.last_inverter_projection] or np.zeros(self.n_renew))
        ess_margin = np.array([min(self.ess_soc[i] - ESS_CONFIGS[i].soc_min, ESS_CONFIGS[i].soc_max - self.ess_soc[i])
                               for i in range(self.n_ess)])
        ess_p_norm = np.array([self.last_physical_slow["ess_p_mw"][i] / max(ESS_CONFIGS[i].max_p_mw, 1e-9)
                               for i in range(self.n_ess)])
        gas_p = self._series_values(net_g, "res_junction", "p_bar", N_GAS_JUNCTIONS,
                                    self.config.gas.network_pressure_target_bar)
        gfg_inlet_p = read_gfg_inlet_pressures_bar(net_g, self.config.gas.network_pressure_target_bar)
        source_mdot, _ = read_gas_source_mdot_kg_s(net_g, self.gas)
        pipe_mdot = self._series_values(net_g, "res_pipe", "mdot_kg_per_s", len(GAS_PIPES), 0.0)
        comp_ratio = self.last_physical_slow["compressor_ratio"]
        if self.last_solve_result and self.last_solve_result.compressor_dispatches:
            comp_ratio = np.asarray(
                [d.applied_pressure_ratio for d in self.last_solve_result.compressor_dispatches],
                dtype=float,
            )
        gfg_mdot = self.last_solve_result.gfg_mdot_kg_s if self.last_solve_result else np.zeros(self.n_gfg)
        p2g_mdot = self.last_solve_result.p2g_mdot_kg_s if self.last_solve_result else np.zeros(self.n_p2g)
        linepack = self.last_solve_result.equivalent_linepack_indicator if self.last_solve_result else 0.0
        gas_age = self.last_solve_result.gas_state_age if self.last_solve_result else 0
        hour = (t * self.config.time.dt_hours) % 24.0
        day_fraction = (t % self.config.time.steps_per_day) / self.config.time.steps_per_day
        time_feat = np.array([
            np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * day_fraction), np.cos(2 * np.pi * day_fraction),
            float(profile["next_hour_load_multiplier"]), float(np.sum(profile["next_hour_renewable_available_mw"])),
        ])
        # parts 的顺序就是观测向量各段的含义，Manager/Worker 会基于它切片。
        # 全局状态固定顺序：电压/线路/新能源/购电与损耗/ESS/气压与流量/时间。
        # 改变此顺序会同时改变 Worker 切片语义，必须同步修改训练脚本。
        parts = [
            (vm - 1.0) / 0.10, loading / 100.0,
            np.array([float(profile["load_multiplier"]), float(np.sum(renewable_avail)), float(np.sum(renewable_actual))]),
            renewable_avail / np.maximum(np.array([r.capacity_mw for r in RENEWABLE_CONFIGS]), 1e-9),
            renewable_actual / np.maximum(np.array([r.capacity_mw for r in RENEWABLE_CONFIGS]), 1e-9),
            renewable_q / np.maximum(np.array([r.s_rated_mva for r in RENEWABLE_CONFIGS]), 1e-9),
            ext_p / 10.0, p_loss / 1.0, self.ess_soc, ess_p_norm, ess_margin,
            (gas_p - self.config.gas.network_pressure_target_bar) / 1.0,
            (gfg_inlet_p - self.config.gas.network_pressure_target_bar) / 1.0,
            comp_ratio / np.array([c.max_pressure_ratio for c in COMPRESSOR_CONFIGS]),
            source_mdot / np.maximum(np.array([s.max_mdot_kg_s for s in GAS_SUPPLIERS]), 1e-9),
            gfg_mdot / 0.12, p2g_mdot / 0.03,
            np.array([gas_age / self.config.time.slow_action_interval_steps, linepack / 200.0]),
            pipe_mdot / 0.25, time_feat,
        ]
        return np.nan_to_num(np.concatenate([np.atleast_1d(p) for p in parts]).astype(np.float32),
                             nan=0.0, posinf=10.0, neginf=-10.0)

    # 【中文导读】Manager 直接使用完整全局状态。
    def get_manager_state(self) -> np.ndarray:
        return self.get_global_state()

    # 【中文导读】按当前电网和新能源数量截取电侧状态作为快 Worker 基础观测。
    def get_fast_worker_state(self) -> np.ndarray:
        cut = 33 + len(IEEE33_LINE_DATA) + 5 + 3 * self.n_renew
        return self.get_global_state()[:cut].copy()

    # 【中文导读】按当前动作和网络维度截取 ESS/气侧状态作为慢 Worker 基础观测。
    def get_slow_worker_state(self) -> np.ndarray:
        start = 33 + len(IEEE33_LINE_DATA) + 5 + 3 * self.n_renew
        return self.get_global_state()[start:].copy()

    # 【中文导读】计算加权成本、物理诊断指标，并返回成本和的负值。
    def _compute_reward(self, solver_failed: bool) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
        """计算一步奖励和诊断指标。

        所有 components 都是成本；最终 reward 取负值，所以越接近 0 越好。
        metrics 保留未加权物理指标，用于训练日志和可视化。
        """

        dt_h = self.config.time.dt_hours
        vm = self._series_values(self.power.net, "res_bus", "vm_pu", 33, 1.0)
        loading = self._series_values(self.power.net, "res_line", "loading_percent", len(IEEE33_LINE_DATA), 0.0)
        gas_p = self._series_values(self.gas.net, "res_junction", "p_bar", N_GAS_JUNCTIONS,
                                    self.config.gas.network_pressure_target_bar)
        pipe_velocity = read_pipe_velocity_m_per_s(self.gas.net, len(GAS_PIPES))
        source_mdot, source_sign = read_gas_source_mdot_kg_s(self.gas.net, self.gas)
        source_caps = np.array([s.max_mdot_kg_s for s in GAS_SUPPLIERS], dtype=float)
        source_capacity_violation = np.maximum(source_mdot - source_caps, 0.0)
        gfg_inlet_pressures = read_gfg_inlet_pressures_bar(self.gas.net, self.config.gas.network_pressure_target_bar)
        gfg_inlet_pressure_violation = np.maximum(self.config.gas.network_pressure_min_bar - gfg_inlet_pressures, 0.0)
        voltage_deviation = float(np.sum(((vm - self.config.power.voltage_target_pu) / 0.05) ** 2))
        voltage_violation = float(np.sum(np.maximum(self.config.power.voltage_min_pu - vm, 0.0) ** 2 +
                                         np.maximum(vm - self.config.power.voltage_max_pu, 0.0) ** 2))
        gas_pressure_deviation = float(np.nanmean((gas_p - self.config.gas.network_pressure_target_bar) ** 2))
        gas_pressure_violation = float(np.nanmean(
            np.maximum(self.config.gas.network_pressure_min_bar - gas_p, 0.0) ** 2 +
            np.maximum(gas_p - self.config.gas.network_pressure_max_bar, 0.0) ** 2
        ))
        pipe_velocity_violation = float(np.nanmean(
            np.maximum(np.abs(pipe_velocity) - self.config.gas.max_pipe_velocity_m_per_s, 0.0) ** 2
        ))
        source_capacity_violation_cost_raw = float(np.sum(source_capacity_violation ** 2))
        line_overload = float(np.sum(np.maximum(loading - self.config.power.max_line_loading_percent, 0.0) ** 2))
        p_loss_mw = float(np.nansum(self._series_values(self.power.net, "res_line", "pl_mw", len(IEEE33_LINE_DATA), 0.0)))
        grid_purchase_mwh = max(float(np.nansum(self._series_values(self.power.net, "res_ext_grid", "p_mw", 1, 0.0))), 0.0) * dt_h
        gas_purchase_kg = float(np.nansum(np.maximum(source_mdot, 0.0))) * dt_h * 3600.0
        curtail_mwh = 0.0
        if self.last_inverter_projection and self.profiles is not None:
            prof = profile_at(self.profiles, min(self.current_step, self.config.time.steps_per_day))
            avail = np.asarray(prof["renewable_available_mw"], dtype=float)
            curtail_mwh = float(np.sum([avail[i] * p.curtailment * dt_h for i, p in enumerate(self.last_inverter_projection)]))
        comp_mwh = 0.0
        comp_dispatches = self.last_solve_result.compressor_dispatches if self.last_solve_result else []
        if self.last_solve_result:
            comp_mwh = float(sum(d.electric_power_mw for d in comp_dispatches) * dt_h)
        comp_signed = np.asarray([d.signed_mdot_kg_s for d in comp_dispatches], dtype=float)
        comp_effective = np.asarray([d.effective_mdot_kg_s for d in comp_dispatches], dtype=float)
        comp_requested_ratio = np.asarray([d.requested_pressure_ratio for d in comp_dispatches], dtype=float)
        comp_applied_ratio = np.asarray([d.applied_pressure_ratio for d in comp_dispatches], dtype=float)
        comp_power = np.asarray([d.electric_power_mw for d in comp_dispatches], dtype=float)
        comp_reverse = np.asarray([d.reverse_flow for d in comp_dispatches], dtype=bool)
        comp_bypassed = np.asarray([d.bypassed for d in comp_dispatches], dtype=bool)
        comp_power_limited = np.asarray([d.power_limited for d in comp_dispatches], dtype=bool)
        comp_projection = np.asarray([d.ratio_projection_magnitude for d in comp_dispatches], dtype=float)
        ess_change = float(np.sum(np.abs(self.last_physical_slow["ess_p_mw"] - self.previous_device_actions["ess_p_mw"])))
        gfg_change = float(np.sum(np.abs(self.last_physical_slow["gfg_p_mw"] - self.previous_device_actions["gfg_p_mw"])))
        p2g_change = float(np.sum(np.abs(self.last_physical_slow["p2g_p_mw"] - self.previous_device_actions["p2g_p_mw"])))
        soc_soft = float(np.sum(np.maximum(self.config.safety.soc_soft_low - self.ess_soc, 0.0) ** 2 +
                                np.maximum(self.ess_soc - self.config.safety.soc_soft_high, 0.0) ** 2))
        terminal_soc = 0.0
        if self.current_step >= self.config.time.steps_per_day:
            terminal_soc = float(np.sum((self.ess_soc - np.array([e.soc_initial for e in ESS_CONFIGS])) ** 2))
        w = self.config.reward
        components = {
            "voltage_deviation": w.voltage_deviation * voltage_deviation,
            "voltage_violation": w.voltage_violation * voltage_violation,
            "gas_pressure_deviation": w.gas_pressure_deviation * gas_pressure_deviation,
            "gas_pressure_violation": w.gas_pressure_violation * gas_pressure_violation,
            "pipe_velocity_violation": w.pipe_velocity_violation * pipe_velocity_violation,
            "source_capacity_violation": w.source_capacity_violation * source_capacity_violation_cost_raw,
            "line_overload": w.line_overload * line_overload,
            "power_loss": w.power_loss * p_loss_mw * dt_h,
            "grid_purchase": w.grid_energy_price * grid_purchase_mwh,
            "gas_purchase": w.gas_price * gas_purchase_kg / 1000.0,
            "renewable_curtailment": w.renewable_curtailment * curtail_mwh,
            "ess_action_change": w.ess_action_change * ess_change,
            "gfg_action_change": w.gfg_action_change * gfg_change,
            "p2g_action_change": w.p2g_action_change * p2g_change,
            "compressor_energy": w.compressor_energy * comp_mwh,
            "soc_soft": w.soc_soft * soc_soft,
            "solver_failure": w.solver_failure if solver_failed else 0.0,
            "terminal_soc": w.terminal_soc * terminal_soc,
        }
        metrics = {
            "vm_min_pu": float(np.nanmin(vm)), "vm_max_pu": float(np.nanmax(vm)),
            "voltage_mean_abs_deviation_pu": float(np.nanmean(np.abs(vm - self.config.power.voltage_target_pu))),
            "voltage_rms_deviation_pu": float(np.sqrt(np.nanmean((vm - self.config.power.voltage_target_pu) ** 2))),
            "max_line_loading_percent": float(np.nanmax(loading)),
            "gas_pressure_min_bar": float(np.nanmin(gas_p)),
            "gas_pressure_max_bar": float(np.nanmax(gas_p)),
            "gas_pressure_mean_bar": float(np.nanmean(gas_p)),
            "gas_pressure_mean_abs_deviation_bar": float(np.nanmean(np.abs(gas_p - self.config.gas.network_pressure_target_bar))),
            "gas_pressure_rms_deviation_bar": float(np.sqrt(np.nanmean((gas_p - self.config.gas.network_pressure_target_bar) ** 2))),
            "gfg_inlet_pressures_bar": gfg_inlet_pressures.copy(),
            "gfg_inlet_pressure_min_bar": float(np.nanmin(gfg_inlet_pressures)),
            "gfg_inlet_pressure_violation_bar": gfg_inlet_pressure_violation.copy(),
            "max_pipe_velocity_m_per_s": float(np.nanmax(np.abs(pipe_velocity))),
            "source_mdot_kg_s": source_mdot.copy(),
            "source_capacity_violation_kg_s": source_capacity_violation.copy(),
            "source_ext_grid_supply_sign": source_sign,
            "compressor_signed_mdot_kg_s": comp_signed.copy(),
            "compressor_effective_mdot_kg_s": comp_effective.copy(),
            "compressor_reverse_flow": comp_reverse.copy(),
            "compressor_bypassed": comp_bypassed.copy(),
            "compressor_requested_ratio": comp_requested_ratio.copy(),
            "compressor_applied_ratio": comp_applied_ratio.copy(),
            "compressor_power_mw": comp_power.copy(),
            "compressor_power_limited": comp_power_limited.copy(),
            "ratio_projection_magnitude": comp_projection.copy(),
            "soc_min": float(np.nanmin(self.ess_soc)), "soc_max": float(np.nanmax(self.ess_soc)),
            "grid_purchase_mwh": float(grid_purchase_mwh), "gas_purchase_kg": float(gas_purchase_kg),
        }
        self.previous_device_actions = {k: v.copy() for k, v in self.last_physical_slow.items()}
        # components 全为非负成本项；因此 reward 越接近 0 表示运行越优。
        return -float(sum(components.values())), components, metrics

    # 【中文导读】暴露 raw/applied 动作、投影、SOC、求解和约束诊断。
    def _build_info(self, components: Dict[str, float], metrics: Dict[str, Any],
                    solver_failed: bool, slow_action_applied: bool) -> Dict[str, Any]:
        r = self.last_solve_result
        return {
            "step": self.current_step,
            "slow_action_applied": slow_action_applied,
            "converged": bool(r.converged if r else False) and not solver_failed,
            "solver_failed": solver_failed,
            "power_converged": bool(r.power_converged if r else False) and not solver_failed,
            "gas_converged": bool(r.gas_converged if r else False) and not solver_failed,
            "gas_solved_this_step": bool(r.gas_solved_this_step if r else False),
            "gas_solve_reason": r.gas_solve_reason if r else "none",
            "gas_state_age": int(r.gas_state_age if r else 0),
            "gas_solve_count": int(self.solver.gas_solve_count),
            "equivalent_linepack_indicator": float(r.equivalent_linepack_indicator if r else 0.0),
            "ess_soc": self.ess_soc.copy(),
            "reward_components": components,
            "constraint_metrics": metrics,
            "raw_action": self.last_raw_action.copy(),
            "applied_action": self.last_applied_action.copy(),
            "action_projection_magnitude": float(np.linalg.norm(self.last_raw_action - self.last_applied_action)),
            "ess_projection": {
                "raw_p_mw": None if self.last_ess_projection is None else self.last_ess_projection.raw_p_mw.copy(),
                "applied_p_mw": None if self.last_ess_projection is None else self.last_ess_projection.applied_p_mw.copy(),
                "projection_magnitude_mw": None if self.last_ess_projection is None else self.last_ess_projection.projection_magnitude_mw.copy(),
                "hit_soc_boundary": None if self.last_ess_projection is None else self.last_ess_projection.hit_soc_boundary.copy(),
            },
            "inverter_projection": [
                {
                    "p_actual_mw": p.p_actual_mw,
                    "q_actual_mvar": p.q_actual_mvar,
                    "curtailment": p.curtailment,
                    "q_limit_mvar": p.q_limit_mvar,
                    "apparent_power_mva": p.apparent_power_mva,
                    "q_was_clipped": p.q_was_clipped,
                    "curtailment_was_clipped": p.curtailment_was_clipped,
                }
                for p in self.last_inverter_projection
            ],
        }

    def _series_values(self, net: Any, table_name: str, column: str, length: int, default: float) -> np.ndarray:
        if not hasattr(net, table_name):
            return np.full(length, default, dtype=float)
        table = getattr(net, table_name)
        if column not in table:
            return np.full(length, default, dtype=float)
        values = np.asarray(table[column].values, dtype=float)
        if values.size < length:
            values = np.pad(values, (0, length - values.size), constant_values=default)
        return values[:length]

    def _copy_tables(self, net: Any, names: Tuple[str, ...]) -> Dict[str, Any]:
        return {name: getattr(net, name).copy(deep=True) for name in names if hasattr(net, name)}

    def _restore_tables(self, net: Any, tables: Dict[str, Any]) -> None:
        for name, table in tables.items():
            setattr(net, name, table.copy(deep=True))

    def _make_snapshot(self) -> Snapshot:
        return Snapshot(
            self.current_step,
            self.ess_soc.copy(),
            self.last_slow_action.copy(),
            self.last_raw_action.copy(),
            self.last_applied_action.copy(),
            copy.deepcopy(self.last_physical_slow),
            copy.deepcopy(self.previous_device_actions),
            int(self.consecutive_solver_failures),
            copy.deepcopy(self.last_ess_projection),
            copy.deepcopy(self.last_inverter_projection),
            copy.deepcopy(self.last_solve_result),
            self.solver.gas_state_age,
            self.solver.gas_solve_count,
            bool(self.solver.last_gas_converged),
            self.solver.last_compressor_mdot_kg_s.copy(),
            copy.deepcopy(self.solver.scheduler.last_state),
            self._copy_tables(self.power.net, ("load", "sgen", "storage", "res_bus", "res_line", "res_load", "res_sgen", "res_storage", "res_ext_grid")),
            self._copy_tables(self.gas.net, ("sink", "source", "compressor", "res_junction", "res_pipe", "res_sink", "res_source", "res_ext_grid", "res_compressor")),
        )

    def _restore_snapshot(self, snapshot: Snapshot) -> None:
        self.current_step = int(snapshot.current_step)
        self.ess_soc = snapshot.ess_soc.copy()
        self.last_slow_action = snapshot.last_slow_action.copy()
        self.last_raw_action = snapshot.last_raw_action.copy()
        self.last_applied_action = snapshot.last_applied_action.copy()
        self.last_physical_slow = copy.deepcopy(snapshot.last_physical_slow)
        self.previous_device_actions = copy.deepcopy(snapshot.previous_device_actions)
        self.consecutive_solver_failures = int(snapshot.consecutive_solver_failures)
        self.last_ess_projection = copy.deepcopy(snapshot.last_ess_projection)
        self.last_inverter_projection = copy.deepcopy(snapshot.last_inverter_projection)
        self.last_solve_result = copy.deepcopy(snapshot.last_solve_result)
        self.solver.gas_state_age = snapshot.solver_gas_state_age
        self.solver.gas_solve_count = snapshot.solver_gas_solve_count
        self.solver.last_gas_converged = bool(snapshot.solver_last_gas_converged)
        self.solver.last_compressor_mdot_kg_s = snapshot.solver_last_compressor_mdot_kg_s.copy()
        self.solver.scheduler.last_state = copy.deepcopy(snapshot.scheduler_last_state)
        self._restore_tables(self.power.net, snapshot.power_tables)
        self._restore_tables(self.gas.net, snapshot.gas_tables)
        if self.solver.scheduler.last_state is not None:
            ratios = np.asarray(self.solver.scheduler.last_state.compressor_ratio, dtype=float)
            for i, idx in enumerate(self.gas.compressor_indices):
                if i < ratios.size:
                    self.gas.net.compressor.at[idx, "pressure_ratio"] = float(ratios[i])


# =============================================================================
# Gas calibration scenarios
# =============================================================================


def _apply_gas_calibration_scenario(name: str, gas_multiplier: float, gfg_fraction: float,
                                    p2g_fraction: float, compressor_ratios: Sequence[float],
                                    config: ProjectConfig | None = None) -> Dict[str, Any]:
    cfg = config or DEFAULT_CONFIG
    gas = build_gas_network(cfg)
    base_load_total = 0.0
    for node in GAS_NODES:
        idx = gas.base_sink_indices_by_node.get(node.node)
        if idx is not None:
            mdot = node.base_mdot_kg_s * gas_multiplier
            gas.net.sink.at[idx, "mdot_kg_per_s"] = mdot
            base_load_total += mdot
    gfg_mdot = []
    for i, gfg in enumerate(GFG_CONFIGS):
        power = gfg.max_p_mw * gfg_fraction
        disp = dispatch_gfg(gfg, power, cfg.gas.hhv_mj_per_kg)
        gas.net.sink.at[gas.gfg_sink_indices[i], "mdot_kg_per_s"] = disp.gas_mdot_kg_s
        gfg_mdot.append(disp.gas_mdot_kg_s)
    p2g_mdot = []
    for i, p2g in enumerate(P2G_CONFIGS):
        power = p2g.max_p_mw * p2g_fraction
        disp = dispatch_p2g(p2g, power, cfg.gas.hhv_mj_per_kg)
        gas.net.source.at[gas.p2g_source_indices[i], "mdot_kg_per_s"] = disp.gas_mdot_kg_s
        p2g_mdot.append(disp.gas_mdot_kg_s)
    gfg_mdot_arr = np.asarray(gfg_mdot, dtype=float)
    p2g_mdot_arr = np.asarray(p2g_mdot, dtype=float)
    auxiliary_mdot = compute_auxiliary_source_mdot_kg_s(gas_multiplier, gfg_mdot_arr, p2g_mdot_arr)
    gas.net.source.at[gas.auxiliary_source_index, "mdot_kg_per_s"] = auxiliary_mdot
    ratios = full_compressor_ratios_from_action(compressor_ratios)
    for i, idx in enumerate(gas.compressor_indices):
        gas.net.compressor.at[idx, "pressure_ratio"] = float(ratios[i])

    import pandapipes as pp
    pp.pipeflow(gas.net, max_iter_hyd=80, tol_p=1e-5, use_numba=False)
    comp_mdot = read_compressor_mdot_estimates(gas.net, gas.compressor_indices)
    comp_dispatches = [
        dispatch_compressor(comp, float(ratios[i]), float(comp_mdot[i]))
        for i, comp in enumerate(COMPRESSOR_CONFIGS)
    ]
    if any(d.power_limited and d.ratio_projection_magnitude > 1e-9 for d in comp_dispatches):
        for i, idx in enumerate(gas.compressor_indices):
            gas.net.compressor.at[idx, "pressure_ratio"] = comp_dispatches[i].applied_pressure_ratio
        pp.pipeflow(gas.net, max_iter_hyd=80, tol_p=1e-5, use_numba=False)
        comp_mdot = read_compressor_mdot_estimates(gas.net, gas.compressor_indices)
        applied_ratios = np.asarray(
            [float(gas.net.compressor.at[idx, "pressure_ratio"]) for idx in gas.compressor_indices],
            dtype=float,
        )
        comp_dispatches = [
            _compressor_dispatch_from_applied(
                comp,
                float(ratios[i]),
                float(applied_ratios[i]),
                float(comp_mdot[i]),
                abs(np.clip(float(ratios[i]), comp.min_pressure_ratio, comp.max_pressure_ratio) -
                    float(ratios[i])) > 1e-9,
                max(np.clip(float(ratios[i]), comp.min_pressure_ratio, comp.max_pressure_ratio) -
                    float(applied_ratios[i]), 0.0) > 1e-9,
            )
            for i, comp in enumerate(COMPRESSOR_CONFIGS)
        ]

    pressure = _table_column_values(gas.net, "res_junction", "p_bar", N_GAS_JUNCTIONS,
                                    cfg.gas.network_pressure_target_bar)
    pipe_mdot = _table_column_values(gas.net, "res_pipe", "mdot_kg_per_s", len(GAS_PIPES), 0.0)
    pipe_velocity = read_pipe_velocity_m_per_s(gas.net, len(GAS_PIPES))
    source_mdot, source_sign = read_gas_source_mdot_kg_s(gas.net, gas)
    source_capacity_violation = np.maximum(
        source_mdot - np.array([s.max_mdot_kg_s for s in GAS_SUPPLIERS], dtype=float),
        0.0,
    )
    mass_balance_error = abs(
        float(np.sum(source_mdot)) + float(np.sum(p2g_mdot_arr)) -
        base_load_total - float(np.sum(gfg_mdot_arr))
    )
    min_pressure_node = int(np.nanargmin(pressure))
    max_velocity_pipe_index = int(np.nanargmax(np.abs(pipe_velocity)))
    result: Dict[str, Any] = {
        "scenario": name,
        "converged": bool(getattr(gas.net, "converged", True)),
        "min_pressure_bar": float(np.nanmin(pressure)),
        "max_pressure_bar": float(np.nanmax(pressure)),
        "min_pressure_node": min_pressure_node,
        "min_pressure_node_name": GAS_NODE_NAMES[min_pressure_node],
        "max_abs_velocity_m_per_s": float(np.nanmax(np.abs(pipe_velocity))),
        "max_velocity_pipe": GAS_PIPES[max_velocity_pipe_index].name,
        "source_mdot_kg_s": source_mdot.copy(),
        "source_capacity_violation_kg_s": source_capacity_violation.copy(),
        "source_ext_grid_supply_sign": source_sign,
        "pressure_ext_grid_count": int(len(gas.net.ext_grid)),
        "auxiliary_source_mdot_kg_s": float(auxiliary_mdot),
        "base_load_mdot_kg_s": float(base_load_total),
        "gfg_mdot_kg_s": gfg_mdot_arr.copy(),
        "p2g_mdot_kg_s": p2g_mdot_arr.copy(),
        "net_demand_kg_s": float(base_load_total + np.sum(gfg_mdot_arr) - np.sum(p2g_mdot_arr)),
        "mass_balance_error_kg_s": float(mass_balance_error),
        "compressor_signed_mdot_kg_s": comp_mdot.copy(),
        "compressor_effective_mdot_kg_s": np.asarray([d.effective_mdot_kg_s for d in comp_dispatches], dtype=float),
        "compressor_reverse_flow": np.asarray([d.reverse_flow for d in comp_dispatches], dtype=bool),
        "compressor_bypassed": np.asarray([d.bypassed for d in comp_dispatches], dtype=bool),
        "compressor_requested_ratio": np.asarray([d.requested_pressure_ratio for d in comp_dispatches], dtype=float),
        "compressor_applied_ratio": np.asarray([d.applied_pressure_ratio for d in comp_dispatches], dtype=float),
        "compressor_power_limited": np.asarray([d.power_limited for d in comp_dispatches], dtype=bool),
        "ratio_projection_magnitude": np.asarray([d.ratio_projection_magnitude for d in comp_dispatches], dtype=float),
        "compressor_mdot_kg_s": np.asarray([d.effective_mdot_kg_s for d in comp_dispatches], dtype=float),
        "compressor_ratio": np.asarray([d.applied_pressure_ratio for d in comp_dispatches], dtype=float),
        "compressor_power_mw": np.asarray([d.electric_power_mw for d in comp_dispatches], dtype=float),
        "pipe_mdot_kg_s": pipe_mdot.copy(),
        "pipe_velocity_m_per_s": pipe_velocity.copy(),
        "pressure_bar": pressure.copy(),
    }
    finite_checks = [
        np.all(np.isfinite(pressure)),
        np.all(np.isfinite(pipe_mdot)),
        np.all(np.isfinite(pipe_velocity)),
        np.all(np.isfinite(source_mdot)),
        np.all(np.isfinite(comp_mdot)),
        np.isfinite(mass_balance_error),
    ]
    if not all(finite_checks):
        raise AssertionError(f"{name}: non-finite gas calibration result")
    return result


def confirm_ext_grid_supply_sign_single_load(config: ProjectConfig | None = None) -> Dict[str, Any]:
    cfg = config or DEFAULT_CONFIG
    gas = build_gas_network(cfg)
    for idx in gas.net.sink.index:
        gas.net.sink.at[idx, "mdot_kg_per_s"] = 0.0
    for idx in gas.net.source.index:
        gas.net.source.at[idx, "mdot_kg_per_s"] = 0.0
    test_sink = gas.base_sink_indices_by_node[2]
    gas.net.sink.at[test_sink, "mdot_kg_per_s"] = 0.05
    for idx in gas.compressor_indices:
        gas.net.compressor.at[idx, "pressure_ratio"] = 1.0
    import pandapipes as pp
    pp.pipeflow(gas.net, max_iter_hyd=80, tol_p=1e-5, use_numba=False)
    supply, sign = read_gas_source_mdot_kg_s(gas.net, gas)
    raw = _table_column_values(gas.net, "res_ext_grid", "mdot_kg_per_s", 1, 0.0)
    if not bool(getattr(gas.net, "converged", True)) or supply[0] <= 0.0:
        raise AssertionError("single-load ext_grid sign confirmation failed")
    return {
        "raw_ext_grid_mdot_kg_s": float(raw[0]),
        "supply_positive_mdot_kg_s": float(supply[0]),
        "source_ext_grid_supply_sign": sign,
    }


def run_fixed_compressor_station_consistency_test(config: ProjectConfig | None = None) -> Dict[str, Any]:
    cfg = config or DEFAULT_CONFIG
    station = COMPRESSOR_CONFIGS[0]
    fixed_ratio = compressor_engineering_ratio(station)
    if station.controllable:
        raise AssertionError("7->8 station must be fixed and absent from RL compressor actions")
    result = _apply_gas_calibration_scenario(
        "fixed_7_to_8_station_consistency", 1.15, 1.0, 0.0, default_compressor_ratios(), config=cfg
    )
    if not result["converged"]:
        raise AssertionError("fixed 7->8 station consistency case did not converge")
    if abs(float(result["compressor_requested_ratio"][0]) - fixed_ratio) > 1e-9:
        raise AssertionError("fixed 7->8 station requested ratio drifted from the engineering value")
    if result["compressor_signed_mdot_kg_s"][0] <= 0.0:
        if result["compressor_power_mw"][0] != 0.0 or not bool(result["compressor_bypassed"][0]):
            raise AssertionError("reverse-flow 7->8 station must bypass with zero compressor power")
    else:
        if result["compressor_power_mw"][0] < -1e-12:
            raise AssertionError("7->8 station compressor power must be non-negative")
        if result["compressor_power_mw"][0] > station.max_power_mw + 1e-6:
            raise AssertionError("7->8 station power exceeds max_power_mw after projection")
        low_cap = replace(station, max_power_mw=max(float(result["compressor_power_mw"][0]) * 0.5, 1e-6))
        limited = dispatch_compressor(
            low_cap, fixed_ratio, float(result["compressor_signed_mdot_kg_s"][0])
        )
        if not limited.power_limited or limited.applied_pressure_ratio >= fixed_ratio - 1e-9:
            raise AssertionError("7->8 fixed station power limit did not project the applied ratio")
    return {
        "scenario": result,
        "fixed_pressure_ratio": fixed_ratio,
        "signed_mdot_kg_s": float(result["compressor_signed_mdot_kg_s"][0]),
        "power_mw": float(result["compressor_power_mw"][0]),
        "bypassed": bool(result["compressor_bypassed"][0]),
    }


def run_controllable_compressor_ratio_sensitivity_test(config: ProjectConfig | None = None) -> Dict[str, Any]:
    cfg = config or DEFAULT_CONFIG
    if CONTROLLED_COMPRESSOR_INDICES != (1,):
        raise AssertionError("Only COMP_17_TO_18 should be controllable in belgian20_derived_mp_v2")
    fixed = compressor_engineering_ratio(COMPRESSOR_CONFIGS[0])
    comp = COMPRESSOR_CONFIGS[1]
    scenario_candidates = (
        ("ratio_sensitivity_16_17_peak", 1.15, 1.0, 0.0),
        ("ratio_sensitivity_16_17_base", 1.00, 1.0, 0.0),
        ("ratio_sensitivity_16_17_half_gfg", 1.00, 0.5, 0.0),
    )
    low = high = None
    for base_name, gas_mult, gfg_frac, p2g_frac in scenario_candidates:
        low_candidate = _apply_gas_calibration_scenario(
            f"{base_name}_r1p00", gas_mult, gfg_frac, p2g_frac,
            np.array([fixed, 1.00], dtype=float), config=cfg
        )
        high_candidate = _apply_gas_calibration_scenario(
            f"{base_name}_r{comp.initial_pressure_ratio:.2f}", gas_mult, gfg_frac, p2g_frac,
            np.array([fixed, comp.initial_pressure_ratio], dtype=float), config=cfg
        )
        if not low_candidate["converged"] or not high_candidate["converged"]:
            continue
        if low_candidate["compressor_signed_mdot_kg_s"][1] > 0.0 and high_candidate["compressor_signed_mdot_kg_s"][1] > 0.0:
            low, high = low_candidate, high_candidate
            break
    if low is None or high is None:
        raise AssertionError("could not construct a positive-flow sensitivity case for COMP_17_TO_18")
    downstream_nodes = np.asarray([17, 18, 19], dtype=int)
    pressure_delta = np.abs(high["pressure_bar"][downstream_nodes] - low["pressure_bar"][downstream_nodes])
    max_downstream_delta = float(np.nanmax(pressure_delta))
    power_delta = float(high["compressor_power_mw"][1] - low["compressor_power_mw"][1])
    if max_downstream_delta <= 1e-5 and power_delta <= 1e-9:
        raise AssertionError("COMP_17_TO_18 ratio change has no downstream pressure or power effect")
    if high["compressor_power_mw"][1] <= low["compressor_power_mw"][1] + 1e-9:
        raise AssertionError("higher COMP_17_TO_18 ratio did not increase compressor power")
    if not low["converged"] or not high["converged"]:
        raise AssertionError("compressor ratio sensitivity cases did not converge")
    return {
        "low": low,
        "high": high,
        "controlled_compressor_index": 1,
        "max_downstream_pressure_delta_bar": max_downstream_delta,
        "node17_pressure_delta_bar": float(high["pressure_bar"][17] - low["pressure_bar"][17]),
        "node18_pressure_delta_bar": float(high["pressure_bar"][18] - low["pressure_bar"][18]),
        "power_delta_mw": power_delta,
    }


def run_compressor_ratio_sensitivity_test(config: ProjectConfig | None = None) -> Dict[str, Any]:
    return run_controllable_compressor_ratio_sensitivity_test(config)


def run_gas_calibration_tests(config: ProjectConfig | None = None) -> Tuple[Dict[str, Any], ...]:
    cfg = config or DEFAULT_CONFIG
    confirm_ext_grid_supply_sign_single_load(cfg)
    initial_ratios = default_compressor_ratios()
    scenarios = (
        ("A_base", 1.00, 0.50, 0.00, initial_ratios),
        ("B_peak", 1.15, 1.00, 0.00, initial_ratios),
        ("C_high_p2g_low_load", 0.35, 0.00, 1.00, initial_ratios),
        ("D_min_ratio", 1.00, 1.00, 0.00, np.array([initial_ratios[0], 1.0], dtype=float)),
    )
    results = tuple(_apply_gas_calibration_scenario(*scenario, config=cfg) for scenario in scenarios)
    for result in results[:3]:
        if not result["converged"]:
            raise AssertionError(f"{result['scenario']}: pipeflow did not converge")
        if result["min_pressure_bar"] < cfg.gas.network_pressure_min_bar - 1e-6:
            raise AssertionError(f"{result['scenario']}: min pressure {result['min_pressure_bar']:.4f} bar below limit")
        if result["max_pressure_bar"] > cfg.gas.network_pressure_max_bar + 1e-6:
            raise AssertionError(f"{result['scenario']}: max pressure {result['max_pressure_bar']:.4f} bar above limit")
        if result["max_abs_velocity_m_per_s"] > cfg.gas.max_pipe_velocity_m_per_s + 1e-6:
            raise AssertionError(f"{result['scenario']}: max velocity {result['max_abs_velocity_m_per_s']:.4f} m/s above limit")
        if np.any(result["source_mdot_kg_s"] < -1e-6):
            raise AssertionError(f"{result['scenario']}: source absorption detected {result['source_mdot_kg_s']}")
        if result["source_mdot_kg_s"][0] > GAS_SUPPLIERS[0].max_mdot_kg_s + 1e-6:
            raise AssertionError(f"{result['scenario']}: main source exceeds capacity")
        if result["source_mdot_kg_s"][1] > GAS_SUPPLIERS[1].max_mdot_kg_s + 1e-6:
            raise AssertionError(f"{result['scenario']}: auxiliary source exceeds capacity")
        if np.max(result["source_capacity_violation_kg_s"]) > 1e-6:
            raise AssertionError(f"{result['scenario']}: source capacity violation is not near zero")
        if result["mass_balance_error_kg_s"] > 1e-4:
            raise AssertionError(f"{result['scenario']}: mass balance error {result['mass_balance_error_kg_s']:.6e} kg/s")
        if result["pressure_ext_grid_count"] != 1:
            raise AssertionError(f"{result['scenario']}: more than one pressure ext_grid is present")
        if result["auxiliary_source_mdot_kg_s"] > 1e-6 and result["compressor_signed_mdot_kg_s"][0] <= 0.0:
            raise AssertionError(f"{result['scenario']}: 7->8 compressor should carry positive forward flow")
    if not results[3]["converged"]:
        raise AssertionError("D_min_ratio: pipeflow did not converge")
    run_fixed_compressor_station_consistency_test(cfg)
    run_controllable_compressor_ratio_sensitivity_test(cfg)
    return results


def _gas_event_state_close(left: GasEventState | None, right: GasEventState | None) -> bool:
    if left is None or right is None:
        return left is right
    return (
        np.allclose(left.gfg_mdot_kg_s, right.gfg_mdot_kg_s)
        and np.allclose(left.p2g_mdot_kg_s, right.p2g_mdot_kg_s)
        and np.allclose(left.compressor_ratio, right.compressor_ratio)
        and abs(left.gas_load_multiplier - right.gas_load_multiplier) <= 1e-12
    )


def _solve_result_close(left: CoupledSolveResult | None, right: CoupledSolveResult | None) -> bool:
    if left is None or right is None:
        return left is right
    if (
        left.power_converged != right.power_converged
        or left.gas_converged != right.gas_converged
        or left.gas_solved_this_step != right.gas_solved_this_step
        or left.gas_solve_reason != right.gas_solve_reason
        or left.gas_state_age != right.gas_state_age
    ):
        return False
    if not np.allclose(left.gfg_mdot_kg_s, right.gfg_mdot_kg_s):
        return False
    if not np.allclose(left.p2g_mdot_kg_s, right.p2g_mdot_kg_s):
        return False
    if len(left.compressor_dispatches) != len(right.compressor_dispatches):
        return False
    for ldisp, rdisp in zip(left.compressor_dispatches, right.compressor_dispatches):
        if abs(ldisp.applied_pressure_ratio - rdisp.applied_pressure_ratio) > 1e-12:
            return False
        if abs(ldisp.signed_mdot_kg_s - rdisp.signed_mdot_kg_s) > 1e-12:
            return False
    return True


def run_forced_solver_failure_rollback_test(config: ProjectConfig | None = None) -> Dict[str, Any]:
    env = ElectricGasMultiScaleEnv(config)
    env.reset(seed=321)
    snapshot = env._make_snapshot()
    before_compressor_ratio = np.asarray(
        env.gas.net.compressor.loc[env.gas.compressor_indices, "pressure_ratio"].values,
        dtype=float,
    ).copy()
    before_source_mdot = np.asarray(env.gas.net.source["mdot_kg_per_s"].values, dtype=float).copy()
    before_sink_mdot = np.asarray(env.gas.net.sink["mdot_kg_per_s"].values, dtype=float).copy()
    original_run_powerflow = env.solver._run_powerflow

    def forced_powerflow_failure() -> None:
        raise RuntimeError("forced powerflow failure after gas solve")

    env.solver._run_powerflow = forced_powerflow_failure  # type: ignore[method-assign]
    try:
        _, _, _, _, failed_info = env.step(np.zeros(env.action_dim, dtype=np.float32))
    finally:
        env.solver._run_powerflow = original_run_powerflow  # type: ignore[method-assign]

    if not failed_info.get("solver_failed", False):
        raise AssertionError("forced rollback test did not trigger a solver failure")

    restored_compressor_ratio = np.asarray(
        env.gas.net.compressor.loc[env.gas.compressor_indices, "pressure_ratio"].values,
        dtype=float,
    ).copy()
    restored_source_mdot = np.asarray(env.gas.net.source["mdot_kg_per_s"].values, dtype=float).copy()
    restored_sink_mdot = np.asarray(env.gas.net.sink["mdot_kg_per_s"].values, dtype=float).copy()
    rollback_ok = (
        np.allclose(restored_compressor_ratio, before_compressor_ratio)
        and np.allclose(restored_source_mdot, before_source_mdot)
        and np.allclose(restored_sink_mdot, before_sink_mdot)
        and np.allclose(env.solver.last_compressor_mdot_kg_s, snapshot.solver_last_compressor_mdot_kg_s)
        and _gas_event_state_close(env.solver.scheduler.last_state, snapshot.scheduler_last_state)
        and _solve_result_close(env.last_solve_result, snapshot.last_solve_result)
    )
    if env.solver.scheduler.last_state is not None:
        scheduler_ratios = np.asarray(env.solver.scheduler.last_state.compressor_ratio, dtype=float)
        rollback_ok = rollback_ok and np.allclose(restored_compressor_ratio, scheduler_ratios)
    if not rollback_ok:
        raise AssertionError("forced solver failure rollback did not restore gas/scheduler state")

    restored_last_compressor_mdot = env.solver.last_compressor_mdot_kg_s.copy()
    restored_last_solve_reason = env.last_solve_result.gas_solve_reason if env.last_solve_result else "none"
    _, _, _, _, next_info = env.step(np.zeros(env.action_dim, dtype=np.float32))
    next_step_ok = not bool(next_info.get("solver_failed", False))
    if not next_step_ok:
        raise AssertionError("next step reused failed gas state after rollback")
    return {
        "rollback_ok": bool(rollback_ok),
        "next_step_ok": bool(next_step_ok),
        "restored_compressor_ratio": restored_compressor_ratio.copy(),
        "restored_compressor_mdot_kg_s": restored_last_compressor_mdot,
        "restored_last_solve_reason": restored_last_solve_reason,
    }


# =============================================================================
# Random policy and visualization
# =============================================================================


# 【中文导读】汇总随机策略仿真 episode 的统计量。
@dataclass
class EpisodeStats:
    """随机策略跑完一天后的统计量。

    这些字段不是训练需要的状态，而是给人看环境是否能稳定运行。
    """

    power_success: List[bool] = field(default_factory=list)
    gas_success: List[bool] = field(default_factory=list)
    vm_min: List[float] = field(default_factory=list)
    vm_max: List[float] = field(default_factory=list)
    voltage_violation_count: int = 0
    max_line_loading: List[float] = field(default_factory=list)
    gas_pressure_min: List[float] = field(default_factory=list)
    gas_pressure_max: List[float] = field(default_factory=list)
    soc_min: List[float] = field(default_factory=list)
    soc_max: List[float] = field(default_factory=list)
    total_power_loss_cost: float = 0.0
    total_curtailment_cost: float = 0.0
    total_grid_purchase_mwh: float = 0.0
    total_gas_purchase_kg: float = 0.0
    gas_solve_count_last: int = 0
    slow_action_count: int = 0
    records: List[Dict[str, float]] = field(default_factory=list)


# 【中文导读】运行随机策略或基线 episode 以检查环境。
def run_episode(seed: int = 42) -> EpisodeStats:
    """用随机动作跑一天，作为环境 smoke test 和可视化数据来源。"""

    env = ElectricGasMultiScaleEnv()
    obs, info = env.reset(seed=seed)
    del obs, info 
    stats = EpisodeStats()
    terminated = truncated = False
    step_id = 0
    dt_hours = env.config.time.dt_hours
    while not (terminated or truncated):
        # 随机策略不追求好回报，只验证动作空间、投影、求解和日志链路都能跑通。
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        del obs
        metrics = info.get("constraint_metrics", {})
        comps = info.get("reward_components", {})
        power_ok = bool(info.get("power_converged", False))
        gas_ok = bool(info.get("gas_converged", False))
        vm_min = float(metrics.get("vm_min_pu", np.nan))
        vm_max = float(metrics.get("vm_max_pu", np.nan))
        stats.power_success.append(power_ok)
        stats.gas_success.append(gas_ok)
        stats.vm_min.append(vm_min)
        stats.vm_max.append(vm_max)
        if vm_min < 0.95 or vm_max > 1.05:
            stats.voltage_violation_count += 1
        stats.max_line_loading.append(float(metrics.get("max_line_loading_percent", np.nan)))
        stats.gas_pressure_min.append(float(metrics.get("gas_pressure_min_bar", np.nan)))
        stats.gas_pressure_max.append(float(metrics.get("gas_pressure_max_bar", np.nan)))
        stats.soc_min.append(float(metrics.get("soc_min", np.nan)))
        stats.soc_max.append(float(metrics.get("soc_max", np.nan)))
        stats.total_power_loss_cost += float(comps.get("power_loss", 0.0))
        stats.total_curtailment_cost += float(comps.get("renewable_curtailment", 0.0))
        stats.total_grid_purchase_mwh += float(metrics.get("grid_purchase_mwh", 0.0))
        stats.total_gas_purchase_kg += float(metrics.get("gas_purchase_kg", 0.0))
        stats.gas_solve_count_last = int(info.get("gas_solve_count", 0))
        if info.get("slow_action_applied", False):
            stats.slow_action_count += 1
        stats.records.append(_build_record(step_id, dt_hours, reward, info, metrics, comps))
        step_id += 1
    return stats


# 【中文导读】把每步物理结果整理为 CSV 记录。
def _build_record(step_id: int, dt_hours: float, reward: float, info: Dict[str, Any],
                  metrics: Dict[str, Any], comps: Dict[str, Any]) -> Dict[str, float]:
    """把嵌套的 info 字典展平为一行，便于写 CSV 和画图。"""

    return {
        "step": float(step_id + 1),
        "hour": float((step_id + 1) * dt_hours),
        "reward": float(reward),
        "power_converged": float(bool(info.get("power_converged", False))),
        "gas_converged": float(bool(info.get("gas_converged", False))),
        "gas_solved_this_step": float(bool(info.get("gas_solved_this_step", False))),
        "slow_action_applied": float(bool(info.get("slow_action_applied", False))),
        "gas_state_age": float(info.get("gas_state_age", 0)),
        "vm_min_pu": float(metrics.get("vm_min_pu", np.nan)),
        "vm_max_pu": float(metrics.get("vm_max_pu", np.nan)),
        "voltage_mean_abs_deviation_pu": float(metrics.get("voltage_mean_abs_deviation_pu", np.nan)),
        "voltage_rms_deviation_pu": float(metrics.get("voltage_rms_deviation_pu", np.nan)),
        "max_line_loading_percent": float(metrics.get("max_line_loading_percent", np.nan)),
        "gas_pressure_min_bar": float(metrics.get("gas_pressure_min_bar", np.nan)),
        "gas_pressure_max_bar": float(metrics.get("gas_pressure_max_bar", np.nan)),
        "gas_pressure_mean_bar": float(metrics.get("gas_pressure_mean_bar", np.nan)),
        "gas_pressure_mean_abs_deviation_bar": float(metrics.get("gas_pressure_mean_abs_deviation_bar", np.nan)),
        "gas_pressure_rms_deviation_bar": float(metrics.get("gas_pressure_rms_deviation_bar", np.nan)),
        "gfg_inlet_pressure_min_bar": float(metrics.get("gfg_inlet_pressure_min_bar", np.nan)),
        "max_pipe_velocity_m_per_s": float(metrics.get("max_pipe_velocity_m_per_s", np.nan)),
        "soc_min": float(metrics.get("soc_min", np.nan)),
        "soc_max": float(metrics.get("soc_max", np.nan)),
        "grid_purchase_mwh": float(metrics.get("grid_purchase_mwh", 0.0)),
        "gas_purchase_kg": float(metrics.get("gas_purchase_kg", 0.0)),
        "voltage_deviation_cost": float(comps.get("voltage_deviation", 0.0)),
        "voltage_violation_cost": float(comps.get("voltage_violation", 0.0)),
        "gas_pressure_deviation_cost": float(comps.get("gas_pressure_deviation", 0.0)),
        "gas_pressure_violation_cost": float(comps.get("gas_pressure_violation", 0.0)),
        "pipe_velocity_violation_cost": float(comps.get("pipe_velocity_violation", 0.0)),
        "source_capacity_violation_cost": float(comps.get("source_capacity_violation", 0.0)),
        "line_overload_cost": float(comps.get("line_overload", 0.0)),
        "power_loss_cost": float(comps.get("power_loss", 0.0)),
        "renewable_curtailment_cost": float(comps.get("renewable_curtailment", 0.0)),
        "compressor_energy_cost": float(comps.get("compressor_energy", 0.0)),
    }


# 【中文导读】保存运行数据和可视化。
def save_episode_artifacts(records: Sequence[Mapping[str, Any]], output_dir: str | Path, prefix: str = "single_file_random_policy") -> Dict[str, Path]:
    """保存随机策略的 CSV 与仪表盘图。"""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{prefix}_timeseries.csv"
    dashboard_path = out / f"{prefix}_dashboard.png"
    _write_csv(records, csv_path)
    _plot_dashboard(records, dashboard_path)
    return {"csv": csv_path, "dashboard": dashboard_path}


# 【中文导读】写出记录表。
def _write_csv(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields = sorted({k for row in records for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({k: row.get(k, "") for k in fields})


# 【中文导读】绘制电压、压力、SOC、功率等运行曲线。
def _plot_dashboard(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = _arr(records, "hour")
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True)
    fig.suptitle("Standalone Electric-Gas Coupled Microgrid Random Policy", fontsize=15)
    axes[0, 0].plot(x, _arr(records, "vm_min_pu"), label="V min", color="#2563eb")
    axes[0, 0].plot(x, _arr(records, "vm_max_pu"), label="V max", color="#dc2626")
    axes[0, 0].axhspan(0.95, 1.05, color="#16a34a", alpha=0.12)
    axes[0, 0].set_title("Bus Voltage Envelope")
    axes[0, 0].set_ylabel("pu")
    axes[0, 0].legend()
    axes[0, 1].plot(x, _arr(records, "max_line_loading_percent"), color="#7c3aed")
    axes[0, 1].axhline(100, color="#dc2626", ls="--")
    axes[0, 1].set_title("Max Line Loading")
    axes[0, 1].set_ylabel("%")
    axes[1, 0].plot(x, _arr(records, "gas_pressure_min_bar"), label="Gas min", color="#0891b2")
    axes[1, 0].plot(x, _arr(records, "gas_pressure_max_bar"), label="Gas max", color="#ea580c")
    axes[1, 0].axhspan(2.5, 5.0, color="#16a34a", alpha=0.12)
    axes[1, 0].set_title("Medium-Pressure Gas Network")
    axes[1, 0].set_ylabel("bar")
    axes[1, 0].legend()
    axes[1, 1].plot(x, _arr(records, "max_pipe_velocity_m_per_s"), color="#0f766e")
    axes[1, 1].axhline(12.0, color="#dc2626", ls="--")
    axes[1, 1].set_title("Max Gas Pipe Velocity")
    axes[1, 1].set_ylabel("m/s")
    axes[2, 0].plot(x, _arr(records, "soc_min"), label="SOC min", color="#2563eb")
    axes[2, 0].plot(x, _arr(records, "soc_max"), label="SOC max", color="#dc2626")
    axes[2, 0].axhspan(0.10, 0.95, color="#16a34a", alpha=0.12)
    axes[2, 0].set_title("ESS SOC Envelope")
    axes[2, 0].set_xlabel("Hour")
    axes[2, 0].legend()
    axes[2, 1].step(x, _arr(records, "gas_state_age"), where="post", color="#4b5563")
    axes[2, 1].scatter(x[_arr(records, "gas_solved_this_step") > 0.5],
                       np.zeros(np.sum(_arr(records, "gas_solved_this_step") > 0.5)),
                       s=14, color="#0891b2", label="pipeflow")
    axes[2, 1].scatter(x[_arr(records, "slow_action_applied") > 0.5],
                       np.ones(np.sum(_arr(records, "slow_action_applied") > 0.5)),
                       s=18, color="#ea580c", label="slow action")
    axes[2, 1].set_title("Event-Driven Gas Solves")
    axes[2, 1].set_xlabel("Hour")
    axes[2, 1].legend()
    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=160)
    plt.close(fig)


# 【中文导读】绘制电网、气网和耦合设备拓扑。
def save_coupled_topology_overview(output_path: str | Path) -> Path:
    """Draw IEEE33 and the Belgian-20-derived medium-pressure micro gas distribution network."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    power_pos = _power_positions()
    gas_pos = _gas_positions()
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.suptitle(f"IEEE 33-Bus Power System and {GAS_MODEL_NAME}", fontsize=17)
    ax.set_title("Gas labels are 0-based Belgian-20 source labels; offset lines show retained parallel pipes", fontsize=11)
    ax.axis("off")
    _draw_edges(ax, power_pos, [(u, v) for u, v, _, _ in IEEE33_LINE_DATA], "#6b7280", 1.8, 0.75)
    _draw_edges(ax, gas_pos, [(p.from_junction, p.to_junction) for p in GAS_PIPES], "#0e7490", 2.0, 0.72)
    _draw_compressors(ax, gas_pos)
    _draw_nodes(ax, power_pos, "#dbeafe", "#1d4ed8", "P", 150)
    _draw_nodes(ax, gas_pos, "#cffafe", "#0e7490", "G", 170)
    _highlight_power_devices(ax, power_pos)
    _highlight_gas_devices(ax, gas_pos)
    _draw_couplings(ax, power_pos, gas_pos, FancyArrowPatch)
    legend = [
        Line2D([0], [0], color="#6b7280", lw=2, label="Power line"),
        Line2D([0], [0], color="#0e7490", lw=2, label="Gas pipe"),
        Line2D([0], [0], color="#7c3aed", lw=2, ls="-.", label="Aggregated gas compressor station"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#16a34a", markeredgecolor="#166534", label="ESS"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#facc15", markeredgecolor="#a16207", label="PV / wind"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#f97316", markeredgecolor="#9a3412", label="GFG"),
        Line2D([0], [0], marker="h", color="w", markerfacecolor="#22c55e", markeredgecolor="#166534", label="P2G"),
    ]
    ax.legend(handles=legend, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.02))
    ax.set_xlim(-1.0, 47.2)
    ax.set_ylim(-4.8, 4.6)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


# 【中文导读】把记录字段转换为 NumPy 数组。
def _arr(records: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, np.nan)) for row in records], dtype=float)


# 【中文导读】格式化数值范围。
def _fmt_range(values: List[float]) -> str:
    arr = np.asarray(values, dtype=float)
    return f"{np.nanmin(arr):.4f} / {np.nanmax(arr):.4f}"


# 【中文导读】打印仿真摘要。
def _print_summary(stats: EpisodeStats, artifacts: Dict[str, Path] | None) -> None:
    n = max(len(stats.power_success), 1)
    print("Standalone random-policy one-day simulation")
    print(f"Power-flow success rate: {100.0 * np.mean(stats.power_success):.2f}%")
    print(f"Gas-flow success rate: {100.0 * np.mean(stats.gas_success):.2f}%")
    print(f"Bus voltage min/max: {_fmt_range(stats.vm_min + stats.vm_max)} pu")
    print(f"Voltage violation count: {stats.voltage_violation_count}")
    print(f"Max line loading: {np.nanmax(np.asarray(stats.max_line_loading, dtype=float)):.2f}%")
    print(f"Medium-pressure gas min/max: {_fmt_range(stats.gas_pressure_min + stats.gas_pressure_max)} bar")
    print(f"SOC min/max: {_fmt_range(stats.soc_min + stats.soc_max)}")
    print(f"Total power-loss cost index: {stats.total_power_loss_cost:.4f}")
    print(f"Total renewable-curtailment cost index: {stats.total_curtailment_cost:.4f}")
    print(f"Total grid purchase: {stats.total_grid_purchase_mwh:.4f} MWh")
    print(f"Total gas purchase: {stats.total_gas_purchase_kg:.4f} kg")
    print(f"Gas pipeflow solve count: {stats.gas_solve_count_last}")
    print(f"Slow action count: {stats.slow_action_count} / {n}")
    if artifacts:
        print("Outputs:")
        for name, path in artifacts.items():
            print(f"  {name}: {path}")


# 【中文导读】生成电网拓扑绘图坐标。
def _power_positions() -> Dict[int, Tuple[float, float]]:
    pos: Dict[int, Tuple[float, float]] = {}
    for bus in range(18):
        pos[bus] = (float(bus) * 0.82, 0.0)
    for offset, bus in enumerate(range(18, 22), start=1):
        pos[bus] = (0.82 * (1 + offset), -1.55)
    for offset, bus in enumerate(range(22, 25), start=1):
        pos[bus] = (0.82 * (2 + offset), 1.45)
    for offset, bus in enumerate(range(25, 33), start=1):
        pos[bus] = (0.82 * (5 + offset), -3.05)
    return pos


# 【中文导读】生成气网拓扑绘图坐标。
def _gas_positions() -> Dict[int, Tuple[float, float]]:
    raw = {
        0: (22.8, 3.0), 1: (25.0, 3.0), 2: (27.2, 3.0), 3: (30.0, 2.35),
        4: (22.8, 0.45), 5: (25.0, 0.45), 6: (27.3, 0.75), 7: (45.4, 1.75),
        8: (43.4, 1.30), 9: (41.2, 1.25), 10: (39.0, 1.35), 11: (36.9, 2.05),
        12: (34.9, 2.35), 13: (32.5, 2.55), 14: (34.7, 3.65), 15: (37.0, 3.80),
        16: (39.0, -0.15), 17: (41.2, -0.95), 18: (43.3, -1.10), 19: (45.2, -1.10),
    }
    return {k: (float(x), float(y)) for k, (x, y) in raw.items()}


# 【中文导读】绘制拓扑边。
def _draw_edges(ax, pos: Mapping[int, Tuple[float, float]], edges: Iterable[Tuple[int, int]], color: str, lw: float, alpha: float) -> None:
    edges_list = list(edges)
    edge_counts = Counter(tuple(sorted(edge)) for edge in edges_list)
    edge_seen: Counter[Tuple[int, int]] = Counter()
    for u, v in edges_list:
        key = tuple(sorted((u, v)))
        index = edge_seen[key]
        edge_seen[key] += 1
        total = edge_counts[key]
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        offset = 0.0
        if total > 1 and length > 1e-9:
            offset = (index - 0.5 * (total - 1)) * 0.12
        nx, ny = ((-dy / length, dx / length) if length > 1e-9 else (0.0, 0.0))
        ax.plot([x1 + nx * offset, x2 + nx * offset],
                [y1 + ny * offset, y2 + ny * offset],
                color=color, lw=lw, alpha=alpha, zorder=1)


# 【中文导读】绘制拓扑节点。
def _draw_nodes(ax, pos: Mapping[int, Tuple[float, float]], face: str, edge: str, prefix: str, size: int) -> None:
    xs = [pos[i][0] for i in sorted(pos)]
    ys = [pos[i][1] for i in sorted(pos)]
    ax.scatter(xs, ys, s=size, c=face, edgecolors=edge, linewidths=1.2, zorder=3)
    for node, (x, y) in pos.items():
        label = str(node)
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5, color="#111827", zorder=4)


# 【中文导读】绘制压缩机。
def _draw_compressors(ax, gas_pos: Mapping[int, Tuple[float, float]]) -> None:
    for idx, comp in enumerate(COMPRESSOR_CONFIGS):
        x1, y1 = gas_pos[comp.from_junction]
        x2, y2 = gas_pos[comp.to_junction]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=2.2, linestyle="-.", mutation_scale=12),
                    zorder=2)
        label = "C0: 2-unit equivalent station" if idx == 0 else f"C{idx}"
        ax.text(0.5 * (x1 + x2), 0.5 * (y1 + y2) + 0.24, label,
                color="#6d28d9", fontsize=8, weight="bold")


# 【中文导读】标记电侧设备。
def _highlight_power_devices(ax, power_pos: Mapping[int, Tuple[float, float]]) -> None:
    _scatter_device(ax, [e.bus for e in ESS_CONFIGS], power_pos, "s", "#16a34a", "#166534", y_offset=0.33)
    _scatter_device(ax, [r.bus for r in RENEWABLE_CONFIGS], power_pos, "D", "#facc15", "#a16207", y_offset=-0.33)
    _scatter_device(ax, [g.power_bus for g in GFG_CONFIGS], power_pos, "^", "#f97316", "#9a3412", y_offset=0.54)
    _scatter_device(ax, [p.power_bus for p in P2G_CONFIGS], power_pos, "h", "#22c55e", "#166534", y_offset=-0.56)
    _scatter_device(ax, COMPRESSOR_POWER_BUSES, power_pos, "P", "#a78bfa", "#6d28d9", y_offset=0.35)


# 【中文导读】标记气侧设备。
def _highlight_gas_devices(ax, gas_pos: Mapping[int, Tuple[float, float]]) -> None:
    _scatter_device(ax, [s.supplier_node for s in GAS_SUPPLIERS], gas_pos, "*", "#38bdf8", "#0369a1", 210, 0.34)
    _scatter_device(ax, [n.node for n in GAS_NODES if n.base_mdot_kg_s > 0.0], gas_pos, "v", "#94a3b8", "#475569", 95, -0.30)
    _scatter_device(ax, [g.gas_junction for g in GFG_CONFIGS], gas_pos, "^", "#f97316", "#9a3412", y_offset=0.52)
    _scatter_device(ax, [p.gas_junction for p in P2G_CONFIGS], gas_pos, "h", "#22c55e", "#166534", y_offset=-0.50)


# 【中文导读】绘制单类设备标记。
def _scatter_device(ax, nodes: Iterable[int], pos: Mapping[int, Tuple[float, float]], marker: str,
                    face: str, edge: str, size: int = 125, y_offset: float = 0.0) -> None:
    xs, ys = [], []
    for node in nodes:
        if node not in pos:
            continue
        xs.append(pos[node][0])
        ys.append(pos[node][1] + y_offset)
    if xs:
        ax.scatter(xs, ys, s=size, marker=marker, c=face, edgecolors=edge, linewidths=1.0, zorder=5)


# 【中文导读】绘制 GFG/P2G/压缩机的跨网耦合。
def _draw_couplings(ax, power_pos: Mapping[int, Tuple[float, float]], gas_pos: Mapping[int, Tuple[float, float]], patch_cls) -> None:
    for idx, gfg in enumerate(GFG_CONFIGS):
        _curved_arrow(ax, gas_pos[gfg.gas_junction], power_pos[gfg.power_bus], "#f97316", patch_cls, -0.16, f"GFG {idx}")
    for idx, p2g in enumerate(P2G_CONFIGS):
        _curved_arrow(ax, power_pos[p2g.power_bus], gas_pos[p2g.gas_junction], "#22c55e", patch_cls, 0.18, f"P2G {idx}")
    for idx, (bus, comp) in enumerate(zip(COMPRESSOR_POWER_BUSES, COMPRESSOR_CONFIGS)):
        _curved_arrow(ax, power_pos[bus], _midpoint(gas_pos[comp.from_junction], gas_pos[comp.to_junction]),
                      "#7c3aed", patch_cls, 0.06, f"Comp {idx}", linestyle=":")


# 【中文导读】绘制跨子图曲线箭头。
def _curved_arrow(ax, start: Tuple[float, float], end: Tuple[float, float], color: str,
                  patch_cls, rad: float, label: str, linestyle: str = "--") -> None:
    arrow = patch_cls(start, end, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                      mutation_scale=13, lw=1.8, linestyle=linestyle, color=color, alpha=0.82, zorder=0)
    ax.add_patch(arrow)
    mx, my = _midpoint(start, end)
    ax.text(mx, my + 0.10, label, fontsize=7, color=color, alpha=0.95, ha="center")


# 【中文导读】计算边中点。
def _midpoint(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))


# 【中文导读】环境独立运行入口。
def main() -> None:
    """单文件脚本入口：可运行随机策略、拓扑图，或两者都运行。"""

    parser = argparse.ArgumentParser(description="Standalone electric-gas coupled microgrid simulator")
    parser.add_argument("--mode", choices=("random", "topology", "both"), default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("single_file_outputs"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    artifacts: Dict[str, Path] = {}
    if args.mode in ("random", "both"):
        stats = run_episode(seed=args.seed)
        random_artifacts = None
        if not args.no_plots: #如果未指定不绘制图表
            random_artifacts = save_episode_artifacts(stats.records, args.output_dir)
            artifacts.update(random_artifacts)
        _print_summary(stats, random_artifacts)
    if args.mode in ("topology", "both") and not args.no_plots:
        topo_path = save_coupled_topology_overview(args.output_dir / "single_file_coupled_topology.png")
        artifacts["topology"] = topo_path
        print(f"Topology: {topo_path}")


if __name__ == "__main__":
    main()
