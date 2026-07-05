"""Standalone IEEE33 + Belgian20 electric-gas coupled microgrid simulator.

This file is a single-file integration of the modular project under `project/`.
It contains:
- IEEE 33-bus distribution network data and pandapower builder
- Belgian 20-node high-pressure gas network data and pandapipes builder
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

import argparse
import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class TimeConfig:
    """时间尺度配置：3 分钟一步，一天 480 步，慢动作每 20 步更新。"""

    dt_minutes: int = 3
    steps_per_day: int = 480
    slow_action_interval_steps: int = 20

    @property
    def dt_hours(self) -> float:
        return self.dt_minutes / 60.0


@dataclass(frozen=True)
class PowerConfig:
    """IEEE 33 节点配电网的电压边界和基准容量。"""

    base_kv: float = 12.66
    base_mva: float = 10.0
    slack_bus: int = 0
    slack_vm_pu: float = 1.0
    voltage_target_pu: float = 1.0
    voltage_min_pu: float = 0.95
    voltage_max_pu: float = 1.05
    max_line_loading_percent: float = 100.0


@dataclass(frozen=True)
class GasConfig:
    """Belgian 20 节点高压气网的压力边界和气体换算参数。"""

    fluid_name: str = "lgas"
    gas_temperature_k: float = 293.15
    high_pressure_min_bar: float = 30.0
    high_pressure_max_bar: float = 70.0
    high_pressure_target_bar: float = 50.0
    source_pressure_bar: float = 60.0
    prs_outlet_pressure_bar: float = 1.5
    prs_outlet_min_bar: float = 1.35
    prs_outlet_max_bar: float = 1.65
    gas_compressibility_z: float = 0.85
    gas_specific_gas_constant_j_per_kg_k: float = 518.28
    # Calibration item: HHV should be confirmed from gas composition.
    hhv_mj_per_kg: float = 50.0


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
    soc_initial: float = 0.50


@dataclass(frozen=True)
class RenewableConfig:
    name: str
    bus: int
    kind: str
    capacity_mw: float
    s_rated_mva: float
    max_curtailment: float = 0.50


@dataclass(frozen=True)
class GFGConfig:
    name: str
    power_bus: int
    gas_junction: int
    max_p_mw: float
    efficiency: float


@dataclass(frozen=True)
class P2GConfig:
    name: str
    power_bus: int
    gas_junction: int
    max_p_mw: float
    efficiency: float


@dataclass(frozen=True)
class CompressorConfig:
    name: str
    from_junction: int
    to_junction: int
    min_pressure_ratio: float = 1.0
    max_pressure_ratio: float = 1.5
    initial_pressure_ratio: float = 1.2
    isentropic_efficiency: float = 0.75
    max_power_mw: float = 0.25
    nominal_flow_kg_s: float = 4.0
    inlet_min_bar: float = 30.0
    outlet_max_bar: float = 70.0
    needs_calibration: bool = True


@dataclass(frozen=True)
class EventConfig:
    gfg_mdot_threshold_kg_s: float = 0.02
    p2g_mdot_threshold_kg_s: float = 0.02
    compressor_ratio_threshold: float = 0.01
    gas_load_relative_threshold: float = 0.03


@dataclass(frozen=True)
class RewardConfig:
    """奖励权重。代码中先计算成本，再用 reward = -sum(cost)。"""

    voltage_deviation: float = 25.0
    voltage_violation: float = 500.0
    high_pressure_deviation: float = 10.0
    high_pressure_violation: float = 20.0
    prs_pressure_deviation: float = 25.0
    prs_pressure_violation: float = 200.0
    line_overload: float = 10.0
    power_loss: float = 20.0
    grid_energy_price: float = 0.0
    gas_price: float = 0.0
    renewable_curtailment: float = 40.0
    ess_action_change: float = 0.5
    gfg_action_change: float = 1.0
    p2g_action_change: float = 1.0
    compressor_energy: float = 60.0
    soc_soft: float = 20.0
    solver_failure: float = 5000.0
    terminal_soc: float = 200.0


@dataclass(frozen=True)
class SafetyConfig:
    max_consecutive_solver_failures: int = 5
    soc_soft_low: float = 0.20
    soc_soft_high: float = 0.90
    solver_iterations: int = 2


@dataclass(frozen=True)
class ProjectConfig:
    time: TimeConfig = field(default_factory=TimeConfig)
    power: PowerConfig = field(default_factory=PowerConfig)
    gas: GasConfig = field(default_factory=GasConfig)
    event: EventConfig = field(default_factory=EventConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    random_seed: int = 42


DEFAULT_CONFIG = ProjectConfig()


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
)


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
COMPRESSOR_POWER_BUSES = (7, 13, 30)
STANDARD_GAS_DENSITY_KG_PER_M3 = 0.80


@dataclass(frozen=True)
class GasNodeData:
    node: int
    demand_mm3_per_day: float
    p_min_bar: float
    p_max_bar: float


@dataclass(frozen=True)
class GasSupplierData:
    name: str
    supplier_node: int
    capacity_mm3_per_day: float
    marginal_cost_musd_per_mm3_day: float
    hourly_ramping_mm3_per_day: float
    needs_calibration: bool = True


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


def mm3_per_day_to_kg_per_s(value_mm3_per_day: float) -> float:
    return value_mm3_per_day * 1_000_000.0 * STANDARD_GAS_DENSITY_KG_PER_M3 / 86_400.0


GAS_NODES: Tuple[GasNodeData, ...] = (
    GasNodeData(0, 0.000, 30.0, 70.0), GasNodeData(1, 0.000, 30.0, 70.0),
    GasNodeData(2, 0.000, 30.0, 70.0), GasNodeData(3, 0.000, 30.0, 70.0),
    GasNodeData(4, 0.000, 30.0, 70.0), GasNodeData(5, 4.034, 30.0, 70.0),
    GasNodeData(6, 5.256, 30.0, 70.0), GasNodeData(7, 0.000, 30.0, 70.0),
    GasNodeData(8, 0.000, 30.0, 70.0), GasNodeData(9, 6.365, 30.0, 70.0),
    GasNodeData(10, 0.000, 30.0, 70.0), GasNodeData(11, 2.120, 30.0, 70.0),
    GasNodeData(12, 1.200, 30.0, 70.0), GasNodeData(13, 0.960, 30.0, 70.0),
    GasNodeData(14, 6.848, 30.0, 70.0), GasNodeData(15, 15.616, 30.0, 70.0),
    GasNodeData(16, 0.000, 30.0, 70.0), GasNodeData(17, 0.000, 30.0, 70.0),
    GasNodeData(18, 0.222, 30.0, 70.0), GasNodeData(19, 1.919, 30.0, 70.0),
)


GAS_SUPPLIERS: Tuple[GasSupplierData, ...] = (
    GasSupplierData("Sup_1", 0, 31.2, 0.03600, 3.12),
    GasSupplierData("Sup_2", 1, 24.0, 0.04320, 2.40),
    GasSupplierData("Sup_3", 2, 8.4, 0.03420, 4.20),
    GasSupplierData("Sup_4", 3, 4.8, 0.03240, 2.40),
    GasSupplierData("Sup_5", 4, 2.4, 0.04104, 1.20),
    GasSupplierData("Sup_6", 5, 2.4, 0.03888, 1.20),
)


def _temporary_pipe_parameters(wmn_reference: float) -> Tuple[float, float, float]:
    # Calibration placeholder. Wmn/Kmn are not used as length/diameter.
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


COMPRESSOR_CONFIGS: Tuple[CompressorConfig, ...] = (
    CompressorConfig("COMP_8_to_18", 7, 17, initial_pressure_ratio=1.15, max_power_mw=0.25),
    CompressorConfig("COMP_14_to_15", 13, 14, initial_pressure_ratio=1.10, max_power_mw=0.25),
    CompressorConfig("COMP_17_to_18", 16, 17, initial_pressure_ratio=1.15, max_power_mw=0.25),
)


def calibration_warning_messages() -> Tuple[str, ...]:
    return (
        "Belgian20 pipe length/diameter/roughness are temporary equivalent parameters.",
        "Wmn/Kmn are retained as references and are not used as pandapipes physical pipe data.",
        "Gas HHV, density, compressor flow and compressor limits still need calibration.",
    )


# =============================================================================
# Profiles
# =============================================================================


@dataclass(frozen=True)
class DailyProfiles:
    """一天内的外生曲线，长度比一天多一个慢动作间隔，方便读取下一小时预测。"""

    load_multiplier: np.ndarray
    gas_multiplier: np.ndarray
    renewable_available_mw: np.ndarray
    next_hour_load_multiplier: np.ndarray
    next_hour_renewable_available_mw: np.ndarray


def _smooth_noise(rng: np.random.Generator, n: int, scale: float, window: int) -> np.ndarray:
    """生成平滑噪声，避免负荷和风光每 3 分钟剧烈跳变。"""

    raw = rng.normal(0.0, scale, n + window)
    kernel = np.ones(window) / window
    return np.convolve(raw, kernel, mode="valid")[:n]


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


@dataclass
class TopologyValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    source_component_nodes: Set[int] = field(default_factory=set)

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ValueError("Gas topology validation failed: " + "; ".join(self.errors))


def validate_belgian20_topology() -> TopologyValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    edges: List[Tuple[int, int]] = []
    seen: Dict[Tuple[int, int], str] = {}
    for pipe in GAS_PIPES:
        if pipe.from_junction == pipe.to_junction:
            errors.append(f"{pipe.name} has identical endpoints")
        if pipe.length_km <= 0 or pipe.diameter_m <= 0 or pipe.roughness_mm <= 0:
            errors.append(f"{pipe.name} has non-positive physical parameter")
        key = tuple(sorted((pipe.from_junction, pipe.to_junction)))
        if key in seen and not pipe.allow_parallel:
            errors.append(f"{pipe.name} duplicates {seen[key]}")
        if key in seen and pipe.allow_parallel:
            warnings.append(f"{pipe.name} is an allowed parallel pipe")
        seen.setdefault(key, pipe.name)
        edges.append((pipe.from_junction, pipe.to_junction))
    for comp in COMPRESSOR_CONFIGS:
        edges.append((comp.from_junction, comp.to_junction))

    adjacency = {i: set() for i in range(N_GAS_JUNCTIONS)}
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    sources = {s.supplier_node for s in GAS_SUPPLIERS}
    visited: Set[int] = set()
    stack = list(sources)
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(sorted(adjacency[node] - visited))

    required = {n.node for n in GAS_NODES if n.demand_mm3_per_day > 0}
    required |= {p.gas_junction for p in P2G_CONFIGS}
    required |= {g.gas_junction for g in GFG_CONFIGS}
    required |= sources
    missing = sorted(required - visited)
    if missing:
        errors.append(f"Gas source component cannot reach nodes {missing}")
    return TopologyValidationResult(not errors, errors, warnings, visited)


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


@dataclass
class GasNetworkArtifacts:
    net: object
    base_sink_indices_by_node: Dict[int, int]
    p2g_source_indices: List[int]
    gfg_sink_indices: List[int]
    compressor_indices: List[int]
    ext_grid_indices: List[int]
    prs_control_indices: List[int]
    prs_junction_indices: List[int]
    high_pressure_junctions: List[int]


def build_power_network(config: ProjectConfig | None = None) -> PowerNetworkArtifacts:
    """构建 pandapower IEEE33 电网，并保存后续 step 需要修改的元件索引。"""

    cfg = config or DEFAULT_CONFIG
    import pandapower as pp

    net = pp.create_empty_network(sn_mva=cfg.power.base_mva)
    # 母线和线路构成基础配电网；负荷/新能源/储能/耦合设备随后挂接上去。
    for bus in range(N_POWER_BUSES):
        pp.create_bus(net, vn_kv=cfg.power.base_kv, name=f"Bus_{bus}",
                      min_vm_pu=cfg.power.voltage_min_pu, max_vm_pu=cfg.power.voltage_max_pu)
    ext_grid = pp.create_ext_grid(net, bus=cfg.power.slack_bus, vm_pu=cfg.power.slack_vm_pu,
                                  va_degree=0.0, name="Utility_Grid")
    for i, (u, v, r, x) in enumerate(IEEE33_LINE_DATA):
        pp.create_line_from_parameters(net, u, v, length_km=1.0, r_ohm_per_km=r,
                                       x_ohm_per_km=x, c_nf_per_km=0.0,
                                       max_i_ka=0.50, name=f"Line_{i}_{u}_{v}")

    load_indices = [int(pp.create_load(net, bus, p, q, name=f"BaseLoad_bus_{bus}"))
                    for bus, p, q in IEEE33_LOAD_DATA]
    renewable_indices = [
        int(pp.create_sgen(net, r.bus, p_mw=0.0, q_mvar=0.0, name=r.name,
                           max_p_mw=r.capacity_mw, min_p_mw=0.0,
                           max_q_mvar=r.s_rated_mva, min_q_mvar=-r.s_rated_mva))
        for r in RENEWABLE_CONFIGS
    ]
    ess_indices = [
        int(pp.create_storage(net, ess.bus, p_mw=0.0, q_mvar=0.0, max_e_mwh=ess.capacity_mwh,
                              soc_percent=100.0 * ess.soc_initial, name=ess.name))
        for ess in ESS_CONFIGS
    ]
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


def build_gas_network(config: ProjectConfig | None = None) -> GasNetworkArtifacts:
    """构建 pandapipes Belgian20 气网，并保存 source/sink/压缩机索引。"""

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

    # junction 是气网节点；显示名使用 1-based Gnode_1..20，内部仍是 0-based。
    for node in GAS_NODES:
        pp.create_junction(net, pn_bar=cfg.gas.source_pressure_bar, tfluid_k=cfg.gas.gas_temperature_k,
                           name=f"Gnode_{node.node + 1}")
    ext_grid_indices = [
        int(pp.create_ext_grid(net, junction=s.supplier_node, p_bar=cfg.gas.source_pressure_bar,
                               t_k=cfg.gas.gas_temperature_k, name=f"{s.name}_pressure_source"))
        for s in GAS_SUPPLIERS
    ]
    # 管道参数是暂定等效值，主要保证第一版准稳态模型可求解。
    for pipe in GAS_PIPES:
        pp.create_pipe_from_parameters(net, pipe.from_junction, pipe.to_junction,
                                       length_km=pipe.length_km, diameter_m=pipe.diameter_m,
                                       k_mm=pipe.roughness_mm, sections=1, name=pipe.name)
    compressor_indices = [
        int(pp.create_compressor(net, c.from_junction, c.to_junction,
                                 pressure_ratio=c.initial_pressure_ratio, name=c.name))
        for c in COMPRESSOR_CONFIGS
    ]
    base_sink_indices: Dict[int, int] = {}
    for node in GAS_NODES:
        if node.demand_mm3_per_day <= 0.0:
            continue
        base_sink_indices[node.node] = int(
            pp.create_sink(net, node.node, mdot_kg_per_s=mm3_per_day_to_kg_per_s(node.demand_mm3_per_day),
                           name=f"BaseGasDemand_Gnode_{node.node + 1}")
        )
    p2g_source_indices = [
        int(pp.create_source(net, p.gas_junction, mdot_kg_per_s=0.0, name=f"{p.name}_gas_source"))
        for p in P2G_CONFIGS
    ]
    # PRS outlet is represented as virtual state in the environment. GFG gas draw is placed
    # on the high-pressure node that supplies the PRS.
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
        ext_grid_indices=ext_grid_indices,
        prs_control_indices=[],
        prs_junction_indices=[-1 for _ in GFG_CONFIGS],
        high_pressure_junctions=list(range(N_GAS_JUNCTIONS)),
    )


# =============================================================================
# Coupling and safety models
# =============================================================================


def p2g_power_to_gas_mdot_kg_s(power_mw: float, efficiency: float, hhv_mj_per_kg: float) -> float:
    return max(0.0, efficiency * max(0.0, power_mw) / hhv_mj_per_kg)


def gfg_power_to_gas_mdot_kg_s(power_mw: float, efficiency: float, hhv_mj_per_kg: float) -> float:
    if efficiency <= 0.0:
        raise ValueError("GFG efficiency must be positive")
    return max(0.0, max(0.0, power_mw) / (efficiency * hhv_mj_per_kg))


@dataclass(frozen=True)
class GFGDispatch:
    electric_power_mw: float
    gas_mdot_kg_s: float


def dispatch_gfg(config: GFGConfig, requested_power_mw: float, hhv_mj_per_kg: float) -> GFGDispatch:
    p_mw = min(max(requested_power_mw, 0.0), config.max_p_mw)
    return GFGDispatch(p_mw, gfg_power_to_gas_mdot_kg_s(p_mw, config.efficiency, hhv_mj_per_kg))


@dataclass(frozen=True)
class P2GDispatch:
    electric_power_mw: float
    gas_mdot_kg_s: float


def dispatch_p2g(config: P2GConfig, requested_power_mw: float, hhv_mj_per_kg: float) -> P2GDispatch:
    p_mw = min(max(requested_power_mw, 0.0), config.max_p_mw)
    return P2GDispatch(p_mw, p2g_power_to_gas_mdot_kg_s(p_mw, config.efficiency, hhv_mj_per_kg))


@dataclass(frozen=True)
class CompressorDispatch:
    pressure_ratio: float
    mdot_kg_s: float
    electric_power_mw: float
    clipped_by_ratio: bool
    clipped_by_power: bool


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


def dispatch_compressor(config: CompressorConfig, requested_ratio: float, mdot_estimate_kg_s: float | None) -> CompressorDispatch:
    ratio = min(max(requested_ratio, config.min_pressure_ratio), config.max_pressure_ratio)
    clipped_by_ratio = abs(ratio - requested_ratio) > 1e-9
    mdot = max(0.0, config.nominal_flow_kg_s if mdot_estimate_kg_s is None else mdot_estimate_kg_s)
    power = estimate_compressor_power_mw(mdot, ratio, config.isentropic_efficiency)
    clipped_by_power = power > config.max_power_mw
    if clipped_by_power:
        power = config.max_power_mw
    return CompressorDispatch(float(ratio), float(mdot), float(power), clipped_by_ratio, clipped_by_power)


@dataclass(frozen=True)
class ActionProjectionResult:
    raw_action: float
    applied_action: float
    projection_magnitude: float
    hit_boundary: bool


@dataclass(frozen=True)
class ESSProjectionBatch:
    raw_p_mw: np.ndarray
    applied_p_mw: np.ndarray
    projection_magnitude_mw: np.ndarray
    hit_soc_boundary: np.ndarray


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


def update_ess_soc(soc: float, p_mw: float, ess: ESSConfig, dt_hours: float) -> float:
    """根据实际执行功率更新 SOC。"""

    if p_mw >= 0.0:
        delta_e_mwh = ess.eta_charge * p_mw * dt_hours
    else:
        delta_e_mwh = p_mw * dt_hours / ess.eta_discharge
    return float(soc + delta_e_mwh / ess.capacity_mwh)


@dataclass(frozen=True)
class InverterProjection:
    p_actual_mw: float
    q_actual_mvar: float
    curtailment: float
    q_limit_mvar: float
    apparent_power_mva: float
    q_was_clipped: bool
    curtailment_was_clipped: bool


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


@dataclass
class GasEventState:
    gfg_mdot_kg_s: np.ndarray
    p2g_mdot_kg_s: np.ndarray
    compressor_ratio: np.ndarray
    gas_load_multiplier: float


@dataclass(frozen=True)
class GasSolveDecision:
    should_solve: bool
    reason: str


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


@dataclass(frozen=True)
class PhysicalActions:
    ess_p_mw: np.ndarray
    gfg_p_mw: np.ndarray
    p2g_p_mw: np.ndarray
    compressor_ratio: np.ndarray
    renewable_p_mw: np.ndarray
    renewable_q_mvar: np.ndarray
    renewable_curtailment: np.ndarray


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


class CoupledSolver:
    """显式手工电-气耦合求解器。

    它把慢/快动作写入 pandapower 和 pandapipes 网络表，然后按事件策略运行
    气网 pipeflow，并每个快速步运行电网 powerflow。
    """

    def __init__(self, power: PowerNetworkArtifacts, gas: GasNetworkArtifacts, config: ProjectConfig):
        self.power = power
        self.gas = gas
        self.config = config
        self.scheduler = EventScheduler(config.time, config.event)
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
        """执行一次 t 到 t+1 的准稳态耦合求解。"""

        # 先把当前外生曲线和设备动作写入两张网络。
        self._write_power_profile_and_actions(profile, actions)
        gfg_mdot = self._write_gfg(actions.gfg_p_mw)
        p2g_mdot = self._write_p2g(actions.p2g_p_mw)
        self._write_gas_loads(float(profile["gas_multiplier"]))
        self._write_compressor_ratios(actions.compressor_ratio)

        decision = self.scheduler.decide(time_index, gfg_mdot, p2g_mdot,
                                         actions.compressor_ratio, float(profile["gas_multiplier"]))
        gas_solved = force_gas or decision.should_solve
        gas_reason = "forced" if force_gas and not decision.should_solve else decision.reason
        # 压缩机电功率依赖流量；气网未重算时沿用上一轮流量估计。
        comp_dispatches = self._write_compressor_power_loads(actions.compressor_ratio)
        gas_converged = self.last_gas_converged
        if gas_solved:
            # 多迭代几次，让气网流量和压缩机电负荷相互刷新。
            for _ in range(self.config.safety.solver_iterations):
                self._run_pipeflow()
                gas_converged = True
                self.last_gas_converged = True
                self.last_compressor_mdot_kg_s = self._read_compressor_mdot_estimates()
                comp_dispatches = self._write_compressor_power_loads(actions.compressor_ratio)
            self.scheduler.mark_solved(gfg_mdot, p2g_mdot, actions.compressor_ratio,
                                       float(profile["gas_multiplier"]))
            self.gas_state_age = 0
            self.gas_solve_count += 1
        else:
            # 不求气网时保留上一轮结果，并记录已复用多少快速步。
            self.gas_state_age += 1

        # 逆变器快动作每步都可能变，所以电网每 3 分钟都重算。
        self._run_powerflow()
        return CoupledSolveResult(True, gas_converged, gas_solved, gas_reason,
                                  self.gas_state_age, gfg_mdot, p2g_mdot,
                                  comp_dispatches, self.compute_equivalent_linepack_indicator())

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
            self.gas.net.sink.at[idx, "mdot_kg_per_s"] = mm3_per_day_to_kg_per_s(node.demand_mm3_per_day) * gas_multiplier

    def _write_compressor_ratios(self, requested_ratio: Sequence[float]) -> None:
        for i, idx in enumerate(self.gas.compressor_indices):
            ratio = float(np.clip(requested_ratio[i], COMPRESSOR_CONFIGS[i].min_pressure_ratio,
                                  COMPRESSOR_CONFIGS[i].max_pressure_ratio))
            self.gas.net.compressor.at[idx, "pressure_ratio"] = ratio

    def _write_compressor_power_loads(self, requested_ratio: Sequence[float]) -> List[CompressorDispatch]:
        dispatches = []
        for i, cfg in enumerate(COMPRESSOR_CONFIGS):
            disp = dispatch_compressor(cfg, float(requested_ratio[i]), float(self.last_compressor_mdot_kg_s[i]))
            self.power.net.load.at[self.power.compressor_load_indices[i], "p_mw"] = disp.electric_power_mw
            self.power.net.load.at[self.power.compressor_load_indices[i], "q_mvar"] = 0.0
            dispatches.append(disp)
        return dispatches

    def _run_pipeflow(self) -> None:
        import pandapipes as pp
        pp.pipeflow(self.gas.net, max_iter_hyd=50, tol_p=1e-5)

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
        net = self.gas.net
        if not hasattr(net, "res_compressor") or len(net.res_compressor) == 0:
            return self.last_compressor_mdot_kg_s.copy()
        table = net.res_compressor
        cols = ("mdot_from_kg_per_s", "mdot_to_kg_per_s", "mdot_kg_per_s", "mf_from_kg_per_s", "mf_to_kg_per_s")
        mdots = []
        for idx, cfg in zip(self.gas.compressor_indices, COMPRESSOR_CONFIGS):
            value = cfg.nominal_flow_kg_s
            for col in cols:
                if col in table.columns and idx in table.index:
                    raw = table.at[idx, col]
                    if np.isfinite(raw):
                        value = abs(float(raw))
                        break
            mdots.append(value)
        return np.asarray(mdots)

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


class SimpleBox:
    def __init__(self, low: float, high: float, shape: Tuple[int, ...], dtype=np.float32):
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high).astype(self.dtype)


@dataclass
class Snapshot:
    current_step: int
    ess_soc: np.ndarray
    last_slow_action: np.ndarray
    last_physical_slow: Dict[str, np.ndarray]
    previous_device_actions: Dict[str, np.ndarray]
    solver_gas_state_age: int
    solver_gas_solve_count: int
    power_tables: Dict[str, Any]
    gas_tables: Dict[str, Any]


class ElectricGasMultiScaleEnv:
    """Gymnasium 风格的电-气耦合环境。

    环境负责把智能体的 [-1, 1] 动作翻译成工程量，调用求解器推进 3 分钟，
    然后返回下一步观测、奖励和 info。训练脚本中的 TD3 并不直接接触 pandapower
    或 pandapipes，它只通过这个类学习控制。
    """

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
        self.n_renew = len(RENEWABLE_CONFIGS)
        self.slow_action_dim = self.n_ess + self.n_gfg + self.n_p2g + self.n_comp
        self.fast_action_dim = 2 * self.n_renew
        self.action_dim = self.slow_action_dim + self.fast_action_dim
        self.action_space = self._make_action_space()
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
            "compressor_ratio": np.array([c.initial_pressure_ratio for c in COMPRESSOR_CONFIGS], dtype=float),
        }
        self.previous_device_actions = {k: v.copy() for k, v in self.last_physical_slow.items()}

    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """开始新 episode。

        reset 会重新生成一天的外生曲线，并强制求解一次初始电/气状态，
        因此返回的初始观测已经包含有效潮流结果。
        """

        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.profiles = generate_daily_profiles(self.config.time, seed=seed or self.config.random_seed)
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
        slow_action_applied = time_index % self.config.time.slow_action_interval_steps == 0
        try:
            # 归一化动作 -> 安全投影后的物理动作 -> 电/气网络求解。
            physical = self._map_and_project_action(action, profile, slow_action_applied)
            attempted_applied_action = self.last_applied_action.copy()
            result = self.solver.solve_step(time_index, profile, physical)
            self.last_solve_result = result
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
            ratios = []
            for i, comp in enumerate(COMPRESSOR_CONFIGS):
                ratios.append(comp.min_pressure_ratio + 0.5 * (slow[cursor + i] + 1.0) *
                              (comp.max_pressure_ratio - comp.min_pressure_ratio))
            self.last_physical_slow["compressor_ratio"] = np.array(ratios)
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
        self.last_applied_action = self._normalized_action_from_physical(physical)
        return physical

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
        for i, cfg in enumerate(COMPRESSOR_CONFIGS):
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
        high_p = self._series_values(net_g, "res_junction", "p_bar", N_GAS_JUNCTIONS, self.config.gas.source_pressure_bar)
        prs_p = np.full(len(GFG_CONFIGS), self.config.gas.prs_outlet_pressure_bar)
        ext_mdot_raw = self._series_values(net_g, "res_ext_grid", "mdot_kg_per_s", len(GAS_SUPPLIERS), 0.0)
        source_mdot = np.maximum(-ext_mdot_raw, 0.0)
        pipe_mdot = self._series_values(net_g, "res_pipe", "mdot_kg_per_s", len(GAS_PIPES), 0.0)
        comp_ratio = self.last_physical_slow["compressor_ratio"]
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
        parts = [
            (vm - 1.0) / 0.10, loading / 100.0,
            np.array([float(profile["load_multiplier"]), float(np.sum(renewable_avail)), float(np.sum(renewable_actual))]),
            renewable_avail / np.maximum(np.array([r.capacity_mw for r in RENEWABLE_CONFIGS]), 1e-9),
            renewable_actual / np.maximum(np.array([r.capacity_mw for r in RENEWABLE_CONFIGS]), 1e-9),
            renewable_q / np.maximum(np.array([r.s_rated_mva for r in RENEWABLE_CONFIGS]), 1e-9),
            ext_p / 10.0, p_loss / 1.0, self.ess_soc, ess_p_norm, ess_margin,
            (high_p - 50.0) / 20.0, (prs_p - self.config.gas.prs_outlet_pressure_bar) / 0.15,
            comp_ratio / np.array([c.max_pressure_ratio for c in COMPRESSOR_CONFIGS]),
            source_mdot / 100.0, gfg_mdot / 2.0, p2g_mdot / 2.0,
            np.array([gas_age / self.config.time.slow_action_interval_steps, linepack / 10_000_000.0]),
            pipe_mdot / 100.0, time_feat,
        ]
        return np.nan_to_num(np.concatenate([np.atleast_1d(p) for p in parts]).astype(np.float32),
                             nan=0.0, posinf=10.0, neginf=-10.0)

    def get_manager_state(self) -> np.ndarray:
        return self.get_global_state()

    def get_fast_worker_state(self) -> np.ndarray:
        cut = 33 + len(IEEE33_LINE_DATA) + 5 + 3 * self.n_renew
        return self.get_global_state()[:cut].copy()

    def get_slow_worker_state(self) -> np.ndarray:
        start = 33 + len(IEEE33_LINE_DATA) + 5 + 3 * self.n_renew
        return self.get_global_state()[start:].copy()

    def _compute_reward(self, solver_failed: bool) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """计算一步奖励和诊断指标。

        所有 components 都是成本；最终 reward 取负值，所以越接近 0 越好。
        metrics 保留未加权物理指标，用于训练日志和可视化。
        """

        dt_h = self.config.time.dt_hours
        vm = self._series_values(self.power.net, "res_bus", "vm_pu", 33, 1.0)
        loading = self._series_values(self.power.net, "res_line", "loading_percent", len(IEEE33_LINE_DATA), 0.0)
        high_p = self._series_values(self.gas.net, "res_junction", "p_bar", N_GAS_JUNCTIONS, self.config.gas.source_pressure_bar)
        prs_p = np.full(len(GFG_CONFIGS), self.config.gas.prs_outlet_pressure_bar)
        voltage_deviation = float(np.sum(((vm - self.config.power.voltage_target_pu) / 0.05) ** 2))
        voltage_violation = float(np.sum(np.maximum(self.config.power.voltage_min_pu - vm, 0.0) ** 2 +
                                         np.maximum(vm - self.config.power.voltage_max_pu, 0.0) ** 2))
        high_pressure_deviation = float(np.sum(((high_p - self.config.gas.high_pressure_target_bar) / 20.0) ** 2))
        high_pressure_violation = float(np.sum(np.maximum(self.config.gas.high_pressure_min_bar - high_p, 0.0) ** 2 +
                                               np.maximum(high_p - self.config.gas.high_pressure_max_bar, 0.0) ** 2))
        prs_pressure_deviation = float(np.sum(((prs_p - self.config.gas.prs_outlet_pressure_bar) / 0.15) ** 2))
        prs_pressure_violation = float(np.sum(np.maximum(self.config.gas.prs_outlet_min_bar - prs_p, 0.0) ** 2 +
                                              np.maximum(prs_p - self.config.gas.prs_outlet_max_bar, 0.0) ** 2))
        line_overload = float(np.sum(np.maximum(loading - self.config.power.max_line_loading_percent, 0.0) ** 2))
        p_loss_mw = float(np.nansum(self._series_values(self.power.net, "res_line", "pl_mw", len(IEEE33_LINE_DATA), 0.0)))
        grid_purchase_mwh = max(float(np.nansum(self._series_values(self.power.net, "res_ext_grid", "p_mw", 1, 0.0))), 0.0) * dt_h
        gas_ext_mdot = self._series_values(self.gas.net, "res_ext_grid", "mdot_kg_per_s", len(GAS_SUPPLIERS), 0.0)
        gas_purchase_kg = float(np.nansum(np.maximum(-gas_ext_mdot, 0.0))) * dt_h * 3600.0
        curtail_mwh = 0.0
        if self.last_inverter_projection and self.profiles is not None:
            prof = profile_at(self.profiles, min(self.current_step, self.config.time.steps_per_day))
            avail = np.asarray(prof["renewable_available_mw"], dtype=float)
            curtail_mwh = float(np.sum([avail[i] * p.curtailment * dt_h for i, p in enumerate(self.last_inverter_projection)]))
        comp_mwh = 0.0
        if self.last_solve_result:
            comp_mwh = float(sum(d.electric_power_mw for d in self.last_solve_result.compressor_dispatches) * dt_h)
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
            "high_pressure_deviation": w.high_pressure_deviation * high_pressure_deviation,
            "high_pressure_violation": w.high_pressure_violation * high_pressure_violation,
            "prs_pressure_deviation": w.prs_pressure_deviation * prs_pressure_deviation,
            "prs_pressure_violation": w.prs_pressure_violation * prs_pressure_violation,
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
            "high_pressure_min_bar": float(np.nanmin(high_p)), "high_pressure_max_bar": float(np.nanmax(high_p)),
            "high_pressure_mean_abs_deviation_bar": float(np.nanmean(np.abs(high_p - self.config.gas.high_pressure_target_bar))),
            "high_pressure_rms_deviation_bar": float(np.sqrt(np.nanmean((high_p - self.config.gas.high_pressure_target_bar) ** 2))),
            "prs_pressure_min_bar": float(np.nanmin(prs_p)), "prs_pressure_max_bar": float(np.nanmax(prs_p)),
            "prs_pressure_mean_abs_deviation_bar": float(np.nanmean(np.abs(prs_p - self.config.gas.prs_outlet_pressure_bar))),
            "prs_pressure_rms_deviation_bar": float(np.sqrt(np.nanmean((prs_p - self.config.gas.prs_outlet_pressure_bar) ** 2))),
            "soc_min": float(np.nanmin(self.ess_soc)), "soc_max": float(np.nanmax(self.ess_soc)),
            "grid_purchase_mwh": float(grid_purchase_mwh), "gas_purchase_kg": float(gas_purchase_kg),
        }
        self.previous_device_actions = {k: v.copy() for k, v in self.last_physical_slow.items()}
        return -float(sum(components.values())), components, metrics

    def _build_info(self, components: Dict[str, float], metrics: Dict[str, float],
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
            {k: v.copy() for k, v in self.last_physical_slow.items()},
            {k: v.copy() for k, v in self.previous_device_actions.items()},
            self.solver.gas_state_age,
            self.solver.gas_solve_count,
            self._copy_tables(self.power.net, ("load", "sgen", "storage", "res_bus", "res_line", "res_load", "res_sgen", "res_storage", "res_ext_grid")),
            self._copy_tables(self.gas.net, ("sink", "source", "compressor", "res_junction", "res_pipe", "res_sink", "res_source", "res_ext_grid", "res_compressor")),
        )

    def _restore_snapshot(self, snapshot: Snapshot) -> None:
        self.ess_soc = snapshot.ess_soc.copy()
        self.last_slow_action = snapshot.last_slow_action.copy()
        self.last_physical_slow = {k: v.copy() for k, v in snapshot.last_physical_slow.items()}
        self.previous_device_actions = {k: v.copy() for k, v in snapshot.previous_device_actions.items()}
        self.solver.gas_state_age = snapshot.solver_gas_state_age
        self.solver.gas_solve_count = snapshot.solver_gas_solve_count
        self._restore_tables(self.power.net, snapshot.power_tables)
        self._restore_tables(self.gas.net, snapshot.gas_tables)


# =============================================================================
# Random policy and visualization
# =============================================================================


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
    high_pressure_min: List[float] = field(default_factory=list)
    high_pressure_max: List[float] = field(default_factory=list)
    prs_pressure_min: List[float] = field(default_factory=list)
    prs_pressure_max: List[float] = field(default_factory=list)
    soc_min: List[float] = field(default_factory=list)
    soc_max: List[float] = field(default_factory=list)
    total_power_loss_cost: float = 0.0
    total_curtailment_cost: float = 0.0
    total_grid_purchase_mwh: float = 0.0
    total_gas_purchase_kg: float = 0.0
    gas_solve_count_last: int = 0
    slow_action_count: int = 0
    records: List[Dict[str, float]] = field(default_factory=list)


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
        stats.high_pressure_min.append(float(metrics.get("high_pressure_min_bar", np.nan)))
        stats.high_pressure_max.append(float(metrics.get("high_pressure_max_bar", np.nan)))
        stats.prs_pressure_min.append(float(metrics.get("prs_pressure_min_bar", np.nan)))
        stats.prs_pressure_max.append(float(metrics.get("prs_pressure_max_bar", np.nan)))
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
        "high_pressure_min_bar": float(metrics.get("high_pressure_min_bar", np.nan)),
        "high_pressure_max_bar": float(metrics.get("high_pressure_max_bar", np.nan)),
        "high_pressure_mean_abs_deviation_bar": float(metrics.get("high_pressure_mean_abs_deviation_bar", np.nan)),
        "high_pressure_rms_deviation_bar": float(metrics.get("high_pressure_rms_deviation_bar", np.nan)),
        "prs_pressure_min_bar": float(metrics.get("prs_pressure_min_bar", np.nan)),
        "prs_pressure_max_bar": float(metrics.get("prs_pressure_max_bar", np.nan)),
        "prs_pressure_mean_abs_deviation_bar": float(metrics.get("prs_pressure_mean_abs_deviation_bar", np.nan)),
        "prs_pressure_rms_deviation_bar": float(metrics.get("prs_pressure_rms_deviation_bar", np.nan)),
        "soc_min": float(metrics.get("soc_min", np.nan)),
        "soc_max": float(metrics.get("soc_max", np.nan)),
        "grid_purchase_mwh": float(metrics.get("grid_purchase_mwh", 0.0)),
        "gas_purchase_kg": float(metrics.get("gas_purchase_kg", 0.0)),
        "voltage_deviation_cost": float(comps.get("voltage_deviation", 0.0)),
        "voltage_violation_cost": float(comps.get("voltage_violation", 0.0)),
        "high_pressure_deviation_cost": float(comps.get("high_pressure_deviation", 0.0)),
        "high_pressure_violation_cost": float(comps.get("high_pressure_violation", 0.0)),
        "prs_pressure_deviation_cost": float(comps.get("prs_pressure_deviation", 0.0)),
        "prs_pressure_violation_cost": float(comps.get("prs_pressure_violation", 0.0)),
        "line_overload_cost": float(comps.get("line_overload", 0.0)),
        "power_loss_cost": float(comps.get("power_loss", 0.0)),
        "renewable_curtailment_cost": float(comps.get("renewable_curtailment", 0.0)),
        "compressor_energy_cost": float(comps.get("compressor_energy", 0.0)),
    }


def save_episode_artifacts(records: Sequence[Mapping[str, Any]], output_dir: str | Path, prefix: str = "single_file_random_policy") -> Dict[str, Path]:
    """保存随机策略的 CSV 与仪表盘图。"""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{prefix}_timeseries.csv"
    dashboard_path = out / f"{prefix}_dashboard.png"
    _write_csv(records, csv_path)
    _plot_dashboard(records, dashboard_path)
    return {"csv": csv_path, "dashboard": dashboard_path}


def _write_csv(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields = sorted({k for row in records for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({k: row.get(k, "") for k in fields})


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
    axes[1, 0].plot(x, _arr(records, "high_pressure_min_bar"), label="HP min", color="#0891b2")
    axes[1, 0].plot(x, _arr(records, "high_pressure_max_bar"), label="HP max", color="#ea580c")
    axes[1, 0].axhspan(30, 70, color="#16a34a", alpha=0.12)
    axes[1, 0].set_title("High-Pressure Gas Network")
    axes[1, 0].set_ylabel("bar")
    axes[1, 0].legend()
    axes[1, 1].plot(x, _arr(records, "prs_pressure_min_bar"), color="#0f766e")
    axes[1, 1].plot(x, _arr(records, "prs_pressure_max_bar"), color="#be123c")
    axes[1, 1].axhspan(1.35, 1.65, color="#16a34a", alpha=0.12)
    axes[1, 1].set_title("PRS Outlet Pressure")
    axes[1, 1].set_ylabel("bar")
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


def save_coupled_topology_overview(output_path: str | Path) -> Path:
    """画出 IEEE33 电网、Belgian20 气网及 GFG/P2G/压缩机耦合关系。"""

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
    fig.suptitle("Standalone IEEE 33-Bus Power and Belgian 20-Node Gas Coupling", fontsize=17)
    ax.set_title("Dashed links show GFG, P2G and electric compressor demand", fontsize=11)
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
        Line2D([0], [0], color="#7c3aed", lw=2, ls="-.", label="Gas compressor"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#16a34a", markeredgecolor="#166534", label="ESS"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#facc15", markeredgecolor="#a16207", label="PV / wind"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#f97316", markeredgecolor="#9a3412", label="GFG"),
        Line2D([0], [0], marker="h", color="w", markerfacecolor="#22c55e", markeredgecolor="#166534", label="P2G"),
    ]
    ax.legend(handles=legend, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.02))
    ax.set_xlim(-1.0, 44.5)
    ax.set_ylim(-5.1, 4.1)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _arr(records: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, np.nan)) for row in records], dtype=float)


def _fmt_range(values: List[float]) -> str:
    arr = np.asarray(values, dtype=float)
    return f"{np.nanmin(arr):.4f} / {np.nanmax(arr):.4f}"


def _print_summary(stats: EpisodeStats, artifacts: Dict[str, Path] | None) -> None:
    n = max(len(stats.power_success), 1)
    print("Standalone random-policy one-day simulation")
    print(f"Power-flow success rate: {100.0 * np.mean(stats.power_success):.2f}%")
    print(f"Gas-flow success rate: {100.0 * np.mean(stats.gas_success):.2f}%")
    print(f"Bus voltage min/max: {_fmt_range(stats.vm_min + stats.vm_max)} pu")
    print(f"Voltage violation count: {stats.voltage_violation_count}")
    print(f"Max line loading: {np.nanmax(np.asarray(stats.max_line_loading, dtype=float)):.2f}%")
    print(f"High-pressure gas min/max: {_fmt_range(stats.high_pressure_min + stats.high_pressure_max)} bar")
    print(f"PRS outlet pressure range: {_fmt_range(stats.prs_pressure_min + stats.prs_pressure_max)} bar")
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


def _gas_positions() -> Dict[int, Tuple[float, float]]:
    raw = {
        0: (23.0, 2.2), 1: (25.2, 3.0), 2: (27.4, 2.8), 3: (29.6, 2.1),
        4: (23.5, -0.8), 5: (25.7, -0.9), 6: (27.9, -0.7), 7: (30.1, 0.4),
        8: (31.5, -1.6), 9: (33.5, -1.7), 10: (35.4, -1.7), 11: (36.8, -0.5),
        12: (38.3, 0.7), 13: (35.5, 1.4), 14: (37.6, 2.2), 15: (39.8, 2.1),
        16: (36.4, -3.25), 17: (39.2, -2.7), 18: (41.0, -2.6), 19: (42.7, -2.2),
    }
    return {k: (float(x), float(y)) for k, (x, y) in raw.items()}


def _draw_edges(ax, pos: Mapping[int, Tuple[float, float]], edges: Iterable[Tuple[int, int]], color: str, lw: float, alpha: float) -> None:
    for u, v in edges:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=color, lw=lw, alpha=alpha, zorder=1)


def _draw_nodes(ax, pos: Mapping[int, Tuple[float, float]], face: str, edge: str, prefix: str, size: int) -> None:
    xs = [pos[i][0] for i in sorted(pos)]
    ys = [pos[i][1] for i in sorted(pos)]
    ax.scatter(xs, ys, s=size, c=face, edgecolors=edge, linewidths=1.2, zorder=3)
    for node, (x, y) in pos.items():
        label = str(node) if prefix == "P" else str(node + 1)
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5, color="#111827", zorder=4)


def _draw_compressors(ax, gas_pos: Mapping[int, Tuple[float, float]]) -> None:
    for idx, comp in enumerate(COMPRESSOR_CONFIGS):
        x1, y1 = gas_pos[comp.from_junction]
        x2, y2 = gas_pos[comp.to_junction]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=2.2, linestyle="-.", mutation_scale=12),
                    zorder=2)
        ax.text(0.5 * (x1 + x2), 0.5 * (y1 + y2) + 0.24, f"C{idx}", color="#6d28d9", fontsize=8, weight="bold")


def _highlight_power_devices(ax, power_pos: Mapping[int, Tuple[float, float]]) -> None:
    _scatter_device(ax, [e.bus for e in ESS_CONFIGS], power_pos, "s", "#16a34a", "#166534", y_offset=0.33)
    _scatter_device(ax, [r.bus for r in RENEWABLE_CONFIGS], power_pos, "D", "#facc15", "#a16207", y_offset=-0.33)
    _scatter_device(ax, [g.power_bus for g in GFG_CONFIGS], power_pos, "^", "#f97316", "#9a3412", y_offset=0.54)
    _scatter_device(ax, [p.power_bus for p in P2G_CONFIGS], power_pos, "h", "#22c55e", "#166534", y_offset=-0.56)
    _scatter_device(ax, COMPRESSOR_POWER_BUSES, power_pos, "P", "#a78bfa", "#6d28d9", y_offset=0.35)


def _highlight_gas_devices(ax, gas_pos: Mapping[int, Tuple[float, float]]) -> None:
    _scatter_device(ax, [s.supplier_node for s in GAS_SUPPLIERS], gas_pos, "*", "#38bdf8", "#0369a1", 210, 0.34)
    _scatter_device(ax, [n.node for n in GAS_NODES if n.demand_mm3_per_day > 0.0], gas_pos, "v", "#94a3b8", "#475569", 95, -0.30)
    _scatter_device(ax, [g.gas_junction for g in GFG_CONFIGS], gas_pos, "^", "#f97316", "#9a3412", y_offset=0.52)
    _scatter_device(ax, [p.gas_junction for p in P2G_CONFIGS], gas_pos, "h", "#22c55e", "#166534", y_offset=-0.50)


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


def _draw_couplings(ax, power_pos: Mapping[int, Tuple[float, float]], gas_pos: Mapping[int, Tuple[float, float]], patch_cls) -> None:
    for idx, gfg in enumerate(GFG_CONFIGS):
        _curved_arrow(ax, gas_pos[gfg.gas_junction], power_pos[gfg.power_bus], "#f97316", patch_cls, -0.16, f"GFG {idx}")
    for idx, p2g in enumerate(P2G_CONFIGS):
        _curved_arrow(ax, power_pos[p2g.power_bus], gas_pos[p2g.gas_junction], "#22c55e", patch_cls, 0.18, f"P2G {idx}")
    for idx, (bus, comp) in enumerate(zip(COMPRESSOR_POWER_BUSES, COMPRESSOR_CONFIGS)):
        _curved_arrow(ax, power_pos[bus], _midpoint(gas_pos[comp.from_junction], gas_pos[comp.to_junction]),
                      "#7c3aed", patch_cls, 0.06, f"Comp {idx}", linestyle=":")


def _curved_arrow(ax, start: Tuple[float, float], end: Tuple[float, float], color: str,
                  patch_cls, rad: float, label: str, linestyle: str = "--") -> None:
    arrow = patch_cls(start, end, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                      mutation_scale=13, lw=1.8, linestyle=linestyle, color=color, alpha=0.82, zorder=0)
    ax.add_patch(arrow)
    mx, my = _midpoint(start, end)
    ax.text(mx, my + 0.10, label, fontsize=7, color=color, alpha=0.95, ha="center")


def _midpoint(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))


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
        if not args.no_plots:
            random_artifacts = save_episode_artifacts(stats.records, args.output_dir)
            artifacts.update(random_artifacts)
        _print_summary(stats, random_artifacts)
    if args.mode in ("topology", "both") and not args.no_plots:
        topo_path = save_coupled_topology_overview(args.output_dir / "single_file_coupled_topology.png")
        artifacts["topology"] = topo_path
        print(f"Topology: {topo_path}")


if __name__ == "__main__":
    main()
