"""全局配置与设备参数 dataclass。

本项目采用 3 分钟快速步长。ESS、GFG、P2G 与压缩机基准设定每 20 个
快速步更新一次，即每小时更新一次；逆变器每个快速步均可动作。

阅读提示：这里的 dataclass 只保存参数，不包含仿真逻辑。环境、求解器和
耦合模型会引用这些参数，因此想改步长、边界、奖励权重时优先从本文件查起。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class TimeConfig:
    """时间尺度配置。

    dt_minutes 决定环境 step 的物理时间长度；steps_per_day=480 对应 24 小时。
    slow_action_interval_steps=20 表示慢速设备每 20*3=60 分钟更新一次。
    """

    dt_minutes: int = 3
    steps_per_day: int = 480
    slow_action_interval_steps: int = 20

    @property
    def dt_hours(self) -> float:
        return self.dt_minutes / 60.0


@dataclass(frozen=True)
class PowerConfig:
    """IEEE 33 节点配电网电压与基准配置。"""

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
    """Belgian 20 节点高压气网与 PRS 配置。"""

    fluid_name: str = "lgas"
    gas_temperature_k: float = 293.15
    high_pressure_min_bar: float = 30.0
    high_pressure_max_bar: float = 70.0
    high_pressure_target_bar: float = 50.0
    source_pressure_bar: float = 60.0
    prs_outlet_pressure_bar: float = 1.5
    prs_outlet_min_bar: float = 1.35
    prs_outlet_max_bar: float = 1.65
    prs_inlet_junction: int = 6
    gas_compressibility_z: float = 0.85
    gas_specific_gas_constant_j_per_kg_k: float = 518.28
    # 待校准参数：应由 Belgian 20 原始文献或气质数据确认。这里集中放置，
    # 禁止在换算公式里散落使用魔法数。
    hhv_mj_per_kg: float = 50.0


@dataclass(frozen=True)
class ESSConfig:
    """储能参数，p_mw>0 表示充电，p_mw<0 表示放电。"""

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
    """光伏/风电逆变器配置。"""

    name: str
    bus: int
    kind: str
    capacity_mw: float
    s_rated_mva: float
    max_curtailment: float = 0.50


@dataclass(frozen=True)
class GFGConfig:
    """燃气轮机配置。"""

    name: str
    power_bus: int
    gas_junction: int
    max_p_mw: float
    efficiency: float


@dataclass(frozen=True)
class P2GConfig:
    """电转气装置配置。"""

    name: str
    power_bus: int
    gas_junction: int
    max_p_mw: float
    efficiency: float


@dataclass(frozen=True)
class CompressorConfig:
    """电驱压缩机配置。

    第一版采用简化等熵压缩功率模型。nominal_flow_kg_s、入口/出口压力约束等
    均应由原始 Belgian 20 文献或工程数据进一步校准。
    """

    name: str
    from_junction: int
    to_junction: int
    min_pressure_ratio: float = 1.0
    max_pressure_ratio: float = 1.5
    initial_pressure_ratio: float = 1.2
    isentropic_efficiency: float = 0.75
    max_power_mw: float = 5.0
    nominal_flow_kg_s: float = 4.0
    inlet_min_bar: float = 30.0
    outlet_max_bar: float = 70.0
    needs_calibration: bool = True


@dataclass(frozen=True)
class EventConfig:
    """事件驱动气网求解阈值。"""

    gfg_mdot_threshold_kg_s: float = 0.02
    p2g_mdot_threshold_kg_s: float = 0.02
    compressor_ratio_threshold: float = 0.01
    gas_load_relative_threshold: float = 0.03


@dataclass(frozen=True)
class RewardConfig:
    """外在奖励权重。所有分量以成本/惩罚形式累加后取负。

    初学者可以把这些权重理解成“控制目标的重要性排序”：权重越大，
    智能体越会避免该项成本，例如电压越限、压力越限或求解失败。
    """

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
    """安全与异常处理配置。"""

    max_consecutive_solver_failures: int = 5
    soc_soft_low: float = 0.20
    soc_soft_high: float = 0.90
    solver_iterations: int = 2


@dataclass(frozen=True)
class ProjectConfig:
    """项目总配置。

    其他模块通常只接收一个 ProjectConfig，再通过 cfg.time、cfg.power 等字段
    访问子配置，避免函数参数列表无限变长。
    """

    time: TimeConfig = field(default_factory=TimeConfig)
    power: PowerConfig = field(default_factory=PowerConfig)
    gas: GasConfig = field(default_factory=GasConfig)
    event: EventConfig = field(default_factory=EventConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    random_seed: int = 42


DEFAULT_CONFIG = ProjectConfig()


def calibration_warning_messages() -> Tuple[str, ...]:
    """返回启动时需要记录的待校准参数提示。"""

    return (
        "Belgian 20 管道缺少可直接用于 pandapipes 的长度、直径、粗糙度；"
        "当前使用 belgian20_data.py 中集中标注的暂定等效参数。",
        "Excel 中 Wmn/Kmn 为稳态流量关系系数，未被当作管径、长度或粗糙度使用。",
        "气体 HHV、压缩机 nominal_flow_kg_s、压缩机效率与压力约束均为待校准参数。",
    )
