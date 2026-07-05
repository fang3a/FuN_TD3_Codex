"""事件驱动气网求解调度。

气网 pipeflow 比电网潮流更慢，且慢速设备通常每小时才变一次。为了加速训练，
环境不是每个 3 分钟步都重算气网，而是当“时间到点或扰动足够大”时才重算。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from project.config import EventConfig, TimeConfig


@dataclass
class GasEventState:
    """上一次气网事件求解的驱动量。"""

    gfg_mdot_kg_s: np.ndarray
    p2g_mdot_kg_s: np.ndarray
    compressor_ratio: np.ndarray
    gas_load_multiplier: float


@dataclass(frozen=True)
class GasSolveDecision:
    """是否需要运行 pipeflow。"""

    should_solve: bool
    reason: str


class EventScheduler:
    """根据时间和扰动阈值决定是否运行气网 pipeflow。"""

    def __init__(self, time_config: TimeConfig, event_config: EventConfig):
        self.time_config = time_config
        self.event_config = event_config
        self.last_state: GasEventState | None = None

    def reset(self) -> None:
        self.last_state = None

    def decide(
        self,
        time_index: int,
        gfg_mdot_kg_s: np.ndarray,
        p2g_mdot_kg_s: np.ndarray,
        compressor_ratio: np.ndarray,
        gas_load_multiplier: float,
    ) -> GasSolveDecision:
        """返回当前快速步是否触发气网求解。"""

        if self.last_state is None:
            # 第一次必须求解，否则没有可复用的气网状态。
            return GasSolveDecision(True, "initial")
        if time_index % self.time_config.slow_action_interval_steps == 0:
            # 每小时强制刷新一次，防止长期复用旧气网状态。
            return GasSolveDecision(True, "hourly")

        # 下面几项是事件触发：GFG/P2G/压缩机/气负荷变化超过阈值则重算。
        if np.max(np.abs(gfg_mdot_kg_s - self.last_state.gfg_mdot_kg_s)) > self.event_config.gfg_mdot_threshold_kg_s:
            return GasSolveDecision(True, "gfg_change")
        if np.max(np.abs(p2g_mdot_kg_s - self.last_state.p2g_mdot_kg_s)) > self.event_config.p2g_mdot_threshold_kg_s:
            return GasSolveDecision(True, "p2g_change")
        if np.max(np.abs(compressor_ratio - self.last_state.compressor_ratio)) > self.event_config.compressor_ratio_threshold:
            return GasSolveDecision(True, "compressor_change")

        previous = max(abs(self.last_state.gas_load_multiplier), 1e-9)
        relative_change = abs(gas_load_multiplier - self.last_state.gas_load_multiplier) / previous
        if relative_change > self.event_config.gas_load_relative_threshold:
            return GasSolveDecision(True, "gas_load_change")

        return GasSolveDecision(False, "hold_previous_gas_state")

    def mark_solved(
        self,
        gfg_mdot_kg_s: np.ndarray,
        p2g_mdot_kg_s: np.ndarray,
        compressor_ratio: np.ndarray,
        gas_load_multiplier: float,
    ) -> None:
        """记录最近一次 pipeflow 的驱动量。"""

        self.last_state = GasEventState(
            gfg_mdot_kg_s=np.asarray(gfg_mdot_kg_s, dtype=float).copy(),
            p2g_mdot_kg_s=np.asarray(p2g_mdot_kg_s, dtype=float).copy(),
            compressor_ratio=np.asarray(compressor_ratio, dtype=float).copy(),
            gas_load_multiplier=float(gas_load_multiplier),
        )
