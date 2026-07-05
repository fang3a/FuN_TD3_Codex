"""显式手工电-气耦合求解器。

这个模块连接两类物理仿真器：
- pandapower 负责电网潮流；
- pandapipes 负责气网 pipeflow。

“显式手工耦合”的意思是：代码先把 GFG/P2G/压缩机等耦合设备写入两张网络，
再分别运行求解器，而不是依赖 pandapower/pandapipes 的 MultiNet 控制器。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np

from project.config import ProjectConfig
from project.coupling.compressor_model import CompressorDispatch, dispatch_compressor
from project.coupling.gfg_model import dispatch_gfg
from project.coupling.p2g_model import dispatch_p2g
from project.data.belgian20_data import COMPRESSOR_CONFIGS, GAS_NODES, GAS_PIPES
from project.data.ieee33_data import (
    ESS_CONFIGS,
    GFG_CONFIGS,
    IEEE33_LOAD_DATA,
    P2G_CONFIGS,
    RENEWABLE_CONFIGS,
)
from project.networks.gas_network import GasNetworkArtifacts
from project.networks.power_network import PowerNetworkArtifacts
from project.simulation.event_scheduler import EventScheduler


@dataclass(frozen=True)
class PhysicalActions:
    """已经映射并投影后的物理动作。"""

    ess_p_mw: np.ndarray
    gfg_p_mw: np.ndarray
    p2g_p_mw: np.ndarray
    compressor_ratio: np.ndarray
    renewable_p_mw: np.ndarray
    renewable_q_mvar: np.ndarray
    renewable_curtailment: np.ndarray


@dataclass
class CoupledSolveResult:
    """耦合求解结果摘要。"""

    power_converged: bool
    gas_converged: bool
    gas_solved_this_step: bool
    gas_solve_reason: str
    gas_state_age: int
    gfg_mdot_kg_s: np.ndarray
    p2g_mdot_kg_s: np.ndarray
    compressor_dispatches: list[CompressorDispatch] = field(default_factory=list)
    equivalent_linepack_indicator: float = 0.0

    @property
    def converged(self) -> bool:
        return self.power_converged and self.gas_converged


class CoupledSolver:
    """pandapower/pandapipes 显式手工耦合求解。

    求解顺序可粗略理解为：
    1. 把当前负荷、可再生出力和慢/快动作写入电网；
    2. 把 GFG 耗气、P2G 注气、基础气负荷和压缩机压力比写入气网；
    3. 根据事件调度器决定是否运行 pipeflow；
    4. 运行电网 powerflow，并返回收敛状态和诊断量。
    """

    def __init__(
        self,
        power: PowerNetworkArtifacts,
        gas: GasNetworkArtifacts,
        config: ProjectConfig,
    ):
        self.power = power
        self.gas = gas
        self.config = config
        self.scheduler = EventScheduler(config.time, config.event)
        self.gas_state_age = 0
        self.gas_solve_count = 0
        self.last_compressor_mdot_kg_s = np.array(
            [comp.nominal_flow_kg_s for comp in COMPRESSOR_CONFIGS],
            dtype=float,
        )
        self.last_gas_converged = False

    def reset(self) -> None:
        """重置事件调度器和上一轮气网状态。"""

        self.scheduler.reset()
        self.gas_state_age = 0
        self.gas_solve_count = 0
        self.last_gas_converged = False
        self.last_compressor_mdot_kg_s = np.array(
            [comp.nominal_flow_kg_s for comp in COMPRESSOR_CONFIGS],
            dtype=float,
        )

    def solve_step(
        self,
        time_index: int,
        profile: Dict[str, np.ndarray | float],
        actions: PhysicalActions,
        force_gas: bool = False,
    ) -> CoupledSolveResult:
        """执行一次 t 到 t+1 的准稳态耦合求解。"""

        # 先写入所有外生曲线和设备动作，使两个网络表处在同一个时间步。
        self._write_power_profile_and_actions(profile, actions)
        gfg_mdot = self._write_gfg(actions.gfg_p_mw)
        p2g_mdot = self._write_p2g(actions.p2g_p_mw)
        self._write_gas_loads(float(profile["gas_multiplier"]))
        self._write_compressor_ratios(actions.compressor_ratio)

        decision = self.scheduler.decide(
            time_index=time_index,
            gfg_mdot_kg_s=gfg_mdot,
            p2g_mdot_kg_s=p2g_mdot,
            compressor_ratio=actions.compressor_ratio,
            gas_load_multiplier=float(profile["gas_multiplier"]),
        )
        gas_solved = force_gas or decision.should_solve
        gas_reason = "forced" if force_gas and not decision.should_solve else decision.reason

        # 压缩机电功率依赖流量估计；气网没重算时沿用上一轮流量估计。
        compressor_dispatches = self._write_compressor_power_loads(actions.compressor_ratio)
        gas_converged = self.last_gas_converged
        if gas_solved:
            # 迭代几次：先 pipeflow 得到更好的压缩机流量，再刷新压缩机电负荷。
            for _ in range(self.config.safety.solver_iterations):
                self._run_pipeflow()
                gas_converged = True
                self.last_gas_converged = True
                self.last_compressor_mdot_kg_s = self._read_compressor_mdot_estimates()
                compressor_dispatches = self._write_compressor_power_loads(actions.compressor_ratio)
            self.scheduler.mark_solved(
                gfg_mdot_kg_s=gfg_mdot,
                p2g_mdot_kg_s=p2g_mdot,
                compressor_ratio=actions.compressor_ratio,
                gas_load_multiplier=float(profile["gas_multiplier"]),
            )
            self.gas_state_age = 0
            self.gas_solve_count += 1
        else:
            # 不求气网时，保留上一轮结果，并记录这个气网状态已经用了多久。
            self.gas_state_age += 1

        # 电网每个快速步都求解，因为逆变器快动作会每 3 分钟变化。
        self._run_powerflow()
        indicator = self.compute_equivalent_linepack_indicator()
        return CoupledSolveResult(
            power_converged=True,
            gas_converged=gas_converged,
            gas_solved_this_step=gas_solved,
            gas_solve_reason=gas_reason,
            gas_state_age=self.gas_state_age,
            gfg_mdot_kg_s=gfg_mdot,
            p2g_mdot_kg_s=p2g_mdot,
            compressor_dispatches=compressor_dispatches,
            equivalent_linepack_indicator=indicator,
        )

    def _write_power_profile_and_actions(self, profile: Dict[str, np.ndarray | float], actions: PhysicalActions) -> None:
        """把电负荷、新能源、ESS 写入 pandapower 网络表。"""

        net = self.power.net
        load_mult = float(profile["load_multiplier"])
        for i, (_, p_base_mw, q_base_mvar) in enumerate(IEEE33_LOAD_DATA):
            idx = self.power.load_indices[i]
            net.load.at[idx, "p_mw"] = p_base_mw * load_mult
            net.load.at[idx, "q_mvar"] = q_base_mvar * load_mult

        for i, idx in enumerate(self.power.renewable_sgen_indices):
            net.sgen.at[idx, "p_mw"] = float(actions.renewable_p_mw[i])
            net.sgen.at[idx, "q_mvar"] = float(actions.renewable_q_mvar[i])

        for i, idx in enumerate(self.power.ess_storage_indices):
            net.storage.at[idx, "p_mw"] = float(actions.ess_p_mw[i])
            net.storage.at[idx, "q_mvar"] = 0.0

    def _write_gfg(self, requested_p_mw: Sequence[float]) -> np.ndarray:
        """写入 GFG：电侧是 sgen，气侧是 sink。"""

        net_power = self.power.net
        net_gas = self.gas.net
        mdot = []
        for i, cfg in enumerate(GFG_CONFIGS):
            dispatch = dispatch_gfg(cfg, float(requested_p_mw[i]), self.config.gas.hhv_mj_per_kg)
            net_power.sgen.at[self.power.gfg_sgen_indices[i], "p_mw"] = dispatch.electric_power_mw
            net_power.sgen.at[self.power.gfg_sgen_indices[i], "q_mvar"] = 0.0
            net_gas.sink.at[self.gas.gfg_sink_indices[i], "mdot_kg_per_s"] = dispatch.gas_mdot_kg_s
            mdot.append(dispatch.gas_mdot_kg_s)
        return np.asarray(mdot, dtype=float)

    def _write_p2g(self, requested_p_mw: Sequence[float]) -> np.ndarray:
        """写入 P2G：电侧是 load，气侧是 source。"""

        net_power = self.power.net
        net_gas = self.gas.net
        mdot = []
        for i, cfg in enumerate(P2G_CONFIGS):
            dispatch = dispatch_p2g(cfg, float(requested_p_mw[i]), self.config.gas.hhv_mj_per_kg)
            net_power.load.at[self.power.p2g_load_indices[i], "p_mw"] = dispatch.electric_power_mw
            net_power.load.at[self.power.p2g_load_indices[i], "q_mvar"] = 0.0
            net_gas.source.at[self.gas.p2g_source_indices[i], "mdot_kg_per_s"] = dispatch.gas_mdot_kg_s
            mdot.append(dispatch.gas_mdot_kg_s)
        return np.asarray(mdot, dtype=float)

    def _write_gas_loads(self, gas_multiplier: float) -> None:
        """按气负荷倍率刷新基础天然气需求。"""

        net = self.gas.net
        from project.data.belgian20_data import mm3_per_day_to_kg_per_s

        for node in GAS_NODES:
            idx = self.gas.base_sink_indices_by_node.get(node.node)
            if idx is None:
                continue
            base_mdot = mm3_per_day_to_kg_per_s(node.demand_mm3_per_day)
            net.sink.at[idx, "mdot_kg_per_s"] = base_mdot * gas_multiplier

    def _write_compressor_ratios(self, requested_ratio: Sequence[float]) -> None:
        """把压缩机压力比写入 pandapipes compressor 元件。"""

        net = self.gas.net
        for i, idx in enumerate(self.gas.compressor_indices):
            ratio = float(np.clip(requested_ratio[i], COMPRESSOR_CONFIGS[i].min_pressure_ratio, COMPRESSOR_CONFIGS[i].max_pressure_ratio))
            net.compressor.at[idx, "pressure_ratio"] = ratio

    def _write_compressor_power_loads(self, requested_ratio: Sequence[float]) -> list[CompressorDispatch]:
        """根据压缩机压力比和流量估计，写入电侧压缩机负荷。"""

        dispatches: list[CompressorDispatch] = []
        net_power = self.power.net
        for i, cfg in enumerate(COMPRESSOR_CONFIGS):
            dispatch = dispatch_compressor(
                cfg,
                requested_ratio=float(requested_ratio[i]),
                mdot_estimate_kg_s=float(self.last_compressor_mdot_kg_s[i]),
            )
            net_power.load.at[self.power.compressor_load_indices[i], "p_mw"] = dispatch.electric_power_mw
            net_power.load.at[self.power.compressor_load_indices[i], "q_mvar"] = 0.0
            dispatches.append(dispatch)
        return dispatches

    def _run_pipeflow(self) -> None:
        """运行 pandapipes 气网水力求解。"""

        import pandapipes as pp

        pp.pipeflow(self.gas.net, max_iter_hyd=50, tol_p=1e-5)

    def _run_powerflow(self) -> None:
        """运行 pandapower 潮流，并按顺序尝试多个算法。"""

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
        """从 pandapipes 结果表读取压缩机流量，供下一步估算压缩机电功率。"""

        net = self.gas.net
        if not hasattr(net, "res_compressor") or len(net.res_compressor) == 0:
            return self.last_compressor_mdot_kg_s.copy()
        mdots = []
        table = net.res_compressor
        candidate_columns = (
            "mdot_from_kg_per_s",
            "mdot_to_kg_per_s",
            "mdot_kg_per_s",
            "mf_from_kg_per_s",
            "mf_to_kg_per_s",
        )
        for idx, cfg in zip(self.gas.compressor_indices, COMPRESSOR_CONFIGS):
            value = cfg.nominal_flow_kg_s
            for col in candidate_columns:
                if col in table.columns and idx in table.index:
                    raw = table.at[idx, col]
                    if np.isfinite(raw):
                        value = abs(float(raw))
                        break
            mdots.append(value)
        return np.asarray(mdots, dtype=float)

    def compute_equivalent_linepack_indicator(self) -> float:
        """计算准稳态压力派生的等效管存指标。

        这不是严格动态 linepack，只是使用当前 pipeflow 压力场和暂定管道体积
        估算的压力状态指标。
        """

        net = self.gas.net
        if not hasattr(net, "res_junction") or "p_bar" not in net.res_junction:
            return 0.0
        total_mass_kg = 0.0
        p_bar = net.res_junction["p_bar"]
        for pipe in GAS_PIPES:
            if pipe.from_junction not in p_bar.index or pipe.to_junction not in p_bar.index:
                continue
            # 理想气体近似：rho = p / (R*T*Z)，再乘以管道体积得到近似管存质量。
            p_avg_pa = 0.5 * (float(p_bar.at[pipe.from_junction]) + float(p_bar.at[pipe.to_junction])) * 1e5
            volume_m3 = np.pi * (pipe.diameter_m / 2.0) ** 2 * pipe.length_km * 1000.0
            rho_kg_m3 = p_avg_pa / (
                self.config.gas.gas_specific_gas_constant_j_per_kg_k
                * self.config.gas.gas_temperature_k
                * self.config.gas.gas_compressibility_z
            )
            total_mass_kg += rho_kg_m3 * volume_m3
        return float(total_mass_kg)
