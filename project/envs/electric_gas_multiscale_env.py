"""Gymnasium 风格的电-气多时间尺度强化学习环境。

强化学习初学者可以把本文件理解成“智能体和物理仿真之间的翻译层”：
- 智能体只会输出 [-1, 1] 的归一化动作；
- 环境把动作映射成 MW、Mvar、压力比等物理量，并做安全投影；
- CoupledSolver 调用 pandapower/pandapipes 求解电网和气网；
- 环境把求解结果整理成下一步观测、奖励和 info 调试字典。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from project.config import DEFAULT_CONFIG, ProjectConfig
from project.data.belgian20_data import COMPRESSOR_CONFIGS, GAS_NODES, GAS_PIPES, GAS_SUPPLIERS, N_GAS_JUNCTIONS
from project.data.ieee33_data import ESS_CONFIGS, GFG_CONFIGS, IEEE33_LINE_DATA, P2G_CONFIGS, RENEWABLE_CONFIGS
from project.data.profile_generator import DailyProfiles, generate_daily_profiles, profile_at
from project.networks.gas_network import GasNetworkArtifacts, build_gas_network
from project.networks.power_network import PowerNetworkArtifacts, build_power_network
from project.networks.topology_validator import validate_belgian20_topology
from project.simulation.coupled_solver import CoupledSolveResult, CoupledSolver, PhysicalActions
from project.simulation.safety_projection import (
    ESSProjectionBatch,
    InverterProjection,
    project_ess_batch,
    project_inverter_action,
    update_ess_soc,
)


class SimpleBox:
    """Gymnasium 不可用时的轻量 Box 替代。

    训练脚本只需要 ``shape`` 和 ``sample()``，所以不强制依赖完整 gymnasium。
    """

    def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype=np.float32):
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high).astype(self.dtype)


@dataclass
class _Snapshot:
    """step 开始时的回滚快照。"""

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
    """电-气耦合微电网多时间尺度 RL 环境。

    step 顺序：
    1. 用当前 t 读取外生曲线；
    2. 接收归一化动作并做安全投影；
    3. 写入网络并求解 t 到 t+1；
    4. 用实际 ESS 功率更新 SOC；
    5. 时间索引变为 t+1；
    6. 返回 t+1 的归一化观测。
    """

    def __init__(self, config: ProjectConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        # 建网之前先做拓扑校验，避免 pandapipes 在更深处报难读的矩阵错误。
        validation = validate_belgian20_topology()
        validation.raise_if_invalid()

        # power/gas 保存 pandapower、pandapipes 网络对象和各类元件索引。
        self.power: PowerNetworkArtifacts = build_power_network(self.config)
        self.gas: GasNetworkArtifacts = build_gas_network(self.config)
        self.solver = CoupledSolver(self.power, self.gas, self.config)

        # 动作维度来自设备数量：慢动作控制 ESS/GFG/P2G/压缩机，快动作控制逆变器。
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
        self.ess_soc = np.array([ess.soc_initial for ess in ESS_CONFIGS], dtype=float)
        self.last_slow_action = np.zeros(self.slow_action_dim, dtype=float)
        self.last_physical_slow: Dict[str, np.ndarray] = {}
        self.previous_device_actions: Dict[str, np.ndarray] = {}
        self.consecutive_solver_failures = 0
        self.last_solve_result: CoupledSolveResult | None = None
        self.last_ess_projection: ESSProjectionBatch | None = None
        self.last_inverter_projection: list[InverterProjection] = []
        self._reset_device_memory()

    @property
    def global_state_dim(self) -> int:
        """全局观测向量长度。

        这里显式计算长度，方便训练脚本提前创建 observation_space 和神经网络。
        """

        power_dim = 33 + len(IEEE33_LINE_DATA) + 5 + 3 * self.n_renew
        ess_dim = 3 * self.n_ess
        gas_dim = N_GAS_JUNCTIONS + len(self.gas.prs_junction_indices if hasattr(self, "gas") else GFG_CONFIGS)
        gas_dim += len(GAS_SUPPLIERS) + len(COMPRESSOR_CONFIGS) + self.n_gfg + self.n_p2g + 2 + len(GAS_PIPES)
        time_dim = 6
        return power_dim + ess_dim + gas_dim + time_dim

    def _make_action_space(self):
        try:
            from gymnasium import spaces

            return spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        except Exception:
            return SimpleBox(-1.0, 1.0, shape=(self.action_dim,), dtype=np.float32)

    def _reset_device_memory(self) -> None:
        """重置慢速设备的“上一时刻设定值”。

        奖励中会惩罚设备动作突变，所以环境需要记住上一次 ESS/GFG/P2G/压缩机动作。
        """

        self.last_physical_slow = {
            "ess_p_mw": np.zeros(self.n_ess, dtype=float),
            "gfg_p_mw": np.zeros(self.n_gfg, dtype=float),
            "p2g_p_mw": np.zeros(self.n_p2g, dtype=float),
            "compressor_ratio": np.array([c.initial_pressure_ratio for c in COMPRESSOR_CONFIGS], dtype=float),
        }
        self.previous_device_actions = {
            "ess_p_mw": np.zeros(self.n_ess, dtype=float),
            "gfg_p_mw": np.zeros(self.n_gfg, dtype=float),
            "p2g_p_mw": np.zeros(self.n_p2g, dtype=float),
            "compressor_ratio": np.array([c.initial_pressure_ratio for c in COMPRESSOR_CONFIGS], dtype=float),
        }

    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """开始一个新 episode，并返回初始观测。

        reset 会重新生成一天的负荷/新能源/气负荷曲线，并强制求解一次气网，
        这样第 0 步的观测不是空表，而是已经有潮流结果的物理状态。
        """

        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.profiles = generate_daily_profiles(self.config.time, seed=seed or self.config.random_seed)
        self.ess_soc = np.array([ess.soc_initial for ess in ESS_CONFIGS], dtype=float)
        self.last_slow_action = np.zeros(self.slow_action_dim, dtype=float)
        self._reset_device_memory()
        self.consecutive_solver_failures = 0
        self.solver.reset()

        initial_fast = self._project_fast_actions(np.zeros(self.fast_action_dim, dtype=float), profile_at(self.profiles, 0))
        actions = PhysicalActions(
            ess_p_mw=self.last_physical_slow["ess_p_mw"].copy(),
            gfg_p_mw=self.last_physical_slow["gfg_p_mw"].copy(),
            p2g_p_mw=self.last_physical_slow["p2g_p_mw"].copy(),
            compressor_ratio=self.last_physical_slow["compressor_ratio"].copy(),
            renewable_p_mw=initial_fast["renewable_p_mw"],
            renewable_q_mvar=initial_fast["renewable_q_mvar"],
            renewable_curtailment=initial_fast["renewable_curtailment"],
        )
        self.last_solve_result = self.solver.solve_step(0, profile_at(self.profiles, 0), actions, force_gas=True)
        obs = self.get_global_state()
        info = self._build_info(reward_components={}, constraint_metrics={}, solver_failed=False, slow_action_applied=True)
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一个 3 分钟快速步。

        Gymnasium 约定返回 ``obs, reward, terminated, truncated, info``：
        terminated 表示自然结束一天，truncated 表示连续求解失败等异常截断。
        """

        if self.profiles is None:
            raise RuntimeError("调用 step 前必须先 reset")
        if self.current_step >= self.config.time.steps_per_day:
            obs = self.get_global_state()
            return obs, 0.0, True, False, {"already_done": True}

        # 求解器可能失败。先快照，失败时回滚网络表和 SOC，再给失败惩罚。
        snapshot = self._make_snapshot()
        time_index = self.current_step
        profile = profile_at(self.profiles, time_index)
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.size != self.action_dim:
            raise ValueError(f"action 维度应为 {self.action_dim}，实际为 {action.size}")

        # 慢动作每 20 个快速步更新一次；其余步继续沿用上一次慢设备设定。
        slow_action_applied = time_index % self.config.time.slow_action_interval_steps == 0
        try:
            # 归一化动作 -> 物理动作 -> 电/气耦合求解。
            physical = self._map_and_project_action(action, profile, slow_action_applied)
            solve_result = self.solver.solve_step(time_index, profile, physical)
            self.last_solve_result = solve_result
            # SOC 必须用“实际执行”的 ESS 功率更新，而不是 Actor 原始请求。
            for i, ess in enumerate(ESS_CONFIGS):
                self.ess_soc[i] = update_ess_soc(
                    float(self.ess_soc[i]),
                    float(physical.ess_p_mw[i]),
                    ess,
                    self.config.time.dt_hours,
                )
            self.consecutive_solver_failures = 0
            self.current_step += 1
            reward, reward_components, constraint_metrics = self._compute_reward(solver_failed=False)
            terminated = self.current_step >= self.config.time.steps_per_day
            truncated = False
        except Exception as exc:
            # 回滚后仍然推进一步，避免训练在同一个坏状态无限卡住。
            self._restore_snapshot(snapshot)
            self.consecutive_solver_failures += 1
            self.current_step = min(snapshot.current_step + 1, self.config.time.steps_per_day)
            reward, reward_components, constraint_metrics = self._compute_reward(solver_failed=True)
            terminated = self.current_step >= self.config.time.steps_per_day
            truncated = self.consecutive_solver_failures >= self.config.safety.max_consecutive_solver_failures
            info = self._build_info(
                reward_components=reward_components,
                constraint_metrics=constraint_metrics,
                solver_failed=True,
                slow_action_applied=slow_action_applied,
            )
            info["exception"] = repr(exc)
            return self.get_global_state(), reward, terminated, truncated, info

        info = self._build_info(
            reward_components=reward_components,
            constraint_metrics=constraint_metrics,
            solver_failed=False,
            slow_action_applied=slow_action_applied,
        )
        return self.get_global_state(), reward, terminated, truncated, info

    def _map_and_project_action(
        self,
        action: np.ndarray,
        profile: Dict[str, np.ndarray | float],
        slow_action_applied: bool,
    ) -> PhysicalActions:
        """把 Actor 的 [-1, 1] 动作映射成求解器需要的物理量。

        慢动作包括 ESS 功率、GFG 出力、P2G 耗电功率和压缩机压力比；
        快动作包括每台新能源逆变器的无功和有功削减。
        """

        slow = np.clip(action[: self.slow_action_dim], -1.0, 1.0)
        fast = np.clip(action[self.slow_action_dim :], -1.0, 1.0)

        if slow_action_applied:
            # ESS 可以充/放电，需要根据当前 SOC 投影到可行功率区间。
            self.last_slow_action = slow.copy()
            cursor = 0
            requested_ess = np.array([slow[cursor + i] * ESS_CONFIGS[i].max_p_mw for i in range(self.n_ess)])
            cursor += self.n_ess
            ess_projection = project_ess_batch(requested_ess, self.ess_soc, ESS_CONFIGS, self.config.time.dt_hours)
            self.last_ess_projection = ess_projection
            self.last_physical_slow["ess_p_mw"] = ess_projection.applied_p_mw.copy()

            self.last_physical_slow["gfg_p_mw"] = np.array(
                [0.5 * (slow[cursor + i] + 1.0) * GFG_CONFIGS[i].max_p_mw for i in range(self.n_gfg)],
                dtype=float,
            )
            cursor += self.n_gfg
            self.last_physical_slow["p2g_p_mw"] = np.array(
                [0.5 * (slow[cursor + i] + 1.0) * P2G_CONFIGS[i].max_p_mw for i in range(self.n_p2g)],
                dtype=float,
            )
            cursor += self.n_p2g
            comp_ratio = []
            for i, comp in enumerate(COMPRESSOR_CONFIGS):
                value = 0.5 * (slow[cursor + i] + 1.0) * (comp.max_pressure_ratio - comp.min_pressure_ratio)
                comp_ratio.append(comp.min_pressure_ratio + value)
            self.last_physical_slow["compressor_ratio"] = np.array(comp_ratio, dtype=float)
        else:
            # 慢动作保持期间，ESS 的 SOC 会变化，所以同一个功率也要重新检查可行性。
            self.last_ess_projection = project_ess_batch(
                self.last_physical_slow["ess_p_mw"],
                self.ess_soc,
                ESS_CONFIGS,
                self.config.time.dt_hours,
            )
            self.last_physical_slow["ess_p_mw"] = self.last_ess_projection.applied_p_mw.copy()

        fast_projected = self._project_fast_actions(fast, profile)
        return PhysicalActions(
            ess_p_mw=self.last_physical_slow["ess_p_mw"].copy(),
            gfg_p_mw=self.last_physical_slow["gfg_p_mw"].copy(),
            p2g_p_mw=self.last_physical_slow["p2g_p_mw"].copy(),
            compressor_ratio=self.last_physical_slow["compressor_ratio"].copy(),
            renewable_p_mw=fast_projected["renewable_p_mw"],
            renewable_q_mvar=fast_projected["renewable_q_mvar"],
            renewable_curtailment=fast_projected["renewable_curtailment"],
        )

    def _project_fast_actions(self, fast: np.ndarray, profile: Dict[str, np.ndarray | float]) -> Dict[str, np.ndarray]:
        """投影快动作，使逆变器满足 P²+Q²<=S² 和削减比例边界。"""

        q_norm = np.asarray(fast[: self.n_renew], dtype=float)
        curtail_norm = np.asarray(fast[self.n_renew :], dtype=float)
        available = np.asarray(profile["renewable_available_mw"], dtype=float)
        projections: list[InverterProjection] = []
        for i, cfg in enumerate(RENEWABLE_CONFIGS):
            q_request = q_norm[i] * cfg.s_rated_mva
            curtail_request = 0.5 * (curtail_norm[i] + 1.0) * cfg.max_curtailment
            projections.append(project_inverter_action(cfg, available[i], q_request, curtail_request))
        self.last_inverter_projection = projections
        return {
            "renewable_p_mw": np.array([p.p_actual_mw for p in projections], dtype=float),
            "renewable_q_mvar": np.array([p.q_actual_mvar for p in projections], dtype=float),
            "renewable_curtailment": np.array([p.curtailment for p in projections], dtype=float),
        }

    def get_global_state(self) -> np.ndarray:
        """返回全局归一化状态。

        观测不是原始工程量，而是适合神经网络学习的量纲化特征：
        电压写成相对 1.0 pu 的偏差，线路负载除以 100%，压力写成相对目标值的偏差。
        """

        if self.profiles is None:
            return np.zeros(self.global_state_dim, dtype=np.float32)
        net_p = self.power.net
        net_g = self.gas.net
        t = min(self.current_step, self.config.time.steps_per_day)
        profile = profile_at(self.profiles, t)

        vm = self._series_values(net_p, "res_bus", "vm_pu", length=33, default=1.0)
        line_loading = self._series_values(net_p, "res_line", "loading_percent", length=len(IEEE33_LINE_DATA), default=0.0)
        ext_grid_p = self._series_values(net_p, "res_ext_grid", "p_mw", length=1, default=0.0)
        p_loss = np.array([np.nansum(self._series_values(net_p, "res_line", "pl_mw", len(IEEE33_LINE_DATA), 0.0))])
        renewable_available = np.asarray(profile["renewable_available_mw"], dtype=float)
        renewable_actual = np.array([p.p_actual_mw for p in self.last_inverter_projection] or np.zeros(self.n_renew), dtype=float)
        renewable_q = np.array([p.q_actual_mvar for p in self.last_inverter_projection] or np.zeros(self.n_renew), dtype=float)

        ess_margin = np.array(
            [min(self.ess_soc[i] - ESS_CONFIGS[i].soc_min, ESS_CONFIGS[i].soc_max - self.ess_soc[i]) for i in range(self.n_ess)],
            dtype=float,
        )
        ess_p_norm = np.array(
            [self.last_physical_slow["ess_p_mw"][i] / max(ESS_CONFIGS[i].max_p_mw, 1e-9) for i in range(self.n_ess)],
            dtype=float,
        )

        high_p_bar = self._series_values(net_g, "res_junction", "p_bar", length=N_GAS_JUNCTIONS, default=self.config.gas.source_pressure_bar)
        prs_p_bar = self._indexed_values(net_g, "res_junction", "p_bar", self.gas.prs_junction_indices, self.config.gas.prs_outlet_pressure_bar)
        source_mdot_raw = self._series_values(net_g, "res_ext_grid", "mdot_kg_per_s", length=len(GAS_SUPPLIERS), default=0.0)
        source_mdot = np.maximum(-source_mdot_raw, 0.0)
        pipe_mdot = self._series_values(net_g, "res_pipe", "mdot_kg_per_s", length=len(GAS_PIPES), default=0.0)
        compressor_ratio = self.last_physical_slow["compressor_ratio"]
        gfg_mdot = self.last_solve_result.gfg_mdot_kg_s if self.last_solve_result else np.zeros(self.n_gfg)
        p2g_mdot = self.last_solve_result.p2g_mdot_kg_s if self.last_solve_result else np.zeros(self.n_p2g)
        linepack = self.last_solve_result.equivalent_linepack_indicator if self.last_solve_result else 0.0
        gas_age = self.last_solve_result.gas_state_age if self.last_solve_result else 0

        hour = (t * self.config.time.dt_hours) % 24.0
        day_fraction = (t % self.config.time.steps_per_day) / self.config.time.steps_per_day
        time_features = np.array(
            [
                np.sin(2 * np.pi * hour / 24.0),
                np.cos(2 * np.pi * hour / 24.0),
                np.sin(2 * np.pi * day_fraction),
                np.cos(2 * np.pi * day_fraction),
                float(profile["next_hour_load_multiplier"]),
                float(np.sum(profile["next_hour_renewable_available_mw"])),
            ],
            dtype=float,
        )

        # parts 的顺序就是观测向量的含义顺序；训练脚本会按这个顺序切分局部观测。
        parts = [
            (vm - 1.0) / 0.10,
            line_loading / 100.0,
            np.array([float(profile["load_multiplier"]), float(np.sum(renewable_available)), float(np.sum(renewable_actual))]),
            renewable_available / np.array([r.capacity_mw for r in RENEWABLE_CONFIGS]),
            renewable_actual / np.maximum(np.array([r.capacity_mw for r in RENEWABLE_CONFIGS]), 1e-9),
            renewable_q / np.maximum(np.array([r.s_rated_mva for r in RENEWABLE_CONFIGS]), 1e-9),
            ext_grid_p / 10.0,
            p_loss / 1.0,
            self.ess_soc,
            ess_p_norm,
            ess_margin,
            (high_p_bar - 50.0) / 20.0,
            (prs_p_bar - self.config.gas.prs_outlet_pressure_bar) / 0.15,
            compressor_ratio / np.array([c.max_pressure_ratio for c in COMPRESSOR_CONFIGS]),
            source_mdot / 100.0,
            gfg_mdot / 2.0,
            p2g_mdot / 2.0,
            np.array([gas_age / self.config.time.slow_action_interval_steps, linepack / 10_000_000.0]),
            pipe_mdot / 100.0,
            time_features,
        ]
        state = np.concatenate([np.atleast_1d(p) for p in parts]).astype(np.float32)
        return np.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0)

    def get_manager_state(self) -> np.ndarray:
        """为 FuN Manager 预留的全局观测。"""

        return self.get_global_state()

    def get_fast_worker_state(self) -> np.ndarray:
        """快速 Worker 观测：偏重电压、线路、新能源与时间。"""

        global_state = self.get_global_state()
        cut = 33 + len(IEEE33_LINE_DATA) + 7 + 3 * self.n_renew
        return global_state[:cut].copy()

    def get_slow_worker_state(self) -> np.ndarray:
        """慢速 Worker 观测：偏重 ESS、气网和预测。"""

        global_state = self.get_global_state()
        start = 33 + len(IEEE33_LINE_DATA) + 7 + 3 * self.n_renew
        return global_state[start:].copy()

    def _compute_reward(self, solver_failed: bool) -> tuple[float, Dict[str, float], Dict[str, float]]:
        """根据当前求解结果计算奖励。

        本项目把各种越限、偏差、损耗和动作变化都看成 cost，最后 reward = -sum(cost)。
        因此 reward 越接近 0 通常表示运行越平稳。
        """

        dt_h = self.config.time.dt_hours
        net_p = self.power.net
        net_g = self.gas.net
        vm = self._series_values(net_p, "res_bus", "vm_pu", 33, 1.0)
        loading = self._series_values(net_p, "res_line", "loading_percent", len(IEEE33_LINE_DATA), 0.0)
        high_p = self._series_values(net_g, "res_junction", "p_bar", N_GAS_JUNCTIONS, self.config.gas.source_pressure_bar)
        prs_p = self._indexed_values(net_g, "res_junction", "p_bar", self.gas.prs_junction_indices, self.config.gas.prs_outlet_pressure_bar)

        voltage_deviation = float(np.sum(((vm - self.config.power.voltage_target_pu) / 0.05) ** 2))
        voltage_violation = float(np.sum(np.maximum(self.config.power.voltage_min_pu - vm, 0.0) ** 2 + np.maximum(vm - self.config.power.voltage_max_pu, 0.0) ** 2))
        high_pressure_deviation = float(np.sum(((high_p - self.config.gas.high_pressure_target_bar) / 20.0) ** 2))
        high_pressure_violation = float(np.sum(np.maximum(self.config.gas.high_pressure_min_bar - high_p, 0.0) ** 2 + np.maximum(high_p - self.config.gas.high_pressure_max_bar, 0.0) ** 2))
        prs_pressure_deviation = float(np.sum(((prs_p - self.config.gas.prs_outlet_pressure_bar) / 0.15) ** 2))
        prs_pressure_violation = float(np.sum(np.maximum(self.config.gas.prs_outlet_min_bar - prs_p, 0.0) ** 2 + np.maximum(prs_p - self.config.gas.prs_outlet_max_bar, 0.0) ** 2))
        line_overload = float(np.sum(np.maximum(loading - self.config.power.max_line_loading_percent, 0.0) ** 2))
        p_loss_mw = float(np.nansum(self._series_values(net_p, "res_line", "pl_mw", len(IEEE33_LINE_DATA), 0.0)))
        grid_purchase_mwh = max(float(np.nansum(self._series_values(net_p, "res_ext_grid", "p_mw", 1, 0.0))), 0.0) * dt_h
        gas_ext_mdot = self._series_values(net_g, "res_ext_grid", "mdot_kg_per_s", len(GAS_SUPPLIERS), 0.0)
        gas_purchase_kg = float(np.nansum(np.maximum(-gas_ext_mdot, 0.0))) * dt_h * 3600.0
        curtailment_mwh = 0.0
        if self.last_inverter_projection and self.profiles is not None:
            profile = profile_at(self.profiles, min(self.current_step, self.config.time.steps_per_day))
            available = np.asarray(profile["renewable_available_mw"], dtype=float)
            curtailment_mwh = float(np.sum([available[i] * p.curtailment * dt_h for i, p in enumerate(self.last_inverter_projection)]))
        comp_energy_mwh = 0.0
        if self.last_solve_result:
            comp_energy_mwh = float(sum(d.electric_power_mw for d in self.last_solve_result.compressor_dispatches) * dt_h)
        ess_action_change = float(np.sum(np.abs(self.last_physical_slow["ess_p_mw"] - self.previous_device_actions["ess_p_mw"])))
        gfg_action_change = float(np.sum(np.abs(self.last_physical_slow["gfg_p_mw"] - self.previous_device_actions["gfg_p_mw"])))
        p2g_action_change = float(np.sum(np.abs(self.last_physical_slow["p2g_p_mw"] - self.previous_device_actions["p2g_p_mw"])))
        soc_soft = float(np.sum(np.maximum(self.config.safety.soc_soft_low - self.ess_soc, 0.0) ** 2 + np.maximum(self.ess_soc - self.config.safety.soc_soft_high, 0.0) ** 2))
        terminal_soc = 0.0
        if self.current_step >= self.config.time.steps_per_day:
            terminal_soc = float(np.sum((self.ess_soc - np.array([ess.soc_initial for ess in ESS_CONFIGS])) ** 2))

        weights = self.config.reward
        # components 保留每项加权成本，方便训练日志分析到底是哪类约束在主导奖励。
        components = {
            "voltage_deviation": weights.voltage_deviation * voltage_deviation,
            "voltage_violation": weights.voltage_violation * voltage_violation,
            "high_pressure_deviation": weights.high_pressure_deviation * high_pressure_deviation,
            "high_pressure_violation": weights.high_pressure_violation * high_pressure_violation,
            "prs_pressure_deviation": weights.prs_pressure_deviation * prs_pressure_deviation,
            "prs_pressure_violation": weights.prs_pressure_violation * prs_pressure_violation,
            "line_overload": weights.line_overload * line_overload,
            "power_loss": weights.power_loss * p_loss_mw * dt_h,
            "grid_purchase": weights.grid_energy_price * grid_purchase_mwh,
            "gas_purchase": weights.gas_price * gas_purchase_kg / 1000.0,
            "renewable_curtailment": weights.renewable_curtailment * curtailment_mwh,
            "ess_action_change": weights.ess_action_change * ess_action_change,
            "gfg_action_change": weights.gfg_action_change * gfg_action_change,
            "p2g_action_change": weights.p2g_action_change * p2g_action_change,
            "compressor_energy": weights.compressor_energy * comp_energy_mwh,
            "soc_soft": weights.soc_soft * soc_soft,
            "solver_failure": weights.solver_failure if solver_failed else 0.0,
            "terminal_soc": weights.terminal_soc * terminal_soc,
        }
        # constraint_metrics 是未加权的物理指标，主要用于画图和诊断策略质量。
        constraint_metrics = {
            "vm_min_pu": float(np.nanmin(vm)),
            "vm_max_pu": float(np.nanmax(vm)),
            "voltage_mean_abs_deviation_pu": float(np.nanmean(np.abs(vm - self.config.power.voltage_target_pu))),
            "voltage_rms_deviation_pu": float(np.sqrt(np.nanmean((vm - self.config.power.voltage_target_pu) ** 2))),
            "max_line_loading_percent": float(np.nanmax(loading)),
            "high_pressure_min_bar": float(np.nanmin(high_p)),
            "high_pressure_max_bar": float(np.nanmax(high_p)),
            "high_pressure_mean_abs_deviation_bar": float(np.nanmean(np.abs(high_p - self.config.gas.high_pressure_target_bar))),
            "high_pressure_rms_deviation_bar": float(np.sqrt(np.nanmean((high_p - self.config.gas.high_pressure_target_bar) ** 2))),
            "prs_pressure_min_bar": float(np.nanmin(prs_p)),
            "prs_pressure_max_bar": float(np.nanmax(prs_p)),
            "prs_pressure_mean_abs_deviation_bar": float(np.nanmean(np.abs(prs_p - self.config.gas.prs_outlet_pressure_bar))),
            "prs_pressure_rms_deviation_bar": float(np.sqrt(np.nanmean((prs_p - self.config.gas.prs_outlet_pressure_bar) ** 2))),
            "soc_min": float(np.nanmin(self.ess_soc)),
            "soc_max": float(np.nanmax(self.ess_soc)),
            "grid_purchase_mwh": float(grid_purchase_mwh),
            "gas_purchase_kg": float(gas_purchase_kg),
        }
        reward = -float(sum(components.values()))
        self.previous_device_actions = {k: v.copy() for k, v in self.last_physical_slow.items()}
        return reward, components, constraint_metrics

    def _build_info(
        self,
        reward_components: Dict[str, float],
        constraint_metrics: Dict[str, float],
        solver_failed: bool,
        slow_action_applied: bool,
    ) -> Dict[str, Any]:
        """构造 info 字典。

        info 不直接参与环境状态转移，但训练和可视化会读取其中的 reward_components、
        constraint_metrics、raw_action、applied_action 等调试信息。
        """

        result = self.last_solve_result
        return {
            "step": self.current_step,
            "slow_action_applied": slow_action_applied,
            "converged": bool(result.converged if result else False) and not solver_failed,
            "solver_failed": solver_failed,
            "power_converged": bool(result.power_converged if result else False) and not solver_failed,
            "gas_converged": bool(result.gas_converged if result else False) and not solver_failed,
            "gas_solved_this_step": bool(result.gas_solved_this_step if result else False),
            "gas_solve_reason": result.gas_solve_reason if result else "none",
            "gas_state_age": int(result.gas_state_age if result else 0),
            "gas_solve_count": int(self.solver.gas_solve_count),
            "equivalent_linepack_indicator": float(result.equivalent_linepack_indicator if result else 0.0),
            "ess_soc": self.ess_soc.copy(),
            "ess_projection": self.last_ess_projection,
            "reward_components": reward_components,
            "constraint_metrics": constraint_metrics,
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

    def _indexed_values(self, net: Any, table_name: str, column: str, indices: list[int], default: float) -> np.ndarray:
        if not hasattr(net, table_name):
            return np.full(len(indices), default, dtype=float)
        table = getattr(net, table_name)
        values = []
        for idx in indices:
            if column in table and idx in table.index:
                values.append(float(table.at[idx, column]))
            else:
                values.append(default)
        return np.asarray(values, dtype=float)

    def _copy_tables(self, net: Any, table_names: tuple[str, ...]) -> Dict[str, Any]:
        copied = {}
        for name in table_names:
            if hasattr(net, name):
                copied[name] = getattr(net, name).copy(deep=True)
        return copied

    def _restore_tables(self, net: Any, tables: Dict[str, Any]) -> None:
        for name, table in tables.items():
            setattr(net, name, table.copy(deep=True))

    def _make_snapshot(self) -> _Snapshot:
        """复制当前环境状态，用于求解失败时回滚。"""

        power_tables = self._copy_tables(
            self.power.net,
            ("load", "sgen", "storage", "res_bus", "res_line", "res_load", "res_sgen", "res_storage", "res_ext_grid"),
        )
        gas_tables = self._copy_tables(
            self.gas.net,
            ("sink", "source", "compressor", "pressure_control", "res_junction", "res_pipe", "res_sink", "res_source", "res_ext_grid", "res_compressor"),
        )
        return _Snapshot(
            current_step=self.current_step,
            ess_soc=self.ess_soc.copy(),
            last_slow_action=self.last_slow_action.copy(),
            last_physical_slow={k: v.copy() for k, v in self.last_physical_slow.items()},
            previous_device_actions={k: v.copy() for k, v in self.previous_device_actions.items()},
            solver_gas_state_age=self.solver.gas_state_age,
            solver_gas_solve_count=self.solver.gas_solve_count,
            power_tables=power_tables,
            gas_tables=gas_tables,
        )

    def _restore_snapshot(self, snapshot: _Snapshot) -> None:
        """恢复 ``_make_snapshot`` 保存的状态。"""

        self.ess_soc = snapshot.ess_soc.copy()
        self.last_slow_action = snapshot.last_slow_action.copy()
        self.last_physical_slow = {k: v.copy() for k, v in snapshot.last_physical_slow.items()}
        self.previous_device_actions = {k: v.copy() for k, v in snapshot.previous_device_actions.items()}
        self.solver.gas_state_age = snapshot.solver_gas_state_age
        self.solver.gas_solve_count = snapshot.solver_gas_solve_count
        self._restore_tables(self.power.net, snapshot.power_tables)
        self._restore_tables(self.gas.net, snapshot.gas_tables)
