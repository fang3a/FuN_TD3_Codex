"""IEEE 33 节点 pandapower 网络构建。

本文件只负责“搭一张电网”。环境每一步会通过返回的元件索引修改负荷、
发电机、储能和压缩机电负荷，然后调用 pandapower 潮流。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from project.config import ProjectConfig
from project.data.ieee33_data import (
    ESS_CONFIGS,
    GFG_CONFIGS,
    IEEE33_LINE_DATA,
    IEEE33_LOAD_DATA,
    N_POWER_BUSES,
    P2G_CONFIGS,
    RENEWABLE_CONFIGS,
)
from project.data.belgian20_data import COMPRESSOR_CONFIGS


@dataclass
class PowerNetworkArtifacts:
    """pandapower 元件索引集合。

    pandapower 的元件存在 DataFrame 里，创建时返回行索引。保存这些索引后，
    求解器就能在每个 step 快速找到对应设备并写入新设定。
    """

    net: object
    load_indices: List[int]
    renewable_sgen_indices: List[int]
    ess_storage_indices: List[int]
    gfg_sgen_indices: List[int]
    p2g_load_indices: List[int]
    compressor_load_indices: List[int]
    ext_grid_index: int


def _require_pandapower():
    try:
        import pandapower as pp
    except ImportError as exc:
        raise RuntimeError("需要安装 pandapower 才能构建电网。建议 pandapower==3.3.3。") from exc
    return pp


def build_power_network(config: ProjectConfig | None = None) -> PowerNetworkArtifacts:
    """构建 IEEE 33 节点配电网和电侧耦合设备。

    GFG 电侧为 sgen；P2G 电侧为 load；电驱压缩机电侧为独立 load；
    这些元件不会通过 MultiNet 控制器重复耦合。
    """

    cfg = config or ProjectConfig()
    pp = _require_pandapower()
    net = pp.create_empty_network(sn_mva=cfg.power.base_mva)

    # 母线是电网节点，IEEE33 在这里用 0..32 编号。
    for bus in range(N_POWER_BUSES):
        pp.create_bus(
            net,
            vn_kv=cfg.power.base_kv,
            name=f"Bus_{bus}",
            min_vm_pu=cfg.power.voltage_min_pu,
            max_vm_pu=cfg.power.voltage_max_pu,
        )

    # ext_grid 是外部电网/slack 母线，负责平衡全网功率。
    ext_grid_index = pp.create_ext_grid(
        net,
        bus=cfg.power.slack_bus,
        vm_pu=cfg.power.slack_vm_pu,
        va_degree=0.0,
        name="Utility_Grid",
    )

    # 线路参数来自 IEEE33 数据表；length_km 固定 1，因此 r/x 直接作为每公里参数。
    for line_id, (from_bus, to_bus, r_ohm, x_ohm) in enumerate(IEEE33_LINE_DATA):
        pp.create_line_from_parameters(
            net,
            from_bus=from_bus,
            to_bus=to_bus,
            length_km=1.0,
            r_ohm_per_km=r_ohm,
            x_ohm_per_km=x_ohm,
            c_nf_per_km=0.0,
            max_i_ka=0.50,
            name=f"Line_{line_id}_{from_bus}_{to_bus}",
        )

    # 基础负荷会在每个 step 按外生负荷倍率缩放。
    load_indices: List[int] = []
    for bus, p_mw, q_mvar in IEEE33_LOAD_DATA:
        idx = pp.create_load(net, bus=bus, p_mw=p_mw, q_mvar=q_mvar, name=f"BaseLoad_bus_{bus}")
        load_indices.append(int(idx))

    # 新能源在 pandapower 中用 sgen 表示，快 Worker 控制其 P/Q。
    renewable_sgen_indices: List[int] = []
    for cfg_ren in RENEWABLE_CONFIGS:
        idx = pp.create_sgen(
            net,
            bus=cfg_ren.bus,
            p_mw=0.0,
            q_mvar=0.0,
            name=cfg_ren.name,
            max_p_mw=cfg_ren.capacity_mw,
            min_p_mw=0.0,
            max_q_mvar=cfg_ren.s_rated_mva,
            min_q_mvar=-cfg_ren.s_rated_mva,
        )
        renewable_sgen_indices.append(int(idx))

    # ESS 用 storage 元件表示；SOC 由环境显式维护。
    ess_storage_indices: List[int] = []
    for ess in ESS_CONFIGS:
        idx = pp.create_storage(
            net,
            bus=ess.bus,
            p_mw=0.0,
            q_mvar=0.0,
            max_e_mwh=ess.capacity_mwh,
            soc_percent=100.0 * ess.soc_initial,
            name=ess.name,
        )
        ess_storage_indices.append(int(idx))

    gfg_sgen_indices: List[int] = []
    for gfg in GFG_CONFIGS:
        idx = pp.create_sgen(
            net,
            bus=gfg.power_bus,
            p_mw=0.0,
            q_mvar=0.0,
            name=gfg.name,
            max_p_mw=gfg.max_p_mw,
            min_p_mw=0.0,
        )
        gfg_sgen_indices.append(int(idx))

    p2g_load_indices: List[int] = []
    for p2g in P2G_CONFIGS:
        idx = pp.create_load(net, bus=p2g.power_bus, p_mw=0.0, q_mvar=0.0, name=p2g.name)
        p2g_load_indices.append(int(idx))

    compressor_load_indices: List[int] = []
    # 第一版将压缩机电源接入 PRS 附近的 IEEE 33 节点，后续可按工程接线校准。
    compressor_power_buses = (7, 13, 30)
    for comp, bus in zip(COMPRESSOR_CONFIGS, compressor_power_buses):
        idx = pp.create_load(net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"{comp.name}_electric_load")
        compressor_load_indices.append(int(idx))

    return PowerNetworkArtifacts(
        net=net,
        load_indices=load_indices,
        renewable_sgen_indices=renewable_sgen_indices,
        ess_storage_indices=ess_storage_indices,
        gfg_sgen_indices=gfg_sgen_indices,
        p2g_load_indices=p2g_load_indices,
        compressor_load_indices=compressor_load_indices,
        ext_grid_index=int(ext_grid_index),
    )
