"""Belgian 20 节点 pandapipes 气网构建。

本文件只负责“搭一张气网”。环境每一步会通过返回的元件索引修改 sink/source、
压缩机压力比等表项，再由 CoupledSolver 决定是否运行 pipeflow。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from project.config import ProjectConfig, calibration_warning_messages
from project.data.belgian20_data import (
    COMPRESSOR_CONFIGS,
    GAS_NODES,
    GAS_PIPES,
    GAS_SUPPLIERS,
    N_GAS_JUNCTIONS,
    mm3_per_day_to_kg_per_s,
)
from project.data.ieee33_data import GFG_CONFIGS, P2G_CONFIGS


@dataclass
class GasNetworkArtifacts:
    """pandapipes 元件索引集合。

    保存索引的原因和电网类似：pandapipes 元件存在表格中，后续 step 需要快速
    找到基础气负荷、P2G source、GFG sink、压缩机等行。
    """

    net: object
    base_sink_indices_by_node: Dict[int, int]
    p2g_source_indices: List[int]
    gfg_sink_indices: List[int]
    compressor_indices: List[int]
    ext_grid_indices: List[int]
    prs_control_indices: List[int]
    prs_junction_indices: List[int]
    high_pressure_junctions: List[int]


def _require_pandapipes():
    try:
        import pandapipes as pp
    except ImportError as exc:
        raise RuntimeError("需要安装 pandapipes 才能构建气网。建议使用本仓库 pandapipes 0.14.x。") from exc
    return pp


def build_gas_network(config: ProjectConfig | None = None) -> GasNetworkArtifacts:
    """构建 Belgian 20 节点高压气网、PRS、P2G、GFG 与压缩机。"""

    cfg = config or ProjectConfig()
    pp = _require_pandapipes()
    for message in calibration_warning_messages():
        import logging

        logging.getLogger(__name__).warning(message)

    try:
        net = pp.create_empty_network(fluid=cfg.gas.fluid_name)
    except TypeError:
        net = pp.create_empty_network()
        pp.create_fluid_from_lib(net, cfg.gas.fluid_name, overwrite=True)

    # junction 是气网节点；这里保持 0-based 编号，但显示名使用 Gnode_1..20。
    for node in GAS_NODES:
        pp.create_junction(
            net,
            pn_bar=cfg.gas.source_pressure_bar,
            tfluid_k=cfg.gas.gas_temperature_k,
            name=f"Gnode_{node.node + 1}",
        )

    # ext_grid 是定压气源，用于给高压气网供气。
    ext_grid_indices: List[int] = []
    for supplier in GAS_SUPPLIERS:
        idx = pp.create_ext_grid(
            net,
            junction=supplier.supplier_node,
            p_bar=cfg.gas.source_pressure_bar,
            t_k=cfg.gas.gas_temperature_k,
            name=f"{supplier.name}_pressure_source",
        )
        ext_grid_indices.append(int(idx))

    # 管道参数是暂定等效值，目的是得到可求解的第一版准稳态模型。
    for pipe in GAS_PIPES:
        pp.create_pipe_from_parameters(
            net,
            from_junction=pipe.from_junction,
            to_junction=pipe.to_junction,
            length_km=pipe.length_km,
            diameter_m=pipe.diameter_m,
            k_mm=pipe.roughness_mm,
            sections=1,
            name=pipe.name,
        )

    # 压缩机改变压力比；其耗电量另在电网中作为 load 写入。
    compressor_indices: List[int] = []
    for comp in COMPRESSOR_CONFIGS:
        idx = pp.create_compressor(
            net,
            from_junction=comp.from_junction,
            to_junction=comp.to_junction,
            pressure_ratio=comp.initial_pressure_ratio,
            name=comp.name,
        )
        compressor_indices.append(int(idx))

    # 基础气负荷作为 sink，之后会乘以日内 gas_multiplier。
    base_sink_indices_by_node: Dict[int, int] = {}
    for node in GAS_NODES:
        if node.demand_mm3_per_day <= 0.0:
            continue
        idx = pp.create_sink(
            net,
            junction=node.node,
            mdot_kg_per_s=mm3_per_day_to_kg_per_s(node.demand_mm3_per_day),
            name=f"BaseGasDemand_Gnode_{node.node + 1}",
        )
        base_sink_indices_by_node[node.node] = int(idx)

    p2g_source_indices: List[int] = []
    for p2g in P2G_CONFIGS:
        idx = pp.create_source(
            net,
            junction=p2g.gas_junction,
            mdot_kg_per_s=0.0,
            name=f"{p2g.name}_gas_source",
        )
        p2g_source_indices.append(int(idx))

    prs_control_indices: List[int] = []
    # 旧版 pandapipes 的 pressure_control 与高压环网、压缩机组合在该等效参数下
    # 容易形成奇异矩阵。第一版将 PRS 出口作为环境中的虚拟低压状态处理；
    # GFG 耗气仍写入对应高压节点 sink，代表高压网向调压站供气。
    prs_junction_indices: List[int] = [-1 for _ in GFG_CONFIGS]
    gfg_sink_indices: List[int] = []
    for gfg in GFG_CONFIGS:
        sink_idx = pp.create_sink(
            net,
            junction=gfg.gas_junction,
            mdot_kg_per_s=0.0,
            name=f"{gfg.name}_gas_consumption",
        )
        gfg_sink_indices.append(int(sink_idx))

    return GasNetworkArtifacts(
        net=net,
        base_sink_indices_by_node=base_sink_indices_by_node,
        p2g_source_indices=p2g_source_indices,
        gfg_sink_indices=gfg_sink_indices,
        compressor_indices=compressor_indices,
        ext_grid_indices=ext_grid_indices,
        prs_control_indices=prs_control_indices,
        prs_junction_indices=prs_junction_indices,
        high_pressure_junctions=list(range(N_GAS_JUNCTIONS)),
    )
