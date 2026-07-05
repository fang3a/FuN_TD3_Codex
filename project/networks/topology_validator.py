"""气网拓扑与参数校验。

建网前先做轻量校验，可以把“节点越界、孤立、重复管道”等问题提前变成
清晰错误信息，而不是等 pandapipes 在数值求解时抛出难定位的异常。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set, Tuple

from project.data.belgian20_data import COMPRESSOR_CONFIGS, GAS_NODES, GAS_PIPES, GAS_SUPPLIERS, N_GAS_JUNCTIONS
from project.data.ieee33_data import GFG_CONFIGS, P2G_CONFIGS


@dataclass
class TopologyValidationResult:
    """拓扑校验结果。"""

    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    source_component_nodes: Set[int] = field(default_factory=set)

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ValueError("气网拓扑校验失败: " + "; ".join(self.errors))


def _build_adjacency(edges: Iterable[Tuple[int, int]]) -> Dict[int, Set[int]]:
    """把管道/压缩机边转换成无向邻接表，用于连通性检查。"""

    adjacency = {node: set() for node in range(N_GAS_JUNCTIONS)}
    for u, v in edges:
        adjacency.setdefault(u, set()).add(v)
        adjacency.setdefault(v, set()).add(u)
    return adjacency


def _component_from_sources(adjacency: Dict[int, Set[int]], sources: Iterable[int]) -> Set[int]:
    """从气源节点出发，找出所有可连通节点。"""

    visited: Set[int] = set()
    stack = list(sources)
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(sorted(adjacency.get(node, set()) - visited))
    return visited


def validate_belgian20_topology() -> TopologyValidationResult:
    """校验 Belgian 20 数据在 pandapipes 建网前是否物理可用。"""

    errors: List[str] = []
    warnings: List[str] = []

    # 逐条检查管道端点、参数正值和是否存在未声明的重复边。
    pipe_edge_seen: Dict[Tuple[int, int], str] = {}
    pipe_edges: List[Tuple[int, int]] = []
    for pipe in GAS_PIPES:
        if pipe.from_junction == pipe.to_junction:
            errors.append(f"{pipe.name} 起点等于终点")
        if not (0 <= pipe.from_junction < N_GAS_JUNCTIONS and 0 <= pipe.to_junction < N_GAS_JUNCTIONS):
            errors.append(f"{pipe.name} 连接节点越界")
        if pipe.length_km <= 0 or pipe.diameter_m <= 0 or pipe.roughness_mm <= 0:
            errors.append(f"{pipe.name} 长度、直径、粗糙度必须为正")
        key = tuple(sorted((pipe.from_junction, pipe.to_junction)))
        if key in pipe_edge_seen and not pipe.allow_parallel:
            errors.append(f"{pipe.name} 与 {pipe_edge_seen[key]} 重复，且未声明并联管道")
        if key in pipe_edge_seen and pipe.allow_parallel:
            warnings.append(f"{pipe.name} 与 {pipe_edge_seen[key]} 为声明允许的并联管道")
        pipe_edge_seen.setdefault(key, pipe.name)
        pipe_edges.append((pipe.from_junction, pipe.to_junction))

    # 压缩机也提供连通边，所以拓扑连通性应同时考虑管道和压缩机。
    compressor_edges: List[Tuple[int, int]] = []
    for comp in COMPRESSOR_CONFIGS:
        if comp.from_junction == comp.to_junction:
            errors.append(f"{comp.name} 压缩机起点等于终点")
        if not (0 <= comp.from_junction < N_GAS_JUNCTIONS and 0 <= comp.to_junction < N_GAS_JUNCTIONS):
            errors.append(f"{comp.name} 压缩机连接节点越界")
        if comp.max_pressure_ratio <= 1.0 or comp.max_power_mw <= 0.0:
            errors.append(f"{comp.name} 压缩机最大压力比和最大功率必须有效")
        compressor_edges.append((comp.from_junction, comp.to_junction))

    source_nodes = {supplier.supplier_node for supplier in GAS_SUPPLIERS}
    if not source_nodes:
        errors.append("至少需要一个气网 ext_grid/source 节点")

    # 所有带负荷、气源或耦合设备的节点，都应能从至少一个气源连通到。
    adjacency = _build_adjacency(pipe_edges + compressor_edges)
    source_component = _component_from_sources(adjacency, source_nodes)

    demand_nodes = {node.node for node in GAS_NODES if node.demand_mm3_per_day > 0.0}
    p2g_nodes = {p2g.gas_junction for p2g in P2G_CONFIGS}
    gfg_nodes = {gfg.gas_junction for gfg in GFG_CONFIGS}
    required_nodes = demand_nodes | p2g_nodes | gfg_nodes | source_nodes
    isolated_required = sorted(required_nodes - source_component)
    if isolated_required:
        errors.append(f"以下带 source/sink/coupling 的节点不在有气源连通分量内: {isolated_required}")

    for node in range(N_GAS_JUNCTIONS):
        if node not in adjacency:
            continue
        if len(adjacency[node]) == 0 and node in required_nodes:
            errors.append(f"节点 {node} 带 source/sink/coupling 但孤立")

    return TopologyValidationResult(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        source_component_nodes=source_component,
    )
