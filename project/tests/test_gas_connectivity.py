"""Belgian 20 气网连通性测试。"""

from __future__ import annotations

from project.data.belgian20_data import GAS_NODES, GAS_SUPPLIERS
from project.data.ieee33_data import GFG_CONFIGS, P2G_CONFIGS
from project.networks.topology_validator import validate_belgian20_topology


def test_gas_topology_connectivity() -> None:
    result = validate_belgian20_topology()
    assert result.ok, result.errors


def test_all_sink_and_source_nodes_reachable_from_supplier() -> None:
    result = validate_belgian20_topology()
    demand_nodes = {node.node for node in GAS_NODES if node.demand_mm3_per_day > 0.0}
    supplier_nodes = {supplier.supplier_node for supplier in GAS_SUPPLIERS}
    p2g_nodes = {p.gas_junction for p in P2G_CONFIGS}
    gfg_nodes = {g.gas_junction for g in GFG_CONFIGS}
    assert demand_nodes | supplier_nodes | p2g_nodes | gfg_nodes <= result.source_component_nodes

