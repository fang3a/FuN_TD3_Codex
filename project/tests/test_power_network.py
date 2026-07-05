"""IEEE 33 电网基础潮流测试。"""

from __future__ import annotations

import pytest


def test_power_network_base_powerflow_converges() -> None:
    pp = pytest.importorskip("pandapower")
    from project.networks.power_network import build_power_network

    artifacts = build_power_network()
    pp.runpp(artifacts.net, algorithm="nr", init="flat")
    assert artifacts.net.converged
    assert len(artifacts.net.bus) == 33

