"""P2G/GFG 能量换算测试。"""

from __future__ import annotations

import pytest

from project.coupling.energy_conversion import gfg_power_to_gas_mdot_kg_s, p2g_power_to_gas_mdot_kg_s


def test_p2g_energy_conversion_units() -> None:
    mdot = p2g_power_to_gas_mdot_kg_s(electric_power_mw=10.0, efficiency=0.7, hhv_mj_per_kg=50.0)
    assert mdot == pytest.approx(0.14)


def test_gfg_energy_conversion_units() -> None:
    mdot = gfg_power_to_gas_mdot_kg_s(electric_power_mw=10.0, efficiency=0.4, hhv_mj_per_kg=50.0)
    assert mdot == pytest.approx(0.5)

