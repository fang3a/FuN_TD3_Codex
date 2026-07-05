"""ESS SOC 更新测试。"""

from __future__ import annotations

from project.config import ESSConfig
from project.simulation.safety_projection import update_ess_soc


def test_ess_charging_increases_soc() -> None:
    ess = ESSConfig("ess", bus=0, max_p_mw=1.0, capacity_mwh=2.0, eta_charge=0.9, eta_discharge=0.9)
    assert update_ess_soc(0.5, p_mw=1.0, ess=ess, dt_hours=0.5) > 0.5


def test_ess_discharging_decreases_soc() -> None:
    ess = ESSConfig("ess", bus=0, max_p_mw=1.0, capacity_mwh=2.0, eta_charge=0.9, eta_discharge=0.9)
    assert update_ess_soc(0.5, p_mw=-1.0, ess=ess, dt_hours=0.5) < 0.5

