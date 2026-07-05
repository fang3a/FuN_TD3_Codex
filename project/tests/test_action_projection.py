"""动作安全投影测试。"""

from __future__ import annotations

import numpy as np

from project.config import ESSConfig, RenewableConfig
from project.simulation.safety_projection import project_ess_power, project_inverter_action


def test_soc_boundary_action_is_projected() -> None:
    ess = ESSConfig("ess", bus=0, max_p_mw=2.0, capacity_mwh=1.0, eta_charge=0.9, eta_discharge=0.9)
    result = project_ess_power(requested_p_mw=-2.0, soc=ess.soc_min, ess=ess, dt_hours=0.25)
    assert result.applied_action >= -1e-12
    assert result.projection_magnitude > 0.0
    assert result.hit_boundary


def test_inverter_apparent_power_constraint() -> None:
    inv = RenewableConfig("pv", bus=0, kind="pv", capacity_mw=1.0, s_rated_mva=1.05, max_curtailment=0.5)
    result = project_inverter_action(inv, p_available_mw=1.0, requested_q_mvar=1.0, requested_curtailment=0.0)
    assert result.p_actual_mw**2 + result.q_actual_mvar**2 <= inv.s_rated_mva**2 + 1e-9
    assert abs(result.q_actual_mvar) <= result.q_limit_mvar + 1e-9

