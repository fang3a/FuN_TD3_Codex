"""可视化输出测试。"""

from __future__ import annotations

import pytest


def test_save_episode_artifacts(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from project.visualization import save_episode_artifacts

    records = []
    for i in range(4):
        records.append(
            {
                "step": i + 1,
                "hour": 0.05 * (i + 1),
                "reward": -float(i),
                "power_converged": 1.0,
                "gas_converged": 1.0,
                "gas_solved_this_step": float(i % 2 == 0),
                "slow_action_applied": float(i == 0),
                "gas_state_age": float(i),
                "vm_min_pu": 0.96,
                "vm_max_pu": 1.02,
                "max_line_loading_percent": 50.0,
                "high_pressure_min_bar": 45.0,
                "high_pressure_max_bar": 65.0,
                "prs_pressure_min_bar": 1.50,
                "prs_pressure_max_bar": 1.50,
                "soc_min": 0.30,
                "soc_max": 0.70,
                "grid_purchase_mwh": 0.1,
                "gas_purchase_kg": 10.0,
                "voltage_violation_cost": 0.0,
                "high_pressure_violation_cost": 0.0,
                "prs_pressure_violation_cost": 0.0,
                "line_overload_cost": 0.0,
                "renewable_curtailment_cost": 1.0,
                "compressor_energy_cost": 0.5,
            }
        )

    artifacts = save_episode_artifacts(records, output_dir=tmp_path)
    assert artifacts["csv"].exists()
    assert artifacts["dashboard"].exists()
    assert artifacts["costs"].exists()
    assert artifacts["dashboard"].stat().st_size > 0
    assert artifacts["costs"].stat().st_size > 0


def test_save_coupled_topology_overview(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from project.visualization import save_coupled_topology_overview

    path = save_coupled_topology_overview(tmp_path / "topology.png")
    assert path.exists()
    assert path.stat().st_size > 0
