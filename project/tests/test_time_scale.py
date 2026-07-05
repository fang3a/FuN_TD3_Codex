"""时间尺度与索引测试。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from project.config import ProjectConfig, TimeConfig
from project.data.profile_generator import generate_daily_profiles, profile_at


def test_profile_one_day_index_does_not_overflow() -> None:
    time_cfg = TimeConfig()
    profiles = generate_daily_profiles(time_cfg, seed=1)
    for idx in (0, 479, 480, 500):
        values = profile_at(profiles, idx)
        assert "next_hour_load_multiplier" in values


def test_slow_action_only_every_20_steps() -> None:
    pytest.importorskip("pandapower")
    pytest.importorskip("pandapipes")
    from project.envs.electric_gas_multiscale_env import ElectricGasMultiScaleEnv

    config = ProjectConfig(time=TimeConfig(steps_per_day=22, slow_action_interval_steps=20))
    env = ElectricGasMultiScaleEnv(config)
    env.reset(seed=2)
    flags = []
    for _ in range(21):
        _, _, _, _, info = env.step(env.action_space.sample())
        flags.append(info["slow_action_applied"])
    assert flags[0] is True
    assert all(flag is False for flag in flags[1:20])
    assert flags[20] is True

