"""环境烟雾测试与失败回滚测试。"""

from __future__ import annotations

import numpy as np
import pytest

from project.config import ProjectConfig, TimeConfig


def test_random_policy_short_episode_does_not_crash() -> None:
    pytest.importorskip("pandapower")
    pytest.importorskip("pandapipes")
    from project.envs.electric_gas_multiscale_env import ElectricGasMultiScaleEnv

    env = ElectricGasMultiScaleEnv(ProjectConfig(time=TimeConfig(steps_per_day=4, slow_action_interval_steps=2)))
    obs, info = env.reset(seed=3)
    assert obs.shape == env.observation_space.shape
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert np.isfinite(reward)
        assert obs.shape == env.observation_space.shape


def test_solver_failure_rolls_back_soc() -> None:
    pytest.importorskip("pandapower")
    pytest.importorskip("pandapipes")
    from project.envs.electric_gas_multiscale_env import ElectricGasMultiScaleEnv

    env = ElectricGasMultiScaleEnv(ProjectConfig(time=TimeConfig(steps_per_day=3, slow_action_interval_steps=2)))
    env.reset(seed=4)
    soc_before = env.ess_soc.copy()

    def fail(*args, **kwargs):
        raise RuntimeError("forced failure")

    env.solver.solve_step = fail  # type: ignore[method-assign]
    _, _, _, _, info = env.step(env.action_space.sample())
    assert info["solver_failed"] is True
    assert np.allclose(env.ess_soc, soc_before)

