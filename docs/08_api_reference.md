# API 参考

本章只列出用户和研究人员最常接触的公开接口，不逐行解释内部实现。

## 1. `ElectricGasMultiScaleEnv`

| 项目 | 内容 |
| --- | --- |
| 文件 | `electric_gas_microgrid_single.py` |
| 功能 | 顶层电-气耦合 Gym 风格环境 |
| 构造 | `env = ElectricGasMultiScaleEnv()` |
| 观测 | 172 维 `np.ndarray` |
| 动作 | 28 维 `np.ndarray`，范围 `[-1, 1]` |
| 副作用 | 内部更新 pandapower/pandapipes 网络、ESS SOC、气网状态年龄 |

常用方法：

```python
obs, info = env.reset(seed=42)
next_obs, reward, terminated, truncated, info = env.step(action)
global_obs = env.get_global_state()
manager_obs = env.get_manager_state()
fast_obs = env.get_fast_worker_state()
slow_obs = env.get_slow_worker_state()
```

异常：

- 动作维度不是 28 时抛出 `ValueError`；
- 求解异常会由环境回滚并转换为失败 transition。

## 2. `TrainConfig`

| 项目 | 内容 |
| --- | --- |
| 文件 | `hierarchical_td3_electric_gas.py` |
| 类型 | dataclass |
| 功能 | 保存训练超参数 |

关键字段：

```python
episodes: int = 3
episode_steps: int = 480
manager_interval: int = 40
slow_interval: int = 20
gamma_fast: float = 0.99
batch_size: int = 256
use_transition_model: bool = False
checkpoint_dir: str = "hierarchical_td3_runs"
```

派生属性：

```python
gamma_slow = gamma_fast ** slow_interval
gamma_manager = gamma_fast ** manager_interval
```

## 3. `ObservationBuilder`

| 方法 | 返回维度 | 说明 |
| --- | ---: | --- |
| `manager_obs()` | 172 | Manager 观测 |
| `fast_obs(manager_age_steps)` | 116 | 快速 Worker 观测，补充时间和慢动作摘要 |
| `slow_obs()` | 84 | 慢速 Worker 观测，补充时间预测摘要 |

## 4. `ManagerTD3`

| 项目 | 内容 |
| --- | --- |
| 输入 | Manager 观测 |
| 输出 | 32 维 goal |
| 主要方法 | `select_goal`, `update`, `state_dict`, `load_state_dict` |
| 网络 | Encoder、Actor、Target Actor、Twin Critic、Target Critic |

示例：

```python
goal = agents.manager.select_goal(manager_obs, previous_goal=None, noise_std=0.0, deterministic=True)
```

## 5. `WorkerTD3`

| 项目 | 内容 |
| --- | --- |
| role | `"fast"` 或 `"slow"` |
| 输入 | Worker 观测 + 24 维 worker goal |
| 输出 | 快速 16 维动作或慢速 12 维动作 |
| 主要方法 | `select_action`, `latent_goal_reward`, `update` |

示例：

```python
fast_action = agents.fast.select_action(fast_obs, goal, noise_std=0.0, deterministic=True)
slow_action = agents.slow.select_action(slow_obs, goal, noise_std=0.0, deterministic=True)
```

## 6. Replay Buffer

### `FastReplayBuffer`

每步一条。字段：

```text
obs, next_obs, raw_actions, executed_actions,
reward_external, reward_intrinsic, reward_total,
goals, next_goals, goal_changed, dones
```

### `SlowReplayBuffer`

每慢速片段一条。字段：

```text
obs_start, obs_end, raw_actions, executed_actions,
discounted_reward, goals, next_goals, dones, duration_steps
```

### `ManagerReplayBuffer`

每 Manager 片段一条。字段：

```text
global_obs_start, global_obs_end, manager_goals,
discounted_external_reward, dones, duration_steps
```

## 7. `RunningMeanStd`

在线观测归一化器。

| 方法 | 功能 |
| --- | --- |
| `update(x)` | 更新均值方差 |
| `normalize(x)` | 标准化并裁剪到 `[-10, 10]` |
| `eval()` | 冻结统计 |
| `train()` | 恢复统计更新 |
| `state_dict()` | checkpoint 保存 |

## 8. Checkpoint

```python
save_checkpoint(path, cfg, agents, episode, global_step, best_return)
payload = load_checkpoint(path, agents, device)
```

`payload` 包含：

```text
config, manager, slow, fast, episode, global_step, best_return
```

安全提示：`.pt` 文件通过 `torch.load` 反序列化，只加载可信文件。

## 9. 评估

```python
stats = evaluate_policy(agents, cfg, episodes=1, max_steps=480, seed=12345)
```

返回：

```text
mean_return, std_return, solver_failures,
power_success_rate, gas_success_rate, steps
```

## 10. 物理耦合函数

| 函数 | 文件 | 功能 |
| --- | --- | --- |
| `p2g_power_to_gas_mdot_kg_s` | `project/coupling/energy_conversion.py` | P2G MW 到 kg/s |
| `gfg_power_to_gas_mdot_kg_s` | `project/coupling/energy_conversion.py` | GFG 发电 MW 到耗气 kg/s |
| `dispatch_p2g` | `project/coupling/p2g_model.py` | P2G 功率裁剪和换算 |
| `dispatch_gfg` | `project/coupling/gfg_model.py` | GFG 功率裁剪和换算 |
| `compute_compressor_power_mw` | `project/coupling/compressor_model.py` | 压缩机功耗估计 |
| `project_ess_power` | `project/simulation/safety_projection.py` | ESS SOC 可行功率投影 |
| `project_inverter_action` | `project/simulation/safety_projection.py` | 逆变器视在功率投影 |

