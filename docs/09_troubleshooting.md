# 故障排查

本章采用“症状 - 可能原因 - 解决方法”格式。

## 1. 安装问题

### pandapower 或 pandapipes 导入失败

症状：

```text
ModuleNotFoundError: No module named 'pandapower'
ModuleNotFoundError: No module named 'pandapipes'
```

可能原因：

- 未安装依赖；
- 当前终端使用的 Python 不是 `python_3_8` 环境；
- `conda run` 在中文输出下遇到编码问题。

解决方法：

```powershell
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' -m pip install -r requirements.txt
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' -c "import pandapower, pandapipes; print(pandapower.__version__, pandapipes.__version__)"
```

### TensorBoard / protobuf 报错

症状：

```text
TypeError: Descriptors cannot be created directly.
```

可能原因：TensorBoard 与 protobuf 版本不兼容。

解决方法：

```powershell
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' -m pip install "protobuf<=3.20.3"
```

代码中已设置：

```python
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
```

## 2. 环境问题

### 动作维度错误

症状：

```text
ValueError: Action dimension should be 28, got ...
```

原因：`env.step(action)` 输入不是 28 维。

检查：

```python
from electric_gas_microgrid_single import ElectricGasMultiScaleEnv
env = ElectricGasMultiScaleEnv()
print(env.action_dim, env.slow_action_dim, env.fast_action_dim)
```

应输出：

```text
28 12 16
```

### 电力潮流不收敛

症状：

- `info["power_converged"] == False`
- `solver_failed == True`
- episode 提前 `truncated`

可能原因：

- 慢速动作或快速动作极端；
- ESS/GFG/P2G/压缩机组合造成过载或电压越限；
- 随机策略本身不可控。

检查：

```python
print(info["constraint_metrics"])
print(info["action_projection_magnitude"])
print(info["reward_components"]["voltage_violation"])
```

解决：

- 先运行短训练或随机策略检查环境；
- 降低探索噪声；
- 观察 `action_projection_magnitude`；
- 使用 `--training-stage all` 而不是从随机初始化直接 `joint_finetune`。

### 气流不收敛

症状：

- `info["gas_converged"] == False`
- `gas_solve_reason` 频繁变化；
- `gas_state_age` 一直为 0。

可能原因：

- P2G/GFG/压缩机扰动过大；
- Belgian 20 临时管道参数不适合当前负荷；
- 压缩机方向或参数需校准。

检查：

```powershell
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' -m pytest project/tests/test_gas_connectivity.py -q
```

### 状态出现 NaN

代码在 `get_global_state()` 中使用 `np.nan_to_num` 做防护，但如果上游表格异常，仍应排查：

```python
import numpy as np
obs, _ = env.reset(seed=42)
print(np.isnan(obs).any(), np.isinf(obs).any())
```

## 3. 训练问题

### 总回报长期极大负值

可能原因：

- 从随机初始化直接 `joint_finetune`；
- 探索噪声过大；
- 安全投影频繁；
- Critic 被灾难回报污染。

解决：

```powershell
.\scripts\train_final_m40.ps1
```

该脚本无 checkpoint 时默认使用 `--training-stage all`。同时保留了：

```text
target_q_clip_abs
worker_reward_clip_abs
manager_reward_clip_abs
noise_decay_episodes
worker_action_l2_weight
```

### Critic loss 或 Q 值爆炸

检查 TensorBoard：

```powershell
tensorboard --logdir runs
```

重点看：

```text
loss/*critic_loss
loss/*q_value
episode/worker_reward_clips
episode/manager_reward_clips
```

解决：

- 增大 `learning_starts`；
- 降低 `updates-per-step`；
- 降低探索噪声；
- 减小学习率；
- 确保 `target_q_clip_abs` 为正。

### Actor 输出饱和

症状：

- `mean_action_projection` 和 `max_action_projection` 长期很高；
- `applied_action` 与 `raw_action` 差异大。

解决：

- 增大 `lambda_projection`；
- 保持 `worker_action_l2_weight > 0`；
- 减小探索噪声；
- 检查动作映射是否符合设备容量。

### Encoder 不更新

内置测试已经检查 Encoder 参数训练后变化。重新运行：

```powershell
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' hierarchical_td3_electric_gas.py --run-tests
```

### Manager goal 不变化

检查：

```text
manager/goal_change
episode/mean_goal_change
episode/max_goal_change
```

如果接近 0：

- Manager 可能还未开始有效学习；
- `manager_buffer` 样本不足；
- `goal_smoothing` 过高；
- Worker 没有响应 goal。

## 4. 性能问题

### GPU 利用率低

原因：环境 step 主要耗时在 pandapower/pandapipes CPU 求解，神经网络训练只是部分计算。

解决方向：

- 并行环境采样；
- 减少气网求解触发；
- 使用代理模型；
- 先做短 episode 调参。

### TensorBoard 日志过大

解决：

- 增大 `eval_interval`；
- 临时加 `--no-tensorboard`；
- 只分析 `episode_log.csv`。

### 后台训练如何查看

```powershell
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -like '*hierarchical_td3_electric_gas.py*' }
Get-Content runs\m40_stability_v1\train_stderr.log -Tail 30
```
