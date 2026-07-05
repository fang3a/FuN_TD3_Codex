# Multi-Timescale Hierarchical Reinforcement Learning for Coupled Electric-Gas Microgrids

# 面向电-气耦合微电网的多时间尺度分层强化学习控制

本项目是一个面向研究与算法原型验证的电-气耦合微电网控制项目。代码以 IEEE 33 节点配电网和 Belgian 20 节点高压天然气网络为底座，使用 `pandapower` 进行电力潮流计算，使用 `pandapipes` 进行天然气稳态/准稳态流计算，并在此基础上构建 Gymnasium 风格的强化学习环境。环境中包含 ESS 储能、光伏/风电逆变器、燃气发电机组 GFG、电转气 P2G、电驱压缩机、调压站 PRS、事件驱动气网求解和安全动作投影。

项目的核心问题是：电力侧逆变器需要在 3 分钟快速尺度上响应电压和线路约束，而 ESS、GFG、P2G、压缩机等设备更适合在 1 小时慢速尺度上调度；更高层的协调目标则可以每约 2 小时更新一次。因此，训练程序 [hierarchical_td3_electric_gas.py](hierarchical_td3_electric_gas.py) 实现了一个 FuN-inspired Hierarchical Multi-Timescale TD3：Manager 输出 32 维组合式 goal，慢速 Worker 输出 12 维慢速动作，快速 Worker 输出 16 维快速动作。该方法不是严格复现原始 FuN 论文，而是面向电-气耦合场景改造的多时间尺度分层 TD3 原型。

当前项目适合学术研究、算法验证和可复现实验，不应直接用于未经校准和安全认证的真实电网/气网控制。

## 核心特点

- IEEE 33 节点、12.66 kV 辐射型配电网，Slack 母线为 0。
- Belgian 20 节点高压气网，包含 23 条管道、6 个气源和 3 台电驱压缩机。
- 3 分钟快速步长、每天 480 步；慢速动作每 20 步更新一次；Manager 默认每 40 步更新一次。
- 动作空间 28 维：慢速 12 维、快速 16 维，均归一化到 `[-1, 1]`。
- 观测空间 172 维；训练中 Manager/fast/slow 观测维度分别为 172、116、84。
- `info` 返回 `raw_action`、`applied_action`、`action_projection_magnitude`，Critic 使用安全投影后的实际执行动作。
- 三个 Replay Buffer：`FastReplayBuffer`、`SlowReplayBuffer`、`ManagerReplayBuffer`。
- TD3 双 Critic、Target Policy Smoothing、Delayed Policy Update、Soft Target Update。
- 可选 Transition Model：`--use-transition-model`。
- 分阶段训练：`fast_pretrain`、`slow_pretrain`、`manager_train`、`joint_finetune`、`all`。
- CSV 与 TensorBoard 日志、checkpoint、无探索噪声评估、随机策略可视化。

## 项目结构

```text
FuN_TD3_project/
  electric_gas_microgrid_single.py      # 顶层单文件电-气耦合环境，训练主程序实际导入此文件
  hierarchical_td3_electric_gas.py      # 分层 TD3 训练、Agent、Replay Buffer、checkpoint 和内置测试
  evaluate_hierarchical_agent.py        # 无探索噪声 checkpoint 评估脚本
  requirements.txt                      # 已验证环境的主要依赖版本
  README.md                             # 项目入口文档
  MIGRATION_MANIFEST.md                 # 从 pandapipes-develop 迁移来的文件清单
  scripts/
    check_project.ps1                   # 自检脚本
    debug_train.ps1                     # 短时调试训练
    train_trial_m40.ps1                 # 40 episode 四阶段试训
    train_final_m40.ps1                 # 优化后的正式训练脚本
    evaluate_checkpoint.ps1             # 评估脚本包装
  project/                              # 模块化建模版本，用于测试、随机策略和可视化
    config.py
    data/
    networks/
    coupling/
    simulation/
    envs/
    visualization/
    tests/
    run_random_policy.py
    plot_coupled_topology.py
  docs/                                 # 中文教程与技术文档
```

## 快速开始

### 1. 环境准备

本项目已在 Windows + Conda 环境中验证：

```powershell
D:\anaconda\anaconda\envs\python_3_8\python.exe
```

关键版本：

```text
Python 3.8
numpy 1.23.5
pandas 2.0.3
pandapower 2.14.10
pandapipes 0.10.0
torch 1.12.1
protobuf <= 3.20.3
```

Windows PowerShell：

```powershell
cd D:\project\FuN_TD3_project
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' -m pip install -r requirements.txt
```

Linux/macOS：

```bash
cd /path/to/FuN_TD3_project
python -m pip install -r requirements.txt
```

说明：Linux/macOS 命令按相同 Python 包接口给出，但本仓库当前实测环境是 Windows。

### 2. 基础自检

Windows PowerShell：

```powershell
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' -m py_compile electric_gas_microgrid_single.py hierarchical_td3_electric_gas.py evaluate_hierarchical_agent.py
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' -m pytest project/tests -q
```

Linux/macOS：

```bash
python -m py_compile electric_gas_microgrid_single.py hierarchical_td3_electric_gas.py evaluate_hierarchical_agent.py
python -m pytest project/tests -q
```

本次文档生成前已验证：

```text
py_compile: passed
project/tests: 15 passed, 1 warning
```

### 3. 最小环境示例

```python
from electric_gas_microgrid_single import ElectricGasMultiScaleEnv

env = ElectricGasMultiScaleEnv()
obs, info = env.reset(seed=42)
action = env.action_space.sample()
next_obs, reward, terminated, truncated, info = env.step(action)

print(obs.shape)                 # (172,)
print(env.action_dim)            # 28
print(reward)
print(info["slow_action_applied"])
print(info["action_projection_magnitude"])
```

### 4. 随机策略一天测试

Windows PowerShell：

```powershell
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' -m project.run_random_policy --output-dir project/outputs/random_policy
```

Linux/macOS：

```bash
python -m project.run_random_policy --output-dir project/outputs/random_policy
```

输出包括 `random_policy_timeseries.csv`、`random_policy_dashboard.png` 和 `random_policy_costs.png`。

### 5. 短时训练

```powershell
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' hierarchical_td3_electric_gas.py --training-stage joint_finetune --episodes 1 --episode-steps 20 --batch-size 8 --learning-starts 5 --use-transition-model --device cpu --checkpoint-dir runs\debug
```

Linux/macOS：

```bash
python hierarchical_td3_electric_gas.py --training-stage joint_finetune --episodes 1 --episode-steps 20 --batch-size 8 --learning-starts 5 --use-transition-model --device cpu --checkpoint-dir runs/debug
```

### 6. 完整四阶段训练

如果没有可确认来源的 checkpoint，推荐使用四阶段训练，而不是直接从随机初始化进入 `joint_finetune`：

```powershell
.\scripts\train_final_m40.ps1
```

等价核心命令：

```powershell
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' hierarchical_td3_electric_gas.py --training-stage all --episodes 300 --episode-steps 480 --manager-interval 40 --batch-size 256 --learning-starts 1000 --updates-per-step 1 --slow-update-interval-steps 5 --manager-update-interval-steps 20 --target-noise 0.08 --target-noise-clip 0.20 --target-q-clip-abs 200000 --use-transition-model --eval-interval 10 --eval-episodes 1 --lambda-projection 10.0 --worker-action-l2-weight 0.01 --projection-imitation-weight 5.0 --checkpoint-dir runs\m40_stability_v1
```

### 7. 从 checkpoint 继续训练

`torch.load` 会反序列化 pickle 格式的 checkpoint。只加载你信任来源的 `.pt` 文件。

```powershell
.\scripts\train_final_m40.ps1 -LoadCheckpoint checkpoints\latest_checkpoint.pt
```

### 8. 无噪声评估

```powershell
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' evaluate_hierarchical_agent.py --checkpoint runs\m40_stability_v1\<run_id>\latest_checkpoint.pt --episodes 1 --episode-steps 480 --device cpu
```

### 9. TensorBoard

```powershell
tensorboard --logdir runs
```

关注曲线：`episode/return`、`eval/return`、`components/*`、`constraints/*`、`reward/*`、`loss/*`、`solver/*`、`episode/mean_action_projection`、`episode/mean_slow_action_projection`、`episode/mean_fast_action_projection`、`episode/mean_ess_action_guard`。

当前默认奖励为稳定性优先：`gas_price=0`，`grid_energy_price=0`，新增 `voltage_deviation`、`high_pressure_deviation`、`prs_pressure_deviation`。重点查看 `episode/mean_voltage_rms_deviation_pu`、`episode/mean_high_pressure_rms_deviation_bar`、`episode/mean_prs_pressure_rms_deviation_bar`。

## 当前限制

- Belgian 20 气网的管道长度、管径、粗糙度来自临时等效参数，`Wmn/Kmn` 只保留为参考，不作为 pandapipes 物理管道数据。
- 气体 HHV、密度、压缩机流量、效率和功率上限仍需基于真实系统校准。
- 当前气网是事件驱动的稳态/准稳态 pipeflow，不是完整瞬态气网。
- `equivalent_linepack_indicator` 是由当前压力场和管道体积估计的准稳态指标，不等于严格动态 linepack。
- 训练程序未实现 HIRO 式 goal relabeling 或 off-policy correction，只预留了组合式 goal 与 Transition Model 接口。
- 当前没有把旧线程中的 300 episode 训练写成最终性能结论；新的优化训练仍需多随机种子重复实验。

## 文档导航

- [docs/index.md](docs/index.md)：文档导航首页
- [docs/01_project_overview.md](docs/01_project_overview.md)：项目总览
- [docs/02_system_model.md](docs/02_system_model.md)：电网、气网和耦合设备模型
- [docs/03_environment_interface.md](docs/03_environment_interface.md)：环境接口和动作/状态说明
- [docs/04_hierarchical_rl_algorithm.md](docs/04_hierarchical_rl_algorithm.md)：分层 TD3 算法
- [docs/05_training_tutorial.md](docs/05_training_tutorial.md)：训练教程
- [docs/06_evaluation_and_visualization.md](docs/06_evaluation_and_visualization.md)：评估和可视化
- [docs/07_code_architecture.md](docs/07_code_architecture.md)：代码架构
- [docs/08_api_reference.md](docs/08_api_reference.md)：API 参考
- [docs/09_troubleshooting.md](docs/09_troubleshooting.md)：故障排查
- [docs/10_reproducibility.md](docs/10_reproducibility.md)：可复现性
- [docs/11_research_extension.md](docs/11_research_extension.md)：研究扩展方向
