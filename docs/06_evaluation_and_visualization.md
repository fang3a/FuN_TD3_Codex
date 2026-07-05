# 评估和可视化

## 1. 评估 checkpoint

评估脚本：

```text
evaluate_hierarchical_agent.py
```

命令：

```powershell
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' evaluate_hierarchical_agent.py --checkpoint runs\m40_stability_v1\<run_id>\latest_checkpoint.pt --episodes 1 --episode-steps 480 --device cpu
```

包装脚本：

```powershell
.\scripts\evaluate_checkpoint.ps1 -Checkpoint runs\m40_stability_v1\<run_id>\latest_checkpoint.pt -Episodes 1 -EpisodeSteps 480 -Device cpu
```

Linux/macOS：

```bash
python evaluate_hierarchical_agent.py --checkpoint runs/m40_stability_v1/<run_id>/latest_checkpoint.pt --episodes 1 --episode-steps 480 --device cpu
```

评估时使用 `evaluate_policy()`，会关闭探索噪声，并暂时冻结 normalizer 统计量，避免评估过程污染训练时的观测归一化。

## 2. 评价指标

建议至少报告：

| 指标 | 来源 |
| --- | --- |
| episode return | `evaluate_policy()` 或 `episode_log.csv` |
| solver failures | `info["solver_failed"]` / CSV |
| power success rate | `info["power_converged"]` |
| gas success rate | `info["gas_converged"]` |
| 电压最小/最大值 | `constraint_metrics["vm_min_pu"]`, `vm_max_pu` |
| 电压 RMS 偏差 | `constraint_metrics["voltage_rms_deviation_pu"]` |
| 最大线路负载率 | `constraint_metrics["max_line_loading_percent"]` |
| 高压气网压力范围 | `high_pressure_min_bar`, `high_pressure_max_bar` |
| 高压气网 RMS 偏差 | `high_pressure_rms_deviation_bar` |
| PRS 压力范围 | `prs_pressure_min_bar`, `prs_pressure_max_bar` |
| PRS RMS 偏差 | `prs_pressure_rms_deviation_bar` |
| SOC 范围 | `soc_min`, `soc_max` |
| 购电量（统计量，非默认优化目标） | `grid_purchase_mwh` |
| 购气量（统计量，非默认优化目标） | `gas_purchase_kg` |
| 安全投影幅度 | `action_projection_magnitude` |
| 慢速/快速投影分解 | `mean_slow_action_projection`, `mean_fast_action_projection` |
| ESS guard 介入幅度 | `mean_ess_action_guard`, `max_ess_action_guard` |
| 气网求解次数 | `gas_solve_count` |

## 3. 随机策略可视化

模块化随机策略入口：

```powershell
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' -m project.run_random_policy --output-dir project/outputs/random_policy
```

输出：

| 文件 | 内容 |
| --- | --- |
| `random_policy_timeseries.csv` | 逐 3 分钟步记录 |
| `random_policy_dashboard.png` | 电压、线路、气压、PRS、SOC、气网求解事件 |
| `random_policy_costs.png` | 奖励和主要成本/惩罚分解 |

只运行仿真不画图：

```powershell
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' -m project.run_random_policy --no-plots
```

## 4. 拓扑总览图

```powershell
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' -m project.plot_coupled_topology
```

默认输出：

```text
project/outputs/topology/coupled_network_overview.png
```

图中电网和气网分侧绘制，GFG、P2G、压缩机电耗以虚线耦合连接表示。

## 5. 读取 CSV 并画曲线

示例：读取训练 `episode_log.csv`。

```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

csv_path = Path("runs/m40_stability_v1/all_stages_YYYYMMDD_HHMMSS/fast_pretrain/<run_id>/episode_log.csv")
df = pd.read_csv(csv_path)

plt.figure()
plt.plot(df["episode"], df["episode_return"])
plt.xlabel("episode")
plt.ylabel("episode return")
plt.grid(True)
plt.show()
```

示例：读取随机策略时序 CSV 中的电压边界。

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("project/outputs/random_policy/random_policy_timeseries.csv")
plt.plot(df["step"], df["vm_min_pu"], label="vm_min_pu")
plt.plot(df["step"], df["vm_max_pu"], label="vm_max_pu")
plt.axhline(0.95, color="red", linestyle="--")
plt.axhline(1.05, color="red", linestyle="--")
plt.legend()
plt.show()
```

## 6. 图表解释

- 电压图：正常情况下应尽量位于 0.95 至 1.05 pu 内。
- 线路负载图：超过 100% 表示过载惩罚。
- 气压图：高压节点应尽量位于 30 至 70 bar 内。
- PRS 图：当前 PRS 是虚拟低压出口，目标范围为 1.35 至 1.65 bar。
- SOC 图：硬范围为 0.10 至 0.95，软惩罚区间为 0.20 至 0.90。
- 事件点：气网不是每步求解，事件点表示实际运行 pipeflow 的步。

## 7. 公平比较基线

建议比较：

| 基线 | 说明 |
| --- | --- |
| 随机策略 | `project.run_random_policy` |
| 规则慢速 + 快速 Worker | 近似 `fast_pretrain` |
| 规则 Manager + 双 Worker | `slow_pretrain` 后组合 |
| Manager only | `manager_train` |
| 完整分层 TD3 | `all` 或 `joint_finetune` |

公平比较应使用相同外生曲线、随机种子、episode 步数和评估脚本。

## 8. 多随机种子

建议至少 5 个随机种子：

```text
42, 2026, 3407, 8801, 10001
```

报告均值和标准差，不只报告单次最优值。
