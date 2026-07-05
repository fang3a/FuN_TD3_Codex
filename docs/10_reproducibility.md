# 可复现性

## 1. 随机种子

训练脚本提供：

```text
--seed
```

`set_seed()` 会设置 Python `random`、NumPy、PyTorch CPU 和 CUDA 随机种子。由于 pandapower/pandapipes 求解、GPU 浮点和系统线程调度可能存在微小差异，仍建议多随机种子报告均值和标准差。

推荐种子：

```text
42, 2026, 3407, 8801, 10001
```

## 2. 环境记录

每次实验建议记录：

```powershell
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' -c "import sys, numpy, pandas, pandapower, pandapipes, torch; print(sys.version); print(numpy.__version__, pandas.__version__, pandapower.__version__, pandapipes.__version__, torch.__version__)"
```

硬件信息建议记录：

```powershell
Get-ComputerInfo | Select-Object CsName, OsName, OsVersion
Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM
```

## 3. 配置保存

每个训练 run 会保存：

```text
config.json
episode_log.csv
latest_checkpoint.pt
best_*.pt
tb/events.out.tfevents...
```

`config.json` 是复现实验的第一入口，包含所有 CLI 参数和训练默认值。

## 4. 标准实验记录模板

```yaml
experiment_name:
date:
code_version:
git_commit: "not available if directory is not a git repo"
python_executable:
python_version:
os:
gpu:
seed:
training_stage:
episodes:
episode_steps:
manager_interval:
slow_interval:
gamma_fast:
batch_size:
learning_starts:
use_transition_model:
reward_weights:
checkpoint_dir:
loaded_checkpoint:
notes:
```

## 5. 多随机种子流程

```powershell
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' hierarchical_td3_electric_gas.py --seed 42 --training-stage all --episodes 300 --checkpoint-dir runs\seed_42
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' hierarchical_td3_electric_gas.py --seed 2026 --training-stage all --episodes 300 --checkpoint-dir runs\seed_2026
```

更完整时，应保持所有非 seed 参数完全一致。

## 6. 结果汇总

示例：汇总多个 `episode_log.csv`。

```python
from pathlib import Path
import pandas as pd

rows = []
for csv_path in Path("runs").rglob("episode_log.csv"):
    df = pd.read_csv(csv_path)
    if "eval_return" in df:
        evals = pd.to_numeric(df["eval_return"], errors="coerce").dropna()
        if len(evals):
            rows.append({
                "run": str(csv_path.parent),
                "best_eval": evals.max(),
                "last_eval": evals.iloc[-1],
                "solver_failures": df["solver_failures"].sum(),
            })

summary = pd.DataFrame(rows)
print(summary)
```

## 7. 论文表格建议

至少报告：

| 指标 | 统计方式 |
| --- | --- |
| return | 多种子均值 ± 标准差 |
| solver failures | 总数或每 episode 平均 |
| voltage violation | 每天累计 |
| gas pressure violation | 每天累计 |
| renewable curtailment | MWh |
| grid purchase | MWh |
| gas purchase | kg |
| action projection | 平均和最大值 |

## 8. 文件保存建议

不要只保存 checkpoint。建议每个实验目录至少包含：

```text
config.json
episode_log.csv
stderr/stdout log
latest_checkpoint.pt
best checkpoint
README或notes.md
```

## 9. 当前复现风险

- `D:\project\FuN_TD3_project` 当前不是 Git 仓库，无法自动记录 commit。
- Belgian 20 气网仍含临时等效参数。
- 旧线程训练结果和当前优化训练不应混合作为同一实验结论。
- 如果后台训练未完成，不应引用其中间结果作为最终性能。

