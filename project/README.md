# 电-气耦合微电网仿真模型

## 现有文件问题摘要

`woker_home/主建模.py` 的主要问题：

- 时间尺度仍为 5 分钟、288 步，不符合 3 分钟、480 步要求。
- 同时使用 MultiNet 控制器和手工耦合，P2G/GFG 容易重复换算或错序更新。
- P2G 与 GFG 使用固定经验系数做 MW 与 kg/s 换算，应统一使用 HHV。
- P2G 被混称为电驱压缩机；压缩机不应产生天然气。
- ESS 先执行越限动作再裁剪 SOC，未在写入 pandapower 前按 SOC 投影可行功率。
- 快速动作包含阀门和压缩机微调，不符合快速 Worker 只控逆变器的接口边界。
- 调压站 1.5 bar 出口与高压气网 30-70 bar 约束混用，奖励和状态归一化不一致。
- 将 `sum(abs(pipe_flow))` 作为气损不物理；当前默认奖励不再优化购气成本，购气量仅作为统计量保留。
- “linepack” 被描述为动态管存；第一版已改为 `equivalent_linepack_indicator`。

`woker_home/建模练习2 扩展.py` 可参考：

- HHV 统一换算思路。
- OutputWriter、时序结果和能量转换校验方式。
- 但其 5 节点小网络不直接搬入本项目。

Excel `IEEE 39-node power system and Belgian 20-node gas system.xlsx`：

- 本项目只读取 Belgian 20 节点气网、供应商、管道端点和 Coupling 参考。
- IEEE 39 电网参数没有混入 IEEE 33 配电网。
- Wmn/Kmn 仅作为文献稳态流量系数保留，不当作长度、管径或粗糙度。

## 目录

```text
project/
  config.py
  data/
    ieee33_data.py
    belgian20_data.py
    profile_generator.py
  networks/
    power_network.py
    gas_network.py
    topology_validator.py
  coupling/
    energy_conversion.py
    gfg_model.py
    p2g_model.py
    compressor_model.py
  simulation/
    coupled_solver.py
    event_scheduler.py
    safety_projection.py
  envs/
    electric_gas_multiscale_env.py
  tests/
    test_power_network.py
    test_gas_connectivity.py
    test_energy_conversion.py
    test_ess_soc.py
    test_action_projection.py
    test_time_scale.py
    test_env_smoke.py
  run_random_policy.py
  requirements.txt
```

## 运行命令

在安装依赖后，从仓库根目录运行：

```powershell
python -m pytest project/tests -q
python -m project.run_random_policy
```

本机已实测的环境：

```powershell
D:\anaconda\anaconda\envs\python_3_8\python.exe
pandapower 2.14.10
pandapipes 0.10.0
numpy 1.23.5
pandas 2.0.3
```

推荐直接用该环境运行，避免 `conda run` 在中文/特殊字符输出下触发 GBK 编码问题：

```powershell
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' -m pytest project/tests -q
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' -m project.run_random_policy
```

## 待校准参数

- Belgian 20 管道长度 `length_km`。
- Belgian 20 管道直径 `diameter_m`。
- Belgian 20 管道粗糙度 `roughness_mm`。
- Wmn/Kmn 到 pandapipes 物理管道参数的标定关系。
- 供应商 Sup 1-6 到具体 gas node 的映射。
- 气体标准密度 `STANDARD_GAS_DENSITY_KG_PER_M3`。
- 气体高位热值 `GasConfig.hhv_mj_per_kg`。
- 压缩机拓扑、方向和物理参数。
- 压缩机 `nominal_flow_kg_s`、等熵效率、最大功率和入口/出口压力约束。
- IEEE 33 配网中压缩机电源接入母线。
- GFG 在 IEEE 33 母线和 Belgian 20 气节点之间的真实耦合映射。
- P2G 在 IEEE 33 母线和 Belgian 20 气节点之间的真实耦合映射。
- PRS 位置、数量、低压出口压力设定和最大供气能力。
- 天然气采购价格与供应商容量约束的训练成本标定。

## 可视化输出

随机策略脚本默认生成一份 CSV 和两张图：

```powershell
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' -m project.run_random_policy --output-dir project/outputs/random_policy
```

输出文件：

- `random_policy_timeseries.csv`：每 3 分钟一步的奖励、越限、购电、购气、SOC、气网事件等指标。
- `random_policy_dashboard.png`：电压包络、线路负载率、高压气网压力、PRS 出口压力、SOC 和气网事件触发。
- `random_policy_costs.png`：步奖励、累计购电、累计购气和主要惩罚/成本分解。

如果只想跑仿真不画图：

```powershell
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' -m project.run_random_policy --no-plots
```

生成电网-气网耦合总体拓扑图：

```powershell
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' -m project.plot_coupled_topology
```

默认输出：

- `project/outputs/topology/coupled_network_overview.png`
