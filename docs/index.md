# 文档导航

本目录是 `FuN_TD3_project` 的中文技术文档。文档以当前工作区真实代码为唯一事实来源，主要面向刚接触电-气耦合微电网、pandapower/pandapipes 和分层强化学习的读者。

推荐阅读顺序：

1. [项目总览](01_project_overview.md)
2. [系统物理模型](02_system_model.md)
3. [环境接口教程](03_environment_interface.md)
4. [分层强化学习算法](04_hierarchical_rl_algorithm.md)
5. [训练教程](05_training_tutorial.md)
6. [评估和可视化](06_evaluation_and_visualization.md)
7. [代码架构](07_code_architecture.md)
8. [API 参考](08_api_reference.md)
9. [故障排查](09_troubleshooting.md)
10. [可复现性](10_reproducibility.md)
11. [研究扩展](11_research_extension.md)

## 当前代码事实摘要

| 项目 | 当前代码事实 |
| --- | --- |
| 主环境类 | `electric_gas_microgrid_single.ElectricGasMultiScaleEnv` |
| 模块化环境类 | `project.envs.electric_gas_multiscale_env.ElectricGasMultiScaleEnv` |
| 训练入口 | `hierarchical_td3_electric_gas.py` |
| 评估入口 | `evaluate_hierarchical_agent.py` |
| 顶层观测维度 | 172 |
| 动作维度 | 28 |
| 慢速动作维度 | 12 |
| 快速动作维度 | 16 |
| Manager 训练观测维度 | 172 |
| 快速 Worker 训练观测维度 | 116 |
| 慢速 Worker 训练观测维度 | 84 |
| goal 维度 | 32 |
| 快速时间尺度 | 3 分钟 |
| 慢速时间尺度 | 20 步，即 1 小时 |
| Manager 默认周期 | 40 步，即 2 小时 |
| 一天步数 | 480 |
| 训练阶段 | `fast_pretrain`、`slow_pretrain`、`manager_train`、`joint_finetune`、`all` |

## 已验证命令

本文档生成前已实际执行：

```powershell
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' -m py_compile electric_gas_microgrid_single.py hierarchical_td3_electric_gas.py evaluate_hierarchical_agent.py
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' -m pytest project/tests -q
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' hierarchical_td3_electric_gas.py --help
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' evaluate_hierarchical_agent.py --help
```

结果：核心文件编译通过，`project/tests` 为 `15 passed, 1 warning`。

