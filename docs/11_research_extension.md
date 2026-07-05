# 研究扩展方向

本章列出适合作为后续论文或工程原型扩展的方向，并说明可能需要修改的模块。

## 1. HIRO 式 goal relabeling

价值：提高高层 Manager 的离策略学习效率，缓解 Worker 策略变化导致的高层经验过期。

需要修改：

- `ManagerReplayBuffer`
- `WorkerTD3.latent_goal_reward`
- 训练循环中 Manager transition 构造

难点：

- 需要定义 goal 与实际 latent transition 的一致性度量；
- 电-气系统多目标约束下，重标记 goal 可能破坏物理语义。

## 2. Off-policy correction

价值：减少高层经验与当前 Worker 策略不匹配的问题。

需要修改：

- Manager 训练目标；
- Worker 行为策略记录；
- Replay Buffer 中增加行为策略信息。

适合作为论文扩展，但实现复杂度较高。

## 3. SAC 替代 TD3

价值：SAC（Soft Actor-Critic）具有熵正则，可能提高探索稳定性。

需要修改：

- `ManagerTD3` 和 `WorkerTD3`；
- Actor 输出概率分布；
- Critic target；
- 温度参数自动调节。

难点：动作安全投影会改变策略分布，需要谨慎处理 log probability。

## 4. MAPPO 或 MASAC

价值：把 Manager、Slow Worker、Fast Worker 更明确地视为多智能体系统。

需要修改：

- 经验结构；
- centralized critic；
- 多智能体 rollout；
- 优势估计或熵正则。

## 5. 图神经网络 Encoder

价值：电网和气网本身是图结构。GNN 可以显式利用拓扑，提高泛化能力。

需要修改：

- `MLPEncoder`；
- 观测构造；
- 网络拓扑输入；
- batch collate。

难点：电网和气网异构图、耦合边和设备节点建模。

## 6. 注意力机制

价值：Manager 可以关注关键母线、关键气节点和关键耦合设备。

需要修改：

- Encoder；
- 状态组织方式；
- 可视化 attention 权重。

## 7. MPC safety filter

价值：在 Actor 输出动作后，用模型预测控制（MPC）或优化层修正动作，减少安全投影造成的不可导/不可解释行为。

需要修改：

- `safe_env_step` 前的动作过滤；
- 设备约束和潮流近似；
- 训练时记录 filtered action。

## 8. Control Barrier Function

价值：为电压、SOC、气压等安全约束提供更形式化的安全保证。

需要修改：

- 安全投影层；
- 约束函数定义；
- 可行性求解器。

## 9. 动态天然气管网和显式 linepack

价值：当前气网是稳态/准稳态。引入动态气网可以研究真实管存、压力波传播和延迟效应。

需要修改：

- `coupled_solver.py`
- `event_scheduler.py`
- `get_global_state()`
- `equivalent_linepack_indicator` 替换为动态状态

难点：瞬态气网数值稳定性和计算成本。

## 10. 代理模型

价值：pandapower/pandapipes 调用较慢。可训练 surrogate model 加速环境 step。

需要修改：

- 数据采集脚本；
- surrogate 网络；
- 训练/验证流程；
- 环境中可切换真实求解器和代理模型。

## 11. 并行环境

价值：提高采样效率，尤其是多随机种子和长 episode 训练。

需要修改：

- rollout 收集；
- Replay Buffer 写入；
- 日志聚合；
- 随机种子管理。

## 12. 域随机化

价值：提高策略对负荷、新能源、气价、设备效率和气网参数不确定性的鲁棒性。

需要修改：

- `profile_generator.py`
- `ProjectConfig`
- reset 时随机化参数

## 13. 离线强化学习

价值：先用规则控制或历史调度生成数据，再离线训练策略，减少在线探索风险。

需要修改：

- 数据集格式；
- 行为策略记录；
- CQL/IQL/TD3+BC 等算法实现。

## 14. 设备故障和预测误差

价值：接近实际运行场景，如新能源预测偏差、压缩机故障、GFG 不可用。

需要修改：

- reset 和 profile 生成；
- action mask；
- reward 和 safety layer。

## 15. IEEE 39 扩展算例

价值：与文献电-气系统对齐，扩展到输电网尺度。

需要注意：

- 当前项目明确没有把 IEEE 39 参数混入 IEEE 33；
- 若要扩展，应新增独立网络构建模块；
- 不应覆盖当前 IEEE 33 配网基准。

