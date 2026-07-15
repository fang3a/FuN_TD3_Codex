# 分层 SMDP-TD3 算法阅读指南

## 1. 项目总体结构

| 文件 | 职责 |
|---|---|
| `electric_gas_microgrid_single.py` | 电—气拓扑、设备参数、外生曲线、状态/动作接口、安全投影、电力潮流与气流求解。阅读接口即可，不在注释版中修改。 |
| `hierarchical_td3_electric_gas.py` | 基础 TD3、三层 observation、基础 Replay、Manager/Worker、训练主循环、评估和兼容 API。 |
| `hierarchical_td3_electric_gas_optimized.py` | 正式训练入口；扩展 PER、严格 SMDP、物理 goal、投影模仿、环境契约、健康检查和严格 checkpoint。 |

优化版保留单一训练主循环：它临时把优化类安装到基础模块，再调用基础 `run_training()`，
作用域结束后恢复原符号。这样不会维护两套语义逐渐漂移的交互循环。

## 2. 推荐阅读顺序

1. **配置与维度**：`TrainConfig`、时间尺度常量、`SlowObservationLayout`。
2. **Actor/Critic 网络**：`MLPEncoder`、`ManagerActor/Critic`、`WorkerActor/Critic`。
3. **Replay Buffer**：先读基础三个 Replay，再读优化版 PER 扩展。
4. **单层 TD3 更新**：`ManagerTD3.update()` 和 `WorkerTD3._update_once_impl()`。
5. **Manager goal**：`normalize_goal_tensor()`、`execute_manager_goal_tensor()`、`worker_goal_tensor()`。
6. **多时间尺度 Pending 片段**：三个 `Pending*`，重点是 `PendingSlowSegment`。
7. **训练主循环**：基础版 `run_training()`。
8. **PER、投影模仿和物理目标**：优化版对应模块。
9. **评估与 checkpoint**：`is_feasible()`、`evaluate_policy()`、`save/load_checkpoint()`。

## 3. 三层结构与一次环境交互

一个快速步为 3 分钟；Fast 每步决策，Slow 每 20 步（1 小时）决策，Manager 每
40 步（2 小时）决策。正式 episode 为 480 步，即 24 小时。

```text
全局状态（环境）
  ↓ ObservationBuilder
Manager observation + previous executed goal
  ↓ Manager Actor
raw goal → 归一化/平滑 → executed goal
  ↓ worker_goal_* 分片
Slow/Fast observation + 各自 24维 goal
  ↓ Worker Actor
raw action
  ↓ ESS 前瞻 guard（主要作用于 Slow）
guarded action
  ↓ 环境 SOC/逆变器/压缩机等安全投影
executed action
  ↓ 电力潮流 + 气流耦合求解
下一状态、环境安全奖励、约束指标、求解器状态
  ↓
Fast 单步入库；Slow/Manager 按真实 duration 聚合后入库
```

编号流程：

1. `env.reset()` 给出新的外生场景和全局 observation。
2. 到 Manager 边界时构造 197 维 observation，Actor 产生 raw goal。
3. `execute_manager_goal_*()` 使用 previous executed goal 做相同的归一化和平滑。
4. 到 Slow 边界时，Slow Actor 依据 49 维 observation 与 24 维子 goal 产生 10 维 raw action。
5. Fast Actor 每步依据 115 维 observation 与 24 维子 goal 产生 16 维 raw action。
6. Slow raw action 可先经 ESS guard；Slow 10 维与 Fast 16 维拼成环境 26 维动作。
7. 环境逐设备投影并执行动作，得到下一状态、reward、terminated/truncated 和 `info`。
8. Fast transition 持续 1 步；Slow/Manager 逐快速步折扣累积 reward。
9. 边界或 episode 结束时，用真实 duration 完成片段并写 Replay。
10. 达到 learning_starts、batch 和 warmup 门槛后才进行梯度更新。

## 4. 一次梯度更新流程

### 4.1 Fast Worker update

1. Critic 从 PER 采样 Fast 单步 transition，duration 为 1。
2. target Actor 产生下一 raw action，并加入 target policy smoothing 噪声。
3. 两个 target Critic 取较小值，构造 SMDP target。
4. Q1/Q2 对 target 做带 importance-sampling 权重的 Smooth L1 回归。
5. 到 delayed update 周期时，Actor 从独立 uniform batch 更新。
6. 动作 L2 和符合 mask 的投影模仿作为辅助项；最后软更新 target 网络。

### 4.2 Slow Worker update

流程与 Fast 相同，但 transition 通常持续 20 步。Replay 还包含整个区间的 executed
action 折扣均值、普通均值、方差、first、last、序列和最大动态投影。TD target 必须
对每条样本使用实际 duration，尾部可能小于 20。

### 4.3 Manager update

1. Replay 样本包含 current/next Manager observation、raw goal、executed goal、previous executed goal、片段奖励和 duration。
2. target Actor 先输出 raw goal，再调用与交互阶段相同的执行变换。
3. target Critic 用 executed goal 估值，并按通常 40 步的 duration bootstrap。
4. Critic 更新后，Manager Actor 使用 uniform batch 延迟更新。

统一公式：

```text
R_segment = Σ(k=0..duration-1) gamma_fast^k * r_(t+k)
y = R_segment + (1-done) * gamma_fast^duration * min(Q1_target, Q2_target)
L_critic = loss(Q1(s,a), y) + loss(Q2(s,a), y)
L_actor = -E[min(Q1,Q2)(s, actor(s))] + 辅助正则
target ← tau * online + (1-tau) * target
```

片段内部已经折扣，bootstrap 只再乘一次片段间的 `gamma_fast**duration`。

## 5. 实际维度表

以下数值由生成脚本实例化真实 `ElectricGasMultiScaleEnv` 并调用
`ObservationBuilder` 动态确认，不是手工猜测。

| 项目 | 实际维度 |
|---|---:|
| 环境全局 state/observation | 165 |
| Manager observation | 197 |
| Slow observation | 49 |
| Fast observation | 115 |
| Manager goal | 32 |
| goal 前部方向向量 | 24 |
| goal 后部物理目标 | 8 |
| Slow action | 10 |
| Fast action | 16 |
| 环境总 action | 26 |
| Slow Worker goal | 24 |
| Fast Worker goal | 24 |

Slow observation 的 49 维命名布局为：

```text
[["soc", 3], ["soc_low_margin", 3], ["soc_high_margin", 3], ["voltage_summary", 3], ["line_summary", 3], ["power_balance", 5], ["power_loss", 1], ["power_forecast", 4], ["gas_pressure_summary", 3], ["pipe_summary", 2], ["source_utilization", 2], ["linepack", 1], ["gas_forecast", 2], ["time", 4], ["held_slow_action", 10]]
```

## 6. Manager goal 与物理目标

32 维 goal = shared direction 8 + Slow direction 8 + Fast direction 8 + physical goal 8。
Worker 实际收到 24 维：shared 8 + 自己的 direction 8 + physical 8。

后 8 维物理目标依次表达：电压偏差、最大线路加载、新能源利用、外购电、平均 SOC、
气压偏差、气源利用率、linepack。Fast 主要负责前 4 项，Slow 主要负责后 4 项。
`PhysicalGoalFeatureExtractor.progress()` 比较动作前后到目标的距离：距离下降则奖励为正。

## 7. raw、guarded、executed 动作

- **raw_action**：Actor 的直接请求，也是 Worker Critic 的动作输入。
- **guarded_action**：训练脚本根据未来保持区间和 SOC 做 ESS 前瞻保护后的请求。
- **executed_action**：环境根据 SOC、逆变器容量、压缩机和其他物理约束投影后真正执行的动作。

executed action 决定真实状态转移，并用于投影惩罚、投影模仿、慢片段统计和日志。
Critic 仍学习 `Q(s, raw_request_action)`，因为 Actor 能直接选择的是 raw action；把历史
executed action 偷换成 Critic 动作会让 Actor/Critic 的动作空间不一致。

## 8. Worker 奖励

- `external reward`：由环境安全分量组合后的外在奖励。
- `global safety reward`：电压、线路、SOC、气压、流速、气源和求解器的共同安全信用。
- `role-specific reward`：Fast 更关注电压/线路/新能源，Slow 更关注 SOC/气压/压缩机和平滑。
- `latent direction reward`：状态 latent 变化是否沿 Manager 指定方向。
- `physical progress`：可解释物理指标到目标的距离是否下降。
- `projection penalty`：请求越不可行，raw 与执行目标差距越大，惩罚越强。
- `action L2`：抑制 Actor 长期饱和在动作边界。
- `reward clipping`：只作为极端保护，并记录裁剪比例，不能隐藏奖励尺度错误。

项目没有电价、气价和套利目标。Slow 设备会改变有功平衡、电压与线路，因此 Slow 也必须
获得非零电—气全局安全奖励。

## 9. PER 与投影模仿

PER priority 由 TD error、安全约束严重度、动作投影和求解失败组成。各分量先做 running
scale 归一化，再按权重组合并裁剪。`alpha` 控制优先程度；importance-sampling `beta`
从初值退火到 1，逐步修正非均匀采样偏差。Critic 使用 PER；Actor 使用 uniform batch，
否则 Actor 容易只优化高 TD error/高违规状态。

投影模仿比较 historical raw、historical imitation/executed 和 current actor action：

1. projection mask：历史动作确实被明显投影；
2. behavior match mask：当前策略仍接近历史 raw；
3. 两者同时满足才模仿安全动作。

权重随训练衰减；模仿只告诉 Actor “相似行为下哪边更可行”，不能代替 Q 值策略梯度。

## 10. 四阶段训练与训练预算

| 阶段 | 训练网络 | 冻结网络 | goal 来源 | 学习率语义 |
|---|---|---|---|---|
| `fast_pretrain` | Fast | Slow、Manager | 固定安全 goal | Fast 预训练 LR |
| `slow_pretrain` | Slow | Fast、Manager | 固定安全 goal | Slow 预训练 LR |
| `manager_train` | Manager | Fast、Slow | Manager Actor | Manager LR |
| `joint_finetune` | 三层 | 无 | Manager Actor | 更保守的 joint Worker/Manager LR |

每阶段探索用全局 `exploration_episode_offset` 延续，避免重新从最大噪声开始。
`validate_training_contract()` 根据 episode、480 步、20/40 边界、learning_starts、batch、
warmup 和更新频率预估样本。短训练可能 Replay 还未达到门槛，三个网络一次梯度都没有，
所以“程序跑完”不等于“发生学习”。

## 11. formal/debug、安全可行性与 done

- formal 强制 480 步完整日和 `feasible_then_return`。
- formal 要求 Fast/Slow 的 global safety weight 都大于 0。
- debug 允许短 episode，但提前截断要补终端 SOC 惩罚。
- 环境 reset 会更换外生场景，因此时间限制截断默认 done，禁止跨 reset bootstrap。

`is_feasible()` 至少统一检查电/气求解成功率、SOC 越限、硬约束、电压 RMS 和气压 RMS，
并扩展到线路、管速、气源和失败次数。比较规则：可行模型永远优于不可行模型；都可行时
再比较综合安全/return；都不可行时比较总违反程度。

## 12. checkpoint 与断点续训

- **policy/lightweight checkpoint**：主要保存网络和归一化统计，用于评估或阶段迁移。
- **full resume checkpoint**：还保存 target 网络、优化器、三层 Replay、写指针、priority、
  更新计数、global step、stage、episode、探索进度、Python/NumPy/PyTorch/CUDA RNG、最佳评价。
- **strict resume**：缺任一关键状态就报错，保证继续一步的结果可复现。
- **stage_transfer**：只迁移允许的网络和 normalizer，重置新阶段 Replay/优化器/调度状态。
- **policy_only**：只用于评估或显式初始化。

observation 布局变化后，即使张量碰巧同形状，旧 schema checkpoint 也不能直接 strict resume。

## 13. 重要类和函数索引

| 类/函数 | 作用 | 优先级 |
|---|---|---|
| `TrainConfig` / `validate_training_contract` | 理解全部训练契约和有效预算 | 必读 |
| `SlowObservationLayout` / `ObservationBuilder` | 理解三层看到什么 | 必读 |
| `ManagerActor/Critic`、`WorkerActor/Critic` | 理解 TD3 网络输入输出 | 必读 |
| `build_smdp_target` | 理解真实 duration 折扣 | 必读 |
| `ManagerTD3._update_once_impl` | Manager 的完整梯度更新 | 必读 |
| `WorkerTD3._update_once_impl` | PER Critic、uniform Actor、投影模仿 | 必读 |
| `PendingSlowSegment` | 20 步执行动作动态聚合 | 必读 |
| `_PrioritizedReplayMixin` | priority、alpha、beta 和 IS 权重 | 重点 |
| `PhysicalGoalFeatureExtractor` | 后 8 维物理目标 | 重点 |
| `run_training` | 三时间尺度数据流 | 必读 |
| `is_feasible` / `evaluate_policy` | 安全评估与最佳模型选择 | 重点 |
| `save_checkpoint` / `load_checkpoint` | 三种恢复模式 | 重点 |
| `run_all_stages` | 四阶段衔接 | 重点 |

## 14. 强化学习术语表

- **state（状态）**：满足 Markov 性的系统信息。
- **observation（观测）**：某层实际得到的状态表示，可能是全局状态的压缩或拼接。
- **action（动作）**：策略对环境或下层发出的控制请求。
- **policy（策略）**：从 observation 到 action 的映射。
- **Actor**：显式生成连续动作的策略网络。
- **Critic**：估计状态—动作长期回报 Q 的网络。
- **Q value**：执行某动作后未来折扣回报的期望。
- **target network**：缓慢更新、用于构造稳定 bootstrap 目标的网络副本。
- **Replay Buffer**：保存历史 transition 并随机复用的经验池。
- **off-policy**：可用旧策略收集的经验训练当前策略。
- **bootstrapping**：用下一状态的估值构造当前 TD 目标。
- **temporal difference**：当前 Q 与 bootstrap target 的差。
- **SMDP**：动作可持续多个底层时间步的半马尔可夫决策过程。
- **PER**：按 TD error/安全信息优先抽样的 Replay。
- **intrinsic reward**：由 latent 方向或目标进展构造的内部塑形奖励。
- **safety projection**：把不满足物理约束的请求映射到可执行动作。
- **checkpoint**：用于评估、迁移或继续训练的持久化状态。

## 15. 常见误区

1. 把 `done`、terminated 和时间截断混为一谈，导致跨外生场景 bootstrap。
2. 把 raw、guarded、executed action 当成同一个动作。
3. 片段内部已经折扣后，又给整个 reward 乘一次 `gamma_slow/manager`。
4. 把 Manager raw goal 不经统一变换直接送给 Critic。
5. 忽略 Slow 片段内 ESS executed action 会随 SOC 动态变化。
6. 只运行几个短 episode 就认为三层均已训练。
7. 只看 return，不检查电压、SOC、气压、线路、管速、气源和求解成功率。
8. observation/schema 改动后继续 strict resume 旧 checkpoint。
9. 让 Actor 直接复用 PER Critic batch，却不处理采样分布偏置。
10. 把投影模仿当成策略梯度的替代品。

## 16. 仍需结合环境文件阅读的接口

- `ElectricGasMultiScaleEnv.reset()` / `step()`：Gymnasium terminated/truncated 语义。
- `get_slow_safety_features()`：49 维 Slow observation 的物理来源。
- `slow_action_dim=10`、`fast_action_dim=16`、`action_dim=26`。
- `info['raw_action']`、`info['applied_action']`、`info['slow_action_applied']`。
- `info['reward_components']`：安全奖励的原始分量。
- `info['constraint_metrics']`：电压、线路、SOC、气压、流速、气源指标。
- `info['solver_failed']`、电/气求解成功标志和失败回滚。
- ESS、逆变器、新能源削减和可控压缩机的动作反归一化与安全投影。
