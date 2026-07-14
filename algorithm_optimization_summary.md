# 分层 SMDP-TD3 最终优化摘要

## 1. 修改边界

本轮实际修改：

- `hierarchical_td3_electric_gas_optimized.py`：主要算法、Replay、PER、checkpoint、评估、测试。
- `hierarchical_td3_electric_gas.py`：兼容 API、训练循环、片段收尾、日志及 checkpoint 公共接口。
- `electric_gas_microgrid_single.py`：仅缩小异常捕获范围，让 API/shape/索引等程序错误保留 traceback；未改变拓扑、设备参数、动作维度、动作映射或求解器数学模型。

本轮没有调整环境奖励公式和奖励权重。工作区中已存在的 RewardConfig 校准值被保留，没有回退或二次修改。

## 2. SMDP 奖励与折扣

Slow 和 Manager 的片段奖励在交互阶段按快速时间尺度累计：

```text
R_segment = sum(gamma_fast ** k * reward[t + k], k=0..duration-1)
```

TD 目标按每条样本的真实持续时间计算：

```text
y = R_segment + (1 - done) * gamma_fast ** duration_steps * Q_target(next_state, next_action)
```

- Fast 的 `duration_steps=1`。
- Slow 正常片段为 20 步，Manager 正常片段为 40 步。
- 提前终止和配置时间上限使用真实 duration。
- 已折扣累计的片段奖励不会再次整体乘 duration discount。
- `episode_steps` 到达时统一生成 time-limit truncation；默认不跨 reset bootstrap。

测试精确覆盖 duration 1、7、20、33、40，并通过 7、20、33、40、47 步真实训练循环验证尾片段收尾。

## 3. Slow 动作语义

Slow 动作链路明确为：

```text
Actor raw command
-> ESS safety guard
-> first-step environment projection
-> interval dynamic projection as SOC changes
```

Slow Replay 保存：

- `raw_action`：Critic 的动作输入，保持 `Q(s, goal, raw_action)`。
- `guarded_action`：投影模仿目标。
- `first_executed_action`、`last_executed_action`。
- `discounted_mean_executed_action`、`executed_action_variance`。
- `max_dynamic_projection`、`duration_steps`。

每个快速步从 `info["applied_action"][:slow_action_dim]` 读取真实执行动作，并按 `gamma_fast ** segment_offset` 统计均值与方差。Transition Model 使用执行动作摘要；Actor 不模仿未来执行均值，只在当前动作接近历史 raw action时模仿 guarded action。

Slow normalized shaping 只缩放端点 latent/physical progress，不重复缩放已在区间内累计的 external reward 和 projection penalty。

## 4. 数值稳定性

- Critic 使用 twin Q、Smooth-L1、target smoothing、delayed actor update 和 Polyak update。
- Worker Critic 与 Actor 都使用 raw action 域。
- target encoder 在采样更新时重新计算 latent reward，Replay 不固化在线 Encoder 的旧 latent reward。
- Transition Model 的 latent delta 两端都来自冻结的 target encoder。
- 所有 target Encoder/Actor/Critic 永久 `requires_grad=False` 且保持 eval mode，只能经 hard/soft update 改变。
- Actor backward 期间临时冻结 Critic 参数，保留动作策略梯度但不产生无效 Critic 梯度。
- `require_finite_tensor()` 和 `require_finite_gradients()` 在任何 optimizer step 前检查输入、奖励、Q、loss、梯度和梯度范数。
- `nonfinite_update_policy=raise` 保存 emergency checkpoint 后抛出；`skip_batch` 清梯度并跳过整批，不更新参数、target 或 PER。

## 5. Replay 与 PER

三层 Replay 均有单调递增的 `total_insertions`。Slow/Manager 用 insertion id 触发边界更新，因此环形 Replay 满容量后仍会继续更新，同一 insertion 不会重复触发。

Fast/Slow PER 分开保存：

```text
td_priorities
constraint_scores
projection_scores
```

TD 分量使用 `log1p(abs(td_error))`，非有限和极大误差在写入前清洗、裁剪。最终 priority 只组合一次。采样概率非法时显式回退均匀采样并增加 `priority_fallback_count`。

Uniform + PER 采用逐 batch 位置 Bernoulli 选择采样源，真实概率为：

```text
p_mix(i) = fraction * p_priority(i) + (1 - fraction) / replay_size
```

因此 batch size 1、2、3 以及 fraction 0/1 的 IS 权重语义一致。Monte Carlo 测试验证经验频率与理论概率一致。

`priority_component_normalization` 支持 `none`、`running_scale`、`rank`。后两种模式在采样时根据全部有效原始分量重算，避免缓存陈旧 scale/rank。日志包含分量 P50/P90/P99、ESS、IS weight 和 fallback 计数。

Manager Replay 可选启用 PER，并保存片段约束均值、最大值及 solver failure 标志。训练主循环会在正常边界、提前截断和 solver failure 路径完整传入这些字段。

## 6. Replay Schema

当前 Replay schema version 为 2，包含：

```text
replay_schema_version, replay_type, capacity, valid_size, idx, full,
total_insertions, obs_dim, action_dim, goal_dim, dtype
```

未满时只序列化 `arrays[:valid_size]`；满载时保存完整环形有效数组。加载时严格检查 Replay 类型、容量、维度、dtype、shape、idx/full/valid_size，不静默截断或补零。

Fast/Slow 额外保存 raw/executed 动作、奖励分量、goal、done、duration、PER 原始分量、sample calls 和诊断状态。Slow 还保存完整执行动作摘要。Manager 保存片段起止状态、goal、折扣奖励、duration 和约束摘要。

## 7. Checkpoint Schema

当前 optimized checkpoint schema version 为 5。

轻量 checkpoint：

```text
best_fast.pt, best_slow.pt, best_manager.pt, best_joint.pt, latest_policy.pt
```

只包含网络、target 网络、normalizer、配置、global step、阶段和评估最佳指标，不含 Replay、RNG 和 optimizer。

完整恢复 checkpoint：

```text
resume_latest.pt
```

包含在线/目标网络、optimizer、三层 Replay/PER、Python/NumPy/Torch/CUDA RNG、`next_episode`、`global_step`、阶段和最佳指标。`full_resume_checkpoint_interval` 控制保存周期。保存使用临时文件加 `os.replace()`，失败不会破坏上一份有效文件。

严格 resume 缺少 Replay、RNG、next episode、global step、任一 optimizer 或 insertion marker 时立即失败。非严格模式返回：

```text
strict_resume_restored
resume_missing_components
```

## 8. 加载与迁移规则

- `resume`：只允许同训练阶段，恢复全部训练状态并从 `next_episode` 继续。
- `stage_transfer`：允许跨阶段；加载网络和 normalizer，保留 Critic 权重，重建 optimizer/update count/Replay/RNG/阶段最佳值并重新应用 trainability。
- `policy_only`：仅加载策略相关网络与 normalizer，其余重新初始化。
- `run_all_stages()` 的后续阶段固定使用 `stage_transfer`。
- 旧共享 `batch_size`、`learning_starts`、`updates_per_step` 按缺失字段逐层迁移，显式新字段优先。
- 旧 `priority_beta_anneal_steps` 迁移到缺失的三层独立退火字段并告警。
- schema 旧或缺核心训练状态的文件可用于 stage transfer/policy-only；在 strict resume 下拒绝。
- Legacy API 固定为 version 3，启动时验证版本、函数签名、RNG API 和三类 Replay state API。

## 9. 评估与最佳模型

评估固定 normalizer、关闭探索噪声且不更新网络、Replay、optimizer 或运行统计。`eval_seed_mode=fixed` 始终使用同一绝对种子集合，`offset` 才按 anchor 平移；`eval_episodes=N` 始终执行 N 个 episode。

评估区分 solver success 与 constraint feasibility，并统计六类硬约束违规率、物理极值、最大越限和 total/per-step/rate 三种 cost。最佳模型比较顺序为：

1. 全局 feasible。
2. solver failure。
3. 硬约束违规率。
4. 最大越限量。
5. 每步归一化约束成本。
6. mean return。

缺失必需安全指标会报错，不会按安全值补零。

## 10. 长期训练支持

- Fast 使用有界随机 warm-up；Slow 使用规则安全动作加受限扰动；Manager 使用随机单位 goal。
- 各层 batch size、learning starts、更新次数、学习率和 PER beta 退火独立。
- 配置启动时验证 interval、容量、batch、learning starts、理论 transition 数、学习率、噪声、gamma、tau、裁剪与正则权重。
- 启动日志打印 legacy 路径/API version、optimized 路径、环境模型版本及 gamma 在 Slow/Manager/episode 尺度下的有效折扣。
- TensorBoard/CSV 记录 Q、target Q、裁剪率、梯度范数、动作饱和、两级投影、solver/约束、Replay insertion/update、PER ESS、normalizer 统计和 checkpoint 大小/耗时。
- 健康监控对长期无更新、高 clipping、高动作饱和、高动态投影及连续 solver failure 发出告警，但不自动修改训练参数。

## 11. 实际验收

执行：

```powershell
& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' -m py_compile `
  electric_gas_microgrid_single.py `
  hierarchical_td3_electric_gas.py `
  hierarchical_td3_electric_gas_optimized.py

& 'D:\anaconda\anaconda\envs\python_3_8\python.exe' `
  hierarchical_td3_electric_gas_optimized.py --run-tests
```

结果：

- 三文件编译通过。
- 完整 optimized 测试通过，最终一次耗时约 105.3 秒。
- 真实 pandapower/pandapipes 电气耦合 smoke test 通过，测试短训练 solver failure 为 0。
- 7/20/33/40/47 步配置上限均正确产生 time-limit truncation，并正确收尾 Fast/Slow/Manager duration。
- NaN reward 与 Inf gradient 测试均未执行 optimizer step、target update 或 PER update。
- strict resume 恢复 Replay/PER/RNG/episode/global step 后，下一次采样、动作、噪声和更新可复现。
- 轻量最佳 checkpoint 不含 Replay/RNG；完整 resume checkpoint 包含全部核心状态。
- 环形 Replay 覆盖后 Slow/Manager 持续更新。

测试中的 expected sample warning 是短程夹具故意使用少量 episode 触发的配置诊断，不是测试失败。

## 12. 尚未实施的研究增强

本轮未引入 GNN 图编码器、Transformer 长时序编码器、CTDE、多智能体通信、Lagrangian/CPO 显式约束优化和可微 Projection Model。这些会改变网络归纳偏置或优化问题，适合在当前可信基线之上做独立消融实验。
