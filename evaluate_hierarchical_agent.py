"""Evaluate a saved hierarchical TD3 checkpoint without exploration noise.

这个脚本只做评估，不训练。它会读取 checkpoint 中保存的 TrainConfig，
重建环境和三个 TD3 智能体，加载权重，然后用 deterministic 动作跑若干 episode。
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace

import torch

from electric_gas_microgrid_single import ElectricGasMultiScaleEnv
from hierarchical_td3_electric_gas import (
    EPISODE_STEPS,
    TrainConfig,
    build_agents,
    evaluate_policy,
    load_checkpoint,
    resolve_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """解析评估所需参数：checkpoint 路径、episode 数、步数和设备。"""

    parser = argparse.ArgumentParser(description="Evaluate hierarchical TD3 electric-gas agent")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode-steps", type=int, default=EPISODE_STEPS)
    parser.add_argument("--manager-interval", type=int, default=40)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    # checkpoint 里包含模型权重和训练时的配置；先读取配置，再按配置创建网络结构。
    payload = torch.load(args.checkpoint, map_location=device)
    saved_cfg = TrainConfig(**payload.get("config", {}))
    cfg = replace(saved_cfg, manager_interval=args.manager_interval, device=args.device)
    env = ElectricGasMultiScaleEnv()
    agents = build_agents(env, cfg, device)
    # load_checkpoint 会把 Manager、slow Worker、fast Worker 的权重和归一化器都恢复。
    load_checkpoint(args.checkpoint, agents, device)
    # evaluate_policy 内部会冻结归一化器，并关闭探索噪声。
    stats = evaluate_policy(agents, cfg, episodes=args.episodes,
                            max_steps=args.episode_steps, seed=args.seed)
    print("Hierarchical TD3 evaluation")
    print(f"Mean return: {stats['mean_return']:.4f}")
    print(f"Std return: {stats['std_return']:.4f}")
    print(f"Power success rate: {100.0 * stats['power_success_rate']:.2f}%")
    print(f"Gas success rate: {100.0 * stats['gas_success_rate']:.2f}%")
    print(f"Solver failures: {int(stats['solver_failures'])}")
    print(f"Steps: {int(stats['steps'])}")
    print(f"Voltage RMS deviation: {stats['mean_voltage_rms_deviation_pu']:.6f} pu")
    print(f"High-pressure RMS deviation: {stats['mean_high_pressure_rms_deviation_bar']:.6f} bar")
    print(f"PRS RMS deviation: {stats['mean_prs_pressure_rms_deviation_bar']:.6f} bar")
    print(f"Voltage deviation cost: {stats['voltage_deviation_cost']:.4f}")
    print(f"High-pressure deviation cost: {stats['high_pressure_deviation_cost']:.4f}")
    print(f"PRS pressure deviation cost: {stats['prs_pressure_deviation_cost']:.4f}")
    print(f"Gas purchase cost: {stats['gas_purchase_cost']:.4f}")


if __name__ == "__main__":
    main()
