"""Visualize hierarchical TD3 training episode logs.

Examples:
    python scripts/visualize_training_results.py --logs runs/.../episode_log.csv --output-dir runs/visualizations/latest
    python scripts/visualize_training_results.py --logs path1.csv path2.csv --labels v2 continue --output-dir runs/visualizations/compare
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NUMERIC_EXCLUDE = {"stage"}


@dataclass
class RunLog:
    label: str
    path: Path
    data: pd.DataFrame


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "run"


def resolve_log_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_file():
        return path
    if path.is_dir():
        matches = sorted(path.rglob("episode_log.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find episode_log.csv from: {path_text}")


def auto_latest_logs(limit: int) -> List[Path]:
    root = Path("runs")
    if not root.exists():
        return []
    return sorted(root.rglob("episode_log.csv"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]


def load_run(path: Path, label: Optional[str] = None) -> RunLog:
    df = pd.read_csv(path)
    for col in df.columns:
        if col not in NUMERIC_EXCLUDE:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "episode" not in df:
        df["episode"] = np.arange(len(df), dtype=float)
    run_label = label or path.parent.name
    return RunLog(run_label, path, df)


def series(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    if name not in df:
        return None
    out = pd.to_numeric(df[name], errors="coerce")
    return out if out.notna().any() else None


def plot_if_present(ax: plt.Axes, df: pd.DataFrame, name: str, label: str, **kwargs: object) -> bool:
    y = series(df, name)
    if y is None:
        return False
    ax.plot(df["episode"], y, label=label, **kwargs)
    return True


def rolling(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window=max(1, window), min_periods=1).mean()


def best_value(df: pd.DataFrame, column: str) -> float:
    y = series(df, column)
    if y is None:
        return float("nan")
    return float(y.max())


def last_value(df: pd.DataFrame, column: str) -> float:
    y = series(df, column)
    if y is None:
        return float("nan")
    valid = y.dropna()
    return float(valid.iloc[-1]) if len(valid) else float("nan")


def summarize_run(run: RunLog) -> dict:
    df = run.data
    ret = series(df, "episode_return")
    eval_ret = series(df, "eval_return")
    best_train_idx = int(ret.idxmax()) if ret is not None and len(ret.dropna()) else -1
    best_eval_idx = int(eval_ret.idxmax()) if eval_ret is not None and len(eval_ret.dropna()) else -1
    return {
        "label": run.label,
        "path": str(run.path),
        "episodes": int(len(df)),
        "stage": str(df["stage"].iloc[0]) if "stage" in df and len(df) else "",
        "last_episode": int(df["episode"].iloc[-1]) if len(df) else -1,
        "best_train_episode": int(df.loc[best_train_idx, "episode"]) if best_train_idx >= 0 else "",
        "best_train_return": best_value(df, "episode_return"),
        "last_train_return": last_value(df, "episode_return"),
        "last10_train_return_mean": float(ret.dropna().tail(10).mean()) if ret is not None and len(ret.dropna()) else float("nan"),
        "best_eval_episode": int(df.loc[best_eval_idx, "episode"]) if best_eval_idx >= 0 else "",
        "best_eval_return": best_value(df, "eval_return"),
        "last_eval_return": last_value(df, "eval_return"),
        "tracked_best_eval_return": last_value(df, "best_eval_return"),
        "last_eval_voltage_rms_deviation_pu": last_value(df, "eval_mean_voltage_rms_deviation_pu"),
        "last_eval_high_pressure_rms_deviation_bar": last_value(df, "eval_mean_high_pressure_rms_deviation_bar"),
        "last_eval_prs_pressure_rms_deviation_bar": last_value(df, "eval_mean_prs_pressure_rms_deviation_bar"),
        "solver_failures_sum": float(series(df, "solver_failures").sum()) if series(df, "solver_failures") is not None else float("nan"),
        "eval_solver_failures_sum": float(series(df, "eval_solver_failures").sum()) if series(df, "eval_solver_failures") is not None else float("nan"),
        "last_mean_projection": last_value(df, "mean_action_projection"),
        "last_slow_projection": last_value(df, "mean_slow_action_projection"),
        "last_fast_projection": last_value(df, "mean_fast_action_projection"),
        "last_ess_guard": last_value(df, "mean_ess_action_guard"),
        "last_voltage_rms_deviation_pu": last_value(df, "mean_voltage_rms_deviation_pu"),
        "last_high_pressure_rms_deviation_bar": last_value(df, "mean_high_pressure_rms_deviation_bar"),
        "last_prs_pressure_rms_deviation_bar": last_value(df, "mean_prs_pressure_rms_deviation_bar"),
        "last_voltage_deviation_cost": last_value(df, "voltage_deviation_cost"),
        "last_high_pressure_deviation_cost": last_value(df, "high_pressure_deviation_cost"),
        "last_prs_pressure_deviation_cost": last_value(df, "prs_pressure_deviation_cost"),
        "last_gas_purchase_cost": last_value(df, "gas_purchase_cost"),
        "last_worker_reward_clips": last_value(df, "worker_reward_clips"),
        "last_manager_reward_clips": last_value(df, "manager_reward_clips"),
    }


def annotate_empty(ax: plt.Axes, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_run_dashboard(run: RunLog, out_dir: Path, window: int) -> Path:
    df = run.data
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f"Training dashboard: {run.label}", fontsize=15)

    ax = axes[0, 0]
    ret = series(df, "episode_return")
    if ret is not None:
        ax.plot(df["episode"], ret, color="#94a3b8", alpha=0.45, label="episode_return")
        ax.plot(df["episode"], rolling(ret, window), color="#2563eb", lw=2, label=f"rolling_mean_{window}")
        ax.axhline(best_value(df, "episode_return"), color="#16a34a", ls="--", lw=1, label="best_train")
        ax.legend()
    else:
        annotate_empty(ax, "episode_return not found")
    ax.set_title("Episode return")
    ax.set_xlabel("episode")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    plotted = False
    plotted |= plot_if_present(ax, df, "eval_return", "eval_return", color="#dc2626", marker="o", ms=3)
    plotted |= plot_if_present(ax, df, "best_eval_return", "best_eval_return", color="#16a34a", lw=2)
    if plotted:
        ax.legend()
    else:
        annotate_empty(ax, "eval columns not found")
    ax.set_title("Evaluation return")
    ax.set_xlabel("episode")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    plotted = False
    for col, label, color in [
        ("mean_action_projection", "total", "#111827"),
        ("mean_slow_action_projection", "slow", "#7c3aed"),
        ("mean_fast_action_projection", "fast", "#ea580c"),
    ]:
        plotted |= plot_if_present(ax, df, col, label, color=color)
    if plotted:
        ax.legend()
    else:
        annotate_empty(ax, "projection columns not found")
    ax.set_title("Mean action projection")
    ax.set_xlabel("episode")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    plotted = False
    plotted |= plot_if_present(ax, df, "mean_ess_action_guard", "mean_ess_guard", color="#0891b2")
    plotted |= plot_if_present(ax, df, "max_ess_action_guard", "max_ess_guard", color="#0f766e", alpha=0.7)
    if plotted:
        ax.legend()
    else:
        annotate_empty(ax, "ESS guard columns not found")
    ax.set_title("ESS action guard")
    ax.set_xlabel("episode")
    ax.grid(True, alpha=0.25)

    ax = axes[2, 0]
    plotted = False
    for col, label, color in [
        ("solver_failures", "solver_failures", "#dc2626"),
        ("worker_reward_clips", "worker_reward_clips", "#9333ea"),
        ("manager_reward_clips", "manager_reward_clips", "#0f766e"),
    ]:
        plotted |= plot_if_present(ax, df, col, label, color=color)
    if plotted:
        ax.legend()
    else:
        annotate_empty(ax, "failure/clip columns not found")
    ax.set_title("Failures and reward clipping")
    ax.set_xlabel("episode")
    ax.grid(True, alpha=0.25)

    ax = axes[2, 1]
    plotted = False
    for col, label, color, scale in [
        ("mean_voltage_rms_deviation_pu", "voltage_rms_x100", "#2563eb", 100.0),
        ("mean_high_pressure_rms_deviation_bar", "hp_rms_bar", "#0891b2", 1.0),
        ("mean_prs_pressure_rms_deviation_bar", "prs_rms_bar", "#16a34a", 1.0),
    ]:
        y = series(df, col)
        if y is not None:
            ax.plot(df["episode"], rolling(y * scale, window), label=label, color=color)
            plotted = True
    if plotted:
        ax.legend()
    else:
        annotate_empty(ax, "stability columns not found")
    ax.set_title("Stability RMS deviation")
    ax.set_xlabel("episode")
    ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = out_dir / f"dashboard_{safe_name(run.label)}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_comparison(runs: List[RunLog], out_dir: Path, window: int) -> Optional[Path]:
    if len(runs) < 2:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Training comparison", fontsize=15)

    ax = axes[0, 0]
    for run in runs:
        ret = series(run.data, "episode_return")
        if ret is not None:
            ax.plot(run.data["episode"], rolling(ret, window), label=run.label)
    ax.set_title(f"Rolling episode return ({window})")
    ax.set_xlabel("episode")
    ax.legend()
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    for run in runs:
        eval_ret = series(run.data, "eval_return")
        if eval_ret is not None:
            ax.plot(run.data["episode"], eval_ret, marker="o", ms=3, label=run.label)
    ax.set_title("Evaluation return")
    ax.set_xlabel("episode")
    ax.legend()
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    width = 0.25
    x = np.arange(len(runs))
    slow = [last_value(run.data, "mean_slow_action_projection") for run in runs]
    fast = [last_value(run.data, "mean_fast_action_projection") for run in runs]
    guard = [last_value(run.data, "mean_ess_action_guard") for run in runs]
    ax.bar(x - width, slow, width, label="slow_projection")
    ax.bar(x, fast, width, label="fast_projection")
    ax.bar(x + width, guard, width, label="ess_guard")
    ax.set_xticks(x)
    ax.set_xticklabels([run.label for run in runs], rotation=20, ha="right")
    ax.set_title("Last projection / guard metrics")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 1]
    best_eval = [max(best_value(run.data, "eval_return"), last_value(run.data, "best_eval_return")) for run in runs]
    best_train = [best_value(run.data, "episode_return") for run in runs]
    ax.bar(x - width / 2, best_train, width, label="best_train_return")
    ax.bar(x + width / 2, best_eval, width, label="best_eval_return")
    ax.set_xticks(x)
    ax.set_xticklabels([run.label for run in runs], rotation=20, ha="right")
    ax.set_title("Best returns")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = out_dir / "comparison_dashboard.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_summary(runs: List[RunLog], out_dir: Path) -> Path:
    rows = [summarize_run(run) for run in runs]
    summary = pd.DataFrame(rows)
    csv_path = out_dir / "summary.csv"
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")

    md_path = out_dir / "summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Training Result Summary\n\n")
        columns = list(summary.columns)
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for _, row in summary.iterrows():
            values = [str(row[col]) for col in columns]
            f.write("| " + " | ".join(values) + " |\n")
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize hierarchical TD3 episode logs.")
    parser.add_argument("--logs", nargs="*", default=[], help="episode_log.csv files or directories containing one")
    parser.add_argument("--labels", nargs="*", default=[], help="Optional labels matching --logs")
    parser.add_argument("--output-dir", default="runs/visualizations/latest", help="Directory for PNG/CSV/MD outputs")
    parser.add_argument("--rolling-window", type=int, default=10)
    parser.add_argument("--auto-latest", type=int, default=0, help="Use N newest runs/**/episode_log.csv files when --logs is omitted")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.logs:
        paths = [resolve_log_path(p) for p in args.logs]
    else:
        paths = auto_latest_logs(max(args.auto_latest, 1))
    if not paths:
        raise SystemExit("No episode_log.csv files found.")

    labels = list(args.labels)
    if labels and len(labels) != len(paths):
        raise SystemExit("--labels length must match --logs length.")
    runs = [load_run(path, labels[i] if labels else None) for i, path in enumerate(paths)]

    dashboard_paths = [plot_run_dashboard(run, out_dir, args.rolling_window) for run in runs]
    comparison_path = plot_comparison(runs, out_dir, args.rolling_window)
    summary_path = write_summary(runs, out_dir)

    print("Wrote:")
    for path in dashboard_paths:
        print(path)
    if comparison_path:
        print(comparison_path)
    print(summary_path)
    print(out_dir / "summary.md")


if __name__ == "__main__":
    main()
