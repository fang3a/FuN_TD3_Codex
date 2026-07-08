"""Generate the overall electric-gas coupled topology figure.

这是一个很薄的命令行包装器，真正的绘图逻辑在
``project.visualization.topology.save_coupled_topology_overview``。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from visualization.topology import save_coupled_topology_overview


def main() -> None:
    """解析输出路径并生成拓扑图。"""

    parser = argparse.ArgumentParser(description="Plot coupled IEEE33-Belgian20 topology overview.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("project/outputs/topology/coupled_network_overview.png"),
    )
    args = parser.parse_args()
    path = save_coupled_topology_overview(args.output)
    print(path)


if __name__ == "__main__":
    main()
