"""Generate the overall electric-gas coupled topology figure."""

from __future__ import annotations

import argparse
from pathlib import Path

from project.visualization.topology import save_coupled_topology_overview


def main() -> None:
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

