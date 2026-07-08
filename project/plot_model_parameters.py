"""Generate static model-parameter dashboards and reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from project.visualization import save_model_parameter_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot static parameters for the electric-gas coupled microgrid model."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("project/outputs/model_parameters"),
        help="Directory for PNG, CSV, JSON, and markdown outputs.",
    )
    parser.add_argument(
        "--skip-topology",
        action="store_true",
        help="Skip the coupled topology overview figure.",
    )
    args = parser.parse_args()

    artifacts = save_model_parameter_artifacts(
        args.output_dir,
        include_topology=not args.skip_topology,
    )
    print("Model parameter visualization artifacts:")
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
