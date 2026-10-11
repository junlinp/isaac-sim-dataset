"""Shared filesystem configuration; environment variables never change sensor frames."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = (
    Path(os.environ.get("ISAAC_DATASET_WORKDIR", str(ROOT / "work")))
    .expanduser()
    .resolve()
)
ASSET_CACHE = (
    Path(
        os.environ.get(
            "ISAAC_DATASET_ASSET_CACHE", str(ROOT / "assets/nvidia_office_5_1")
        )
    )
    .expanduser()
    .resolve()
)
OFFICE_ENTRY = (
    ASSET_CACHE / "cloud/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd"
)
ROUTE_PATH = WORK_DIR / "route_spline.npz"
if not ROUTE_PATH.exists():
    ROUTE_PATH = ROOT / "config/office_route_spline.npz"
