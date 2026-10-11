"""Package a fully validated dataset with source snapshot and checksums."""

import argparse, json, hashlib, shutil, subprocess, time
from pathlib import Path
import sys

sys.path = [
    p
    for p in sys.path
    if p != "/usr/lib/python3/dist-packages" and not p.startswith("/opt/ros/")
]
from settings import ROOT
from pxr import UsdUtils, Sdf, Usd, UsdGeom, Gf
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--project", default=str(ROOT))
p.add_argument("--archive")
a = p.parse_args()
root = Path(a.dataset)
project = Path(a.project)
validation = json.loads((root / "validation_report.json").read_text())
native = json.loads((root / "native_playback_report.json").read_text())
assert validation["status"] == native["status"] == "passed"
layers, assets, unresolved = UsdUtils.ComputeAllDependencies(
    Sdf.AssetPath(str(root / "scene/room.usda"))
)
assert not unresolved, unresolved
outside = [
    x
    for x in assets
    if not Path(str(x)).resolve().is_relative_to((root / "scene").resolve())
]
assert not outside, outside
cal = json.loads((root / "calibration.json").read_text())
stage = Usd.Stage.Open(str(root / "scene/room.usda"))
projection_checks = []
poses = []
for name in ["Left", "Right"]:
    camera = UsdGeom.Camera(stage.GetPrimAtPath("/World/" + name))
    P = camera.GetCamera().frustum.ComputeProjectionMatrix()
    pixels = []
    for point in [(0, 0, -1), (0.1, 0, -1), (0, -0.1, -1)]:
        v = P.Transform(Gf.Vec3d(*point))
        pixels.append(
            np.array([(v[0] + 1) * cal["width"] / 2, (1 - v[1]) * cal["height"] / 2])
        )
    measured = np.array(
        [
            *(pixels[0]),
            (pixels[1][0] - pixels[0][0]) / 0.1,
            (pixels[2][1] - pixels[0][1]) / 0.1,
        ]
    )
    expected = np.array(
        [cal["K"][0][2], cal["K"][1][2], cal["K"][0][0], cal["K"][1][1]]
    )
    assert np.max(np.abs(measured - expected)) < 0.001, (measured, expected)
    projection_checks.append(
        dict(
            camera=name,
            cx=float(measured[0]),
            cy=float(measured[1]),
            fx=float(measured[2]),
            fy=float(measured[3]),
            max_error_px=float(np.max(np.abs(measured - expected))),
        )
    )
    T = np.array(
        UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    ).T
    T[:3, :3] = T[:3, :3] @ np.diag([1, -1, -1])
    poses.append(T)
assert np.allclose(np.linalg.inv(poses[0]) @ poses[1], cal["T_left_right"], atol=1e-9)
truth = np.load(root / "sensor_truth.npz")
index = (validation["stereo_pairs"] - 1) * 10
assert np.allclose(poses[0][:3, :3], truth["R"][index], atol=1e-9)
assert np.allclose(poses[0][:3, 3], truth["p"][index], atol=1e-9)
(root / "camera_projection_report.json").write_text(
    json.dumps(
        dict(
            status="passed",
            cameras=projection_checks,
            actual_stereo_extrinsics_match=True,
            last_render_camera_pose_matches_ground_truth=True,
        ),
        indent=2,
    )
)
source = root / "source"
source.mkdir(exist_ok=True)
for name in ["scripts", "config", "docs", "integrations"]:
    if (project / name).is_dir():
        shutil.copytree(
            project / name,
            source / name,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "._*"),
        )
for name in ["README.md", "LICENSE", "requirements.txt", "requirements-render.txt"]:
    if (project / name).is_file():
        shutil.copy2(project / name, source / name)
m = json.loads((root / "manifest.json").read_text())
m["status"] = "generated_validated_and_native_playback_verified"
m["native_ros2_humble_playback"] = "passed, all topic counts exact"
m["scene_dependencies_resolved"] = True
m["scene_external_assets"] = False
m["source_snapshot"] = "source/"
m["rendering_note"] = (
    "GPU selected by --gpu; automatic multi-GPU rendering explicitly disabled."
)
(root / "manifest.json").write_text(json.dumps(m, indent=2))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


files = sorted(f for f in root.rglob("*") if f.is_file() and f.name != "SHA256SUMS")
lines = []
for f in files:
    lines.append(f"{sha(f)}  {f.relative_to(root)}\n")
(root / "SHA256SUMS").write_text("".join(lines))
archive = Path(a.archive) if a.archive else root.with_suffix(".tar.gz")
assert not archive.exists()
subprocess.run(
    [
        "tar",
        "-I",
        "pigz -p 4 -1",
        "-cf",
        str(archive),
        "-C",
        str(root.parent),
        root.name,
    ],
    check=True,
)
archive_sha = sha(archive)
Path(str(archive) + ".sha256").write_text(f"{archive_sha}  {archive.name}\n")
delivery = dict(
    status="complete",
    dataset=str(root),
    rosbag=str(root / "rosbag"),
    archive=str(archive),
    archive_bytes=archive.stat().st_size,
    archive_sha256=archive_sha,
    dataset_bytes=sum(f.stat().st_size for f in root.rglob("*") if f.is_file()),
    file_count=len(files) + 1,
    stereo_pairs=m["camera_pairs"],
    imu_samples=m["imu_samples"],
    ground_truth_messages=m["ground_truth_samples"],
    native_ros2_humble_playback="passed",
    scene_dependencies_resolved=True,
    validation_report=str(root / "validation_report.json"),
    preview=str(root / "stereo_preview.mp4"),
)
(root.parent / (root.name + "_delivery.json")).write_text(
    json.dumps(delivery, indent=2)
)
print("DELIVERY_COMPLETE", json.dumps(delivery), flush=True)
