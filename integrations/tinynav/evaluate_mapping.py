import argparse
import json
import shelve
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize(x):
    return dict(
        rmse=float(np.sqrt(np.mean(x * x))),
        median=float(np.median(x)),
        p95=float(np.percentile(x, 95)),
        maximum=float(np.max(x)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--map", required=True)
    args = parser.parse_args()
    data, out = Path(args.dataset), Path(args.map)
    with np.load(data / "sensor_truth.npz") as z:
        truth = {k: z[k] for k in ["timestamps_ns", "p", "R"]}
    gt_t = truth["timestamps_ns"]
    poses = np.load(out / "visual_odometry.npy", allow_pickle=True).item()
    ts = np.array(sorted(poses), dtype=np.int64)
    est = np.stack([poses[t] for t in ts])
    indices = np.searchsorted(gt_t, ts)
    assert np.array_equal(gt_t[indices], ts)
    gt = truth["p"][indices]
    ep = est[:, :3, 3]
    u, s, vt = np.linalg.svd((ep - ep.mean(0)).T @ (gt - gt.mean(0)))
    align = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
    offset = gt.mean(0) - align @ ep.mean(0)
    aligned = ep @ align.T + offset
    errors = np.linalg.norm(aligned - gt, axis=1)
    ori = Rotation.from_matrix(
        truth["R"][indices].transpose(0, 2, 1) @ align @ est[:, :3, :3]
    ).magnitude()
    j = np.searchsorted(ts, ts + 1_000_000_000)
    valid = j < len(ts)
    i = np.nonzero(valid)[0]
    j = j[valid]
    relative_est = np.linalg.inv(est[i]) @ est[j]
    gt_poses = np.tile(np.eye(4), (len(ts), 1, 1))
    gt_poses[:, :3, :3] = truth["R"][indices]
    gt_poses[:, :3, 3] = gt
    relative_gt = np.linalg.inv(gt_poses[i]) @ gt_poses[j]
    relative_error = np.linalg.inv(relative_gt) @ relative_est
    grid = np.load(out / "occupancy_grid.npy")
    meta = np.load(out / "occupancy_meta.npy")
    occupied = np.argwhere(grid == 2) * meta[3] + meta[:3] + meta[3] / 2
    occupied = occupied @ align.T + offset
    scene = json.loads((data / "scene/scene_manifest.json").read_text())
    from mesh_query import load_scene, distance, ray_depth

    mesh = load_scene(data / "scene/office_mesh.npz")
    distances = distance(mesh, occupied)
    pg = np.load(out / "poses.npy", allow_pickle=True).item()
    pg_t = np.array(sorted(pg), dtype=np.int64)
    pg_p = np.stack([pg[t][:3, 3] for t in pg_t]) @ align.T + offset
    pg_gt = truth["p"][np.searchsorted(gt_t, pg_t)]
    report = dict(
        alignment="rigid SE(3), scale fixed at 1; transform fitted to visual odometry, reused for map and pose graph",
        alignment_rotation=align.tolist(),
        alignment_translation=offset.tolist(),
        visual_samples=len(ts),
        time_span_seconds=float((ts[-1] - ts[0]) / 1e9),
        trajectory_length_m=float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum()),
        visual_ate_m=summarize(errors),
        visual_orientation_error_degrees=summarize(np.degrees(ori)),
        relative_pose_error_1s_translation_m=summarize(
            np.linalg.norm(relative_error[:, :3, 3], axis=1)
        ),
        relative_pose_error_1s_rotation_degrees=summarize(
            np.degrees(Rotation.from_matrix(relative_error[:, :3, :3]).magnitude())
        ),
        keyframes=len(pg),
        map_pose_graph_ate_m=summarize(np.linalg.norm(pg_p - pg_gt, axis=1)),
        occupancy_shape=list(grid.shape),
        occupancy_resolution_m=float(meta[3]),
        occupancy_counts={str(k): int((grid == k).sum()) for k in [0, 1, 2]},
        occupied_surface_distance_m=summarize(distances),
        occupied_fraction_within_20cm=float(np.mean(distances < 0.2)),
        finite_poses=bool(np.isfinite(est).all()),
        proper_rotations=bool(np.allclose(np.linalg.det(est[:, :3, :3]), 1)),
        baseline_m=float(np.load(out / "baseline.npy")),
        map_geometry_note="Distance to nearest triangle surface in NVIDIA Office mesh, after rigid trajectory alignment; occupancy is local raycast accumulation and is not a complete mesh.",
    )
    report["assessment"] = (
        "usable"
        if report["visual_ate_m"]["rmse"] < 0.2
        and report["occupied_fraction_within_20cm"] > 0.7
        else "mapping completed, accuracy insufficient for acceptance"
    )
    depth_errors, relative_depth_errors = [], []
    K = np.load(out / "intrinsics.npy")
    with np.load(out / "verification_depth_samples.npz") as depths:
        for t in pg_t[np.linspace(0, len(pg_t) - 1, min(12, len(pg_t))).astype(int)]:
            depth = depths[str(t)]
            yy, xx = np.mgrid[8 : depth.shape[0] : 16, 8 : depth.shape[1] : 16]
            rays = np.stack(
                [(xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1], np.ones_like(xx)],
                axis=-1,
            ).reshape(-1, 3)
            gi = np.searchsorted(gt_t, t)
            directions = rays @ truth["R"][gi].T
            origin = truth["p"][gi]
            expected = ray_depth(
                mesh, np.broadcast_to(origin, directions.shape), directions
            )
            measured = depth[yy, xx].ravel()
            valid_depth = (
                np.isfinite(expected)
                & np.isfinite(measured)
                & (measured > 0.1)
                & (measured < 30)
            )
            delta = np.abs(measured[valid_depth] - expected[valid_depth])
            depth_errors.extend(delta)
            relative_depth_errors.extend(delta / expected[valid_depth])
    report["stereo_depth_absolute_error_m"] = summarize(np.array(depth_errors))
    report["stereo_depth_relative_error"] = summarize(np.array(relative_depth_errors))
    report["stereo_depth_samples"] = len(depth_errors)
    report["stereo_depth_reference"] = (
        "First ray-triangle intersection with NVIDIA Office geometry using ground-truth camera pose; sampled pixels across up to 12 keyframes; ground truth used only here after mapping."
    )
    report["relative_pose_error_actual_interval_seconds"] = dict(
        minimum=float((ts[j] - ts[i]).min() / 1e9),
        median=float(np.median(ts[j] - ts[i]) / 1e9),
        maximum=float((ts[j] - ts[i]).max() / 1e9),
        convention="First available estimate at least 1 s later; no pose interpolation.",
    )
    report["proper_rotations"] = bool(
        report["proper_rotations"]
        and np.allclose(est[:, :3, :3].transpose(0, 2, 1) @ est[:, :3, :3], np.eye(3))
    )
    run = json.loads((out / "run_report.json").read_text())
    report["loop_closure_constraints"] = run["constraints"] - max(0, len(pg) - 1)
    report["acceptance_criteria"] = (
        "Diagnostic criteria chosen for this run: visual ATE RMSE < 0.20 m and >70% of occupied voxels within 0.20 m of a reference surface; not a TinyNav upstream benchmark standard."
    )
    with (out / "occupied_voxels_world.ply").open("w") as f:
        f.write(
            "ply\nformat ascii 1.0\nelement vertex "
            + str(len(occupied))
            + "\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
        )
        np.savetxt(f, occupied, fmt="%.5f")
    (out / "verification_report.json").write_text(json.dumps(report, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].plot(gt[:, 0], gt[:, 1], label="Ground truth")
    axes[0].plot(aligned[:, 0], aligned[:, 1], label="TinyNav visual odometry")
    axes[0].plot(pg_p[:, 0], pg_p[:, 1], ".", ms=2, label="Map keyframes")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f'Trajectory: ATE RMSE {report["visual_ate_m"]["rmse"]:.3f} m')
    axes[0].axis("equal")
    axes[1].plot((ts - ts[0]) / 1e9, errors)
    axes[1].set(
        xlabel="Time (s)",
        ylabel="Position error (m)",
        title="Rigid alignment, metric scale retained",
    )
    shown = occupied[:: max(1, len(occupied) // 30000)]
    axes[2].scatter(
        shown[:, 0], shown[:, 1], c=shown[:, 2], s=1, cmap="viridis", vmin=0, vmax=3
    )
    floorplan = np.load(data / "scene/floorplan.npz")
    yy, xx = np.indices(floorplan["obstacles"].shape)
    axes[2].contour(
        xx * float(floorplan["resolution"]) + floorplan["origin"][0],
        yy * float(floorplan["resolution"]) + floorplan["origin"][1],
        floorplan["obstacles"],
        levels=[0.5],
        colors="red",
        linewidths=0.4,
    )
    axes[2].axis("equal")
    axes[2].set_title("Occupied voxels; red = reference objects")
    fig.tight_layout()
    fig.savefig(out / "verification.png", dpi=160)
    for name, times, points in [
        ("estimated_visual_aligned", ts, aligned),
        ("map_keyframes_aligned", pg_t, pg_p),
    ]:
        matrices = (
            est[:, :3, :3]
            if name.startswith("estimated")
            else np.stack([pg[t][:3, :3] for t in times])
        )
        quats = Rotation.from_matrix(align @ matrices).as_quat()
        with (out / (name + ".tum")).open("w") as f:
            for t, p, q in zip(times, points, quats):
                f.write(
                    f"{t//1000000000}.{t%1000000000:09d} "
                    + " ".join(f"{v:.10f}" for v in [*p, *q])
                    + "\n"
                )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
