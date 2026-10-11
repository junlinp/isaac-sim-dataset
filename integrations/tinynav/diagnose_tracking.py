"""Locate relative tracking errors without changing estimator parameters."""

import argparse, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--map", required=True)
a = p.parse_args()
data = Path(a.dataset)
out = Path(a.map)
z = np.load(data / "sensor_truth.npz")
poses = np.load(out / "visual_odometry.npy", allow_pickle=True).item()
ts = np.array(sorted(poses), dtype=np.int64)
E = np.stack([poses[t] for t in ts])
idx = np.searchsorted(z["timestamps_ns"], ts)
G = np.tile(np.eye(4), (len(ts), 1, 1))
G[:, :3, :3] = z["R"][idx]
G[:, :3, 3] = z["p"][idx]
err = np.linalg.inv(np.linalg.inv(G[:-1]) @ G[1:]) @ (np.linalg.inv(E[:-1]) @ E[1:])
d = np.linalg.norm(err[:, :3, 3], axis=1)
r = np.degrees(Rotation.from_matrix(err[:, :3, :3]).magnitude())
order = np.argsort(d)[-10:][::-1]
report = dict(
    visual_interval_s=float(np.median(np.diff(ts)) / 1e9),
    native_image_throttle_s=0.1333,
    median_relative_translation_error_per_frame_m=float(np.median(d)),
    largest_relative_translation_errors=[
        dict(
            start_s=float((ts[i] - z["timestamps_ns"][0]) / 1e9),
            end_s=float((ts[i + 1] - z["timestamps_ns"][0]) / 1e9),
            translation_error_m=float(d[i]),
            rotation_error_deg=float(r[i]),
            ground_truth_speed_m_s=float(np.linalg.norm(z["v_world"][idx[i]])),
            ground_truth_angular_speed_rad_s=float(np.linalg.norm(z["w_body"][idx[i]])),
        )
        for i in order
    ],
    maximum_ground_truth_speed_m_s=float(np.linalg.norm(z["v_world"], axis=1).max()),
    maximum_ground_truth_angular_speed_rad_s=float(
        np.linalg.norm(z["w_body"], axis=1).max()
    ),
    interpretation="Tracking has localized relative-pose failures; high motion between native throttled visual updates and repetitive/glossy office appearance are hypotheses, not isolated causes. No estimator tuning or ground-truth feedback was applied.",
)
(out / "tracking_diagnostics.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
