"""Verify both cameras along the complete trajectory against the actual USD mesh."""

import json
import numpy as np
from settings import WORK_DIR
from trajectory import evaluate, BASELINE
from mesh_query import load_scene, distance

state = evaluate(np.arange(24000) / 200)
right = state["p"] + state["R"][:, :, 0] * BASELINE
scene = load_scene(WORK_DIR / "office_mesh.npz")
minimum = float(distance(scene, np.concatenate([state["p"], right])).min())
assert minimum > 0.30, f"Camera mesh clearance is only {minimum:.3f} m"
report_path = WORK_DIR / "route_report.json"
report = json.loads(report_path.read_text())
report.update(
    status="passed",
    actual_mesh_clearance_pending=False,
    minimum_camera_mesh_clearance_m=minimum,
    maximum_speed_m_s=float(np.linalg.norm(state["v_world"], axis=1).max()),
    maximum_angular_speed_rad_s=float(np.linalg.norm(state["w_body"], axis=1).max()),
    trajectory_bounds=[state["p"].min(0).tolist(), state["p"].max(0).tolist()],
)
report_path.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
