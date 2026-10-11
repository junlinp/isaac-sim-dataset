"""CPU checks for the bundled 120 s route and optical-frame inertial convention."""

import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from trajectory import evaluate, calibration, GRAVITY


def test_stationary_optical_imu_and_closed_route():
    s = evaluate([0.0, 2.0, 5.0, 115.0, 118.0, 119.995])
    assert np.allclose(s["v_world"], 0, atol=1e-10)
    assert np.allclose(s["w_body"], 0, atol=1e-7)
    assert np.allclose(s["p"], s["p"][0], atol=1e-9)
    assert np.allclose(s["R"], s["R"][0], atol=1e-9)
    # Bring specific force back into the world: stationary + gravity = zero.
    world_acc = np.einsum("nij,nj->ni", s["R"], s["f_body"]) + GRAVITY
    assert np.allclose(world_acc, 0, atol=1e-8)
    assert np.array_equal(calibration()["T_left_imu"], np.eye(4))


def test_pose_derivatives_and_angular_velocity():
    t = np.array([12.3, 28.4, 48.7, 73.1, 94.2, 108.5])
    h = 1e-4
    s, before, after = evaluate(t), evaluate(t - h), evaluate(t + h)
    assert np.allclose((after["p"] - before["p"]) / (2 * h), s["v_world"], atol=1e-5)
    assert np.allclose(
        (after["v_world"] - before["v_world"]) / (2 * h), s["a_world"], atol=1e-4
    )
    increment = Rotation.from_matrix(
        before["R"].transpose(0, 2, 1) @ after["R"]
    ).as_rotvec() / (2 * h)
    assert np.allclose(increment, s["w_body"], atol=1e-5)


def test_discrete_ideal_imu_recovers_full_trajectory():
    s = evaluate(np.arange(24000) / 200)
    R, p, v = s["R"][0].copy(), s["p"][0].copy(), s["v_world"][0].copy()
    max_p = max_v = max_r = 0.0
    dt = 0.005
    for i in range(len(s["t"]) - 1):
        w = (s["w_body"][i] + s["w_body"][i + 1]) / 2
        f = (s["f_body"][i] + s["f_body"][i + 1]) / 2
        acceleration = R @ Rotation.from_rotvec(w * dt / 2).as_matrix() @ f + GRAVITY
        p = p + v * dt + 0.5 * acceleration * dt**2
        v = v + acceleration * dt
        R = R @ Rotation.from_rotvec(w * dt).as_matrix()
        max_p = max(max_p, np.linalg.norm(p - s["p"][i + 1]))
        max_v = max(max_v, np.linalg.norm(v - s["v_world"][i + 1]))
        max_r = max(max_r, Rotation.from_matrix(R.T @ s["R"][i + 1]).magnitude())
    assert max_p < 0.01
    assert max_v < 0.01
    assert max_r < 0.0001
