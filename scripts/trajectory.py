"""Continuous trajectory and calibrated optical-frame IMU for the VIO pilot."""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import splev
from pathlib import Path
from settings import ROUTE_PATH

_route = np.load(ROUTE_PATH)
_TCK = (_route["knots"], list(_route["coefficients"]), int(_route["degree"]))
WIDTH, HEIGHT = 544, 640
FX = FY = 305.453491
CX = 269.162323
CY = 313.403473
BASELINE = 0.099401839
T0_NS = 1_800_000_000_000_000_000
CAMERA_DT_NS = 50_000_000
IMU_DT_NS = 5_000_000
OPTICAL_TO_FORWARD = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
GRAVITY = np.array([0.0, 0.0, -9.81])


def phase(t):
    t = np.asarray(t, float)
    s = np.clip((t - 5) / 110, 0, 1)
    q = 10 * s**3 - 15 * s**4 + 6 * s**5
    dq = (30 * s**2 - 60 * s**3 + 30 * s**4) / 110
    ddq = (60 * s - 180 * s**2 + 120 * s**3) / 110**2
    return 2 * np.pi * q, 2 * np.pi * dq, 2 * np.pi * ddq


def rotation(t):
    theta, _, _ = phase(t)
    dxy = np.array(splev(theta / (2 * np.pi), _TCK, der=1)).T
    yaw = np.arctan2(dxy[..., 1], dxy[..., 0])
    pitch = 0.04 * np.sin(theta)
    roll = 0.03 * np.sin(2 * theta)
    return (
        Rotation.from_euler("ZYX", np.stack([yaw, pitch, roll], axis=-1)).as_matrix()
        @ OPTICAL_TO_FORWARD
    )


def evaluate(t):
    t = np.atleast_1d(np.asarray(t, float))
    th, dth, ddth = phase(t)
    u = th / (2 * np.pi)
    du = dth / (2 * np.pi)
    ddu = ddth / (2 * np.pi)
    xy = np.array(splev(u, _TCK)).T
    dxy = np.array(splev(u, _TCK, der=1)).T
    ddxy = np.array(splev(u, _TCK, der=2)).T
    p = np.column_stack([xy, 1.35 + 0.06 * np.sin(2 * th)])
    v = np.column_stack([dxy * du[:, None], 0.12 * np.cos(2 * th) * dth])
    a = np.column_stack(
        [
            ddxy * du[:, None] ** 2 + dxy * ddu[:, None],
            -0.24 * np.sin(2 * th) * dth**2 + 0.12 * np.cos(2 * th) * ddth,
        ]
    )
    R = rotation(t)
    h = 1e-4
    dR = (
        rotation(t - 2 * h)
        - 8 * rotation(t - h)
        + 8 * rotation(t + h)
        - rotation(t + 2 * h)
    ) / (12 * h)
    skew = R.transpose(0, 2, 1) @ dR
    skew = (skew - skew.transpose(0, 2, 1)) / 2
    w = np.column_stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]])
    f = np.einsum("nji,nj->ni", R, a - GRAVITY)
    vb = np.einsum("nji,nj->ni", R, v)
    return dict(
        t=t,
        p=p,
        R=R,
        q_xyzw=Rotation.from_matrix(R).as_quat(),
        v_world=v,
        v_body=vb,
        a_world=a,
        w_body=w,
        f_body=f,
    )


def calibration():
    return dict(
        width=WIDTH,
        height=HEIGHT,
        camera_rate_hz=20,
        imu_rate_hz=200,
        K=[[FX, 0, CX], [0, FY, CY], [0, 0, 1]],
        D=[0.0] * 5,
        baseline_m=BASELINE,
        T_left_imu=np.eye(4).tolist(),
        T_left_right=[[1, 0, 0, BASELINE], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        world_frame="world",
        left_frame="left_camera_optical_frame",
        right_frame="right_camera_optical_frame",
        frame_convention="T_A_B maps coordinates in B into A; world Z up; camera/IMU x right, y down, z forward",
        gravity_world_m_s2=GRAVITY.tolist(),
        ground_truth_pose="T_world_left = T_world_imu",
        ground_truth_twist_frame="left_camera_optical_frame",
    )
