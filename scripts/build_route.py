import heapq
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from settings import WORK_DIR

P = WORK_DIR
g = np.load(P / "floorplan.npz")
origin = g["origin"]
res = float(g["resolution"])
free = g["free"]
clear = g["distance"]
waypoints = [
    (-17, 30),
    (-4, 28.5),
    (-1, 30),
    (-3.2, 36),
    (-2, 46),
    (-2.7, 49.2),
    (-1, 52),
    (-8, 52.3),
    (-16, 53.5),
    (-18, 49),
    (-22.6, 47.5),
    (-23, 39),
    (-22.5, 33),
]
cells = np.argwhere(free)


def cell(point):
    target = (np.array(point) - origin)[::-1] / res
    distances = np.linalg.norm(cells - target, axis=1)
    index = distances.argmin()
    assert distances[index] * res < 1.5, (point, distances[index] * res)
    return tuple(cells[index])


def astar(start, end):
    pending = [(0.0, start)]
    cost = {start: 0.0}
    parents = {}
    while pending:
        _, a = heapq.heappop(pending)
        if a == end:
            route = [a]
            while a in parents:
                a = parents[a]
                route.append(a)
            return route[::-1]
        for dy, dx in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]:
            b = (a[0] + dy, a[1] + dx)
            if not (
                0 <= b[0] < free.shape[0] and 0 <= b[1] < free.shape[1] and free[b]
            ):
                continue
            if dx and dy and not (free[a[0] + dy, a[1]] and free[a[0], a[1] + dx]):
                continue
            step = np.hypot(dx, dy) * (1 + 0.25 / max(clear[b], 0.05) ** 2)
            value = cost[a] + step
            if value < cost.get(b, np.inf):
                cost[b] = value
                parents[b] = a
                heapq.heappush(
                    pending, (value + np.hypot(b[0] - end[0], b[1] - end[1]), b)
                )
    raise RuntimeError(("No route", start, end))


nodes = [cell(p) for p in waypoints]
points = []
for a, b in zip(nodes, nodes[1:] + nodes[:1]):
    points.extend(astar(a, b)[:-1])
xy = np.array(points)[:, ::-1] * res + origin
distance = np.r_[
    0, np.linalg.norm(np.diff(np.vstack([xy, xy[0]]), axis=0), axis=1).cumsum()
]
sample = np.arange(0, distance[-1], 0.08)
closed = np.vstack([xy, xy[0]])
uniform = np.column_stack([np.interp(sample, distance, closed[:, j]) for j in [0, 1]])
for sigma in [6, 4, 3, 2]:
    smooth = gaussian_filter1d(uniform, sigma, axis=0, mode="wrap")
    nearest = np.rint((smooth - origin) / res).astype(int)
    if clear[nearest[:, 1], nearest[:, 0]].min() > 0.32:
        break
else:
    raise RuntimeError("No smooth route with clearance")
d1 = np.gradient(smooth, axis=0)
d2 = np.gradient(d1, axis=0)
curvature = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(
    np.linalg.norm(d1, axis=1) ** 3, 1e-9
)
ds = np.linalg.norm(np.roll(smooth, -1, axis=0) - smooth, axis=1)
weighted = np.r_[0, (ds * (1 + 3 * curvature)).cumsum()]
u = weighted / weighted[-1]
tck, _ = splprep(np.vstack([smooth, smooth[0]]).T, u=u, k=5, s=1e-5, per=True)
np.savez(
    P / "route_spline.npz", knots=tck[0], coefficients=np.array(tck[1]), degree=tck[2]
)
uv = np.linspace(0, 1, 12000)
route = np.array(splev(uv, tck)).T
nearest = np.rint((route - origin) / res).astype(int)
minimum = float(clear[nearest[:, 1], nearest[:, 0]].min())
assert minimum > 0.30, minimum
report = dict(
    status="planned",
    one_loop_length_m=float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum()),
    minimum_raster_clearance_m=minimum,
    waypoints=waypoints,
    smoothing_sigma_m=sigma * 0.08,
    phase_loops=1,
    timing="quintic easing over 110 s, parameter weighted by 1+3*curvature to slow turns",
    actual_mesh_clearance_pending=True,
)
(P / "route_report.json").write_text(json.dumps(report, indent=2))
fig, ax = plt.subplots(figsize=(10, 11))
ax.imshow(
    clear,
    origin="lower",
    extent=[
        origin[0],
        origin[0] + res * free.shape[1],
        origin[1],
        origin[1] + res * free.shape[0],
    ],
    vmin=0,
    vmax=2,
    cmap="gray",
)
ax.plot(route[:, 0], route[:, 1], color="dodgerblue", lw=2)
ax.plot(route[0, 0], route[0, 1], "go", label="Start/end")
ax.set(
    xlim=(-25, 7),
    ylim=(float(route[:, 1].min()) - 3, float(route[:, 1].max()) + 3),
    xlabel="World x (m)",
    ylabel="World y (m)",
    title="NVIDIA Office: smooth route through open office and meeting-area corridors",
)
ax.legend()
fig.tight_layout()
fig.savefig(P / "route.png", dpi=140)
print(json.dumps(report, indent=2))
