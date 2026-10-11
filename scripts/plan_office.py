"""Rasterize authored floor and furniture to inspect navigable office space."""

import json
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, label
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from settings import WORK_DIR

PROJECT = WORK_DIR
mesh = json.loads((PROJECT / "mesh_manifest.json").read_text())
triangles = np.load(PROJECT / "office_mesh.npz")["triangles"]
floor_parts = []
indoor_parts = []
for obj in mesh["meshes"]:
    part = triangles[obj["start"] : obj["start"] + obj["triangles"]]
    if obj["indoor"]:
        indoor_parts.append(part)
    if "/SM_Floor" in obj["path"]:
        floor_parts.append(part[np.max(np.abs(part[:, :, 2]), axis=1) < 0.05])
floors = np.concatenate(floor_parts)
origin = np.floor(floors[:, :, :2].min((0, 1))) - 1
maximum = np.ceil(floors[:, :, :2].max((0, 1))) + 1
resolution = 0.1
shape = np.ceil((maximum - origin) / resolution).astype(int)
floor = np.zeros(shape[::-1], np.uint8)
obstacles = np.zeros_like(floor)
for tri in floors:
    xy = np.rint((tri[:, :2] - origin) / resolution).astype(np.int32)
    cv2.fillConvexPoly(floor, xy, 1)
indoor = np.concatenate(indoor_parts)
obstacle_tri = indoor[(indoor[:, :, 2].min(1) < 1.9) & (indoor[:, :, 2].max(1) > 0.25)]
for tri in obstacle_tri:
    xy = np.rint((tri[:, :2] - origin) / resolution).astype(np.int32)
    cv2.fillConvexPoly(obstacles, xy, 1)
distance = distance_transform_edt((floor > 0) & (obstacles == 0)) * resolution
free = distance > 0.45
components, n = label(free)
sizes = np.bincount(components.ravel())
sizes[0] = 0
largest = components == sizes.argmax()
np.savez_compressed(
    PROJECT / "floorplan.npz",
    origin=origin,
    resolution=resolution,
    floor=floor,
    obstacles=obstacles,
    distance=distance,
    free=largest,
)
fig, ax = plt.subplots(figsize=(10, 14))
image = np.ones((*floor.shape, 3)) * 0.25
image[floor > 0] = [0.84, 0.87, 0.91]
image[largest] = 1
image[(floor > 0) & (obstacles > 0)] = [0.1, 0.15, 0.2]
ax.imshow(image, origin="lower", extent=[origin[0], maximum[0], origin[1], maximum[1]])
ax.set(
    xlabel="World x (m)",
    ylabel="World y (m)",
    title="NVIDIA Office: white = free with 0.45 m body clearance",
)
ax.set_xticks(np.arange(origin[0], maximum[0] + 1, 2))
ax.set_yticks(np.arange(origin[1], maximum[1] + 1, 3))
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(PROJECT / "floorplan.png", dpi=130)
print(
    "FLOORPLAN",
    origin.tolist(),
    maximum.tolist(),
    "free_area",
    float(largest.sum() * resolution**2),
    "components",
    n,
    flush=True,
)
