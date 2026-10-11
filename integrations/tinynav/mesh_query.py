from pathlib import Path
import sys
import numpy as np
import open3d as o3d


def load_scene(path):
    triangles = np.load(Path(path))["triangles"]
    vertices = triangles.reshape(-1, 3).astype(np.float32)
    faces = np.arange(len(vertices), dtype=np.uint32).reshape(-1, 3)
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(faces))
    return scene


def distance(scene, points):
    return scene.compute_distance(
        o3d.core.Tensor(np.asarray(points, dtype=np.float32))
    ).numpy()


def ray_depth(scene, origins, directions):
    rays = np.column_stack([origins, directions]).astype(np.float32)
    return scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
