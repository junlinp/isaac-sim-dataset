"""Export a deterministic verification subset using the mapper's DB backend."""

import argparse
import shelve
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--map", required=True)
a = p.parse_args()
out = Path(a.map)
poses = np.load(out / "poses.npy", allow_pickle=True).item()
times = np.array(sorted(poses), dtype=np.int64)
selected = times[np.linspace(0, len(times) - 1, min(12, len(times))).astype(int)]
with shelve.open(str(out / "depths"), flag="r") as db:
    samples = {str(t): db[str(t)] for t in selected}
np.savez_compressed(out / "verification_depth_samples.npz", **samples)
print("DEPTH_VERIFICATION_SAMPLES", len(samples), flush=True)
