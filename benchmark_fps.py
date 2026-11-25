"""python benchmark_fps.py output/EXP  -- inference speed (test cameras, training resolution)."""
import sys
import time
import types

import numpy as np
import torch
import warp as wp
import yaml

from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene

out = sys.argv[1]
wp.init()
args = types.SimpleNamespace(**yaml.safe_load(open(f"{out}/config.yaml")))
args.init_points = 1000
train = DataHandler(args)
train.reload("train", downsample=args.downsample[-1])
test = DataHandler(args)
test.reload("test", downsample=args.downsample[-1])
model = PowerfoamScene(args)
model.initialize_from_dataset(train, device="cuda")
model.load_pt(f"{out}/model.pt")
model.rebuild_adjacency()
black = torch.zeros(3, device="cuda")
cams = test.cameras
H, W = cams[0].height, cams[0].width


def bench(label, reps=3):
    with torch.no_grad():
        for c in cams[:3]:  # warm-up (kernel compilation, allocator)
            model.forward(c, black)
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            for c in cams:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model.forward(c, black)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1000
    print(
        f"{label:34s} {ts.mean():7.1f} ms/frame  ({1000 / ts.mean():5.2f} FPS)   "
        f"median {np.median(ts):.1f}  p10 {np.percentile(ts, 10):.1f}  p90 {np.percentile(ts, 90):.1f}"
    )


print(f"N={model.points.shape[0]} cells, Cech degree {model.adjacency.shape[0] / model.points.shape[0]:.1f}, "
      f"{W}x{H} px, {torch.cuda.get_device_name()}")
bench("forward (BVH rebuilt every frame)")
model.update_bvh()
model.update_bvh = lambda: None  # static scene: BVH built once
bench("forward (BVH prebuilt, static scene)")
print(f"peak VRAM allocated by torch: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

# inference without BVH / Cech: exact adjacency from qhull (computed once), cell-to-cell walk
t0 = time.perf_counter()
model.build_exact_adjacency()
torch.cuda.synchronize()
print(f"qhull adjacency (once): {time.perf_counter() - t0:.1f} s, degree {model.exact_adj.shape[0] / model.points.shape[0]:.1f}")
_forward = model.forward
model.forward = lambda c, b: (model.forward_walk(c, b), None)
bench("walk on qhull adjacency (no BVH)")
print(f"peak VRAM allocated by torch: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
