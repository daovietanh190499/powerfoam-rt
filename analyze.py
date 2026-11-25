"""python analyze.py output/EXP -- train-view vs test-view PSNR, cell statistics."""
import sys
import types

import numpy as np
import torch
import warp as wp
import yaml

from data_loader import DataHandler
from powerfoam.metrics import psnr
from powerfoam.scene import PowerfoamScene

out = sys.argv[1]
wp.init()
args = types.SimpleNamespace(**yaml.safe_load(open(f"{out}/config.yaml")))
args.init_points = 1000
train = DataHandler(args)
train.reload("train", downsample=8)
test = DataHandler(args)
test.reload("test", downsample=8)
model = PowerfoamScene(args)
model.initialize_from_dataset(train, device="cuda")
model.load_pt(f"{out}/model.pt")
model.rebuild_adjacency()
black = torch.zeros(3, device="cuda")


def run(data, name, n):
    ps = []
    with torch.no_grad():
        for i in np.linspace(0, len(data.cameras) - 1, n).astype(int):
            rgb, _ = model.forward(data.cameras[i], black)
            ps.append(psnr(rgb.clamp(0, 1), data.rgbs[i].cuda()).item())
    print(f"{name}: PSNR {np.mean(ps):.2f} over {len(ps)} views")


run(train, "train", 24)
run(test, "test ", 24)
r = model.get_radii().detach()
s = model.get_density().detach()
print(f"N={r.numel()}  radius median {r.median():.4f} p90 {r.quantile(0.9):.4f}  density median {s.median():.2f} p90 {s.quantile(0.9):.2f}")
print(f"Cech degree {model.adjacency.shape[0] / r.numel():.1f}")
print(f"diffuse logit std {model.diffuse.detach().std():.3f}  specular logit std {model.specular.detach().std():.3f}")
