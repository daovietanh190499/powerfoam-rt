"""python compare.py  -- GT | PowerFoam (original) | powerfoam_rt  + error maps, per-image PSNR."""
import sys
import types

import numpy as np
import torch
import yaml
from PIL import Image

from data_loader import DataHandler

A = "../powerfoam-main/output/garden_powerfoam"
B = sys.argv[1] if len(sys.argv) > 1 else "output/garden_rt_cech"
args = types.SimpleNamespace(**yaml.safe_load(open(f"{B}/config.yaml")))
test = DataHandler(args)
test.reload("test" if args.eval else "all", downsample=args.downsample[-1])


def psnr(x, y):
    return -10 * np.log10(((x - y) ** 2).mean())


rows, pa, pb = [], [], []
for i in range(len(test.cameras)):
    gt = test.rgbs[i].numpy()
    a = np.asarray(Image.open(f"{A}/test/{i:03d}.png"))[:, :648, :3] / 255.0
    b = np.asarray(Image.open(f"{B}/test/{i:03d}.png"))[:, :, :3] / 255.0
    pa.append(psnr(a, gt))
    pb.append(psnr(b, gt))
    if i in (0, 5, 12, 20):
        ea = np.abs(a - gt).mean(-1, keepdims=True).repeat(3, -1) * 4
        eb = np.abs(b - gt).mean(-1, keepdims=True).repeat(3, -1) * 4
        rows.append(np.concatenate([gt, a, b, ea.clip(0, 1), eb.clip(0, 1)], axis=1))
print("img   PowerFoam   rt")
for i, (x, y) in enumerate(zip(pa, pb)):
    print(f"{i:3d}   {x:6.2f}   {y:6.2f}   diff {x - y:+.2f}")
print(f"mean  {np.mean(pa):.2f}   {np.mean(pb):.2f}")
Image.fromarray((np.concatenate(rows, axis=0) * 255).astype(np.uint8)).save(f"{B}/compare.png")
