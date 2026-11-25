"""python evaluate.py output/EXPERIMENT [--fresh]  -- test metrics of a saved model.

The adjacency stored in model.pt is replaced by a freshly computed exact one
when --fresh is given (the stored one may be up to ``adj_interval`` its stale).
"""
import sys
import time
import types

import numpy as np
import torch
import warp as wp
import yaml

from data_loader import DataHandler
from powerfoam.metrics import lpips_eval, psnr, ssim_eval
from powerfoam.scene import PowerfoamScene


def main():
    out_dir = sys.argv[1]
    fresh = "--fresh" in sys.argv
    wp.init()
    args = types.SimpleNamespace(**yaml.safe_load(open(f"{out_dir}/config.yaml")))
    args.init_points = 1000
    test_split = "test" if args.eval else "all"
    train = DataHandler(args)
    train.reload("train" if args.eval else "all", downsample=args.downsample[-1])
    test = DataHandler(args)
    test.reload(test_split, downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(train, device="cuda")
    model.load_pt(f"{out_dir}/model.pt")
    if fresh:
        model.rebuild_adjacency()
    black = torch.zeros(3, device=model.device)
    ps, ss, ls, steps = [], [], [], []
    with torch.no_grad():
        for i, cam in enumerate(test.cameras):
            gt = test.rgbs[i].cuda()
            rgb, alpha = model.forward(cam, black)
            rgb = rgb.clamp(0, 1)
            ps.append(psnr(rgb, gt).item())
            ss.append(ssim_eval(rgb, gt).item())
            ls.append(lpips_eval(rgb, gt).item())
            steps.append(model.raytracer.last_count.float().mean().item())
    print(f"N={model.points.shape[0]} fresh={fresh} PSNR {np.mean(ps):.4f} SSIM {np.mean(ss):.4f} "
          f"LPIPS {np.mean(ls):.4f} cells/ray {np.mean(steps):.1f}")
    print("per-image PSNR:", " ".join(f"{p:.1f}" for p in ps))


if __name__ == "__main__":
    main()
