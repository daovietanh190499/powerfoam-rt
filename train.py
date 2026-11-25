"""PowerFoam trained with its own ray tracer (no rasterizer).

Loss = L2 + 0.2 * (1 - SSIM) + w(t) * interpenetration on rays traced through the Cech complex (BVH ray queries).
Densify / prune (PowerFoam's resample) is driven by the tracer's backward pass.
"""
import os
import time
import uuid

import configargparse
import numpy as np
import torch
import tqdm
import warp as wp
import yaml
from PIL import Image

from configs import *
from data_loader import DataHandler
from powerfoam.metrics import lpips_eval, psnr, ssim, ssim_eval
from powerfoam.scene import PowerfoamScene
from powerfoam.scheduling import get_exp_scheduler

torch.manual_seed(42)
np.random.seed(42)


def train(args):
    wp.init()
    if len(args.experiment_name) == 0:
        args.experiment_name = f"{args.scene}@{str(uuid.uuid4())[:8]}"
    out_dir = f"output/{args.experiment_name}"
    os.makedirs(f"{out_dir}/test", exist_ok=True)
    with open(f"{out_dir}/config.yaml", "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    log = open(f"{out_dir}/log.txt", "a")

    def P(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    train_split = "train" if args.eval else "all"
    test_split = "test" if args.eval else "all"
    test_data = DataHandler(args)
    test_data.reload(test_split, downsample=args.downsample[-1])
    train_data = DataHandler(args)
    train_data.reload(train_split, downsample=args.downsample[0])
    train_iter = train_data.get_iter()
    cur_downsample = args.downsample[0]
    P("Loaded dataset")

    model = PowerfoamScene(args)
    model.initialize_from_dataset(train_data, device="cuda")
    model.declare_optimizers(args, args.iterations)
    model.sort_points()
    model.rebuild_adjacency()
    args.init_points = model.points.shape[0]
    black = torch.zeros(3, device=model.device)

    def deg():
        return model.adjacency.shape[0] / model.points.shape[0]

    def test_loop(final=False):
        ps, ss, ls = [], [], []
        t0 = time.time()
        with torch.no_grad():
            for i, camera in enumerate(test_data.cameras):
                gt = test_data.rgbs[i].cuda()
                rgb, _ = model.forward(camera, black)
                rgb = rgb.clamp(0, 1)
                ps.append(psnr(rgb, gt).item())
                ss.append(ssim_eval(rgb, gt).item())
                if final:
                    ls.append(lpips_eval(rgb, gt).item())
                    im = (rgb.cpu() * 255).to(torch.uint8).numpy()
                    Image.fromarray(im).save(f"{out_dir}/test/{i:03d}.png")
        torch.cuda.synchronize()
        dt = (time.time() - t0) / len(ps)
        return (
            float(np.mean(ps)),
            float(np.mean(ss)),
            float(np.mean(ls)) if ls else None,
            dt,
        )

    P(f"init: {model.points.shape[0]} cells, adjacency degree {deg():.1f}")
    t_train = 0.0
    since_tri, tri_interval = 0, 1
    interp_sched = get_exp_scheduler(
        args.interpenetration_weight, 1e-3 * args.interpenetration_weight, args.iterations
    )
    with tqdm.trange(args.iterations, desc="Training") as bar:
        for it in bar:
            if it and it in args.downsample_iterations:
                ds = args.downsample[args.downsample_iterations.index(it)]
                if ds != cur_downsample:
                    train_data.reload(train_split, downsample=ds)
                    train_iter = train_data.get_iter()
                    cur_downsample = ds

            camera, rgb_gt, alpha_gt, _ = next(train_iter)
            rand_bg = torch.rand_like(rgb_gt)
            rgb_gt = rgb_gt + (1 - alpha_gt[..., None]) * rand_bg

            torch.cuda.synchronize()
            t0 = time.time()
            model.optimizer.zero_grad(set_to_none=True)
            if it == args.t_thresh_after:
                model.raytracer.t_thresh = args.t_thresh_late
            # PowerFoam: Cech complex rebuilt every `tri_interval` iterations (1 -> 20)
            since_tri += 1
            if since_tri % tri_interval == 0:
                model.rebuild_adjacency()
                since_tri = 0
                tri_interval = min(tri_interval + 1, 20)

            rgb, alpha = model.forward(camera, rand_bg, gt=rgb_gt, with_stats=True)
            rgb_loss = ((rgb - rgb_gt) ** 2).sum(-1).mean()
            interp_loss = model.interpenetration().sum()
            loss = (
                rgb_loss
                + 0.2 * (1 - ssim(rgb, rgb_gt))
                + interp_sched(it) * interp_loss
            )
            loss.backward()
            model.texture_regularizers()
            model.optimizer.step()
            model.update_learning_rate(it)
            st = model.stats
            model.update_stats(st["contrib"], st["point_error"])

            if it >= args.densify_from and it < args.densify_until:
                a = (args.final_points / args.init_points) ** (
                    1 / (args.densify_until - args.densify_from - 1)
                )
                target = int(args.init_points * (a ** (it - args.densify_from)))
            else:
                target = model.points.shape[0]
            if (
                it < int(0.95 * args.iterations)
                and it % args.resample_interval == args.resample_interval - 1
            ):
                model.resample(target)
                model.sort_points()
                model.rebuild_adjacency()
                since_tri = 0
            torch.cuda.synchronize()
            t_train += time.time() - t0

            if it % 20 == 0:
                bar.set_postfix(
                    N=model.points.shape[0],
                    deg=f"{deg():.0f}",
                    psnr=f"{psnr(rgb.detach(), rgb_gt).item():.2f}",
                    cells=f"{model.raytracer.last_count.float().mean().item():.0f}",
                    dist=f"{model.raytracer.last_dist_loss.item():.4f}",
                    ms=f"{1000 * t_train / (it + 1):.0f}",
                )
            if it % args.test_every == args.test_every - 1:
                p, s, _, dt = test_loop()
                P(
                    f"it {it + 1:5d}  N={model.points.shape[0]:6d}  deg={deg():.0f}  "
                    f"test PSNR {p:.2f}  SSIM {s:.3f}  ({1000 * dt:.0f} ms/img)  "
                    f"train {1000 * t_train / (it + 1):.0f} ms/it"
                )
                torch.cuda.empty_cache()
            if it % 1000 == 999:
                model.save_pt(f"{out_dir}/model.pt")

    model.rebuild_adjacency()
    model.save_pt(f"{out_dir}/model.pt")
    p, s, l, dt = test_loop(final=True)
    P(
        f"FINAL  PSNR {p:.4f}  SSIM {s:.4f}  LPIPS {l:.4f}  N={model.points.shape[0]}  "
        f"train time {t_train:.0f}s  render {1000 * dt:.0f} ms/img"
    )
    with open(f"{out_dir}/metrics.txt", "w") as f:
        f.write(f"Average PSNR:  {p:.4f}\nAverage SSIM:  {s:.4f}\nAverage LPIPS: {l:.4f}\n")
        f.write(f"Train time (s): {t_train:.0f}\n")


if __name__ == "__main__":
    parser = configargparse.ArgParser()
    get_params = add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True, help="Path to config file")
    train(get_params(parser.parse_args()))
