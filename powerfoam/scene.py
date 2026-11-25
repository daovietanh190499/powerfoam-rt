import math
import fpsample
import numpy as np
from scipy.spatial import KDTree

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import InterpenetrationFunction, morton_sort
from .bvh import AABBTree
from .raytrace import RayTracer
from .walk import WalkTracer, exact_adjacency
from .scheduling import get_cosine_scheduler


def camera_rays(camera, device):
    """Pinhole rays of a PowerFoam camera (same convention as get_ray_dir)."""
    eye = camera.eye.to(device).float()
    right = camera.right.to(device).float()
    up = camera.up.to(device).float()
    H, W = camera.height, camera.width
    fwd = torch.cross(up, right, dim=-1)
    fwd = fwd / fwd.norm()
    i = torch.arange(H, device=device, dtype=torch.float32)
    j = torch.arange(W, device=device, dtype=torch.float32)
    y = 1.0 - 2.0 * i / (H - 1.0)
    x = 2.0 * j / (W - 1.0) - 1.0
    dirs = (
        x[None, :, None] * right[None, None, :]
        + y[:, None, None] * up[None, None, :]
        + fwd[None, None, :]
    )
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    dirs = dirs.reshape(-1, 3).contiguous()
    origins = eye[None, :].expand(dirs.shape[0], 3).contiguous()
    return eye, origins, dirs


def init_points_sfm(data_handler, num_points):
    init_points = data_handler.points3D.float()
    cpu_points = init_points.cpu().numpy()

    sample_points = min(num_points, int(0.95 * init_points.shape[0]))
    print(f"Sampling {sample_points} points from {init_points.shape[0]} sfm points")
    point_inds = fpsample.bucket_fps_kdtree_sampling(cpu_points, sample_points)
    point_inds = torch.tensor(point_inds, dtype=torch.long)

    if num_points < 0.9 * init_points.shape[0]:
        init_points = init_points[point_inds]

    else:
        kdtree = KDTree(cpu_points)
        _, knn_i = kdtree.query(cpu_points, k=6)
        w = torch.rand(num_points - sample_points, 6, device=init_points.device)
        w = w / w.sum(dim=-1, keepdim=True)
        j = torch.randint(0, knn_i.shape[0], (num_points - sample_points,))
        knn_i = torch.tensor(knn_i, dtype=torch.long, device=init_points.device)
        knn_i = knn_i[j]
        _init_points = (w[:, :, None] * init_points[knn_i]).sum(dim=1)

        init_points = torch.cat([init_points[point_inds], _init_points], dim=0)

    return init_points


def init_points_bounded(data_handler, num_points):
    # LOCAL EDIT (smoke test): upstream samples in [0,1]^3, which misses the
    # Blender lego scene (centered at origin, extent ~[-1.5, 1.5]^3).
    points = torch.rand((num_points, 3), dtype=torch.float32) * 3.0 - 1.5
    return points


def init_points_unbounded(data_handler, num_points):
    cameras = data_handler.cameras
    camera_centers = torch.stack([cam.eye for cam in cameras], dim=0)

    camera_mean = camera_centers.mean(dim=0)
    camera_std = camera_centers.std(dim=0)

    points = torch.randn((num_points, 3), dtype=torch.float32) * camera_std * 3.0
    points = points + camera_mean[None, :]
    return points


PARAM_NAMES = ["points", "density", "radii", "diffuse", "specular"]


class PowerfoamScene(nn.Module):
    """Cells = power cell ∩ own sphere; per cell: site, radius, density, SH colour."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.raytracer = RayTracer(args.t_thresh, args.max_cells)
        self.aabb_tree = AABBTree(None)
        self.stats = {}

    @property
    def device(self):
        return self.points.device

    # ------------------------------------------------------------------ init
    def initialize_from_dataset(self, data_handler, device):
        a = self.args
        if a.init_type == "sfm":
            if data_handler.points3D is None:
                raise ValueError("No 3D points found in COLMAP output.")
            pts = init_points_sfm(data_handler, a.init_points)
        elif a.init_type == "random_bounded":
            pts = init_points_bounded(data_handler, a.init_points)
        elif a.init_type == "random_unbounded":
            pts = init_points_unbounded(data_handler, a.init_points)
        else:
            raise ValueError(f"Unknown init_type {a.init_type}")
        pts = pts.float().to(device)
        self.points = nn.Parameter(pts)

        # PowerFoam init: radius = mean distance to the 8 nearest sites, clamped so
        # that a cell seen by a camera projects to at most ~10 % of the image
        cpu = pts.cpu().numpy()
        knn_d, _ = KDTree(cpu).query(cpu, k=8)
        radii = torch.tensor(knn_d.mean(axis=1), dtype=torch.float32)
        max_radii = 100 * torch.ones(pts.shape[0])
        pc = pts.cpu()
        for cam in data_handler.cameras:
            v = cam.eye.cpu()[None, :] - pc
            x, y = cam.right.cpu(), cam.up.cpu()
            z = torch.cross(x, y, dim=-1)
            v_x = (v * x[None, :]).sum(-1) / x.norm()
            v_y = (v * y[None, :]).sum(-1) / y.norm()
            v_z = (v * z[None, :]).sum(-1) / z.norm()
            m_c = 0.1 * v_z * y.norm()
            mask = (v_z > 0) & ((v_x / v_z).abs() < x.norm()) & ((v_y / v_z).abs() < y.norm())
            max_radii[mask] = torch.minimum(m_c[mask], max_radii[mask])
        radii = torch.minimum(radii, max_radii)
        self.radii = nn.Parameter(radii.to(device))
        self.density = nn.Parameter(torch.full((pts.shape[0],), math.log(a.density_init), device=device))
        # octahedral textures (logits, sigmoid -> colour): diffuse indexed by the direction
        # site -> entry point of the ray, specular residual indexed by the view direction
        n = pts.shape[0]
        self.diffuse = nn.Parameter(torch.zeros(n, 3 * a.oct_res**2, device=device))
        self.specular = nn.Parameter(torch.zeros(n, 3 * a.spec_res**2, device=device))

    # ------------------------------------------------------------ attributes
    def get_radii(self):
        return F.softplus(self.radii, beta=100)

    def get_density(self):
        """sigma = exp(rho): the ray-segment length cancels in dL/d rho (VoroTracing 5.4)."""
        return torch.exp(self.density.clamp(max=12.0))

    # ------------------------------------------------------------- adjacency
    def update_bvh(self):
        """BVH over the spheres of the current sites (used for the ray queries)."""
        with torch.no_grad():
            self.aabb_tree.device = self.device
            self.aabb_tree.update(self.points.detach(), self.get_radii().detach())
            self.raytracer.bvh_id = self.aabb_tree.tree.id

    def rebuild_adjacency(self):
        """Cech complex of the spheres (PowerFoam): i ~ j iff |p_i - p_j| < r_i + r_j."""
        self.update_bvh()
        with torch.no_grad():
            self.adjacency, self.adjacency_offsets = self.aabb_tree.build_cech_complex()

    # ------------------------------------------------------------- rendering
    def forward(self, camera, bg, gt=None, with_stats=False):
        """camera -> (rgb (H,W,3), alpha (H,W)).  bg: (H,W,3) or (3,) background."""
        dev = self.device
        eye, origins, dirs = camera_rays(camera, dev)
        H, W = camera.height, camera.width
        self.update_bvh()
        if bg.dim() == 3:
            bgr = bg.reshape(-1, 3).contiguous()
        else:
            bgr = bg[None].expand(H * W, 3).contiguous()
        stats = {} if with_stats else None
        training = torch.is_grad_enabled()
        self.raytracer.lam_dist = self.args.distortion_weight if training else 0.0
        self.raytracer.skip_thresh = 0.0 if training else self.args.skip_thresh
        rgb = self.raytracer.render(
            self.points,
            self.get_radii(),
            self.get_density(),
            self.diffuse,
            self.specular,
            origins,
            dirs,
            bgr,
            self.adjacency,
            self.adjacency_offsets,
            gt=None if gt is None else gt.reshape(-1, 3),
            stats=stats,
        )
        self.stats = stats if stats is not None else {}
        alpha = 1.0 - self.raytracer.last_T
        return rgb.view(H, W, 3), alpha.view(H, W)

    def build_exact_adjacency(self):
        """Inference: exact power-diagram adjacency (qhull), once; no BVH / Cech needed afterwards."""
        with torch.no_grad():
            self.exact_adj, self.exact_off = exact_adjacency(self.points, self.get_radii())

    @torch.no_grad()
    def forward_walk(self, camera, bg):
        """Inference render by walking cell to cell through the exact adjacency (no BVH, no Cech)."""
        if not hasattr(self, "walker"):
            self.walker = WalkTracer(self.raytracer.t_thresh, self.args.skip_thresh)
        eye, origins, dirs = camera_rays(camera, self.device)
        H, W = camera.height, camera.width
        bgr = bg.reshape(-1, 3).contiguous() if bg.dim() == 3 else bg[None].expand(H * W, 3).contiguous()
        rgb = self.walker.render(
            self.points, self.get_radii(), self.get_density(), self.diffuse, self.specular,
            origins, dirs, bgr, self.exact_adj, self.exact_off,
        )
        return rgb.view(H, W, 3)

    def interpenetration(self):
        """Per cell: sum over Cech neighbours of (r_i + r_j - d_ij)^2 (overlapping spheres)."""
        return InterpenetrationFunction.apply(
            self.points, self.get_radii(), self.adjacency, self.adjacency_offsets
        )

    # ---------------------------------------------------------- optimisation
    def declare_optimizers(self, args, iterations):
        groups = [
            dict(params=self.points, lr=args.points_lr_init, name="points"),
            dict(params=self.density, lr=args.density_lr_init, name="density"),
            dict(params=self.radii, lr=args.radii_lr_init, name="radii"),
            dict(params=self.diffuse, lr=args.attr_lr_init, name="diffuse"),
            dict(params=self.specular, lr=args.attr_lr_init, name="specular"),
        ]
        self.optimizer = torch.optim.Adam(groups, eps=1e-15)
        self.schedulers = {
            "points": get_cosine_scheduler(args.points_lr_init, args.points_lr_final, max_steps=iterations),
            "density": get_cosine_scheduler(
                args.density_lr_init, args.density_lr_final, warmup_steps=1_000, max_steps=iterations
            ),
            "radii": get_cosine_scheduler(
                args.radii_lr_init, args.radii_lr_final, warmup_steps=1_000, max_steps=iterations
            ),
            "diffuse": get_cosine_scheduler(args.attr_lr_init, args.attr_lr_final, max_steps=iterations),
            "specular": get_cosine_scheduler(args.attr_lr_init, args.attr_lr_final, max_steps=iterations),
        }

    def update_learning_rate(self, it):
        for g in self.optimizer.param_groups:
            lr = self.schedulers[g["name"]](it)
            if g["name"] == "specular" and it < self.args.specular_start:
                lr *= (it / self.args.specular_start) ** 2  # specular fades in (VoroTracing)
            g["lr"] = lr

    def texture_regularizers(self):
        """VoroTracing manual gradients: specular -> 0, diffuse texels -> their per-cell mean."""
        a = self.args
        with torch.no_grad():
            if a.specular_reg_weight > 0 and self.specular.grad is not None:
                self.specular.grad.add_(self.specular, alpha=2.0 * a.specular_reg_weight / self.specular.numel())
            if a.diffuse_mean_pull_weight > 0 and self.diffuse.grad is not None:
                R = a.oct_res
                diff = self.diffuse.view(-1, R * R, 3)
                mean = diff.mean(dim=1, keepdim=True)
                scale = 2.0 * a.diffuse_mean_pull_weight / diff.numel()
                g = self.diffuse.grad.view(-1, R * R, 3)
                g.add_(diff, alpha=scale)
                g.add_(mean, alpha=-scale)

    def _reindex(self, idx):
        """Re-index every parameter (and its Adam state) with ``idx``."""
        self._gen = getattr(self, "_gen", 0) + 1
        new = {}
        for g in self.optimizer.param_groups:
            old = g["params"][0]
            st = self.optimizer.state.get(old, None)
            if st is not None:
                st["exp_avg"] = st["exp_avg"][idx]
                st["exp_avg_sq"] = st["exp_avg_sq"][idx]
                del self.optimizer.state[old]
            p = nn.Parameter(old.detach()[idx].requires_grad_(True))
            g["params"][0] = p
            if st is not None:
                self.optimizer.state[p] = st
            new[g["name"]] = p
        for n in PARAM_NAMES:
            setattr(self, n, new[n])

    def sort_points(self):
        perm = morton_sort(self.points)
        self._reindex(perm)
        for n in ("contrib_ema", "point_error_ema"):
            if hasattr(self, n):
                setattr(self, n, getattr(self, n)[perm])

    # ----------------------------------------------------- densify / prune
    def update_stats(self, contrib, point_error):
        with torch.no_grad():
            vis = (contrib > 0).float()
            alpha = 0.99 * vis + (1 - vis)
            if hasattr(self, "contrib_ema"):
                self.contrib_ema = alpha * self.contrib_ema + (1 - alpha) * contrib
                self.point_error_ema = 0.99 * self.point_error_ema + 0.01 * point_error
            else:
                self.contrib_ema = torch.ones_like(contrib) * 1e-5
                self.point_error_ema = torch.ones_like(point_error) * 1e-5

    def resample(self, target):
        """PowerFoam's resample: drop low-contribution cells, duplicate high-error ones."""
        with torch.no_grad():
            q = torch.quantile(self.contrib_ema, torch.tensor([0.1, 0.99], device=self.device), dim=0)
            thr = torch.min(torch.tensor(1 / (target * 25), device=self.device), q[0])
            valid = torch.nonzero(self.contrib_ema > thr, as_tuple=False)[:, 0]
            num = target - valid.shape[0]
            if num == 0:
                return 0
            eq = torch.quantile(self.point_error_ema, 0.99, dim=0)
            prob = self.point_error_ema[valid].clamp(max=eq)
            pick = torch.multinomial(prob, num, replacement=False)
            dup = torch.ones_like(valid)
            dup.index_add_(0, pick, torch.ones_like(pick))
            dup_mask = dup > 1
            cells = valid[pick]
            new_idx = torch.cat([valid, cells])
            new_dup_mask = torch.cat([dup_mask, dup_mask[pick]]).bool()
            new_dup_cnt = torch.cat([dup, dup[pick]])

            self._reindex(new_idx)
            direction = torch.randn_like(self.points)
            direction = direction / direction.norm(dim=-1, keepdim=True)
            pert = 0.05 * self.get_radii().detach()[:, None] * direction
            self.points[new_dup_mask] += pert[new_dup_mask]
            self.contrib_ema = self.contrib_ema[new_idx] / new_dup_cnt.float()
            self.point_error_ema = self.point_error_ema[new_idx] / new_dup_cnt.float()
        return num

    # -------------------------------------------------------------------- io
    def save_pt(self, path):
        torch.save(
            {
                "points": self.points.detach().cpu(),
                "radii": self.radii.detach().cpu(),
                "density": self.density.detach().cpu(),
                "diffuse": self.diffuse.detach().cpu(),
                "specular": self.specular.detach().cpu(),
                "adjacency": self.adjacency.cpu(),
                "adjacency_offsets": self.adjacency_offsets.cpu(),
            },
            path,
        )

    def load_pt(self, path):
        d = torch.load(path)
        self.raytracer.t_thresh = self.args.t_thresh_late  # trained with early termination
        for n in PARAM_NAMES:
            getattr(self, n).data = d[n].to(self.device)
        self.adjacency = d["adjacency"].to(self.device)
        self.adjacency_offsets = d["adjacency_offsets"].to(self.device)
