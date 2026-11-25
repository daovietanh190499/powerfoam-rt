"""Inference-only cell-to-cell walk through the *exact* power diagram.

No BVH and no Cech complex: the adjacency is the exact one of the weighted Delaunay
triangulation (qhull, ``exact_adj.py``), computed once before inference.  A ray starts
in the power cell that contains the camera and repeatedly leaves the current cell
through its exit face (smallest t among the neighbours).  The segment inside a cell
is  [t_near, t_far] = sphere ∩ {n_ij . x <= h_ij for all neighbours j}, exactly the
cell of the training tracer (``raytrace.py``), so the image is the same.

Same appearance model as training (octahedral diffuse + specular textures, sigmoid),
early termination at ``t_thresh`` and the low-contribution skip (``skip_thresh``).
"""

import numpy as np
import torch
import warp as wp

from .exact_adj import exact_edges_np
from .raytrace import BIG, diffuse_uv, oct_uv, sigmoid3, tex_bilinear

wp.config.quiet = True


def exact_adjacency(points, radii):
    """CSR (adjacency, offsets) of the regular triangulation of the power diagram."""
    N = points.shape[0]
    keys = torch.as_tensor(
        exact_edges_np(points.detach().cpu().numpy(), radii.detach().cpu().numpy()),
        device=points.device,
    )
    a, b = keys // N, keys % N
    S, D = torch.cat([a, b]), torch.cat([b, a])
    order = torch.argsort(S, stable=True)
    off = torch.zeros(N + 1, dtype=torch.int32, device=points.device)
    off[1:] = torch.bincount(S, minlength=N).cumsum(0).int()
    return D[order].int().contiguous(), off


@wp.func
def cell_walk(
    o: wp.vec3f,
    d: wp.vec3f,
    i: int,
    t_cur: float,
    points: wp.array(dtype=wp.vec3f),
    radii: wp.array(dtype=wp.float32),
    adjacency: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
):
    """(valid, t_near, t_far, next_cell, t_next) of the ray in cell i."""
    pi = points[i]
    ri = radii[i]
    y0 = o - pi
    t_near = float(0.0)
    t_far = float(BIG)
    valid = int(1)

    b = wp.dot(y0, d)
    disc = b * b - (wp.dot(y0, y0) - ri * ri)
    if disc <= 0.0:
        valid = 0
    else:
        s = wp.sqrt(disc)
        if -b + s <= 0.0:
            valid = 0
        else:
            t_far = -b + s
            if -b - s > 0.0:
                t_near = -b - s

    next_cell = int(-1)
    t_next = float(BIG)
    ri2 = ri * ri
    for k in range(offsets[i], offsets[i + 1]):
        j = adjacency[k]
        n = points[j] - pi
        rj = radii[j]
        h = 0.5 * (wp.dot(n, n) + ri2 - rj * rj)
        dp = wp.dot(n, d)
        num = h - wp.dot(n, y0)
        if wp.abs(dp) < 1.0e-12:
            if num < 0.0:
                valid = 0
        else:
            tf = num / dp
            if dp > 0.0:
                if tf < t_far:
                    t_far = tf
                if tf < t_next and tf > t_cur - 1.0e-5:
                    t_next = tf
                    next_cell = j
            else:
                if tf > t_near:
                    t_near = tf
    return valid, t_near, t_far, next_cell, t_next


@wp.kernel(enable_backward=False)
def walk_fwd(
    origins: wp.array(dtype=wp.vec3f),
    dirs: wp.array(dtype=wp.vec3f),
    starts: wp.array(dtype=wp.int32),
    bg: wp.array(dtype=wp.vec3f),
    points: wp.array(dtype=wp.vec3f),
    radii: wp.array(dtype=wp.float32),
    sigma: wp.array(dtype=wp.float32),
    diffuse: wp.array(dtype=wp.float32),
    specular: wp.array(dtype=wp.float32),
    rd: int,
    rs: int,
    adjacency: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    t_thresh: float,
    skip_thresh: float,
    max_steps: int,
    out_rgb: wp.array(dtype=wp.vec3f),
    out_steps: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    o = origins[tid]
    d = dirs[tid]
    i = starts[tid]
    spec_uv = oct_uv(-d, rs)

    T = float(1.0)
    C = wp.vec3f(0.0, 0.0, 0.0)
    t_cur = float(-1.0e-5)
    steps = int(0)
    while i >= 0 and steps < max_steps and T > t_thresh:
        valid, t_near, t_far, nxt, tn = cell_walk(o, d, i, t_cur, points, radii, adjacency, offsets)
        sg = sigma[i]
        dt = t_far - t_near
        if valid == 1 and dt > 0.0 and sg >= 0.0:
            e = wp.exp(-wp.min(sg * dt, 60.0))
            w = T * (1.0 - e)
            if w >= skip_thresh:
                duv = diffuse_uv(o, d, t_near, points[i], rd)
                logit = tex_bilinear(diffuse, i * 3 * rd * rd, rd, duv) + tex_bilinear(
                    specular, i * 3 * rs * rs, rs, spec_uv
                )
                C += w * sigmoid3(logit)
            T = T * e
        t_cur = wp.max(t_cur, tn)
        i = nxt
        steps += 1
    out_rgb[tid] = C + T * bg[tid]
    out_steps[tid] = steps


class WalkTracer:
    def __init__(self, t_thresh, skip_thresh, max_steps=4096):
        self.t_thresh, self.skip_thresh, self.max_steps = float(t_thresh), float(skip_thresh), max_steps
        self.last_steps = None

    @torch.no_grad()
    def render(self, points, radii, sigma, diffuse, specular, origins, dirs, bg, adjacency, offsets):
        dev = points.device
        R = origins.shape[0]
        rd = int(round((diffuse.shape[1] // 3) ** 0.5))
        rs = int(round((specular.shape[1] // 3) ** 0.5))
        eye = origins[0]
        start = torch.argmin(((eye[None] - points) ** 2).sum(-1) - radii**2).int()
        starts = start.expand(R).contiguous()
        out = torch.empty(R, 3, device=dev)
        steps = torch.empty(R, device=dev, dtype=torch.int32)
        v3 = lambda t: wp.from_torch(t.contiguous().float(), dtype=wp.vec3f)
        f1 = lambda t: wp.from_torch(t.contiguous().view(-1).float(), dtype=wp.float32)
        i1 = lambda t: wp.from_torch(t.contiguous().int(), dtype=wp.int32)
        wp.launch(
            walk_fwd,
            dim=R,
            inputs=[
                v3(origins), v3(dirs), i1(starts), v3(bg), v3(points), f1(radii), f1(sigma),
                f1(diffuse), f1(specular), rd, rs, i1(adjacency), i1(offsets),
                self.t_thresh, self.skip_thresh, self.max_steps, v3(out), i1(steps),
            ],
            block_dim=64,
            device=str(dev),
            stream=wp.stream_from_torch(torch.cuda.current_stream()),
        )
        self.last_steps = steps
        return out
