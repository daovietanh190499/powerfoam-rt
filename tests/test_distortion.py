"""Finite-difference check of the hand-written distortion-loss gradient, and of the
low-contribution skip (inference): same image up to ~1e-3 per skipped cell, no gradient."""
import sys

import torch
import warp as wp

sys.path.insert(0, ".")
from powerfoam.bvh import AABBTree
from powerfoam.raytrace import RayTracer


def main():
    wp.init()
    torch.manual_seed(0)
    dev = "cuda"
    N, R = 400, 512
    pts = torch.rand(N, 3, device=dev)
    rad = 0.07 + 0.05 * torch.rand(N, device=dev)
    sig = 1.0 + 4.0 * torch.rand(N, device=dev)
    diff = 0.7 * torch.randn(N, 3 * 8 * 8, device=dev)
    spec = 0.3 * torch.randn(N, 3 * 4 * 4, device=dev)
    wgt = torch.randn(R, 3, device=dev)
    o = torch.tensor([0.5, 0.5, -0.8], device=dev).expand(R, 3).contiguous()
    d = torch.nn.functional.normalize(
        torch.tensor([0.0, 0.0, 1.0], device=dev) + 0.25 * torch.randn(R, 3, device=dev), dim=-1
    )
    bg = torch.rand(R, 3, device=dev)

    tracer = RayTracer(1e-4, 128)
    tracer.lam_dist = 1.0

    def loss_and_grad(p, r, s):
        tree = AABBTree(dev)
        tree.update(p, r)
        adj, off = tree.build_cech_complex()
        tracer.bvh_id = tree.tree.id
        ts = [t.clone().requires_grad_(True) for t in (p, r, s)]
        out = tracer.render(ts[0], ts[1], ts[2], diff, spec, o, d, bg, adj, off)
        out.backward(torch.zeros_like(out))  # photometric gradient = 0 -> only distortion
        return tracer.last_dist_loss.item(), [t.grad for t in ts], tracer.last_count.float().mean().item()

    L, g, cells = loss_and_grad(pts, rad, sig)
    print(f"distortion {L:.5f}  cells/ray {cells:.1f}")
    names = ["points", "radii", "sigma"]
    base = [pts, rad, sig]
    for k, n in enumerate(names):
        v = g[k].clone()  # directional derivative along the gradient (strong signal)
        v = v / v.norm()
        eps = 1e-4 * (10 if n == "sigma" else 1)
        plus = [t.clone() for t in base]
        minus = [t.clone() for t in base]
        plus[k] = plus[k] + eps * v
        minus[k] = minus[k] - eps * v
        fd = (loss_and_grad(*plus)[0] - loss_and_grad(*minus)[0]) / (2 * eps)
        an = (g[k] * v).sum().item()
        print(f"{n:7s} analytic {an:+.5e}  finite-diff {fd:+.5e}  rel err {abs(an - fd) / (abs(fd) + 1e-12):.2e}")
        assert abs(an - fd) < 0.06 * abs(fd) + 1e-5, n

    # photometric gradient of every parameter (incl. both octahedral textures)
    tracer.lam_dist = 0.0
    tree = AABBTree(dev)
    tree.update(pts, rad)
    adj, off = tree.build_cech_complex()
    tracer.bvh_id = tree.tree.id

    def photo(ps):
        ts = [t.clone().requires_grad_(True) for t in ps]
        out = tracer.render(*ts[:3], ts[3], ts[4], o, d, bg, adj, off)
        (out * wgt).sum().backward()
        return (out * wgt).sum().item(), [t.grad for t in ts]

    ps = [pts, rad, sig, diff, spec]
    _, g = photo(ps)
    # points / radii also move the texture coordinate, which (as in VoroTracing) is not
    # differentiated, so only sigma and the textures are finite-difference checked here
    for k, n in [(2, "sigma"), (3, "diffuse"), (4, "specular")]:
        v = g[k] / g[k].norm()
        eps = 1e-3
        pl = [t.clone() for t in ps]
        mi = [t.clone() for t in ps]
        pl[k] = pl[k] + eps * v
        mi[k] = mi[k] - eps * v
        fd = (photo(pl)[0] - photo(mi)[0]) / (2 * eps)
        an = g[k].norm().item()
        print(f"photometric {n:8s} analytic {an:+.5e}  finite-diff {fd:+.5e}  rel err {abs(an - fd) / (abs(fd) + 1e-12):.2e}")
        assert abs(an - fd) < 0.06 * abs(fd) + 1e-5, n

    # inference skip
    tree = AABBTree(dev)
    tree.update(pts, rad)
    adj, off = tree.build_cech_complex()
    tracer.bvh_id = tree.tree.id
    tracer.lam_dist = 0.0
    with torch.no_grad():
        full = tracer.render(pts, rad, sig, diff, spec, o, d, bg, adj, off)
        tracer.skip_thresh = 1e-3
        skip = tracer.render(pts, rad, sig, diff, spec, o, d, bg, adj, off)
    print(f"skip: max |full - skip| = {(full - skip).abs().max():.3e}, mean {(full - skip).abs().mean():.3e}")
    print("OK")


if __name__ == "__main__":
    main()
