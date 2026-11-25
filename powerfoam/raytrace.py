"""Differentiable ray tracer of the PowerFoam cell model with the Cech complex.

Cell i = power cell (site p_i, power weight r_i^2) clipped by its own sphere
|x - p_i| <= r_i, constant density sigma_i, view dependent colour
c_i(d) = max(0.5 + sum_k SH_k(d) sh_ik, 0).

Neighbours (PowerFoam, no inflation of the radii)
-------------------------------------------------
Neighbours only *clip* a cell: the face between i and j, n . x = h with
n = p_j - p_i, h = (|n|^2 + r_i^2 - r_j^2) / 2, cuts the sphere of i iff
|n| < r_i + r_j, i.e. iff the two spheres overlap (Cech complex, built on the GPU
with a BVH every few iterations).  Hence  sphere ∩ {all Cech faces}  is exactly
the cell  power cell ∩ sphere.

Tracing
-------
No walk (a Cech complex cannot be walked through the empty space between the
spheres).  Every ray queries the BVH of the spheres, computes the interval
[t_near, t_far] of each hit cell (sphere ∩ Cech faces), keeps the K nearest by
t_near (cells are disjoint, so this is the depth order), and volume-renders them

    alpha_i = 1 - exp(-sigma_i dt_i),  C += T alpha_i c_i,  T *= 1 - alpha_i.

Back-propagation is hand written.  The backward kernel re-evaluates the stored
cell list and uses the "rest of the ray" identity C = C_before + T_i [alpha_i c_i
+ (1 - alpha_i) R_i] (R_i from the float64 accumulated colour).  Gradients:
sigma, SH, sites and radii (through t_near / t_far: a sphere root or a face).
"""

import torch
import warp as wp

wp.config.quiet = True

BIG = 1.0e30


@wp.func
def oct_uv(dir: wp.vec3f, R: int) -> wp.vec2f:
    """Octahedral encoding of a unit direction -> continuous texel coordinates (VoroTracing)."""
    l1 = wp.abs(dir[0]) + wp.abs(dir[1]) + wp.abs(dir[2])
    inv = 1.0 / wp.max(l1, 1.0e-10)
    u = dir[0] * inv
    v = dir[1] * inv
    if dir[2] < 0.0:
        su = wp.where(u >= 0.0, 1.0, -1.0)
        sv = wp.where(v >= 0.0, 1.0, -1.0)
        ou = (1.0 - wp.abs(v)) * su
        ov = (1.0 - wp.abs(u)) * sv
        u = ou
        v = ov
    return wp.vec2f((u * 0.5 + 0.5) * float(R) - 0.5, (v * 0.5 + 0.5) * float(R) - 0.5)


@wp.func
def oct_index(u: int, v: int, R: int) -> int:
    """Texel index with the octahedral wrap (outside texels mirror across the border)."""
    wu = u
    wv = v
    ou = int(0)
    ov = int(0)
    if u < 0:
        wu = u + R
        ou = 1
    elif u >= R:
        wu = u - R
        ou = 1
    if v < 0:
        wv = v + R
        ov = 1
    elif v >= R:
        wv = v - R
        ov = 1
    if ou != ov:
        wu = R - 1 - wu
        wv = R - 1 - wv
    return wv * R + wu


@wp.func
def tex_bilinear(tex: wp.array(dtype=wp.float32), base: int, R: int, uv: wp.vec2f):
    u0 = int(wp.floor(uv[0]))
    v0 = int(wp.floor(uv[1]))
    fu = uv[0] - float(u0)
    fv = uv[1] - float(v0)
    i00 = base + 3 * oct_index(u0, v0, R)
    i10 = base + 3 * oct_index(u0 + 1, v0, R)
    i01 = base + 3 * oct_index(u0, v0 + 1, R)
    i11 = base + 3 * oct_index(u0 + 1, v0 + 1, R)
    w00 = (1.0 - fu) * (1.0 - fv)
    w10 = fu * (1.0 - fv)
    w01 = (1.0 - fu) * fv
    w11 = fu * fv
    return wp.vec3f(
        w00 * tex[i00] + w10 * tex[i10] + w01 * tex[i01] + w11 * tex[i11],
        w00 * tex[i00 + 1] + w10 * tex[i10 + 1] + w01 * tex[i01 + 1] + w11 * tex[i11 + 1],
        w00 * tex[i00 + 2] + w10 * tex[i10 + 2] + w01 * tex[i01 + 2] + w11 * tex[i11 + 2],
    )


@wp.func
def tex_scatter(grad: wp.array(dtype=wp.float32), base: int, R: int, uv: wp.vec2f, g: wp.vec3f):
    u0 = int(wp.floor(uv[0]))
    v0 = int(wp.floor(uv[1]))
    fu = uv[0] - float(u0)
    fv = uv[1] - float(v0)
    i00 = base + 3 * oct_index(u0, v0, R)
    i10 = base + 3 * oct_index(u0 + 1, v0, R)
    i01 = base + 3 * oct_index(u0, v0 + 1, R)
    i11 = base + 3 * oct_index(u0 + 1, v0 + 1, R)
    w00 = (1.0 - fu) * (1.0 - fv)
    w10 = fu * (1.0 - fv)
    w01 = (1.0 - fu) * fv
    w11 = fu * fv
    for c in range(3):
        wp.atomic_add(grad, i00 + c, w00 * g[c])
        wp.atomic_add(grad, i10 + c, w10 * g[c])
        wp.atomic_add(grad, i01 + c, w01 * g[c])
        wp.atomic_add(grad, i11 + c, w11 * g[c])


@wp.func
def diffuse_uv(o: wp.vec3f, d: wp.vec3f, t_near: float, p: wp.vec3f, R: int) -> wp.vec2f:
    """Direction from the site to the point where the ray enters the cell -> octahedral uv."""
    h = o + t_near * d - p
    ln = wp.length(h)
    dirv = d
    if ln > 1.0e-8:
        dirv = h / ln
    return oct_uv(dirv, R)


@wp.func
def sigmoid3(x: wp.vec3f) -> wp.vec3f:
    return wp.vec3f(
        1.0 / (1.0 + wp.exp(-x[0])), 1.0 / (1.0 + wp.exp(-x[1])), 1.0 / (1.0 + wp.exp(-x[2]))
    )


@wp.func
def cell_interval(
    o: wp.vec3f,
    d: wp.vec3f,
    i: int,
    points: wp.array(dtype=wp.vec3f),
    radii: wp.array(dtype=wp.float32),
    adjacency: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
):
    """Segment of the ray inside cell i = sphere ∩ Cech faces.

    near_src / far_src: -2 = none (t_near = 0), -1 = sphere, k >= 0 = adjacency
    element k (the face between i and adjacency[k]).
    """
    pi = points[i]
    ri = radii[i]
    y0 = o - pi

    t_near = float(0.0)
    t_far = float(BIG)
    near_src = int(-2)
    far_src = int(-2)
    valid = int(1)

    b = wp.dot(y0, d)
    disc = b * b - (wp.dot(y0, y0) - ri * ri)
    if disc <= 0.0:
        valid = 0
    else:
        s = wp.sqrt(disc)
        sf = -b + s
        sn = -b - s
        if sf <= 0.0:
            valid = 0
        else:
            t_far = sf
            far_src = -1
            if sn > 0.0:
                t_near = sn
                near_src = -1

    if valid == 1:
        ri2 = ri * ri
        k0 = offsets[i]
        k1 = offsets[i + 1]
        for k in range(k0, k1):
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
                        far_src = k
                else:
                    if tf > t_near:
                        t_near = tf
                        near_src = k
        if t_far <= t_near:
            valid = 0

    return valid, t_near, t_far, near_src, far_src


@wp.kernel(enable_backward=False)
def collect_cells(
    bvh_id: wp.uint64,
    origins: wp.array(dtype=wp.vec3f),
    dirs: wp.array(dtype=wp.vec3f),
    points: wp.array(dtype=wp.vec3f),
    radii: wp.array(dtype=wp.float32),
    adjacency: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    cap: int,
    ids: wp.array(dtype=wp.int32),
    keys: wp.array(dtype=wp.float32),
    count: wp.array(dtype=wp.int32),
    overflow: wp.array(dtype=wp.int32),
):
    """BVH ray query; keeps the ``cap`` cells with the smallest t_near, sorted."""
    tid = wp.tid()
    o = origins[tid]
    d = dirs[tid]
    base = tid * cap
    cnt = int(0)

    query = wp.bvh_query_ray(bvh_id, o, d)
    j = int(0)
    while wp.bvh_query_next(query, j):
        valid, tn, tf, ns, fs = cell_interval(o, d, j, points, radii, adjacency, offsets)
        if valid == 1:
            pos = int(-1)
            if cnt < cap:
                pos = cnt
                cnt += 1
            else:
                overflow[tid] = 1
                if tn < keys[base + cap - 1]:
                    pos = cap - 1
            if pos >= 0:
                while pos > 0 and keys[base + pos - 1] > tn:
                    keys[base + pos] = keys[base + pos - 1]
                    ids[base + pos] = ids[base + pos - 1]
                    pos -= 1
                keys[base + pos] = tn
                ids[base + pos] = j
    count[tid] = cnt


@wp.kernel(enable_backward=False)
def render_fwd(
    origins: wp.array(dtype=wp.vec3f),
    dirs: wp.array(dtype=wp.vec3f),
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
    cap: int,
    ids: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    t_thresh: float,
    skip_thresh: float,
    store_dist: int,
    dist_w: wp.array(dtype=wp.float32),
    dist_s: wp.array(dtype=wp.float32),
    dist_d: wp.array(dtype=wp.float32),
    proc: wp.array(dtype=wp.int32),
    out_rgb: wp.array(dtype=wp.vec3f),
    out_rgb64: wp.array(dtype=wp.vec3d),
    out_T: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    o = origins[tid]
    d = dirs[tid]
    spec_uv = oct_uv(-d, rs)
    base = tid * cap
    n = count[tid]

    T = float(1.0)
    C = wp.vec3d()
    m = int(0)
    while m < n and T > t_thresh:
        i = ids[base + m]
        valid, t_near, t_far, ns, fs = cell_interval(o, d, i, points, radii, adjacency, offsets)
        sg = sigma[i]
        dt = t_far - t_near
        if store_dist == 1:
            dist_w[base + m] = 0.0
        if valid == 1 and dt > 0.0 and sg >= 0.0:
            mm = wp.min(sg * dt, 60.0)
            e = wp.exp(-mm)
            a = 1.0 - e
            w = T * a
            # inference only: a cell whose weight cannot change the colour keeps
            # updating T (the ray goes on) but skips the texture loads / sigmoid
            if w >= skip_thresh:
                duv = diffuse_uv(o, d, t_near, points[i], rd)
                logit = tex_bilinear(diffuse, i * 3 * rd * rd, rd, duv) + tex_bilinear(
                    specular, i * 3 * rs * rs, rs, spec_uv
                )
                col = sigmoid3(logit)
                C += wp.vec3d(
                    wp.float64(w * col[0]), wp.float64(w * col[1]), wp.float64(w * col[2])
                )
            if store_dist == 1:
                sa = 1.0 - 1.0 / (1.0 + t_near)
                sb = 1.0 - 1.0 / (1.0 + t_far)
                dist_w[base + m] = w
                dist_s[base + m] = 0.5 * (sa + sb)
                dist_d[base + m] = sb - sa
            T = T * e
        m += 1
    proc[tid] = m

    b = bg[tid]
    Cf = C + wp.vec3d(wp.float64(T * b[0]), wp.float64(T * b[1]), wp.float64(T * b[2]))
    out_rgb64[tid] = Cf
    out_rgb[tid] = wp.vec3f(wp.float32(Cf[0]), wp.float32(Cf[1]), wp.float32(Cf[2]))
    out_T[tid] = T


@wp.func
def face_grad(
    o: wp.vec3f,
    d: wp.vec3f,
    i: int,
    k: int,
    scale: float,
    points: wp.array(dtype=wp.vec3f),
    radii: wp.array(dtype=wp.float32),
    adjacency: wp.array(dtype=wp.int32),
    g_points: wp.array(dtype=wp.vec3f),
    g_radii: wp.array(dtype=wp.float32),
):
    pi = points[i]
    ri = radii[i]
    j = adjacency[k]
    pj = points[j]
    rj = radii[j]
    y0 = o - pi
    n = pj - pi
    h = 0.5 * (wp.dot(n, n) + ri * ri - rj * rj)
    dp = wp.dot(n, d)
    tf = (h - wp.dot(n, y0)) / dp
    dtdn = (n - y0) / dp - tf * d / dp
    dti = n / dp - dtdn
    wp.atomic_add(g_points, j, scale * dtdn)
    wp.atomic_add(g_points, i, scale * dti)
    wp.atomic_add(g_radii, i, scale * ri / dp)
    wp.atomic_add(g_radii, j, -scale * rj / dp)


@wp.func
def sphere_grad(
    o: wp.vec3f,
    d: wp.vec3f,
    i: int,
    far: int,
    scale: float,
    points: wp.array(dtype=wp.vec3f),
    radii: wp.array(dtype=wp.float32),
    g_points: wp.array(dtype=wp.vec3f),
    g_radii: wp.array(dtype=wp.float32),
):
    pi = points[i]
    ri = radii[i]
    y0 = o - pi
    b = wp.dot(y0, d)
    disc = b * b - (wp.dot(y0, y0) - ri * ri)
    s = wp.max(wp.sqrt(wp.max(disc, 0.0)), 1.0e-3)
    q = (b * d - y0) / s
    dtdy = wp.vec3f(0.0, 0.0, 0.0)
    dtdr = float(0.0)
    if far == 1:
        dtdy = -d + q
        dtdr = ri / s
    else:
        dtdy = -d - q
        dtdr = -ri / s
    wp.atomic_add(g_points, i, -scale * dtdy)
    wp.atomic_add(g_radii, i, scale * dtdr)


@wp.kernel(enable_backward=False)
def render_bwd(
    origins: wp.array(dtype=wp.vec3f),
    dirs: wp.array(dtype=wp.vec3f),
    bg: wp.array(dtype=wp.vec3f),
    out_rgb64: wp.array(dtype=wp.vec3d),
    grad_rgb: wp.array(dtype=wp.vec3f),
    ray_err: wp.array(dtype=wp.float32),
    points: wp.array(dtype=wp.vec3f),
    radii: wp.array(dtype=wp.float32),
    sigma: wp.array(dtype=wp.float32),
    diffuse: wp.array(dtype=wp.float32),
    specular: wp.array(dtype=wp.float32),
    rd: int,
    rs: int,
    adjacency: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    cap: int,
    ids: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    t_thresh: float,
    inv_rays: float,
    lam: float,
    dist_w: wp.array(dtype=wp.float32),
    dist_s: wp.array(dtype=wp.float32),
    dist_d: wp.array(dtype=wp.float32),
    proc: wp.array(dtype=wp.int32),
    dist_loss: wp.array(dtype=wp.float32),
    g_points: wp.array(dtype=wp.vec3f),
    g_radii: wp.array(dtype=wp.float32),
    g_sigma: wp.array(dtype=wp.float32),
    g_diff: wp.array(dtype=wp.float32),
    g_spec: wp.array(dtype=wp.float32),
    contrib: wp.array(dtype=wp.float32),
    point_err: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    o = origins[tid]
    d = dirs[tid]
    g = grad_rgb[tid]
    bgc = bg[tid]
    err = ray_err[tid]
    Ctot = out_rgb64[tid]
    spec_uv = oct_uv(-d, rs)
    base = tid * cap
    n = count[tid]

    # distortion loss of this ray: sum_{k,l} w_k w_l |s_k - s_l| + 1/3 sum_k w_k^2 ds_k
    # (Mip-NeRF 360), w_k = T_k alpha_k.  Wtot = sum_k w_k dL/dw_k  (for the T-chain rule)
    # Cells of one ray have disjoint intervals, so the slots are sorted by s_k and
    # sum_l w_l |s_k - s_l| = s_k (Wl - Wr) - Sl + Sr  with prefix/suffix sums (O(K)).
    npr = proc[tid]
    Wsum = float(0.0)
    Ssum = float(0.0)
    Wtot = float(0.0)
    if lam > 0.0:
        for k in range(npr):
            wk = dist_w[base + k]
            Wsum += wk
            Ssum += wk * dist_s[base + k]
        Lr = float(0.0)
        Wl0 = float(0.0)
        Sl0 = float(0.0)
        for k in range(npr):
            wk = dist_w[base + k]
            if wk > 0.0:
                sk = dist_s[base + k]
                Wr0 = Wsum - Wl0 - wk
                Sr0 = Ssum - Sl0 - wk * sk
                acc = sk * (Wl0 - Wr0) - Sl0 + Sr0
                Lr += wk * acc + wk * wk * dist_d[base + k] / 3.0
                Wtot += wk * (2.0 * acc + 2.0 * wk * dist_d[base + k] / 3.0)
                Wl0 += wk
                Sl0 += wk * sk
        wp.atomic_add(dist_loss, 0, Lr * inv_rays)
    Wl = float(0.0)
    Sl = float(0.0)
    pref = float(0.0)

    T = float(1.0)
    C = wp.vec3d()
    m = int(0)
    while m < n and T > t_thresh:
        i = ids[base + m]
        valid, t_near, t_far, ns, fs = cell_interval(o, d, i, points, radii, adjacency, offsets)
        sg = sigma[i]
        dt = t_far - t_near
        if valid == 1 and dt > 0.0 and sg >= 0.0:
            mraw = sg * dt
            mm = wp.min(mraw, 60.0)
            e = wp.exp(-mm)
            a = 1.0 - e
            duv = diffuse_uv(o, d, t_near, points[i], rd)
            logit = tex_bilinear(diffuse, i * 3 * rd * rd, rd, duv) + tex_bilinear(
                specular, i * 3 * rs * rs, rs, spec_uv
            )
            col = sigmoid3(logit)
            w = T * a
            C += wp.vec3d(
                wp.float64(w * col[0]), wp.float64(w * col[1]), wp.float64(w * col[2])
            )
            T_after = T * e

            # colour of everything behind this cell, as if it started at T = 1
            Rp = bgc
            if T_after > 1.0e-10:
                rest = Ctot - C
                inv = wp.float64(1.0) / wp.float64(T_after)
                Rp = wp.vec3f(
                    wp.float32(rest[0] * inv),
                    wp.float32(rest[1] * inv),
                    wp.float32(rest[2] * inv),
                )
                Rp = wp.vec3f(
                    wp.clamp(Rp[0], 0.0, 4.0),
                    wp.clamp(Rp[1], 0.0, 4.0),
                    wp.clamp(Rp[2], 0.0, 4.0),
                )

            # texture gradients: both maps share the sigmoid, d logit = g w col (1 - col)
            gl = wp.vec3f(
                g[0] * w * col[0] * (1.0 - col[0]),
                g[1] * w * col[1] * (1.0 - col[1]),
                g[2] * w * col[2] * (1.0 - col[2]),
            )
            tex_scatter(g_diff, i * 3 * rd * rd, rd, duv, gl)
            tex_scatter(g_spec, i * 3 * rs * rs, rs, spec_uv, gl)

            dLda = T * wp.dot(g, col - Rp)
            dLdtn = float(0.0)
            dLdtf = float(0.0)
            if lam > 0.0:
                # distortion: dL/dw_i, dL/d(s_i), dL/d(ds_i)
                si = dist_s[base + m]
                di = dist_d[base + m]
                Wr = Wsum - Wl - w
                Sr = Ssum - Sl - w * si
                acc = si * (Wl - Wr) - Sl + Sr
                sgn = Wl - Wr
                Wl += w
                Sl += w * si
                Di = 2.0 * acc + 2.0 * w * di / 3.0
                pref += w * Di
                suffix = wp.max(Wtot - pref, 0.0)
                # w_k = T_k alpha_k:  d w_i / d alpha_i = T_i,  d w_k / d alpha_i = -w_k / (1 - alpha_i)
                dLda += lam * (T * Di - suffix / wp.max(e, 1.0e-12))
                dLds = lam * 2.0 * w * sgn
                dLdd = lam * w * w / 3.0
                gta = 1.0 / ((1.0 + t_near) * (1.0 + t_near))
                gtb = 1.0 / ((1.0 + t_far) * (1.0 + t_far))
                dLdtn = dLds * 0.5 * gta - dLdd * gta
                dLdtf = dLds * 0.5 * gtb + dLdd * gtb
            if mraw < 60.0:
                wp.atomic_add(g_sigma, i, dLda * e * dt)
                G = dLda * e * sg
                dLdtn += -G
                dLdtf += G
            if fs == -1:
                sphere_grad(o, d, i, 1, dLdtf, points, radii, g_points, g_radii)
            elif fs >= 0:
                face_grad(o, d, i, fs, dLdtf, points, radii, adjacency, g_points, g_radii)
            if ns == -1:
                sphere_grad(o, d, i, 0, dLdtn, points, radii, g_points, g_radii)
            elif ns >= 0:
                face_grad(o, d, i, ns, dLdtn, points, radii, adjacency, g_points, g_radii)

            wp.atomic_add(contrib, i, w * inv_rays)
            wp.atomic_add(point_err, i, w * err * inv_rays)
            T = T_after
        m += 1


def _v3(t):
    return wp.from_torch(t, dtype=wp.vec3f)


def _f1(t):
    return wp.from_torch(t, dtype=wp.float32)


def _i1(t):
    return wp.from_torch(t, dtype=wp.int32)


class _TraceFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tracer, points, radii, sigma, diffuse, specular, origins, dirs, bg, adjacency, offsets, gt, stats):
        dev = points.device
        R = origins.shape[0]
        N = points.shape[0]
        rd = int(round((diffuse.shape[1] // 3) ** 0.5))
        rs = int(round((specular.shape[1] // 3) ** 0.5))
        cap = tracer.cap
        a = dict(
            origins=origins.contiguous().float(),
            dirs=dirs.contiguous().float(),
            bg=bg.contiguous().float(),
            points=points.detach().contiguous().float(),
            radii=radii.detach().contiguous().float(),
            sigma=sigma.detach().contiguous().float(),
            diffuse=diffuse.detach().contiguous().float().view(-1),
            specular=specular.detach().contiguous().float().view(-1),
            adjacency=adjacency,
            offsets=offsets,
        )
        ids = torch.zeros(R * cap, device=dev, dtype=torch.int32)
        keys = torch.zeros(R * cap, device=dev, dtype=torch.float32)
        count = torch.zeros(R, device=dev, dtype=torch.int32)
        overflow = torch.zeros(R, device=dev, dtype=torch.int32)
        out_rgb = torch.empty(R, 3, device=dev, dtype=torch.float32)
        out_rgb64 = torch.empty(R, 3, device=dev, dtype=torch.float64)
        out_T = torch.empty(R, device=dev, dtype=torch.float32)
        store = tracer.lam_dist > 0.0
        proc = torch.zeros(R, device=dev, dtype=torch.int32)
        if store:
            dist_w = torch.zeros(R * cap, device=dev, dtype=torch.float32)
            dist_s = torch.zeros(R * cap, device=dev, dtype=torch.float32)
            dist_d = torch.zeros(R * cap, device=dev, dtype=torch.float32)
        else:
            dist_w = dist_s = dist_d = torch.zeros(1, device=dev, dtype=torch.float32)
        stream = wp.stream_from_torch(torch.cuda.current_stream())
        wp.launch(
            collect_cells,
            dim=R,
            inputs=[
                tracer.bvh_id, _v3(a["origins"]), _v3(a["dirs"]), _v3(a["points"]),
                _f1(a["radii"]), _i1(adjacency), _i1(offsets), cap,
                _i1(ids), _f1(keys), _i1(count), _i1(overflow),
            ],
            block_dim=64,
            device=str(dev),
            stream=stream,
        )
        wp.launch(
            render_fwd,
            dim=R,
            inputs=[
                _v3(a["origins"]), _v3(a["dirs"]), _v3(a["bg"]), _v3(a["points"]),
                _f1(a["radii"]), _f1(a["sigma"]), _f1(a["diffuse"]), _f1(a["specular"]), rd, rs,
                _i1(adjacency), _i1(offsets), cap, _i1(ids), _i1(count), tracer.t_thresh,
                tracer.skip_thresh, int(store), _f1(dist_w), _f1(dist_s), _f1(dist_d), _i1(proc),
                _v3(out_rgb), wp.from_torch(out_rgb64, dtype=wp.vec3d), _f1(out_T),
            ],
            block_dim=64,
            device=str(dev),
            stream=stream,
        )
        ctx.tracer, ctx.a, ctx.res, ctx.gt, ctx.stats = tracer, a, (rd, rs), gt, stats
        ctx.num = (N, R)
        ctx.lists = (ids, count)
        ctx.dist = (store, dist_w, dist_s, dist_d, proc)
        ctx.save_for_backward(out_rgb64, out_rgb)
        tracer.last_T = out_T
        tracer.last_count = count
        tracer.last_overflow = overflow
        return out_rgb

    @staticmethod
    def backward(ctx, grad_rgb):
        out_rgb64, out_rgb = ctx.saved_tensors
        tracer, a = ctx.tracer, ctx.a
        rd, rs = ctx.res
        ids, count = ctx.lists
        N, R = ctx.num
        store, dist_w, dist_s, dist_d, proc = ctx.dist
        dist_loss = torch.zeros(1, device=out_rgb.device)
        dev = out_rgb.device
        grad_rgb = grad_rgb.contiguous().float()
        if ctx.gt is not None:
            ray_err = (out_rgb - ctx.gt).abs().mean(-1).detach().contiguous()
        else:
            ray_err = torch.zeros(R, device=dev)
        g_points = torch.zeros(N, 3, device=dev)
        g_radii = torch.zeros(N, device=dev)
        g_sigma = torch.zeros(N, device=dev)
        g_diff = torch.zeros(N * 3 * rd * rd, device=dev)
        g_spec = torch.zeros(N * 3 * rs * rs, device=dev)
        contrib = torch.zeros(N, device=dev)
        point_err = torch.zeros(N, device=dev)
        stream = wp.stream_from_torch(torch.cuda.current_stream())
        wp.launch(
            render_bwd,
            dim=R,
            inputs=[
                _v3(a["origins"]), _v3(a["dirs"]), _v3(a["bg"]),
                wp.from_torch(out_rgb64, dtype=wp.vec3d), _v3(grad_rgb), _f1(ray_err),
                _v3(a["points"]), _f1(a["radii"]), _f1(a["sigma"]), _f1(a["diffuse"]), _f1(a["specular"]), rd, rs,
                _i1(a["adjacency"]), _i1(a["offsets"]), tracer.cap, _i1(ids), _i1(count),
                tracer.t_thresh, 1.0 / float(R),
                tracer.lam_dist / float(R) if store else 0.0,
                _f1(dist_w), _f1(dist_s), _f1(dist_d), _i1(proc), _f1(dist_loss),
                _v3(g_points), _f1(g_radii), _f1(g_sigma), _f1(g_diff), _f1(g_spec),
                _f1(contrib), _f1(point_err),
            ],
            block_dim=64,
            device=str(dev),
            stream=stream,
        )
        tracer.last_dist_loss = dist_loss
        if ctx.stats is not None:
            ctx.stats["contrib"] = contrib
            ctx.stats["point_error"] = point_err
        return (
            None, g_points, g_radii, g_sigma,
            g_diff.view(N, -1), g_spec.view(N, -1),
            None, None, None, None, None, None, None,
        )


class RayTracer:
    """BVH ray query + Cech-complex cell clipping (see module docstring).

    ``bvh_id`` must be set (``tree.id`` of a wp.Bvh over the sphere AABBs of the
    points / radii passed to ``render``) before rendering.
    """

    def __init__(self, t_thresh=1e-3, cap=96):
        self.t_thresh = float(t_thresh)
        self.cap = int(cap)
        self.bvh_id = None
        self.last_T = None
        self.last_count = None
        self.last_overflow = None
        # distortion loss weight (training) / low-contribution skip threshold (inference)
        self.lam_dist = 0.0
        self.skip_thresh = 0.0
        self.last_dist_loss = None

    def render(self, points, radii, sigma, diffuse, specular, origins, dirs, bg, adjacency, offsets, gt=None, stats=None):
        """points (N,3), radii (N,), sigma (N,), diffuse (N,3 rd^2), specular (N,3 rs^2) -> rgb (R,3)."""
        return _TraceFn.apply(
            self, points, radii, sigma, diffuse, specular, origins, dirs, bg, adjacency, offsets, gt, stats
        )
