"""Light module (numpy/scipy only) so the spawn worker does not import torch/warp."""

def exact_edges_np(points, radii):
    """Edges (as sorted keys a*N+b, a<b) of the weighted Delaunay triangulation
    (regular triangulation) of the power diagram: lower convex hull of the
    sites lifted to 4D by |p|^2 - r^2.  Pure numpy/qhull => runs in a worker."""
    import numpy as np
    from scipy.spatial import ConvexHull, QhullError

    P = np.asarray(points, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    P = P - P.mean(0, keepdims=True)
    L = np.concatenate([P, (P**2).sum(1, keepdims=True) - r[:, None] ** 2], axis=1)
    try:
        hull = ConvexHull(L)
    except QhullError:
        hull = ConvexHull(L, qhull_options="QJ")
    sim = hull.simplices[hull.equations[:, 3] < 0].astype(np.int64)
    N = P.shape[0]
    keys = []
    for a in range(4):
        for b in range(a + 1, 4):
            lo = np.minimum(sim[:, a], sim[:, b])
            hi = np.maximum(sim[:, a], sim[:, b])
            keys.append(lo * N + hi)
    return np.unique(np.concatenate(keys))


def die_with_parent():
    """Worker initializer: get SIGKILLed when the training process dies (Linux),
    so an OOM-killed run does not leave a multi-GB qhull worker behind."""
    import ctypes
    import signal

    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
