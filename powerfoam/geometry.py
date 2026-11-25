from typing import Any
import torch
import torch.nn.functional as F
import warp as wp


@wp.func
def expand_bits(v: wp.uint64):
    v = v & wp.uint64(0x00000000001FFFFF)
    v = (v | (v << wp.uint64(32))) & wp.uint64(0x001F00000000FFFF)
    v = (v | (v << wp.uint64(16))) & wp.uint64(0x001F0000FF0000FF)
    v = (v | (v << wp.uint64(8))) & wp.uint64(0x100F00F00F00F00F)
    v = (v | (v << wp.uint64(4))) & wp.uint64(0x10C30C30C30C30C3)
    v = (v | (v << wp.uint64(2))) & wp.uint64(0x1249249249249249)
    return v


@wp.kernel
def morton_code_kernel(
    points: wp.array(dtype=wp.vec3f),
    lower: wp.vec3f,
    upper: wp.vec3f,
    codes: wp.array(dtype=wp.int64),
):
    tid = wp.tid()
    p = points[tid]

    # Normalize to [0, 1]
    dims = upper - lower
    # Avoid division by zero
    dims = wp.vec3f(wp.max(dims[0], 1e-6), wp.max(dims[1], 1e-6), wp.max(dims[2], 1e-6))

    normalized = wp.cw_div(p - lower, dims)

    # Quantize to 21 bits [0, 2097151]
    resolution = float(2097152.0)

    x = wp.min(wp.max(normalized[0] * resolution, 0.0), resolution - 1.0)
    y = wp.min(wp.max(normalized[1] * resolution, 0.0), resolution - 1.0)
    z = wp.min(wp.max(normalized[2] * resolution, 0.0), resolution - 1.0)

    xx = expand_bits(wp.uint64(x))
    yy = expand_bits(wp.uint64(y))
    zz = expand_bits(wp.uint64(z))

    code = xx * wp.uint64(4) + yy * wp.uint64(2) + zz
    codes[tid] = wp.int64(code)


def morton_sort(points: torch.Tensor):
    """
    Sort points using Morton codes (Z-order curve) and Radix sort.
    Returns the sorted indices.
    """
    if points.shape[0] == 0:
        return torch.zeros(0, dtype=torch.int32, device=points.device)

    # Compute bounding box
    min_p = points.min(dim=0)[0]
    max_p = points.max(dim=0)[0]

    # Warp inputs
    device = points.device
    num_points = points.shape[0]

    with wp.ScopedDevice(str(device)):
        wp_points = wp.from_torch(points, dtype=wp.vec3f)
        indices = torch.arange(num_points, dtype=torch.int32, device=device)

        # Warp radix sort requires 2*N buffer
        wp_indices = wp.zeros(2 * num_points, dtype=wp.int32)
        wp.copy(wp_indices, wp.from_torch(indices), count=num_points)

        wp_codes = wp.zeros(2 * num_points, dtype=wp.int64)
        lower = wp.vec3f(float(min_p[0]), float(min_p[1]), float(min_p[2]))
        upper = wp.vec3f(float(max_p[0]), float(max_p[1]), float(max_p[2]))

        wp.launch(
            kernel=morton_code_kernel,
            dim=num_points,
            inputs=[wp_points, lower, upper, wp_codes],
        )

        wp.utils.radix_sort_pairs(wp_codes, wp_indices, num_points)
        return wp.to_torch(wp_indices)[:num_points]



@wp.kernel(enable_backward=True)
def interpenetration_kernel(
    spheres: wp.array(dtype=wp.vec4f),
    adjacency: wp.array(dtype=wp.int32),
    adjacency_offsets: wp.array(dtype=wp.int32),
    areas: wp.array(dtype=wp.float32),
):
    thread_idx = wp.tid()
    sphere = spheres[thread_idx]
    center = wp.vec3f(sphere[0], sphere[1], sphere[2])
    radius = sphere[3]

    prim_offset_start = adjacency_offsets[thread_idx]
    prim_offset_end = adjacency_offsets[thread_idx + 1]
    total_prims = wp.int32(prim_offset_end - prim_offset_start)

    for intersection_idx in range(total_prims):
        current_offset = prim_offset_start + intersection_idx
        neighbor_idx = adjacency[current_offset]

        neighbor_sphere = spheres[neighbor_idx]
        neighbor_center = wp.vec3f(
            neighbor_sphere[0], neighbor_sphere[1], neighbor_sphere[2]
        )
        neighbor_radius = neighbor_sphere[3]

        d = wp.length(center - neighbor_center)

        if d < (radius + neighbor_radius):
            areas[thread_idx] += (radius + neighbor_radius - d) ** 2.0


class InterpenetrationFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        points,
        radii,
        adjacency,
        adjacency_offsets,
    ):
        with wp.ScopedDevice(str(points.device)):
            num_points = points.shape[0]

            spheres = torch.cat([points, radii[:, None]], dim=-1).to(torch.float32)

            ctx.spheres = wp.from_torch(spheres, dtype=wp.vec4f, requires_grad=True)
            ctx.adjacency = adjacency
            ctx.adjacency_offsets = adjacency_offsets
            ctx.areas = wp.zeros(
                num_points,
                dtype=wp.float32,
                device=str(points.device),
                requires_grad=True,
            )

            wp.launch(
                kernel=interpenetration_kernel,
                dim=num_points,
                inputs=[
                    ctx.spheres,
                    ctx.adjacency,
                    ctx.adjacency_offsets,
                ],
                outputs=[ctx.areas],
                block_dim=128,
            )

            return wp.to_torch(ctx.areas)

    @staticmethod
    def backward(ctx, areas_grad_in):
        with wp.ScopedDevice(str(ctx.spheres.device)):

            areas_grad_in = areas_grad_in.contiguous()
            ctx.areas.grad = wp.from_torch(areas_grad_in, dtype=wp.float32)
            num_points = ctx.spheres.shape[0]

            wp.launch(
                kernel=interpenetration_kernel,
                dim=num_points,
                inputs=[
                    ctx.spheres,
                    ctx.adjacency,
                    ctx.adjacency_offsets,
                ],
                outputs=[ctx.areas],
                adj_inputs=[
                    ctx.spheres.grad,
                    None,
                    None,
                ],
                adj_outputs=[
                    ctx.areas.grad,
                ],
                block_dim=128,
                adjoint=True,
            )

            spheres_grad_out = wp.to_torch(ctx.spheres.grad)

            return spheres_grad_out[..., :3], spheres_grad_out[..., 3], None, None

