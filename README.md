# powerfoam_rt

A **ray-traced, differentiable** version of the [Power Foam](https://powerfoam.github.io/) cell model
([arXiv:2604.24994](https://arxiv.org/abs/2604.24994)), with the main ideas of
[VoroTracing](https://github.com/zenseact/VoroTracing) ([arXiv:2608.17682](https://arxiv.org/abs/2608.17682)).

It started as a copy of the official Power Foam code. The rasterizer and its losses were removed and only the
ray-tracing renderer is kept, written in [NVIDIA Warp](https://github.com/NVIDIA/warp) with a **hand-written
backward pass**. Training and inference are both ray tracing.

The code is a research prototype, developed and tested only on the Mip-NeRF 360 **garden** scene at 1/8
resolution on one RTX 2060 (6 GB).

## What the model is

Every cell `i` has a site `p_i`, a radius `r_i`, a density `sigma_i = exp(rho_i)` and two octahedral textures.

* **Geometry (Power Foam).** Cell `i` is the power cell of the site (power weight `r_i^2`) clipped by its own sphere
  `|x - p_i| <= r_i`. A neighbour `j` only clips cell `i` through the face `n.x = h` with `n = p_j - p_i` and
  `h = (|n|^2 + r_i^2 - r_j^2) / 2`. That face cuts the sphere iff the spheres overlap, so the **Cech complex**
  (sphere overlap, built with a BVH on the GPU, radii not inflated) gives exactly `power cell ∩ sphere`.
* **Appearance (VoroTracing).** Per cell, an 8x8 RGB *diffuse* octahedral texture indexed by the direction from
  the site to the point where the ray enters the cell, plus an 8x8 *specular* texture indexed by the view direction
  (a residual, regularised to stay small). Colour = `sigmoid(diffuse + specular)`, both bilinearly filtered.
  The texture coordinate is not differentiated (same as VoroTracing). SH is no longer used.
* **Volume rendering.** `alpha_i = 1 - exp(-sigma_i * dt_i)`, `C += T alpha_i c_i`, `T *= 1 - alpha_i`,
  `C += T * background`.

## What was changed compared with the official Power Foam

1. **Ray-traced training, no rasterizer.**
   Each ray queries the BVH of the spheres, computes the interval `[t_near, t_far]` of every hit cell
   (sphere ∩ Cech faces), keeps the `max_cells` (96) nearest ones by `t_near` (cells are disjoint, so this is the depth
   order) and composites them front to back. No walk during training, so the Cech complex is enough and no Delaunay
   triangulation is needed.
2. **Hand-written backward.**
   Uses the "rest of the ray" identity with float64 accumulation. Gradients: density, both textures, sites and radii
   (through `t_near` / `t_far`, each being a sphere root or a Cech face).
3. **Losses.**
   `L2 + 0.2 (1 - SSIM)` on the image, an **interpenetration loss** `sum_j max(0, r_i + r_j - d_ij)^2` over Cech
   neighbours (weight `1e-4`, exponentially decayed to `1e-7`; it keeps the Cech degree and the speed under control),
   and the **distortion loss** below. Power Foam's densify / prune (`resample`) is kept.
4. **Exponential density** `sigma = exp(rho)` (VoroTracing 5.4). `d loss / d rho` does not depend on the cell size.
   `rho` is initialised to `log(density_init)`, lr `0.1 -> 0.01` (the old softplus lr `1.0` must not be used with it;
   all configs in `configs/` were updated).
5. **Distortion loss** (Mip-NeRF 360) on the per-ray weights `w_k = T_k alpha_k`, with `s(t) = 1 - 1/(1+t)`:
   `sum_{k,l} w_k w_l |s_k - s_l| + 1/3 sum_k w_k^2 ds_k`. Cells of a ray are disjoint, so the double sum is
   computed in O(K) with prefix sums. Forward and backward (through `w`, `T`, `t_near`, `t_far`) are in
   `raytrace.py`; the weight is `distortion_weight` (default `0.01`, `0.02` in `garden_smoke.yaml`). It is injected
   in the backward kernel, so it is not a term of `loss` in `train.py` (its value is shown as `dist` in the progress bar).
6. **Early ray termination** `t_thresh_late` (default `1e-2`) from iteration `t_thresh_after` (default 3000) on,
   instead of `t_thresh` (`1e-3`). It stays active at inference.
7. **Low-contribution skip at inference** (`skip_thresh`, default `1e-3`): cells with `T alpha` below it still update
   `T` and the ray continues, but the texture loads and the sigmoid are skipped. Used only without gradients.
8. **Inference without BVH and Cech** (`powerfoam/walk.py`): the exact adjacency of the weighted Delaunay
   triangulation is computed once with qhull (`build_exact_adjacency()`, ~9 s for 300k cells), then a ray starts in the
   power cell that contains the camera and walks from cell to cell through the exit face. The cells and the image
   are the same as in training.

## Results (garden, 1/8 resolution, 5000 iterations, 300k cells, RTX 2060)

| Model | PSNR | SSIM | LPIPS | Train time |
|---|---|---|---|---|
| Power Foam original (1 texel), reference | 25.32 | 0.719 | 0.318 | - |
| SH + Cech + interpenetration | 23.73 | 0.631 | 0.382 | 17 min |
| **octahedral textures + exp density + distortion + early termination** | **26.17** | **0.785** | **0.204** | 17 min |

Inference speed of the last model (test cameras, 648x420, 300k cells):

| Renderer | PSNR | ms / frame | FPS |
|---|---|---|---|
| BVH + Cech (training tracer) | 26.168 | 290 | 3.4 |
| qhull adjacency + walk (no BVH) | 26.167 | 22.4 | 44.7 |

Notes and limits:

* The textures are by far the largest gain (about +2.5 dB over SH). With the SH model, exp density + distortion alone
  made things worse in this short schedule (fewer cells could become exactly transparent).
* About 32% of the cells have `1 - exp(-2 r sigma) > 0.9` (9% before), i.e. the opacity is more bimodal.
* The low-contribution skip does not change PSNR (26.169 -> 26.168) but gives **no speed-up here**. In the BVH tracer
  the cost is the BVH query and the cell clipping (done for all nearest cells before compositing); in the walk
  it is the cell stepping. Early termination also does not help the BVH tracer for the same reason.
* A single run per setting, one scene, one seed. Nothing was tuned on the test views beyond what is listed here,
  but the numbers are not averaged over scenes or seeds.
* Training memory is dominated by the textures (`384` floats per cell plus Adam state): about 5 GB of VRAM at
  300k cells on the 6 GB card.

## Installation

Linux, Python 3.10+, CUDA 12.x GPU (compute capability 7.0+). Reference build: `torch==2.9.1+cu128`,
`warp-lang==1.10.0`.

```bash
conda create -n powerfoam python=3.11 -y && conda activate powerfoam
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The viewer and Metric3D parts of the original code are not included in this repository.

## Data

COLMAP format, e.g. Mip-NeRF 360:

```
data/mipnerf360/garden/{images,images_2,images_4,images_8,sparse/0}
```

`data_path` in the config is relative to where you start `train.py`
(`configs/garden_smoke.yaml` uses `../data/mipnerf360`). For your own images, run
`python prepare_colmap_data.py --data_dir data/your_own_data` first.

## Training

```bash
python train.py -c configs/garden_smoke.yaml --experiment_name garden_rt_oct
```

`garden_smoke.yaml` is the reduced setting used for the results above (1/8 images, 5000 iterations, 100k -> 300k
cells, held-out test views). The other files in `configs/` are the original Power Foam configs with the density lr
fixed for the exponential density; they were **not** run with this code. Any field can be overridden on the command line
(`--iterations 10000 --distortion_weight 0.05`). The full list is in `configs/__init__.py`.

Main options of this version:

| Option | Default | Meaning |
|---|---|---|
| `oct_res`, `spec_res` | 8, 8 | diffuse / specular octahedral texture resolution |
| `attr_lr_init`, `attr_lr_final` | 2e-2, 5e-4 | texture learning rate |
| `specular_start` | 1000 | the specular lr ramps up as `(it / start)^2` |
| `specular_reg_weight`, `diffuse_mean_pull_weight` | 1e-2, 5e-3 | pull specular to 0 / diffuse texels to their per-cell mean |
| `density_init` | 0.1 | initial `sigma` (`rho = log sigma`) |
| `density_lr_init`, `density_lr_final` | per config (0.1, 0.01) | lr of `rho` |
| `distortion_weight` | 0.01 | distortion loss weight, `0` disables it |
| `t_thresh`, `t_thresh_late`, `t_thresh_after` | 1e-3, 1e-2, 3000 | ray termination threshold, and its late value |
| `skip_thresh` | 1e-3 | inference-only low-contribution skip |
| `max_cells` | 96 | nearest cells kept per ray |
| `interpenetration_weight` | 1e-4 | interpenetration loss weight |

Outputs go to `output/<experiment_name>/`: `config.yaml`, `model.pt`, `log.txt` (test PSNR/SSIM every
`test_every` iterations), `metrics.txt` (final PSNR / SSIM / LPIPS and train time), `test/` (renders).

## Inference, evaluation, benchmark

```bash
# test metrics of a saved model (BVH + Cech renderer; --fresh recomputes the Cech complex)
python evaluate.py output/garden_rt_oct

# FPS: BVH + Cech renderer, then the qhull-adjacency walk renderer (no BVH)
python benchmark_fps.py output/garden_rt_oct
```

From code:

```python
model.build_exact_adjacency()                 # qhull, once (a few seconds for 300k cells)
with torch.no_grad():
    rgb = model.forward_walk(camera, bg)      # (H, W, 3), no BVH / Cech
    rgb, alpha = model.forward(camera, bg)    # BVH + Cech tracer
```

`load_pt` switches the tracer to `t_thresh_late`, since the model was trained with it.
`analyze.py` prints radii / density / texture statistics and train vs. test PSNR of a saved model.
`compare.py` writes side-by-side images against the original Power Foam output; it expects the original repository
next to this one (`../powerfoam-main/output/garden_powerfoam`).

## Tests

```bash
python tests/test_distortion.py
```

Finite-difference check of the distortion-loss gradient (points, radii, density), of the photometric gradient of
density and both textures, and of the low-contribution skip. The photometric gradient of points / radii is not
finite-difference tested, because the texture coordinate depends on them and is not differentiated.
The earlier version of the tracer (SH) was checked to give the same image and gradients as an exact-adjacency
walk; that test was removed together with SH.

## Layout

```
train.py                    training loop
evaluate.py, benchmark_fps.py, analyze.py, compare.py
powerfoam/raytrace.py       BVH + Cech tracer: forward, backward, distortion loss, textures (Warp)
powerfoam/walk.py           inference walk on the exact (qhull) adjacency
powerfoam/scene.py          parameters, densify / prune, optimisers, save / load
powerfoam/bvh.py            sphere BVH and Cech complex construction
powerfoam/geometry.py       Morton sort, interpenetration loss
powerfoam/exact_adj.py      qhull weighted Delaunay edges
configs/                    configs (garden_smoke.yaml is the one that was run)
tests/test_distortion.py
```

## Credits

Power Foam (Govindarajan et al.) and Radiant Foam; VoroTracing (Taveira et al.) for the octahedral textures,
exponential density, distortion loss and the low-contribution skip. See `LICENSE`.
