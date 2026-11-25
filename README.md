# Power Foam: Unifying Real-Time Differentiable Ray Tracing and Rasterization

## Shrisudhan Govindarajan\*, Daniel Rebain\*, Dor Verbin, Kwang Moo Yi, Anish Prabhu, Andrea Tagliasacchi

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=githubpages&logoColor=white)](https://powerfoam.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.24994)

</div>

This repository contains the official implementation of **Power Foam: Unifying Real-Time Differentiable Ray Tracing and Rasterization**. It provides scripts for training, evaluation and benchmarking, plus an interactive viewer that can be used both to inspect trained checkpoints and to watch a model converge live during training. The code includes scripts for training and evaluation, as well as a real-time viewer that can be used to visualize trained models, or optionally to observe the progression of models as they train. Everything in this repository is non-final and subject to change as the project is still being actively developed.

Warning: this is an organic, free-range research codebase, and should be treated with the appropriate care when integrating it into any other software.

## Getting started

Start by cloning the repository:

```bash
git clone https://github.com/theialab/powerfoam.git
cd powerfoam
```

You will need a Linux environment with **Python 3.10**, **CUDA 12.x** and a CUDA-compatible GPU of **Compute Capability 7.0 or higher**. The reference build is `torch==2.9.1+cu128`, `warp-lang==1.10.0` on CUDA 12.8.

After creating your Python virtual environment, install PyTorch matching your CUDA build (cu128 below):

```bash
conda create -n powerfoam python=3.11 -y
conda activate powerfoam

pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

Then install the rest of the Python dependencies:

```bash
pip install -r requirements.txt
```

### Optional: Metric3D normal/depth supervision

If you plan to use Metric3D-derived normals (`--use_metric3d True`, optionally combined with `--normal_supervision`), install the extra dependencies on top of the base requirements:

```bash
pip install -r metric3d_requirements.txt
```

The Metric3D ViT-Large checkpoint is downloaded from `torch.hub` on first use (`yvanyin/metric3d`).

### Optional: Viewer

The interactive viewer (`view.py`, and `--viewer` during training) needs `imgui-bundle`, `glfw` and `PyOpenGL`. They are listed in `requirements.txt` and installed by the command above.

## Dataset layout

Power Foam supports two dataset types, selected by the `dataset:` field in the yaml config:

| Config value | Reader                                                   |
| ------------ | -------------------------------------------------------- |
| `colmap`     | COLMAP `images/` + `sparse/0/` (DTU, MipNeRF-360, DL3DV) |
| `blender`    | Synthetic Blender (`transforms_*.json` + `*/r_*.png`)    |

We have tested our model on the [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) and [DL3DV](https://huggingface.co/datasets/DL3DV/DL3DV-10K-Sample) datasets.

Place each scene under `<data_path>/<scene>/`, e.g. `data/mipnerf360/garden/{images,sparse}/`. If your raw data is just a folder of images, you can run COLMAP on it via:

```bash
python prepare_colmap_data.py --data_dir data/your_own_data
```

This creates `data/your_own_data/sparse/0/{cameras,images,points3D}.bin` so the COLMAP loader can pick it up.

When `--use_metric3d True` is passed, the dataset reader also expects (or auto-creates on first run) a sibling `metric3d/` directory containing per-image `.pt` files with predicted depth, normals and confidence.

## Configuration files

Per-dataset defaults live under `configs/`:

```
configs/dl3dv.yaml
configs/mipnerf360_indoor.yaml
configs/mipnerf360_outdoor.yaml
configs/dtu.yaml
```

Add your own `configs/<dataset>.yaml` for new datasets — copy one of the shipped files and edit `dataset:`, `data_path:`, `scene:` and any optimisation overrides.

Every entry can also be overridden from the command line, e.g. `--scene garden`, `--iterations 30000`. The full schema is defined in `configs/__init__.py` (`Params` dataclass).

> **Camera model default**
> For the standard datasets shipped in `configs/`, `is_pinhole: true`. The rasterizer therefore uses the fast pinhole pipeline. **If your dataset has a non-pinhole camera (fisheye, distortion, etc.), pass `is_pinhole: false`** in your config file so the renderer switches to the generic precomputed-cone pipeline.

## Training

Training is launched with:

```bash
python train.py -c configs/<config_file>.yaml
```

where `<config_file>` is one of the shipped files in `configs/` or your own. You can optionally include the `--viewer` flag to train interactively, or use `view.py` to inspect saved checkpoints.

```bash
# Train MipNeRF-360 garden with the defaults from configs/mipnerf360_outdoor.yaml
python train.py -c configs/mipnerf360_outdoor.yaml

# Train with the live viewer attached
python train.py -c configs/mipnerf360_outdoor.yaml --viewer
```

Outputs are written to `output/<experiment_name>/` and contain:

- `config.yaml` — frozen copy of all CLI args
- `model.pt`    — periodic checkpoint
- `points.ply`  — point cloud snapshot
- `test/`       — composed RGB / normal / depth previews
- TensorBoard event files (`tensorboard --logdir output/<experiment>`)

## Evaluation

The standard test metrics can be computed with:

```bash
python test.py -c output/<checkpoint_directory>/config.yaml
```

## Benchmarking

Rendering speed can be computed with:

```bash
python benchmark.py -c output/<checkpoint_directory>/config.yaml
```

`benchmark.py` supports both the rasterizer and the ray tracer over the same set of power-diagram primitives:

```bash
# Rasterizer benchmark (default)
python benchmark.py -c output/<checkpoint_directory>/config.yaml

# Ray tracer benchmark (uses adjacency-walk ray traversal)
python benchmark.py -c output/<checkpoint_directory>/config.yaml --render_type raytrace
```

## Viewer

```bash
python view.py -c output/<checkpoint_directory>/config.yaml
```

The viewer can also be attached *during* training by passing `--viewer` to `train.py` — the training loop pauses cleanly while the renderer holds the state lock so you can inspect convergence in real time.

## Important flags reference

| Flag                    | Default       | Description                                                                                                                                                |
| ----------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--is_pinhole`          | `False`*      | When `True`, use the fast pinhole rasterizer. The standard configs ship with `is_pinhole: true`. **Set to `False` for fisheye / distorted cameras.**       |
| `--use_metric3d`        | `False`       | Pre-compute / load Metric3D depth + normals into the dataset. Required to use Metric3D normals as supervision.                                             |
| `--normal_supervision`  | `False`       | Enable external normal supervision. Combine with `--use_metric3d True` for Metric3D normals; otherwise uses finite-difference normals from rendered depth. |
| `--eval`                | `False`       | Use `train`/`test` splits. When `False`, the loader uses all views as `all`.                                                                               |
| `--viewer`              | `False`       | Launch the interactive viewer alongside training.                                                                                                          |
| `--dry_run`             | `False`       | Don't write any checkpoints or TensorBoard events.                                                                                                         |
| `--experiment_name`     | random uuid   | Subdirectory name under `output/`.                                                                                                                         |
| `--render_type`         | `rasterize`   | (`benchmark.py` only) `rasterize` or `raytrace`.                                                                                                           |

*`Params.is_pinhole` defaults to `False` in code, but every shipped yaml overrides it to `True` because the standard datasets use pinhole models. Override on the CLI if you need otherwise.

### Flag combinations for normal supervision

| `normal_supervision` | `use_metric3d` | What happens                                                                        |
| -------------------- | -------------- | ----------------------------------------------------------------------------------- |
| False (default)      | any            | Only the renderer's internal `normal_err` term — no external normals are consulted. |
| True                 | False          | External target = finite-difference normals from the rendered median-depth map.     |
| True                 | True           | External target = Metric3D predicted normals.                                       |

When `--normal_supervision` is on, training also evaluates the rendered depth map at the median quantile (`0.5`) so the bilateral / FD path or the validity mask can be computed.

## SurfaceFoam: occupancy-MLP guided Voronoi-walk surface reconstruction (new)

`SurfaceFoam` is an additional algorithm built on top of Power Foam's Voronoi/Cech-complex ray walk.
Instead of volumetric densities it learns an **occupancy MLP** `o(x) in [0,1]` and the reconstructed
surface *is* a set of Voronoi faces, so a closed polygonal mesh is read off the diagram directly
(no TSDF fusion / marching cubes).

**Per ray**

1. *Voronoi walk* (Warp kernel `voronoi_walk_kernel`): the ray is walked cell by cell through the diagram
   (`s_0, s_1, ...`); the ray crosses the bisector of `(s_k, s_{k+1})` at the intersection `x_k`.
2. Every visited site gets an occupancy `MLP(site position)`.  The first consecutive pair with
   occupancy going from `< 0.5` to `>= 0.5` defines the **surface**: their bisector is a mesh face and its
   intersection with the ray is `x_m`.
3. Around `x_m`, `K_b in [2,5]` intersections before and `K_a in [2,5]` after it are collected
   (`x_{m-K_b} .. x_{m+K_a}`).  The MLP is evaluated at the *intersection coordinates*; the occupancy is
   used as opacity `alpha_k = MLP(x_k)` and colours are alpha-composited as in UNISURF:
   `C = sum_k alpha_k prod_{j<k} (1-alpha_j) c_k` (colour of an intersection = mean of the two
   sites' view-dependent SH colours).  Rays without a surface crossing (initialisation, empty space)
   composite `W` intersections spread over the walked path, which bootstraps / carves the occupancy.

**Losses and gradient routing** (checked by `tests/test_surface_foam.py`)

| loss | definition | gradient goes to |
| --- | --- | --- |
| colour | MSE of the composited colour | MLP weights + site colours only |
| normal | for the two sites `(a,b)` holding the surface bisector, `p_b - p_a` must be parallel to the MLP normal `grad_x o` at `x_m` (smoothed over a few jittered points) | site positions only |
| intersection occupancy | `o(x_m) = 0.5` for the ray/bisector intersection of the surface face | site positions only (MLP frozen) |

**Surface-weighted compositing.** As in UNISURF the colour of the intersection whose bisector is the
surface (`x_m`) must dominate: its opacity is boosted, `alpha_m <- alpha_m + (1-alpha_m) * surface_boost`
(default 0.5), which guarantees `w_m = alpha_m T_m >= 0.5 T_m >= w_k` for every later intersection `k > m`
(`surface_boost: 0` disables it).  The training log reports `surf_max`, the fraction of crossing rays
where `x_m` has the largest weight.

**Rays without a surface (background / free space), from UNISURF.**  UNISURF needs no mask: rays without
a root use stratified samples on the *whole* ray (coarse volume rendering, so the colour loss carves the
occupancy to 0), and rays with a surface additionally get random samples between the camera and the surface
("free-space" samples) which reduce floaters.  Here: rays without a crossing composite
`free_samples + 2 win_max + 1` intersections spread uniformly over the walked path; rays with a crossing
get `free_samples` stratified intersections between the camera and the local window, which enter the
transmittance (any occupancy in front of the surface hides it and is penalised by the colour loss).
Training rays use a random background colour.

**Sphere initialisation (SAL adapted to occupancy)** (`OccupancyMLP.init_sphere`, `init_sphere: 1`).
SAL (Atzmon & Lipman, Thm. 1) sets `b=0`, `W ~ N(0, sqrt(2)/sqrt(d_out))`, `w = sqrt(pi)/sqrt(d_out) 1`,
`c = -r`, so that `f(x) ~ |x| - r` (SDF of a sphere; `E[w . relu(Wx)] = |x|`).  An occupancy is the sigmoid
of a *logit*, so the logit takes the role of the (sign-flipped, scaled) SDF:
`logit(x) = -kappa f(x) = kappa (r - |x|)`, `o = sigmoid(logit)` (1 inside, 0 outside, `o = 0.5` on the
sphere, `|grad o| = kappa/4`, `grad o/|grad o| = -x/|x|`), i.e. `w = -kappa sqrt(pi)/sqrt(d_out) 1`,
`c = +kappa r` (with `coord_scale s`: `kappa' = kappa/s`).  As in IDR/VolSDF the first layer only sees the raw
coordinates (Fourier channels start at zero and are switched on by the BARF annealing).  Because the theorem is
asymptotic in the width (and for ReLU), the output layer is then refined by a closed-form weighted ridge fit
(shrunk towards the SAL solution) to `kappa (r - |x|)`; this gives radius `0.50 +- 0.01`, `|grad logit| ~ kappa`,
normal vs. `-x/|x|` cosine 0.999 with the default 128x4 softplus MLP.  The MLP learning rate is warmed up
(`mlp_warmup`) so that the first Adam steps do not destroy the sphere.  Set `sphere_radius` to the object
radius (`init_sphere: 0` disables it).

**Surface smoothness** (`SurfaceFoam.mlp_regularization`, gradients reach the MLP only).
* UNISURF's surface regulariser `|| n(x_s) - n(x_s + eps) ||_2` (`smooth_weight`, `smooth_eps` x local site
  spacing, optionally decayed with `smooth_final_frac`) at the ray/bisector intersections.
* SDFoam's eikonal term adapted to the occupancy (`eik_weight`, `sharp_lr`): `(|grad f| / kappa - 1)^2` with
  `logit = exp(log_sharp) f` and a learnable sharpness (SDFoam's beta).  On lego it *hurt* (PSNR -2 dB, see the
  ablation), so it is off by default.
* The Voronoi-face mesh can additionally be projected onto the MLP level set (`python test_surface.py ... --snap`),
  and `--mc` / `mc_res` extracts a marching-cubes reference mesh of the MLP (`mc_mesh.ply`);
  `--roughness` prints the normal roughness (mean angle between a surface normal and the mean normal within
  3 cm) of every mesh.

**Densification** (every `resample_interval` iterations, `SurfaceFoam.resample` / `error_surface_pairs`):

1. render random training rays and keep the per-ray colour error;
2. UNISURF root finding on the MLP along those rays (`root_samples` samples, first `<0.5 -> >=0.5`
   bracket, `root_steps` secant/bisection steps) gives `x*` with `o(x*) = 0.5`;
3. hits are sampled proportionally to the error with a spatial non-maximum suppression (`densify_nms`);
4. at each `x*` with MLP normal `n` two sites are added, `x* + delta n` and `x* - delta n`
   (`delta = densify_offset` x local site spacing), so that their bisector passes through `x*` with normal `n`;
5. the SH colours of the new sites are copied (inverse-distance weighted) from the `copy_knn` nearest existing sites.

Low-contribution sites are pruned first; if too few surface hits are found the remaining budget is filled
by jittered duplicates of high-error sites.  The adjacency used by the walk is the union of a GPU Cech complex and exact
Delaunay edges (kNN-radius Cech alone loses rays after densification, see `rebuild_adjacency`).

```bash
python train_surface.py -c configs/surface_lego.yaml            # train, evaluate, export mesh
python test_surface.py  -c output/lego_surface/config.yaml      # re-evaluate / re-export mesh
python tests/test_surface_foam.py                               # unit tests (walk, grads, compositing)
```

Outputs in `output/<experiment>/`: `model.pt`, `metrics.txt`, `test/*.png` (render | gt | normal | depth),
`surface_mesh.ply` (Voronoi-face mesh, vertex colours) and `surface_mesh_preview.png`.
Implementation: `powerfoam/surface_foam.py`, `powerfoam/surface_mesh.py`, `powerfoam/ray_bank.py`,
`SurfaceParams` in `configs/__init__.py`.

## BibTeX

```bibtex
@article{govindarajan2026powerfoam,
  title   = {Power Foam: Unifying Real-Time Differentiable Ray Tracing and Rasterization},
  author  = {Govindarajan, Shrisudhan and Rebain, Daniel and Verbin, Dor and
             Yi, Kwang Moo and Prabhu, Anish and Tagliasacchi, Andrea},
  journal = {arXiv},
  year    = {2026},
}
```
