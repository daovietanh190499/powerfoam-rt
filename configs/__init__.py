from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
import inspect
from typing import Optional, Union

import configargparse


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == "true":
        return True
    elif v.lower() == "false":
        return False
    else:
        raise configargparse.ArgumentTypeError("Boolean value expected. True or False.")


def add_group(parser: configargparse.ArgParser, group_class):
    fields = dict(inspect.getmembers(group_class))["__dataclass_fields__"]
    for name, field in fields.items():
        required = field.default == dataclasses.MISSING
        t = field.type
        nargs = None
        if hasattr(t, "__origin__") and hasattr(t, "__args__"):
            origin = t.__origin__
            if origin is Union and len(t.__args__) == 2 and type(None) in t.__args__:
                for arg in t.__args__:
                    if arg != type(None):
                        t = arg
                        break
                if required:
                    field.default = None
                    required = False
            elif issubclass(origin, Sequence):
                t = t.__args__[0]
                nargs = "+"
            else:
                raise ValueError(f"Unsupported type {t} for field {name}")

        if required and t == bool:
            parser.add_argument(
                f"--{name}", type=str_to_bool, nargs="?", const=True, required=True
            )
        elif required:
            parser.add_argument(f"--{name}", nargs=nargs, type=t, required=True)
        elif t == bool:
            assert field.default == False
            parser.add_argument(f"--{name}", action="store_true")
        else:
            parser.add_argument(f"--{name}", nargs=nargs, type=t, default=field.default)

    def extract(args):
        kwargs = {}
        for arg in vars(args).items():
            if arg[0] in fields:
                kwargs[arg[0]] = arg[1]
        return group_class(**kwargs)

    return extract


@dataclass(kw_only=True)
class Params:
    # Experiement parameters
    iterations: int
    densify_from: int
    densify_until: int
    experiment_name: str = ""
    dry_run: bool = False
    resample_interval: int = 100
    interpenetration_weight: float = 1e-4  # PowerFoam value; decays x1e-3 over training
    test_every: int = 500
    # Dataset parameters
    dataset: str
    data_path: str
    scene: str
    alpha_format_on_disk: str
    downsample: list[int]
    downsample_iterations: list[int]
    use_metric3d: bool = False
    is_pinhole: bool = False
    eval: bool = False

    # Model parameters
    init_type: str
    init_points: int
    final_points: int
    bkgd_color: list[float]
    oct_res: int = 8  # diffuse octahedral texture (per side), indexed by site -> entry point
    spec_res: int = 8  # specular (view dependent) octahedral texture
    density_init: float = 0.1
    t_thresh: float = 1e-3
    max_cells: int = 96  # nearest cells kept per ray
    t_thresh_late: float = 1e-2  # early ray termination once T < this ...
    t_thresh_after: int = 3000  # ... from this iteration on (kept for inference)
    distortion_weight: float = 0.01  # Mip-NeRF 360 distortion loss on the per-ray compositing weights
    skip_thresh: float = 1e-3  # inference: cells with T*alpha below this skip the SH evaluation

    # Optimizer parameters
    points_lr_init: float
    points_lr_final: float
    density_lr_init: float
    density_lr_final: float
    radii_lr_init: float
    radii_lr_final: float
    attr_lr_init: float = 2e-2
    attr_lr_final: float = 5e-4
    specular_start: int = 1_000  # specular lr ramps up as (it / start)^2
    specular_reg_weight: float = 1e-2
    diffuse_mean_pull_weight: float = 5e-3
