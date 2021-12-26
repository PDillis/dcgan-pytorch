import os
import click
import json
import re
from locale import atof

from typing import Union, Optional, Tuple, List
from collections import OrderedDict
import sys

import numpy as np
import torch.nn as nn


# ----------------------------------------------------------------------------


# init_dict for initializing the neural networks
init_dict = {
        'uniform': nn.init.uniform_,    # U([0, 1])
        'normal': nn.init.normal_,      # N(0, 1)
        'constant': nn.init.constant_,
        'ones': nn.init.ones_,
        'zeros': nn.init.zeros_,
        'eye': nn.init.eye_,
        'dirac': nn.init.dirac_,
        'xavier_uniform': nn.init.xavier_uniform_,
        'xavier': nn.init.xavier_normal_,
        'kaiming_uniform': nn.init.kaiming_uniform_,
        'kaiming': nn.init.kaiming_normal_,
        'orthogonal': nn.init.orthogonal_,
        'sparse': nn.init.sparse_
}


def weights_init(module: nn.Module, init_type: str = 'kaiming', const: float = None, bias_init: float = 0.0) -> None:
    """
    Auxiliary function to initialize the parameters of a network
    Args:
        module (nn.Module): Network to initialize
        init_type (str): type of initialization to apply; must be in init_dict (though I've listed all available inits)
        const (float): constant value to fill the weight, used if init_type='constant'
        bias_init (float): value to initialize the bias (we zero-initialize the bias by default)
    Output:
        (NoneType), applies the desired initialization to the module's layers
    """
    # We will use a set of available initializations
    if init_type not in init_dict:
        print(f'{init_type} not available.')
        sys.exit(1)
    if isinstance(module, nn.Linear):
        # The special case will be the constant initialization
        if init_type == 'constant':
            # Make sure the user has provided the constant value
            # (guard against user forgetting and using a default value)
            assert const is not None, 'Please provide the constant value! (const)'
            # Then, initialize the weight with the provided constant
            init_dict[init_type](module.weight, const)
        else:
            # Else, it's one of the other initialization methods:
            init_dict[init_type](module.weight)
        # Initialize the bias with zeros (in-place); can be changed if so desired
        module.bias.data.fill_(bias_init)


# ----------------------------------------------------------------------------

def create_image_grid(images: np.ndarray, grid_size: Optional[Tuple[int, int]] = None):
    """
    Create a grid with the fed images
    Args:
        images (np.array): array of images
        grid_size (tuple(int)): size of grid (grid_width, grid_height)
    Returns:
        grid (np.array): image grid of size grid_size
    """
    # Sanity check
    assert images.ndim == 3 or images.ndim == 4, f'Images has {images.ndim} dimensions (shape: {images.shape})!'
    num, img_h, img_w, c = images.shape
    # If user specifies the grid shape, use it
    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
        # If one of the sides is None, then we must infer it (this was divine inspiration)
        if grid_w is None:
            grid_w = num // grid_h + min(num % grid_h, 1)
        elif grid_h is None:
            grid_h = num // grid_w + min(num % grid_w, 1)

    # Otherwise, we can infer it by the number of images (priority is given to grid_w)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    # Sanity check
    assert grid_w * grid_h >= num, 'Number of rows and columns must be greater than the number of images!'
    # Get the grid
    grid = np.zeros([grid_h * img_h, grid_w * img_h] + list(images.shape[-1:]), dtype=images.dtype)
    # Paste each image in the grid
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[y:y + img_h, x:x + img_w, ...] = images[idx]
    return grid


# ----------------------------------------------------------------------------


def parse_fps(fps: Union[str, int]) -> int:
    """Return FPS for the video; at worst, video will be 1 FPS, but no lower.
    Useful if we don't have Click, else simply use Click.IntRange(min=1)"""
    if isinstance(fps, int):
        return max(fps, 1)
    try:
        fps = int(atof(fps))
        return max(fps, 1)
    except ValueError:
        print(f'Typo in "--fps={fps}", will use default value of 30')
        return 30


def num_range(s: str, remove_repeated: bool = True) -> List[int]:
    """
    Extended helper function from the original (original is contained here).
    Accept a comma separated list of numbers 'a,b,c', a range 'a-c', or a combination
    of both 'a,b-c', 'a-b,c', 'a,b-c,d,e-f,...', and return as a list of ints.
    """
    str_list = s.split(',')
    nums = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for el in str_list:
        match = range_re.match(el)
        if match:
            # Sanity check 1: accept ranges 'a-b' or 'b-a', with a<=b
            lower, upper = int(match.group(1)), int(match.group(2))
            if lower <= upper:
                r = list(range(lower, upper + 1))
            else:
                r = list(range(upper, lower + 1))
            # We will extend nums as r is also a list
            nums.extend(r)
        else:
            # It's a single number, so just append it (if it's an int)
            try:
                nums.append(int(atof(el)))
            except ValueError:
                continue  # we ignore bad values
    # Sanity check 2: delete repeating numbers by default, but keep order given by user
    if remove_repeated:
        nums = list(OrderedDict.fromkeys(nums))
    return nums


def parse_slowdown(slowdown: Union[str, int]) -> int:
    """Function to parse the 'slowdown' parameter by the user. Will approximate to the nearest power of 2."""
    # TODO: slowdown should be any int
    if not isinstance(slowdown, int):
        try:
            slowdown = atof(slowdown)
        except ValueError:
            print(f'Typo in "{slowdown}"; will use default value of 1')
            slowdown = 1
    assert slowdown > 0, '"slowdown" cannot be negative or 0!'
    # Let's approximate slowdown to the closest power of 2 (nothing happens if it's already a power of 2)
    slowdown = 2**int(np.rint(np.log2(slowdown)))
    return max(slowdown, 1)  # Guard against 0.5, 0.25, ... cases


# ----------------------------------------------------------------------------


def save_config(ctx: click.Context, run_dir: Union[str, os.PathLike], save_name: str = 'config.json') -> None:
    """Save the configuration stored in ctx.obj into a JSON file at the output directory."""
    with open(os.path.join(run_dir, save_name), 'w') as f:
        json.dump(ctx.obj, f, indent=4, sort_keys=True)


# ----------------------------------------------------------------------------


def make_run_dir(outdir: Union[str, os.PathLike], desc: str, dry_run: bool = False) -> str:
    """Reject modernity, return to automatically create the run dir."""
    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):  # sanity check, but click.Path() should clear this one
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1  # start with 00000
    run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(run_dir)  # make sure it doesn't already exist

    # Don't create the dir if it's a dry-run
    if not dry_run:
        print('Creating output directory...')
        os.makedirs(run_dir)
    return run_dir
