import click
import os
from typing import Union

import numpy as np
from networks import Generator, Discriminator
import torch
import torch.optim as optim


# ----------------------------------------------------------------------------


@click.command()
# TODO: separate into network architecture, hyperparams, training (iterations, loss, etc.)...
@click.option('--latent-dim', '-ld', type=click.IntRange(min=1), help='Size of the latent dimension', default=100, show_default=True)
@click.option('--gpus', help='GPUs to use (numbering according to CUDA_VISIBLE_DEVICES)', default=0)
@click.option('--pretrained-pth', type=click.Path(file_okay=True, dir_okay=False), help='Path to pretrained model')  # TODO: allow https?
@click.option('--dataset-resolution', type=click.IntRange(min=8), help='Dataset resolution', default=64)
def main(latent_dim: int, gpus: int, pretrained_pth: Union[str, os.PathLike], dataset_resolution: int):
    num_intermediate_blocks = int(np.log2(dataset_resolution / 8))  # 8 is the minimum possible resolution
    generator = Generator(latent_dim=latent_dim, ngf=8, ngc=3, num_intermediate_blocks=num_intermediate_blocks)
    pass


# ----------------------------------------------------------------------------


if __name__ == '__main__':
    main()


# ----------------------------------------------------------------------------
