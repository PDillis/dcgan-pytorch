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
# Training options
@click.option('--learning-rate', '-lr', type=click.FloatRange(min=0.0, min_open=True), help='Learning rate of the networks', default=0.0002)  # TODO: use different lr's for G and D
@click.option('--latent-dim', '-ld', type=click.IntRange(min=1), help='Size of the latent dimension', default=100, show_default=True)
@click.option('--gpus', help='GPUs to use (numbering according to CUDA_VISIBLE_DEVICES)', default=0)
@click.option('--pretrained-pth', type=click.Path(file_okay=True, dir_okay=False), help='Path to pretrained model')  # TODO: allow https?
@click.option('--dataset-resolution', type=click.IntRange(min=8), help='Dataset resolution', default=64)
# GAN network architecture; if None, will use auto config
@click.option('--g-blocks', help='Number of intermediate blocks for the Generator', default=None)
@click.option('--d-blocks', help='Number of intermediate blocks for the Discriminator', default=None)
def main(latent_dim: int, gpus: int, pretrained_pth: Union[str, os.PathLike], dataset_resolution: int,
         g_blocks: int, d_blocks: int):
    # Set the number of intermediate blocks automatically if user doesn't specify it
    num_intermediate_blocks = int(np.log2(dataset_resolution / 8))  # 8 is the minimum possible resolution
    if g_blocks is None:
        g_blocks = num_intermediate_blocks
    if d_blocks is None:
        d_blocks = num_intermediate_blocks
    assert dataset_resolution == 8 * 2 ** g_blocks and dataset_resolution == 8 * 2 ** d_blocks  # Sanity check

    # Setup the networks
    generator = Generator(latent_dim=latent_dim, ngf=128, ngc=3, num_intermediate_blocks=num_intermediate_blocks)
    discriminator = Discriminator(nc=3, ndf=128, num_intermediate_blocks=num_intermediate_blocks)

    # Init weights
    generator.init_weights()
    discriminator.init_weights()

    # TODO: put all of the above into a Train class and then do a .run_epoch() here

    pass


# ----------------------------------------------------------------------------


if __name__ == '__main__':
    main()


# ----------------------------------------------------------------------------
