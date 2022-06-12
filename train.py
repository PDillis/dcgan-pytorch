import click
import os
from typing import Union

import time

import numpy as np
from networks import Generator, Discriminator
import utils

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import tensorboard

# ----------------------------------------------------------------------------


class DCGANTrainer:
    """Trainer class based on Niantic Lab's Trainer found in: https://github.com/nianticlabs/monodepth2"""
    def __init__(self, options):
        self.options = options
        self.epoch = 0
        self.step = 0
        self.start_time = 0
        self.device = None
        self.models = None
        self.fixed_latents = None
        self.optimizer_generator = None
        self.optimizer_discriminator = None
        self.real_label = None
        self.fake_label = None
        self.criterion = None
        self.writer = None
        self.outdir = None
        self.desc = None

        # Setup the whole networks, optimizers, etc.
        self.setup()

    def setup(self):
        # Check dataset resolution is a power of 2
        assert self.options.dataset_resolution & (self.options.dataset_resolution - 1) == 0

        # Use CUDNN benchmark
        torch.backends.cudnn.benchmark = self.options.cudnn_benchmark

        # Automatically create outdir; setup experiment description
        self.desc = f'{self.options.dataset_resolution}res'
        self.desc = f'{self.desc}-{self.options.gpus}gpus' if self.options.gpus > 0 else f'{self.desc}-cpu'
        if self.options.desc is not None:
            self.desc = f'{self.desc}-{self.options.desc}'
        self.outdir = utils.make_run_dir(self.options.outdir, self.desc, self.options.dry_run)

        # Set the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set models, move them to the set device, and initialize the weights
        self.models = utils.AttrDict()
        # Generator setup and weight initialization
        self.models.generator = Generator(self.options.latent_dim,
                                          self.options.ngf,
                                          self.options.nc,
                                          self.options.dataset_resolution).to(self.device)
        self.models.generator.init_weights()

        # Discriminator setup and weight initialization
        self.models.discriminator = Discriminator(self.options.nc,
                                                  self.options.ndf,
                                                  self.options.dataset_resolution).to(self.device)
        self.models.discriminator.init_weights()  # TODO: start from a given checkpoint!

        # Fix a latent
        self.fixed_latents = torch.randn(8, self.options.latent_dim, 1, 1, device=self.device)

        # Optimizers
        self.optimizer_generator = optim.Adam(self.models.generator.parameters(),
                                              lr=self.options.learning_rate,
                                              betas=(0.5, 0.999))
        self.optimizer_discriminator = optim.Adam(self.models.discriminator.parameters(),
                                                  lr=self.options.learning_rate,
                                                  betas=(0.5, 0.999))

        # Labels to use for real and fake data, 1.0 and 0.0; for soft labels is set to 0.9 and 0.1, respectively
        self.real_label = 1.0 - 0.1 * self.options.soft_labels
        self.fake_label = 0.0 + 0.1 * self.options.soft_labels

        # Criterion for loss
        self.criterion = nn.BCELoss().to(self.device)

        # Train logs
        self.writer = tensorboard.SummaryWriter(os.path.join(self.outdir, 'logs'))

    def train(self):
        """Run the training"""
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()

        self.generate_fake_images()
        for self.epoch in range(self.options.num_epochs):
            self.run_epoch()
            # Don't save the model if the save frequency is zero
            if self.options.save_frequency == 0:
                pass
            # Save the model whenever the use desires and at the end of training
            elif (self.epoch + 1) % self.options.save_frequency == 0 or self.epoch + 1 == self.options.num_epochs:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training through the Generator and Discriminator"""
        print(f'Training, {self.epoch}')

    def save_model(self):
        pass

    def load_model(self):
        pass

    def log(self):
        """Log events to Tensorboard"""
        pass

    def compute_loss(self):
        pass

    def generate_fake_images(self, iteration: int):
        """Generate a fake batch of images to log"""
        g = torch.Generator(device=self.device).manual_seed(self.options.seed)
        grid_width, grid_height = utils.get_grid_dims(self.options.dataset_resolution,  # TODO: rectangular images!
                                                      self.options.dataset_resolution,  # TODO: rectangular images!
                                                      self.options.snapshot_resolution)
        z = torch.randn(grid_width * grid_height, self.options.latent_dim, 1, 1, generator=g).to(self.device)
        images = self.models.generator(z)
        print('min', images.min(), 'max', images.max())
        utils.save_image_grid(images.detach().numpy(),
                              os.path.join(self.outdir, f'fakes_epoch{iteration:06d}.jpg'),
                              [0, 1],
                              (grid_width, grid_height))
# ----------------------------------------------------------------------------


@click.command()
@click.pass_context
# TODO: separate into network architecture, hyperparams, training (iterations, loss, etc.)...
# Training options
@click.option('--num-epochs', type=click.IntRange(min=1), help='Number of epochs to train for', default=10, show_default=True)
@click.option('--learning-rate', '-lr', type=click.FloatRange(min=0.0, min_open=True), help='Learning rate of the networks', default=0.0002, show_default=True)  # TODO: use different lr's for G and D
@click.option('--latent-dim', '-ld', type=click.IntRange(min=1), help='Size of the latent dimension', default=100, show_default=True)
@click.option('--gpus', help='GPUs to use (numbering according to CUDA_VISIBLE_DEVICES)', default=0, show_default=True)
@click.option('--pretrained-pth', type=click.Path(file_okay=True, dir_okay=False), help='Path to pretrained model', default='')  # TODO: allow https?
@click.option('--dataset-resolution', type=click.IntRange(min=8), help='Dataset resolution', default=64, show_default=True)
@click.option('--dataset-channels', 'nc', type=click.IntRange(min=1), help='Channels in image dataset; RGB by default', default=3, show_default=True)
@click.option('--use-soft-labels', 'soft_labels', is_flag=True, help='Use soft labels for real and fake images (0.9/0.1 instead of 1.0/0.0, respectively)')
@click.option('--cudnn-benchmark', type=bool, help='Use CUDNN benchmark for faster training, given that images are of same shape', default=True, show_default=True)
@click.option('--seed', type=int, help='Random seed to use for training', default=0)
@click.option('--snapshot-res', 'snapshot_resolution', type=click.Choice(['480p', '720p', '1080p', '4k', '8k']), help='Set the resolution of the training snapshot', default='1080p', show_default=True)
@click.option('--snapshot-save', 'snapshot_save_frequency', type=click.IntRange(min=0))
@click.option('--dry-run', is_flag=True, help='Print the network and training options and exit')
# GAN network architecture; if None, will use auto config
@click.option('--g-blocks', type=click.IntRange(min=0), help='Number of intermediate blocks for the Generator', default=None)
@click.option('--d-blocks', type=click.IntRange(min=0), help='Number of intermediate blocks for the Discriminator', default=None)
@click.option('--g-filters', 'ngf', type=click.IntRange(min=1), help='Number of filters per block of the Generator', default=128, show_default=True)
@click.option('--d-filters', 'ndf', type=click.IntRange(min=1), help='Number of filters per block of the Discriminator', default=128, show_default=True)
@click.option('--save-freq', 'save_frequency', type=click.IntRange(min=0), help='How often to save the model (w.r.t. epochs)', default=1, show_default=True)
@click.option('--outdir', type=click.Path(exists=False, file_okay=False), help='Root directory path to save results', default=os.path.join(os.getcwd(), 'training_runs'), show_default=True)
@click.option('--desc', type=str, help='String to include in the outdir name', default=None)
def main(ctx, **kwargs):
    # Get the parameters obtained by click and pass them to the DCGANTrainer
    params = utils.AttrDict(ctx.params)
    trainer = DCGANTrainer(options=params)
    # Print the networks

    # Train when not running a dry run
    if not params.dry_run:
        trainer.train()


# ----------------------------------------------------------------------------


if __name__ == '__main__':
    main()


# ----------------------------------------------------------------------------
