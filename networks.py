import os
from typing import Union

import torch
import torch.nn as nn

from utils import weights_init


class GeneratorBlock(nn.Module):
    """Block of the Generator that will take a tensor of the form [batch_size, in_channels, """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, relu_inplace: bool = True):
        super(GeneratorBlock, self).__init__()
        self.convt = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False)  # TODO: make sure to fix checkerboard artifacts
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.tanh = nn.Tanh()

    def forward(self, x, is_final_block: bool = False):
        x = self.convt(x)
        # Apply BatchNorm and ReLU for first and intermediate blocks
        if not is_final_block:
            x = self.bn(x)
            x = self.relu(x)
        # Use Tanh for final output block
        else:
            x = self.tanh(x)
        return x


# Baseline; TODO: make improvements?
class Generator(nn.Module):
    def __init__(self, latent_dim: int, ngf: int, ngc: int, num_intermediate_blocks: int = 3):
        super(Generator, self).__init__()
        self.num_intermediate_blocks = num_intermediate_blocks

        # First and last blocks
        setattr(self, 'block0', GeneratorBlock(latent_dim, ngf * 2 ** num_intermediate_blocks, 4, 1, 0))
        setattr(self, f'block{num_intermediate_blocks+1}', GeneratorBlock(ngf, ngc, 4, 2, 1))

        # Setting intermediate blocks allows for flexibility (i.e., set as many as desired)
        for i in range(num_intermediate_blocks, 0, -1):
            setattr(self, f'block{num_intermediate_blocks-i+1}',
                    GeneratorBlock(ngf * 2 ** i, ngf * 2 ** (i - 1), 4, 2, 1))

    def train(self):
        pass

    def init_weights(self, pretrained_weights_path: Union[str, os.PathLike] = None) -> None:
        """Initialize the weights of the Generator, lest there exists a pretrained model"""
        if pretrained_weights_path is not None:
            print(f'Loading weights from {pretrained_weights_path}...')
            # TODO: this, so see how I save the .pth files
        else:
            print(f'\t=> Initializing Generator with Normal weights and zero bias...')
            for module in self.modules():
                if isinstance(module, nn.ConvTranspose2d):
                    nn.init.normal_(module.weight, 0.0, 0.02)   # N(0, 0.02^2)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.normal_(module.weight, 1.0, 0.02)   # N(1.0, 0.02^2)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        # Apply the first and intermediate blocks to the input
        for i in range(self.num_intermediate_blocks+1):
            x = getattr(self, f'block{i}')(x)
        # Apply the final block
        x = getattr(self, f'block{self.num_intermediate_blocks+1}')(x, is_final_block=True)

        return x


class DiscriminatorBLock(nn.Module):
    def __init__(self):
        super(DiscriminatorBLock, self).__init__()

    def forward(self, x):
        pass


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def train(self):
        pass

    def init_weights(self):
        pass

    def forward(self):
        pass
