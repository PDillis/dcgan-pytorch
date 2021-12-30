import os
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import weights_init


class GeneratorBlock(nn.Module):
    """Base block of the Generator"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super(GeneratorBlock, self).__init__()

        # For the rest of the default values, refer to: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
        # TODO: make sure to fix checkerboard artifacts: https://distill.pub/2016/deconv-checkerboard/
        # Basically, do a resize, a pad, then a convolution
        # self.upsample = nn.Upsample(mode='nearest', size=tbd)
        # for padding, use: nn.functional.pad(x, pad=[0, y, 0, y])   TBD
        # self.conv = nn.Conv2d()
        self.convt = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, is_final_block: bool = False):
        x = self.convt(x)
        # Apply BatchNorm and ReLU for first and intermediate blocks
        if not is_final_block:
            x = self.bn(x)
            x = F.relu(x)
        # Use Tanh for final output block
        else:
            x = F.tanh(x)
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
        for i in range(num_intermediate_blocks + 1, 0, -1):
            output_channels = ngc if i == num_intermediate_blocks else ngf * 2 ** (i - 1)
            setattr(self, f'block{num_intermediate_blocks-i+1}',
                    GeneratorBlock(ngf * 2 ** i, output_channels, 4, 2, 1))

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
    """Base block of the Discriminator"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super(DiscriminatorBLock, self).__init__()

        # For the rest of the default values, refer to: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, is_intermediate_block: bool = False, is_final_block: bool = False):
        # Apply Convolution at the beginning of every block
        x = self.conv(x)
        # Apply Batchnorm and Leaky ReLU for intermediate block
        if is_intermediate_block:
            x = self.bn(x)
            x = F.leaky_relu(x, negative_slope=0.2)
        # Use sigmoid for final block
        elif is_final_block:
            x = F.sigmoid(x)
        # Only apply Leaky ReLU for the first block
        else:
            x = F.leaky_relu(x, negative_slope=0.2)
        return x


class Discriminator(nn.Module):
    def __init__(self, nc: int, ndf: int, num_intermediate_blocks: int = 3):
        super(Discriminator, self).__init__()
        self.num_intermediate_blocks = num_intermediate_blocks

        # Setup first and intermediate blocks
        for i in range(num_intermediate_blocks + 1):
            input_channels = nc if i == 0 else ndf * 2 ** (i - 1)  # First block has nc channels as input
            setattr(self, f'block{i}', DiscriminatorBLock(input_channels, ndf * 2 ** i, 4, 2, 1))

        setattr(self, f'block{num_intermediate_blocks+1}', DiscriminatorBLock(ndf * 2 ** num_intermediate_blocks, 1, 4, 1, 0))

    def init_weights(self, pretrained_weights_path: str = None):
        """Initialize the weights of the Discriminator, lest there exists a pretrained model"""
        if pretrained_weights_path is not None:
            print(f'Loading weights from {pretrained_weights_path}...')
            # TODO: this, so see how I save the .pth files
        else:
            print(f'\t=> Initializing Discriminator with Normal weights and zero bias...')
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.normal_(module.weight, 0.0, 0.02)  # N(0, 0.02^2)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.normal_(module.weight, 1.0, 0.02)  # N(1.0, 0.02^2)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        # Apply the first and intermediate blocks to the input
        x = self.block0(x)
        for i in range(1, self.num_intermediate_blocks+1):
            x = getattr(self, f'block{i}')(x, is_intermediate_block=True)
        # Final block
        x = getattr(self, f'block{self.num_intermediate_blocks+1}')(x, is_final_block=True)

        return x
