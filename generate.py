import torch
import numpy as np

import matplotlib.pyplot as plt
import PIL.Image

from networks import Generator, Discriminator
import utils


# With the resolution, set the number of blocks for G and D
dataset_resolution = 128
num_intermediate_blocks = int(np.log2(dataset_resolution / 8))
# Set the seed
torch.manual_seed(0)

z = torch.randn(1, 100, 1, 1)

gen = Generator(100, 16, 3, num_intermediate_blocks)
disc = Discriminator(3, 1, num_intermediate_blocks)

img = utils.z_to_img(gen, z)
img = utils.create_image_grid(img)
PIL.Image.fromarray(img, 'RGB').show()

gen.init_weights()
disc.init_weights()

img = utils.z_to_img(gen, z)
img = utils.create_image_grid(img)
PIL.Image.fromarray(img, 'RGB').show()