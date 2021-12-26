import PIL.Image
import numpy as np
import cv2

import os
from typing import Union
from multiprocessing import Pool


def check_file_is_image(image_path: Union[str, os.PathLike]):
    """Confirm the given path is an image, irrespective of its file format"""
    img = cv2.imread(image_path)
    if img is not None:
        return True
    return False


def check_directory(directory_path: Union[str, os.PathLike], summarize_dir: bool = True):
    """Check that all images in the path are of the same shape, print out a summary in the end if needed"""
    pass


def summarize_dataset(directory_path: Union[str, os.PathLike]):
    """Summarize the dataset (shapes of images, channels, outliers)"""
    pass
