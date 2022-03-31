#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @auther:FelixFu
# @Date: 2021.10.1
# @github:https://github.com/felixfu520

from PIL import Image
import numpy as np

from albumentations.core.transforms_interface import ImageOnlyTransform


class Custom(ImageOnlyTransform):
    def apply(self, img, **params):
        return histogram_fn(img)


def histogram_fn(image):
    """
    histogram stretch a gray image(ndarray)
    """
    rows, cols, channels = image.shape
    assert channels == 1
    flat_gray = image.reshape((cols * rows,)).tolist()
    A = min(flat_gray)
    B = max(flat_gray)
    image = np.uint8(255 / (B - A + 0.1) * (image - A) + 0.5)
    return image
