#!/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'Jinyang Shao'

import skimage.io as io
import skimage.transform as transform
import matplotlib.pyplot as plt
import numpy as np


img = io.imread('../images/bgfirst.jpg')
newimg = img[:, 900:, :]
plt.imshow(newimg)
plt.show()
io.imsave('../images/bgfirst1.jpg', newimg)
