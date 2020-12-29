#!/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'Jinyang Shao'

import json
import glob
import random

import skimage.io as io
import skimage.transform as transform
import matplotlib.pyplot as plt
import numpy as np


def loadImageInfo(fileprefix='nadaer'):
    """
    A generator, every time called return one object's image and its information
    """
    imgFiles = glob.glob(f'../results/images/{fileprefix}_*.jpg')
    for i in range(len(imgFiles)):
        objectImg = io.imread(f'../results/images/{fileprefix}_{i}.jpg')
        f = open(f'../results/json/{fileprefix}_{i}.json', "r")
        data = json.load(f)
        f.close()
        category = data['id']
        mask = [np.array(w) for w in data['mask']]
        mask = np.array(mask)
        bbox = data['bbox']
        score = float(data['score'])
        yield objectImg, category, score, bbox, mask


imgObjGenerator = loadImageInfo()
objImg, category, score, bbox, mask = next(imgObjGenerator)
print(category, bbox)
# plt.imshow(objImg)
# plt.imshow(mask)
# plt.show()
