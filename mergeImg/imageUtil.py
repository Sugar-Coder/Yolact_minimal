#!/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'Jinyang Shao'


import skimage.transform as transform
from skimage import img_as_ubyte
import numpy as np

from scipy import ndimage


# 按scale比例来缩放图片
def rescaleImage(origin, scale):
    # return ndimage.zoom(origin, [scale, scale, 1])  # too slow
    newimg = transform.rescale(origin, (scale, scale, 1))  # 变换后为float类型
    return img_as_ubyte(newimg)


# mask是一个二维数组，用skimage来放缩会出现数据失真
def zoomMask(origin, zoom):
    return ndimage.zoom(origin, zoom)


# 两张任意大小的图片的合成
def compose(foreground, mask, background, translateX=None, translateY=None):
    """
    translateX,Y: 在背景图上插入的点的位置（左上角）
    """
    originalSize = mask.shape
    # 横向平移的距离
    if translateX is None:
        translateX = np.random.randint(background.shape[0] // 3, background.shape[0] - originalSize[0] - 1)
    # 纵向平移的距离
    if translateY is None:
        translateY = np.random.randint(background.shape[1] // 3, background.shape[1] - originalSize[1] - 1)

    bgSize = (background.shape[0], background.shape[1])  # (height, width)
    # extract the object from the foreground
    foreground = foreground * mask.reshape(mask.shape[0], mask.shape[1], 1)

    # expand mask, 注意mask是二维的
    row = np.zeros((bgSize[0] - foreground.shape[0], foreground.shape[1]), dtype=foreground.dtype)
    col = np.zeros((bgSize[0], bgSize[1] - foreground.shape[1]), dtype=foreground.dtype)
    temp = np.concatenate((mask, row), axis=0)
    mask = np.concatenate((temp, col), axis=1)
    mask = np.roll(mask, translateX, axis=1)  # 向右平移
    mask = np.roll(mask, translateY, axis=0)  # 向下平移

    # subtract the foreground area from the background
    background = background * (1 - mask.reshape(mask.shape[0], mask.shape[1], 1))

    # expand foreground, 用0去补充
    row = np.zeros((bgSize[0] - foreground.shape[0], foreground.shape[1], 3), dtype=foreground.dtype)
    col = np.zeros((bgSize[0], bgSize[1] - foreground.shape[1], 3), dtype=foreground.dtype)
    temp = np.concatenate((foreground, row), axis=0)
    foreground = np.concatenate((temp, col), axis=1)

    # 平移
    foreground = np.roll(foreground, translateX, axis=1)
    foreground = np.roll(foreground, translateY, axis=0)

    # add together
    composed_image = background + foreground
    composed_image = composed_image.astype(np.uint8)
    return composed_image
