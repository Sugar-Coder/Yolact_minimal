#!/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'Jinyang Shao'


import skimage.transform as transform
from skimage import img_as_ubyte
import numpy as np

from scipy import ndimage


# 按scale比例来缩放图片
# TODO: exception
def rescaleImage(origin, scale):
    # return ndimage.zoom(origin, [scale, scale, 1])  # too slow
    newimg = transform.rescale(origin, (scale, scale, 1))  # 变换后为float类型
    return img_as_ubyte(newimg)


# mask是一个二维数组，用skimage来放缩会出现数据失真
def zoomMask(origin, zoom):
    return ndimage.zoom(origin, zoom)


# 两张任意大小的图片的合成
def compose(foreground, mask, background, translateX: int = None, translateY: int = None):
    """
    translateX,Y: foreground需要平移的距离（左上角）
    mask: 放缩后的mask
    foreground: 放缩后的对象图
    """
    originalSize = mask.shape
    objBox = getBBoxFromMask(mask)
    # print("The mask shape:", mask.shape)
    # print("The foreground shape:", foreground.shape)
    # print("The objBox:", objBox)
    bgSize = (background.shape[0], background.shape[1])  # (height, width)
    # 横向平移的距离
    if translateX is None:
        translateX = np.random.randint(background.shape[0] // 3, background.shape[0] - originalSize[0] - 1)
    # 纵向平移的距离
    if translateY is None:
        translateY = np.random.randint(background.shape[1] // 3, background.shape[1] - originalSize[1] - 1)

    translateX += objBox[0]
    translateY += objBox[1]  # 将mask平移到左上角
    # prune mask
    mask = mask[objBox[1]:, objBox[0]:]  # 剪裁mask左上角的矩形 mask[height,width]
    mask = mask[:bgSize[0], :bgSize[1]]  # 剪裁超出背景框的mask
    # print(mask.shape)
    # pune foreground
    foreground = foreground[objBox[1]:, objBox[0]:, :]
    foreground = foreground[:bgSize[0], :bgSize[1], :]
    # print(foreground.shape)

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


# 获取mask的边界框
def getBBoxFromMask(mask):
    nz = np.nonzero(mask)  # 提取mask中非零像素的坐标
    bbox = [np.min(nz[1]), np.min(nz[0]), np.max(nz[1]), np.max(nz[0])]
    return bbox  # [x1, y1, x2, y2]
