#!/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'Jinyang Shao'

import os
import numpy as np
import skimage.io as io
import skimage.transform as transform
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

"""
将手工扣图的对象去掉白底后插入的目标图片上，但存在mask有缺陷的情况
"""


# 对插入对象的变换
def foregroundAug(foreground):
    angle = np.random.randint(-10, 10) * (np.pi/180.0)
    zoom = np.random.random() * 0.4 + 0.8  # Zoom in range [0.8,1.2)
    t_x = np.random.randint(0, int(foreground.shape[1] / 3))
    t_y = np.random.randint(0, int(foreground.shape[0] / 3))

    tform = transform.AffineTransform(scale=(zoom, zoom),  # 放缩
                                      rotation=angle,  # 旋转
                                      translation=(t_x, t_y))  # 平移
    foreground = transform.warp(foreground, tform.inverse)

    # Random horizontal flip with 0.5 probability
    if np.random.randint(0, 100) >= 50:
        foreground = foreground[:, ::-1]

    return foreground


# mask[i]为1 表示是原图片的像素
def getForegroundMask(foreground):
    mask_new = foreground.copy()[:, :, 0]  # 只拷贝前两维
    mask_new[mask_new > 0] = 1
    return mask_new


def compose(foreground, mask, background):
    originalSize = mask.shape
    # 横向平移的距离
    translateX = np.random.randint(originalSize[0], background.shape[0] - originalSize[0] - 1)
    # 纵向平移的距离
    translateY = np.random.randint(originalSize[1], background.shape[1] - originalSize[1] - 1)

    # resize background, now is the 4 times bigger than the object
    bgSize = (foreground.shape[0] * 4, foreground.shape[1] * 4)
    background = transform.resize(background, bgSize)
    # expand mask, 注意mask是二维的
    row = np.zeros((bgSize[0] - foreground.shape[0], foreground.shape[1]), dtype=foreground.dtype)
    col = np.zeros((bgSize[0], bgSize[1] - foreground.shape[1]), dtype=foreground.dtype)
    temp = np.concatenate((mask, row), axis=0)
    mask = np.concatenate((temp, col), axis=1)
    # 平移
    mask = np.roll(mask, translateX, axis=1)  # 向右平移
    mask = np.roll(mask, translateY, axis=0)  # 向下平移

    # subtract the foregraound area from the background
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
    return composed_image, mask


def pltImgRec(mask, composed_image, saved=False):
    nz = np.nonzero(mask)  # 提取mask中非零像素的坐标
    bbox = [np.min(nz[0]), np.min(nz[1]), np.max(nz[0]), np.max(nz[1])]

    x = bbox[1]
    y = bbox[0]
    width = bbox[3] - bbox[1]
    height = bbox[2] - bbox[0]

    # Display the image
    plt.imshow(composed_image)

    # draw bbox on the image
    plt.gca().add_patch(Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none'))
    plt.axis('off')
    if saved:
        plt.savefig('../results/images/syntheManually.jpg')
    plt.show()


I = io.imread('../results/images/nadaer.jpg_0.jpg')/255.0

foreground = I.copy()
foreground[foreground >= 0.9] = 0  # Setting surrounding pixels to zero

# plt.axis('off')
# plt.imshow(foreground)
# plt.show()

foreground_new = foregroundAug(foreground)  # 对象变换

mask = getForegroundMask(foreground)  # 获取mask

background = io.imread('./background/road.jpg')/255.0
print("Read background image finish")

composedImg, mask = compose(foreground, mask, background)

# plt.imshow(composedImg)
# plt.axis('off')
# plt.show()

pltImgRec(mask, composedImg, True)


# Visualize the foreground
# plt.imshow(foreground_new)
# plt.axis('off')
# plt.show()
