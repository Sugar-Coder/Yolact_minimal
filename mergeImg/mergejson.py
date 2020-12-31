#!/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'Jinyang Shao'

import json
import glob
import random
import os

import skimage.io as io
import skimage.transform as transform
import matplotlib.pyplot as plt
import numpy as np


def loadImageInfo(fileprefix='nadaer'):
    """
    A generator, every time called return one object's image and its information
    从images文件夹下面读取指定文件前缀的图片,作为提取对象的源
    @return: cutout后的小图片，类别，置信度，边界框
    """
    imgFiles = glob.glob(f'../results/images/{fileprefix}_*.jpg')
    for i in range(len(imgFiles)):
        print(f'{fileprefix}_{i} loaded.')
        objectImg = io.imread(f'../results/images/{fileprefix}_{i}.jpg')  # cutout后的合成的时候其实不需要
        f = open(f'../results/json/{fileprefix}_{i}.json', "r")
        data = json.load(f)
        f.close()
        category = data['id']
        mask = [np.array(w) for w in data['mask']]
        mask = np.array(mask)
        bbox = data['bbox']
        score = float(data['score'])
        yield objectImg, category, score, bbox, mask


def randomInsertion(bgPath, Img, mask, bbox):
    """
    background: 背景图名称: xxx.jpg
    Img: 包含待插入对象的原始图片
    mask: 对象掩码
    bbox: 掩码边界
    """
    background = io.imread(bgPath)
    bgName = bgPath.split('/')[-1].split('.')[0]
    fname = f'../results/json/{bgName}_background.json'
    bBoxes = []  # 背景图上的边界框
    if os.path.exists(fname):  # 如果背景图上存在边界框
        f = open(fname, 'r')
        data = json.load(f)
        f.close()
        bBoxes = [w['bbox'] for w in data]
    if len(bBoxes) == 0:
        compose(Img, mask, background)
    else:
        pass


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

    bgSize = (background.shape[0], background.shape[1])

    # extract the object from the foreground
    foreground = foreground * mask.reshape(mask.shape[0], mask.shape[1], 1)

    # expand mask, 注意mask是二维的
    row = np.zeros((bgSize[0] - foreground.shape[0], foreground.shape[1]), dtype=foreground.dtype)
    col = np.zeros((bgSize[0], bgSize[1] - foreground.shape[1]), dtype=foreground.dtype)
    temp = np.concatenate((mask, row), axis=0)
    mask = np.concatenate((temp, col), axis=1)
    print(mask.shape)
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
    return composed_image


# imgObjGenerator = loadImageInfo()
# objImg, category, score, bbox, mask = next(imgObjGenerator)
# print(category, bbox)
# plt.imshow(objImg)
# plt.imshow(mask)
# plt.show()

if __name__ == '__main__':
    gen = loadImageInfo()
    _, _, score, bbox, mask = next(gen)
    foreground = io.imread('../images/nadaer.jpg')
    # plt.imshow(foreground)
    background = io.imread('./background/bgfirst.jpg')
    # plt.imshow(background)
    res = compose(foreground, mask, background)
    plt.imshow(res)
    plt.show()
