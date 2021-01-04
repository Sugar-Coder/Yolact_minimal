#!/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'Jinyang Shao'

import json
import glob
import random
import os
import logging

import skimage.io as io
import skimage.transform as transform
import matplotlib.pyplot as plt
import numpy as np


logging.basicConfig(level=logging.INFO)
os.makedirs('../results/synthesis', exist_ok=True)


def loadObjImageInfo(fileprefix='nadaer'):
    """
    A generator, every time called return one object's image and its information
    从images文件夹下面读取指定文件前缀的图片,作为提取对象的源
    @return: cutout后的小图片，类别，置信度，边界框
    """
    fileprefix = fileprefix.split('.')[0]
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


def guidedInsertion(bgSize, bBoxes, box):
    """
    bgSize: (imgHeight, imgWidth)
    """
    objWidth = box[2] - box[0]  # x2 - x1
    objHeight = box[3] - box[1]  # y2 - y1

    def checkIntersect(bx, by):
        """
        检验在背景图上(bx,by)的位置插入对象，是否与其他bbox相交
        """
        bx1 = bx
        bx2 = bx + objWidth
        by1 = by
        by2 = by + objHeight
        points = [[bx1, by1], [bx1, by2], [bx2, by1], [bx2, by2]]
        for box in bBoxes:
            for point in points:  # 对于矩形的四个角
                if point[0] < 0 or point[0] >= bgSize[1] or point[1] < 0 or point[1] >= bgSize[0]:
                    return False
                if box[0] < point[0] < box[2] and box[1] < point[1] < box[3]:  # 存在一个点在背景中的框内
                    return False
        return True


    while True:
        bgObjBox = bBoxes[random.randint(0, len(bBoxes)-1)]  # 选择背景图上的一个边界框
        left, right, down, up = bgObjBox[0] - objWidth, bgObjBox[2], bgObjBox[3], bgObjBox[1] - objHeight
        # 在b的四周随机选择一个位置
        choice = random.randint(0, 3)
        if choice == 0:
            # 考虑在原对象的左边插入
            places = [p for p in range(down, up, objHeight)]
            random.shuffle(places)
            for y in places:
                if checkIntersect(left, y):
                    yield left, y
                    break
        elif choice == 1:
            places = [p for p in range(down, up, objHeight)]
            random.shuffle(places)
            for y in places:
                if checkIntersect(right, y):
                    yield right, y
                    break
        # 考虑上下位置
        elif choice == 2:
            places = [p for p in range(left, right, objHeight)]
            random.shuffle(places)
            for x in places:
                if checkIntersect(x, down):
                    yield x, down
                    break
        elif choice == 3:
            places = [p for p in range(left, right, objHeight)]
            random.shuffle(places)
            for x in places:
                if checkIntersect(x, up):
                    yield x, up
                    break


def insertion(bgPath, Img, mask, bbox, genNum=1):
    """
    background: 背景图名称: xxx.jpg or xxx
    Img: 包含待插入对象的原始图片
    mask: 对象掩码
    bbox: 待插入对象的掩码边界
    genNum: 需要生成的合成图片个数
    """
    background = io.imread(bgPath)
    bgSize = background.shape[:2]
    bgName = bgPath.split('/')[-1].split('.')[0]
    fname = f'../results/json/{bgName}_background.json'
    bBoxes = []  # 背景图上的边界框
    bId = []  # 背景上对象类别
    if os.path.exists(fname):  # 如果背景图上存在边界框
        f = open(fname, 'r')
        data = json.load(f)
        f.close()
        bBoxes = [w['bbox'] for w in data]
        bId = [w['id'] for w in data]
    if len(bBoxes) == 0:
        logging.info('random insertion')
        return compose(Img, mask, background)
    else:
        logging.info('guided insertion')
        placeGener = guidedInsertion(bgSize, bBoxes, bbox)
        for i in range(genNum):
            posx, posy = next(placeGener)
            synImg = compose(Img, mask, background, posx - bbox[0], posy - bbox[1])
            plt.imshow(synImg)
            plt.show()
            io.imsave(f'../results/synthesis/output_{i}.jpg', synImg)


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


# tested
def testGuidedInsertion():
    background = io.imread('../images/bgfirst.jpg')
    bgSize = background.shape[:2]
    fname = '../results/json/bgfirst_background.json'
    f = open(fname, 'r')
    data = json.load(f)
    f.close()
    bBoxes = [w['bbox'] for w in data]
    objGen = loadObjImageInfo()
    _, _, _, box, mask = next(objGen)
    placeGener = guidedInsertion(bgSize, bBoxes, box)
    posx, posy = next(placeGener)
    print(posx, posy)
    Img = io.imread('./nadaer.jpg')
    res = compose(Img, mask, background, posx - box[0], posy - box[1])
    plt.imshow(res)
    plt.show()


# tested
def testCompose():
    gen = loadObjImageInfo()
    _, _, score, bbox, mask = next(gen)
    foreground = io.imread('./nadaer.jpg')
    # plt.imshow(foreground)
    background = io.imread('../images/bgfirst.jpg')
    # plt.imshow(background)
    res = compose(foreground, mask, background)
    plt.imshow(res)
    plt.show()


def testInsertion():
    bgFilepath = '../images/bgfirst.jpg'
    srcImgname = 'nadaer.jpg'
    Img = io.imread(srcImgname)
    objGen = loadObjImageInfo('nadaer')
    _, _, _, box, mask = next(objGen)  # 读取带插入对象的信息
    insertion(bgFilepath, Img, mask, box, 2)


if __name__ == '__main__':
    # testGuidedInsertion()
    testInsertion()
    # testCompose()
