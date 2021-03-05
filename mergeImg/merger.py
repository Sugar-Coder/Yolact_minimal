#!/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'Jinyang Shao'


import json
import glob

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow

import mergeImg.imageUtil as imageUtil

"""
bounding box in the image:
+-----------------> x
|  (x1, y1)
|       o----+
|       |    |
|       +----o (x2, y2)
|
V
y

bounding box data format:
[x1, y1, x2, y2]
"""


class Merger:
    def __init__(self, bgName: str, objImgName: str):
        self.vps = []
        self.bgName = bgName  # 背景图路径
        self.objImgName = objImgName  # 待插入对象所在图片路径
        self.bgImg = None
        self.objImg = None
        self.bgData = None
        self.objData = None
        self.bgBoxes = None
        self.prefix = self.bgName.split('/')[-1].split('.')[0]  # 背景图名无后缀

    # 加载背景图的灭点信息，如果不指定，则选背景图生成的灭点
    # TODO: 生成的灭点按照原名称命名
    def loadvps(self, prefix: str = ""):
        if prefix == "":
            prefix = self.prefix
        jsonFile = f'./jsondata/vp/{prefix}_vps.json'
        with open(jsonFile, "r") as f:
            data = json.load(f)
            vps = []
            for i in range(1, 4):
                vps.append(data[f'vp{i}'])
            self.vps = vps

    # 加载背景图的边界框信息
    def loadImageData(self):
        self.bgImg = io.imread(self.bgName)
        with open(f"./jsondata/background/{self.prefix}.json", "r") as f:
            self.bgData = json.load(f)

    # 从图片中提取特定的一个对象作为待插入对象
    def loadObjectData(self, cocoClass: str = "car"):
        self.objImg = io.imread(self.objImgName)
        objPrefix = self.objImgName.split('/')[-1].split('.')[0]  # 待插入对象文件名（无后缀）
        jsonFiles = glob.glob(f"./jsondata/object/{objPrefix}_{cocoClass}_*.json")
        # TODO: 选取对象，目前默认选取第一个；可能的筛选规则：大小、类别
        file = jsonFiles[1]
        with open(file, "r") as f:
            self.objData = json.load(f)
            print(f"{file} loaded.")

    # 根据透视原理合成图片
    def mergeByVP(self):
        if len(self.vps) == 0:
            print("Loading Vanishing Points Info...")
            self.loadvps()
        # 筛选合适的vp, 即非无限远的灭点
        bgWidth = self.bgImg.shape[1]
        bgHeight = self.bgImg.shape[0]
        VP = None
        for vp in self.vps:
            if 0 < vp[0] < bgWidth and 0 < vp[1] < bgHeight:
                VP = [int(v) for v in vp]
                break
        # 背景图上的边界框 list of bounding boxes
        self.bgBoxes = [w['bbox'] for w in self.bgData]
        # 待插入对象的信息
        # category = self.objData['id']
        mask = [np.array(w) for w in self.objData['mask']]
        mask = np.array(mask)
        bbox = self.objData['bbox']
        # score = float(self.objData['score'])
        objWidth = bbox[2] - bbox[0]
        objHeight = bbox[3] - bbox[1]

        count = 0
        for bgBox in self.bgBoxes:

            X2, Y1, scale = self.calcuPosition(VP, bgBox, bbox)

            newObjWidth = int(objWidth * scale)  # 放缩后对象的宽
            newObjHeight = int(objHeight * scale)  # 放缩后对象的高

            newMask = imageUtil.zoomMask(mask, scale)  # 放缩掩码
            zoomedBox = imageUtil.getBBoxFromMask(newMask)

            if self.checkInsertion(X2 - newObjWidth, Y1, newObjWidth, newObjHeight):
                print(f"Insert to (x1={X2-newObjWidth} y1={Y1} width={newObjWidth} height={newObjHeight})")
                newObjImg = imageUtil.rescaleImage(self.objImg, scale)
                # TODO: 放大之后，前景图过大，需要剪裁
                resultImg = imageUtil.compose(newObjImg, newMask, self.bgImg, X2 - zoomedBox[2], Y1 - zoomedBox[1])

                filename = f"./image/result/{self.prefix}_out_{count}.jpg"
                print(f"Synthesis image: {filename} created.")

                plt.imshow(resultImg)
                # 画出作为映射的背景对象
                plt.gca().add_patch(
                    Rectangle((bgBox[0], bgBox[1]), bgBox[2]-bgBox[0], bgBox[3]-bgBox[1], linewidth=1,
                              edgecolor='r', facecolor='none'))
                # 新插入的对象
                plt.gca().add_patch(
                    Rectangle((X2-newObjWidth, Y1), newObjWidth, newObjHeight, linewidth=1,
                              edgecolor='b', facecolor='none'))
                plt.show()
                # io.imsave(filename, resultImg)
                plt.pause(5)

                choice = input("Continue?(y/n):")
                if choice != 'Y' and choice != 'y':
                    break
                count += 1
            else:
                print(f"The position(x1={X2-newObjWidth} y1={Y1} width={newObjWidth} height={newObjHeight}) cause overlapping: ")

    # 查看在插入点(x, y)是否会与背景的bounding box相交
    # TODO: optimize
    def checkInsertion(self, x, y, objWidth, objHeight):
        bgWidth = self.bgImg.shape[1]
        bgHeight = self.bgImg.shape[0]
        bx1 = x
        bx2 = x + objWidth
        by1 = y
        by2 = y + objHeight
        points = [[bx1, by1], [bx1, by2], [bx2, by2], [bx2, by1]]

        for i, point in enumerate(points):  # 对于矩形的四个角 是否在背景图内
            if i % 2 == 0:
                if point[0] < 0 or point[0] >= bgWidth or point[1] < 0 or point[1] >= bgHeight:
                    return False
        for box in self.bgBoxes:  # 对于背景图上所有边界框
            for point in points:  # 对于矩形的四个角
                if box[0] < point[0] < box[2] and box[1] < point[1] < box[3]:  # 存在一个点在背景中的框内
                    return False
        return True

    def calcuPosition(self, vp, bgBox, objBox) -> (int, int, float):
        """计算插入点位置(x, y)，以及放缩大小scale
        Parameters
        ----------
        vp: 背景上的一个灭点
        bgBox: 作为映射的背景框
        objBox: 待插入的物体边界框

        Returns
        ----------
        tuple: (x, y, scale) 返回插入的在背景图上的点 和 放缩的大小
        """
        bgWidth = self.bgImg.shape[1]  # 背景宽
        bgHeight = self.bgImg.shape[0]  # 背景高
        bgBoxWidth = bgBox[2] - bgBox[0]
        bgBoxHeight = bgBox[3] - bgBox[1]
        objOriginHeight = objBox[3] - objBox[1]

        isLeft = False
        # 以灭点为原点
        newBgBox = [bgBox[0]-vp[0], bgBox[1]-vp[1], bgBox[2]-vp[0], bgBox[3]-vp[1]]
        rightEdge = Edge(newBgBox)
        if rightEdge.upX < 0:
            isLeft = True  # 原框在灭点的左侧
        # 从灭点引出的两条射线
        k1 = rightEdge.upY / rightEdge.upX
        k2 = rightEdge.downY / rightEdge.downX

        deltaX = -bgBoxWidth if isLeft else bgBoxWidth
        deltaX *= 1.5  # distance between the origin box in background
        y1 = (rightEdge.upX + deltaX) * k1
        y2 = (rightEdge.upX + deltaX) * k2
        scale = abs(y1 - y2) / objOriginHeight
        # 计算相对于原来objbox位置的偏移，右上角的点(x2, y1)
        finalX = int(rightEdge.upX + deltaX + vp[0])
        finalY = int(y1 + vp[1])
        return finalX, finalY, scale


# 边界框的右边点
class Edge:
    def __init__(self, box):
        self.upX = box[2]
        self.upY = box[1]
        self.downX = box[2]
        self.downY = box[3]


if __name__ == "__main__":
    merger = Merger("./image/background/bgfirst.jpg", "./image/object/objsrc2.jpg")
    # print(merger.prefix)
    merger.loadvps()
    merger.loadImageData()
    # io.imshow(merger.bgImg)
    merger.loadObjectData()
    # print(merger.objData["id"])
    # plt.show()
    merger.mergeByVP()

