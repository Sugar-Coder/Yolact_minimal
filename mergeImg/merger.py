#!/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'Jinyang Shao'


import json
import glob

import numpy as np
import skimage.io as io

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

    # 加载背景图的灭点信息，如果不指定，则选背景图生成的灭点
    # TODO: 生成的灭点按照原名称命名
    def loadvps(self, prefix: str = ""):
        if prefix == "":
            prefix = self.bgName.split('/')[-1].split('.')[0]
        jsonFile = f'{prefix}_vps.json'
        with open(jsonFile, "r") as f:
            data = json.load(f)
            vps = []
            for i in range(1, 4):
                vps.append(data[f'vp{i}'])
            self.vps = vps

    # 加载背景图的边界框信息
    def loadImageData(self):
        bgPrefix = self.bgName.split('/')[-1].split('.')[0]  # 背景图文件名（无后缀）
        self.bgImg = io.imread(self.bgName)
        # self.objImg = io.imread(self.objImgName)  # 等到合成的时候再读取
        with open(f"./image/background/{bgPrefix}.json", "r") as f:
            self.bgData = json.load(f)

    # 从图片中提取特定的一个对象作为待插入对象
    def loadObjectData(self, cocoClass: str = "car"):
        objPrefix = self.objImgName.split('/')[-1].split('.')[0]  # 待插入对象文件名（无后缀）
        jsonFiles = glob.glob(f"./jsondata/object/{objPrefix}_{cocoClass}_*.json")
        # TODO: 选取对象，目前默认选取第一个；可能的筛选规则：大小、类别
        file = jsonFiles[0]
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
        bgBoxes = [w['bbox'] for w in self.bgData]
        # 待插入对象的信息
        category = self.objData['id']
        mask = [np.array(w) for w in self.objData['mask']]
        mask = np.array(mask)
        bbox = self.objData['bbox']
        score = float(self.objData['score'])

        for bgBox in bgBoxes:
            bgBoxWidth = bgBox[2] - bgBox[0]
            bgBoxHeight = bgBox[3] - bgBox[1]
            # TODO: 面积？原来对象的位置？

