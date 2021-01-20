#!/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'Jinyang Shao'


import torch
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

img1 = Image.open('../results/synthesis/output_0.jpg')
img2 = Image.open('../results/synthesis/output_1.jpg')

imgs = [img1, img2]

results = model(imgs, size=640)

results.print()  # print results to screen
results.show()  # display results
# results.save()  # save as results1.jpg, results2.jpg... etc.

print('\n', results.xyxy[0])  # print img1 predictions
