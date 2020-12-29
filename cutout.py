#!/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'Jinyang Shao'

"""
This script is to cut out objects from images. And output the relevant box and mask data
"""
import argparse
import re
import json
import os

import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import cv2

from config import get_config, COCO_CLASSES
from modules.yolact import Yolact
from utils.coco import COCODetection, detect_collate
from utils.output_utils import nms, after_nms, draw_img


os.makedirs('results/json', exist_ok=True)

parser = argparse.ArgumentParser(description='YOLACT Detection.')
parser.add_argument('--background', default=False, action='store_true', help='Is the image is background, only output the bbox')
parser.add_argument('--weight', default='weights/best_29.3_res50_coco_400001.pth', type=str, help='The model for detection.')
parser.add_argument('--image', default='images', type=str, help='The folder of images for detecting.')
parser.add_argument('--video', default=None, type=str, help='The path of the video to evaluate.')
parser.add_argument('--img_size', type=int, default=550, help='The image size for validation.')
parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
parser.add_argument('--hide_mask', default=False, action='store_true', help='Hide masks in results.')
parser.add_argument('--hide_bbox', default=False, action='store_true', help='Hide boxes in results.')
parser.add_argument('--hide_score', default=False, action='store_true', help='Hide scores in results.')
parser.add_argument('--cutout', default=True, action='store_true', help='Cut out each object and save.')
parser.add_argument('--save_lincomb', default=False, action='store_true', help='Show the generating process of masks.')
parser.add_argument('--no_crop', default=False, action='store_true',
                    help='Do not crop the output masks with the predicted bounding box.')
parser.add_argument('--real_time', default=False, action='store_true', help='Show the detection results real-timely.')
parser.add_argument('--visual_thre', default=0.3, type=float,
                    help='Detections with a score under this threshold will be removed.')

args = parser.parse_args()
args.cfg = re.findall(r'res.+_[a-z]+', args.weight)[0]  # 用权重文件名来命名cfg
cfg = get_config(args, mode='detect')

net = Yolact(cfg)
net.load_weights(cfg.weight, cfg.cuda)
net.eval()
print(f'Model loaded with {cfg.weight}.\n')


def save(ids_p, class_p, box_p, masks_p, img_name):
    """
    Dump to json about the Objects in image
    注意，这里的mask是在原图片上的mask
    """
    ids_p = ids_p.cpu().numpy()
    class_p = class_p.cpu().numpy()
    box_p = box_p.cpu().numpy()
    masks_p = masks_p.cpu().numpy()
    masks_p = masks_p.astype(np.uint8)

    for i, COCOId in enumerate(ids_p):
        # print(i, COCOclassId)
        data = {"id": COCO_CLASSES[COCOId],  # 类名
                "score": str(class_p[i]),
                "bbox": box_p[i].tolist(),
                "mask": [row.tolist() for row in masks_p[i]]}
        f = open(f'results/json/{img_name}_{i}.json', 'w')
        json.dump(data, f)
        print(f'{f.name} created.')
        f.close()


def saveBackground(ids_p, class_p, box_p, img_name):
    """
    Dump to json about the background object boxes
    """
    ids_p = ids_p.cpu().numpy()
    class_p = class_p.cpu().numpy()
    box_p = box_p.cpu().numpy()

    f = open(f'results/json/{img_name}_background.json', 'a')

    for i, COCOID in enumerate(ids_p):
        data = {"id": COCO_CLASSES[COCOID],
                "score": str(class_p[i]),
                "bbox": box_p[i].tolist()}
        json.dump(data, f)

    print(f'{f.name} created.')
    f.close()


with torch.no_grad():
    # detect the image
    if cfg.image is not None:
        # 待识别图片
        dataset = COCODetection(cfg, mode='detect')  # Map-Style dataset
        data_loader = data.DataLoader(dataset, 1, num_workers=0, shuffle=False,
                                      pin_memory=True, collate_fn=detect_collate)

        # img是被正规化的550 * 550图片，img_origin是从cv2中读取的BGR图片
        for i, (img, img_origin, img_name) in enumerate(data_loader):
            img_name = img_name.split('.')[0]
            print("the {} image : {}".format(i, img_name))
            print("img size:", img.shape)
            img_h, img_w = img_origin.shape[0:2]

            class_p, box_p, coef_p, proto_p, anchors = net(img)
            ids_p, class_p, box_p, coef_p, proto_p = nms(class_p, box_p, coef_p, proto_p, anchors, cfg)
            ids_p, class_p, boxes_p, masks_p = after_nms(ids_p, class_p, box_p, coef_p,
                                                         proto_p, img_h, img_w, cfg, img_name=img_name)
            # 种类id, 置信度，bbox[n, 4]，mask[n, img_h, img_w]
            print(ids_p.shape, class_p.shape, boxes_p.shape, masks_p.shape)

            if args.background:
                saveBackground(ids_p, class_p, boxes_p, img_name)
            else:
                save(ids_p, class_p, boxes_p, masks_p, img_name)

            # output the image
            img_numpy = draw_img(ids_p, class_p, boxes_p, masks_p, img_origin, cfg, img_name=img_name)
            cv2.imwrite(f'results/images/{img_name}.jpg', img_numpy)
