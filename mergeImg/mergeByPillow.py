#!/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'Jinyang Shao'

from PIL import Image

bgImage = Image.open('./mergeImg/background.png')
eleImage = Image.open('./mergeImg/elephant.png')
# bgImage.show()
bgSize = bgImage.size

newImage = Image.new('RGB', bgSize)
eleImage = eleImage.resize((50, 30))
newImage.paste(bgImage, (0, 0))
newImage.paste(eleImage, (bgSize[0] - 60, bgSize[1] - 40))
newImage.save("mergeImg/result.jpg", "JPEG")
newImage.show()
