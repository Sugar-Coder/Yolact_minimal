#!/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'Jinyang Shao'

from PIL import Image

bgImage = Image.open('background.png')
eleImage = Image.open('elephant.png')
# bgImage.show()
bgSize = bgImage.size()

newImage = Image.new('RGB', )