#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as py
import cv2 as cv
from matplotlib import pyplot as plt

left_image = cv.imread('example1.png', cv.IMREAD_GRAYSCALE)
right_image = cv.imread('example2.png', cv.IMREAD_GRAYSCALE)

stereo = cv.StereoBM_create(numDisparities=16, blockSize=21)

depth = stereo.compute(left_image, right_image)

cv.imshow("left", left_image)
cv.imshow("right", right_image)

plt.imshow(depth)
plt.axix('off')
plt.show()
