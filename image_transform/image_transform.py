# -*- coding: utf-8 -*-
import cv2
import numpy as np

if __name__ == '__main__':
    #读入图片
    img = cv2.imread("./u2dark.png")

    #转换成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray)
