# -*- coding: utf-8 -*-
import cv2
import numpy as np

if __name__ == '__main__':
    #读入图片
    img = cv2.imread("./u2dark.png")

    #转换成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #获取最大值、最小值、均值并输出
    gray_max = np.max(gray)
    gray_min = np.min(gray)
    gray_ave = np.average(gray)
    print(gray_max)
    print(gray_min)
    print(gray_ave)

    #尺度变换
    gray_norm = np.zeros(gray.shape)
    gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX) #归一化到0~255

    #改变对比度
    gray_sharp = 2 * (gray_norm - 128) + 128

    #效果显示
    print(gray)
    print(gray_norm)
    print(gray_sharp)
    cv2.imshow("gray", gray)
    cv2.imshow("norm", gray_norm)
    cv2.imshow("sharp", gray_sharp)
    cv2.waitKey(0)
