# -*- coding: utf-8 -*-
import cv2
import numpy as np

if __name__ == "__main__":
    #读入图片
    img = cv2.imread("./black_kitten.jpg")

    #将二维像素转换为一维
    data = np.float32(img.reshape(-1, 3))

    #kmeans类聚
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    compactness, lables, centers = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    dst = centers[lables.flatten()].reshape(img.shape)
    dst = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX)


    #效果显示
    cv2.imshow("gray", img)
    cv2.imshow("kmeans", dst)
    cv2.waitKey(0)