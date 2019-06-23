# -*- coding: utf-8 -*-
import cv2
import numpy as np
from numpy import uint8
from sklearn.cluster import KMeans
import pandas as pd

if __name__ == "__main__":
    #读入图片
    cat = cv2.imread("./black_kitten.jpg")
    beach = cv2.imread("./beach.jpg")

    #添加位置信息
    cols = cat.shape[0] #列数
    rows = cat.shape[1] #行数
    print("cat shape:", cat.shape)

    data = np.zeros([cols,rows,5])

    newcols1 = np.uint8(np.zeros([rows, 1]))
    newcols2 = np.mat(np.arange(rows)).reshape(-1, 1)
    for x in range(0, cols):
        data[x] = np.c_[newcols1+x, newcols2, cat[x]]

    # 将二维像素转换为一维
    data = np.float32(data.reshape(-1, 5))
    print("data shape:", data.shape)

    # sklearn kmeans聚类
    # data = cv2.normalize(data, None, 0, 1, cv2.NORM_MINMAX)
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data)
    print("kmeans result shape:", kmeans.labels_.shape)

    # 获取聚类中心的数据
    resSeries = pd.Series(kmeans.labels_)
    cate0 = resSeries[resSeries.values == 0] #获取类别0的数据
    cate1 = resSeries[resSeries.values == 1] #获取类别1的数据

    #数据量少的类别是黑猫
    if (cate0.index.__len__() < cate1.index.__len__()):
        blackcat = cate0.index
    else:
        blackcat = cate1.index

    #定义起始点
    u = np.uint16(beach.shape[1]/4.0)
    v = np.uint16(beach.shape[0]/4.0)
    print("start point:", u, v)

    #替换像素
    for i in range(blackcat.__len__()):
        u_offset = np.int16(data[blackcat[i]][0])
        v_offset = np.int16(data[blackcat[i]][1])
        pix_value = np.uint8([data[blackcat[i]][2], data[blackcat[i]][3], data[blackcat[i]][4]])
        beach[u+u_offset, v+v_offset] = pix_value

    #将分类结果恢复成图片
    dst = kmeans.labels_.reshape([cat.shape[0], cat.shape[1]])
    dst = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX)
    dst = np.uint8(dst)

    # 效果显示
    cv2.imshow("color", cat)
    cv2.imshow("kmeans", dst)
    cv2.imshow("beach_cat", beach)
    cv2.waitKey(0)
    cv2.imwrite("./kmeans_result.jpg", dst)
    cv2.imwrite("./beach_cat.jpg", beach)


