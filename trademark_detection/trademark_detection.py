#-*- coding:utf-8 -*-
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

#============================================滑动窗口(64*64)===========================================
def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield(x, y, image[y:y+windowSize[1], x:x+windowSize[0]])

# ===============================================图像分块===============================================
def divideImg(img, m, n):  # 分割成m行n列
    m=m+1
    n=n+1
    h, w = img.shape[0], img.shape[1]
    grid_h = int(h * 1.0 / (m - 1) + 0.5)  # 每个网格的高
    grid_w = int(w * 1.0 / (n - 1) + 0.5)  # 每个网格的宽

    # 满足整除关系时的高、宽
    h = grid_h * (m - 1)
    w = grid_w * (n - 1)

    # 图像缩放
    img_re = cv2.resize(img, (w, h),
                        cv2.INTER_LINEAR)  # 也可以用img_re=skimage.transform.resize(img, (h,w)).astype(np.uint8)
    # plt.imshow(img_re)
    gx, gy = np.meshgrid(np.linspace(0, w, n), np.linspace(0, h, m))
    gx = gx.astype(np.int)
    gy = gy.astype(np.int)

    divide_image = np.zeros([m - 1, n - 1, grid_h, grid_w],
                            np.uint8)  # 这是一个五维的张量，前面两维表示分块后图像的位置（第m行，第n列），后面三维表示每个分块后的图像信息

    for i in range(m - 1):
        for j in range(n - 1):
            divide_image[i, j] = img_re[gy[i][j]:gy[i + 1][j + 1], gx[i][j]:gx[i + 1][j + 1]]
    return divide_image

# =============================================按金字塔分割图片===========================================
def divideImg2Cells(img, mn):
    cells = divideImg(img, mn, mn)
    return cells

# ===========================================计算窗口图片相似度总数=========================================
def spmCalc(template, img, level=4, bin=[10]):
    L_l = np.zeros(level, np.float32)
    w = np.zeros(level, np.float32)
    for l in range(level):
        # 分割图片成2^l个cells
        cells1 = divideImg2Cells(template, pow(2, l))
        cells2 = divideImg2Cells(img, pow(2, l))
        hist_sum = np.zeros([pow(2, l),pow(2, l)], np.uint32)
        for x in range(pow(2, l)):
            for y in range(pow(2, l)):
                # 计算每个cell的直方图
                hist1 = cv2.calcHist(cells1[x][y], [0], None, bin, [0, 256])
                hist2 = cv2.calcHist(cells2[x][y], [0], None, bin, [0, 256])

                # 得到尺度l下单个cell的match总数
                hist = np.zeros(bin[0], np.uint16)
                # 统计一个bin中match的总数
                for i in range(bin[0]):
                    hist[i] = min(hist1[i], hist2[i])
                # 统计l尺度下单个cell的match总数
                hist_sum[x][y] = sum(hist)
        # 统计尺度l下的match总数
        L_l[l] = sum(hist_sum.reshape([pow(2, l)*pow(2, l), 1]))
        w[l] = 1/pow(2, (level-1)-l)


    # 计算匹配程度
    a = list(L_l)
    a.pop()
    a = np.array(a)

    b = list(L_l)
    b.pop(0)
    b = np.array(b)

    c = list(w)
    c.pop()
    w = np.array(c)

    k_L = L_l[level-1] + sum((a - b) * w.transpose())

    return k_L

if __name__ == "__main__":
    #读入图像
    template = cv2.imread("./template1.jpg", 0)
    starbucks_color = cv2.imread("./starbucks3.jpg")
    starbucks = cv2.cvtColor(starbucks_color, cv2.COLOR_BGR2GRAY)

    print(starbucks.shape)

    sw = sliding_window(starbucks, 12, np.array([96, 96]))
    match_rate = np.zeros(10000, np.float)
    img_cord = np.zeros([10000,2], np.uint16)
    x = 0
    print("start detecting.....")
    for windows in sw:
        img_cord[x] = [windows[0], windows[1]]
        match_rate[x] = spmCalc(template, windows[2], 4, [100])
        x += 1

    list_rate = match_rate.tolist()
    index = list_rate.index(max(list_rate))

    # 显示效果
    result = cv2.rectangle(starbucks_color,
                           (img_cord[index][0], img_cord[index][1]),
                           (img_cord[index][0] + 96, img_cord[index][1] + 96),
                           (255, 0, 0), 2)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.imwrite("starbucks3_detect.jpg", result)

