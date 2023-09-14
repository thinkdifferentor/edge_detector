# -*- coding: UTF-8 -*-

# 对canny阈值做了一个自适应的过程

import numpy as np
import cv2
import gradient as g
import loop as l
import copy

# 读取一张图片,做灰度化处理,转为二维矩阵。虽然原图是灰度图,但仍需此步骤,不然会是三维矩阵
image = cv2.imread(r'D:\Documents\Postgraduate\Project\edge_detector\images\Prostate\BMC_2.png', 0)

# 使用高斯滤波消除噪声
# (5,5)是高斯核的大小,0是sigmaX,0的意思是让方程自己给你算对应的sigma
image_b = cv2.GaussianBlur(image, (5, 5), 0)

image_t = copy.deepcopy(image_b)

# 返回的是经过非极大值抑制后的矩阵
image_g = g.gradient(image_t)

# 返回的是迭代后的最优阈值
image_T = l.loop(image_g)

# ret是返回的最优阈值,image是处理后的图
# THRESH_BINARY就是三角法（这句话存疑,忘记哪里看来的了,但实际使用起来似乎使用的阈值是传入的阈值）
# THRESH_OTSU是大津法
# 这一个函数会根据直方图选择最优的阈值,即otus或传入的阈值,可以从返回值ret中获取
# image_ret, new1 = cv2.threshold(image_b, image_T, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 这一条就只会使用大津法算出的阈值,但大津有缺点,直方图没有明显双峰时效果可能不好
image_ret, _ = cv2.threshold(image_b, image_T, 255, cv2.THRESH_OTSU)

# 因为想要有更多的边被画出来,所以取更小的（我自己一厢情愿的,可能没有理论依据）
image_T = min(image_ret, image_T)

image = cv2.Canny(image_b, image_T / 2, image_T)
image[image == 255] = 1

# np.save(r'D:\Documents\Postgraduate\Project\edge_detector\images\Prostate\BMC_2_edge.npy', image)
np.savetxt(r'D:\Documents\Postgraduate\Project\edge_detector\images\Prostate\BMC_2_edge.txt', image, fmt='%d',newline='\n')

# 这是自适应阈值的边缘切割,用高斯核做窗口去给每一个像素一个自适应阈值
# 最后一个参数会影响切割质量,具体原理请参考文档
# 倒数第二个会影响切割出来的边的宽度,原理见文档
# image_two = cv2.adaptiveThreshold(image_b, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)

# 将图片做一个反色处理
# image[image == 0] = 1
# image[image == 255] = 0
# image[image == 1] = 255

cv2.imshow('image', image)
# cv2.imshow('image_two', image_two)

# 保存图片
# cv2.imwrite('imageQ1.png', image)

cv2.waitKey(0)

exit(1)

# 后面部分是切割质量的一个定量评估方法
# 注释掉的原因是因我没有读懂文档中计算八连通成分数的部分
# 我结合网上的资料,决定用matlab封装的函数去定量分析结果
# 只需要把opencv割出来的图保存,用matlab读取即可


# def calA(image):
#     cnt = 0.0
#     for i in range(len(image)):
#         for j in range(len(image[0])):
#             if image[i][j] == 255:  # 边的颜色是几就填多少,白边是255
#                 cnt = cnt + 1
#     return cnt


# test_matrix = [[0,0,255,255],[0,255,0,255],[255,255,255,0],[0,0,0,0]]

# # 计算4邻域
# bofCanny = e.evaluate4(canny)
# aofCanny = calA(canny)
# print('canny -> a: ', aofCanny, ' b: ', bofCanny, ' a/b: ', aofCanny / bofCanny)
