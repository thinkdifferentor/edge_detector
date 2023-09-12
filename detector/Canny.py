import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

####### 参数理解 #######

# 1.提前做高斯模糊可以去除噪点，会使得检测的边缘变少，对于医疗图像这里不使用，一方面相比自然图像噪声较少；另一方面使用的话可能会使得边缘细节信息丢失，不利于医学诊断。

# 2.较大的阈值2用于检测图像中明显的边缘，但会使得边缘检测出来是断断续续的；较小的阈值2用于将这些间断的边缘连接起来。
# 从理论上理解：Canny算法中应用了一种叫双阈值的技术。即设定一个阈值上界maxVal和阈值下界minVal，图像中的像素点如果大于阈值上界则认为必然是边界（称为强边界，strong edge），
# 小于阈值下界则认为必然不是边界，两者之间的则认为是候选项（称为弱边界，weak edge），需进行进一步处理，如果与确定为边缘的像素点邻接，则判定为边缘；否则为非边缘。

# 3.L2gradient指计算梯度强度和方向的算法，默认算法(False)比较简单，会使检测边缘变多；若使用True，则梯度和方向计算更为准确，会使检测边缘变少。

# 4.apertureSize指sobel算子大小，默认为3，扩大算子，会获得更多的细节，使得检测边缘变多。


####### 预期效果 #######

# 目标：通过引入边缘检测分支，迫使网络学习不同模态和机构的Domain-invariant信息，从而提升模型泛化性。 

# 边缘检测效果：
# 1.正常组织边缘 & 病变组织边缘
# 2.在满足第一条基础上，提取的边缘尽可能简单，过于复杂可能不利于训练


img_path = '/data/jiangjun/myprojects/edge_detector/images/Nature/ci_1.png'

# root = os.path.dirname(img_path)
# name = img_path.split('\\')[-1].split('.')[0]

img = cv.imread(img_path, 0)
# print(img.shape, img.dtype)
# img = cv.GaussianBlur(img,(3,3),0)
edges = cv.Canny(img, threshold1=40, threshold2=100, L2gradient=True, apertureSize=3)

# edges[edges != 0] = 1 # the edge pixel value is 255
# np.savetxt(r'D:\Documents\Postgraduate\Project\visualization\CannyEdgeDetector\BraTS\{}_edge.txt'.format(name), edges, fmt='%d',newline='\n')
# np.save(r'D:\Documents\Postgraduate\Project\visualization\CannyEdgeDetector\BraTS\{}_edge.npy'.format(name), edges)
# plt.imsave(r'D:\Documents\Postgraduate\Project\visualization\CannyEdgeDetector\BraTS\{}_edge.png'.format(name), edges, cmap='gray')

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
