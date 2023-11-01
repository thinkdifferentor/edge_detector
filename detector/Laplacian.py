import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os


# img_path = r'D:\Documents\Postgraduate\Project\edge_detector\edge\Edge_Prostate\case2\RUNMC_2.png'

# # root = os.path.dirname(img_path)
# name = img_path.split('\\')[-1].split('.')[0]

# img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
# # print(img.shape, img.dtype)
# img = cv.GaussianBlur(img,(3,3),0)
# edges = cv.Laplacian(img, -1)
# edges =  cv.convertScaleAbs(edges)
# print(edges.shape, edges.dtype)
# # np.savetxt(r'D:\Documents\Postgraduate\Project\edge_detector\images\Nature\{}_edge.txt'.format(name), edges, fmt='%d',newline='\n')

# # edges[edges != 0] = 1 # the edge pixel value is 255
# # np.savetxt(r'D:\Documents\Postgraduate\Project\visualization\CannyEdgeDetector\BraTS\{}_edge.txt'.format(name), edges, fmt='%d',newline='\n')
# # np.save(r'D:\Documents\Postgraduate\Project\edge_detector\images\BraTS\{}_edge.npy'.format(name), edges)
# # plt.imsave(r'D:\Documents\Postgraduate\Project\edge_detector\images\BraTS\{}_edge.png'.format(name), edges, cmap='gray')

# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])

# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

# plt.show()

def processOneImage(img_path, target_dir):
    name = img_path.split('\\')[-1].split('.')[0]

    img = cv.imread(img_path, 0)
    # print(img.shape, img.dtype)
    img = cv.GaussianBlur(img,(3,3),0)
    edges = cv.Laplacian(img, -1)
    edges =  cv.convertScaleAbs(edges)

    # np.savetxt('{}/{}_edge.txt'.format(root, name), edges, fmt='%d',newline='\n')
    # np.save('{}/{}_edge.npy'.format(target_dir, name), edges)
    plt.imsave('{}/{}_la.png'.format(target_dir, name), edges, cmap='gray')

if __name__ == '__main__':
    img_path = r'D:\Documents\Postgraduate\Project\edge_detector\edge\Edge_MMWHS\case2\MRI2.png'
    tgt_dir = r'D:\Documents\Postgraduate\Project\edge_detector\edge\Edge_MMWHS\case2'
    processOneImage(img_path, tgt_dir)
