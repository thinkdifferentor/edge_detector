"""
Written in Python 2.7!
This module takes an image and converts it to grayscale, then applies a
Roberts Cross operator.
"""

__author__ = "Kevin Gay"

from PIL import Image
import math
import numpy as np

class Roberts(object):

    def __init__(self, imPath):

        im = Image.open(imPath).convert('L')
        print(im.size)
        self.width, self.height = im.size
        mat = im.load()

        robertsx = [[1,0],[0,-1]]
        robertsy = [[0,1],[-1,0]]

        self.robertsIm = Image.new('L', (self.width, self.height))
        pixels = self.robertsIm.load()

        linScale = .7

        #For each pixel in the image
        for row in range(self.width-len(robertsx)):
            for col in range(self.height-len(robertsy)):
                Gx = 0
                Gy = 0
                for i in range(len(robertsx)):
                    for j in range(len(robertsy)):
                        val = mat[row+i, col+j] * linScale
                        Gx += robertsx[i][j] * val
                        Gy += robertsy[i][j] * val

                pixels[row+1,col+1] = int(math.sqrt(Gx*Gx + Gy*Gy))

    def saveIm(self, name):
        self.robertsIm.save(name)


def test():
    img_path = r'D:\Documents\Postgraduate\Project\edge_detector\edge\Edge_MMWHS\case2\MRI2.png'
    # img_path = r'D:\Documents\Postgraduate\Project\edge_detector\images\BraTS\BraTS19_2013_7_1_t1ce_z_100.png'
    # img_path = r'D:\Documents\Postgraduate\Project\edge_detector\images\Nature\ci_1.png'
    name = img_path.split('\\')[-1].split('.')[0]
    out_name = name + '_ro.png'
    roberts = Roberts(img_path)

    # print(np.array(roberts.robertsIm).shape)
    # np.savetxt(r'D:\Documents\Postgraduate\Project\edge_detector\images\Nature\{}_edge.txt'.format(name), np.array(roberts.robertsIm), fmt='%d',newline='\n')

    roberts.saveIm(out_name)

if __name__ == '__main__':
    test()