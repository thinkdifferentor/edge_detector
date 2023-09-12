"""
Written in Python 2.7!
This module takes an image and converts it to grayscale, then applies a
Prewitt operator.
"""

__author__ = "Kevin Gay"

from PIL import Image
import math
import numpy as np

class Prewitt(object):

    def __init__(self, imPath):

        im = Image.open(imPath).convert('L')
        self.width, self.height = im.size
        mat = im.load()

        prewittx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
        prewitty = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]

        self.prewittIm = Image.new('L', (self.width, self.height))
        pixels = self.prewittIm.load()

        linScale = .3

        #For each pixel in the image
        for row in range(self.width-len(prewittx)):
            for col in range(self.height-len(prewittx)):
                Gx = 0
                Gy = 0
                for i in range(len(prewittx)):
                    for j in range(len(prewitty)):
                        val = mat[row+i, col+j] * linScale
                        Gx += prewittx[i][j] * val
                        Gy += prewitty[i][j] * val

                pixels[row+1,col+1] = int(math.sqrt(Gx*Gx + Gy*Gy))

    def saveIm(self, name):
        self.prewittIm.save(name)

def test():
    img_path = r'D:\Documents\Postgraduate\Project\edge_detector\images\Nature\ci_1.png'
    name = img_path.split('\\')[-1].split('.')[0]
    out_name = name + '_prewitt.jpg'
    prewitt = Prewitt(img_path)
    
    print(np.array(prewitt.prewittIm).shape)
    np.savetxt(r'D:\Documents\Postgraduate\Project\edge_detector\images\Nature\{}_edge.txt'.format(name), np.array(prewitt.prewittIm), fmt='%d',newline='\n')

    prewitt.saveIm(out_name)

if __name__ == '__main__':
    test()