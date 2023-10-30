import cv2, torchvision
from ciconv2d import CIConv2d
import torch
from matplotlib import pyplot as plt
import numpy as np

img = r'D:\Documents\Postgraduate\Project\summary_sdg\edge\Edge_BraTS\BraTS19_2013_7_1_t2_z_74.png'
inv = 'W'

# load and preprocess input image
i = torchvision.transforms.functional.to_tensor(cv2.imread(img)[:,:,::-1].copy())[None,:,:,:]
print(i.shape)
# get max value of output
w=CIConv2d(inv, k=3, scale=0.)(i)
r = torch.max(torch.abs(w.detach()[0,0,:,:]))
print(r)
w=CIConv2d(inv, k=3, scale=-2.5)(i)
r = torch.max(torch.abs(w.detach()[0,0,:,:]))
print(r)

plt.figure(figsize=(15,8))
plt.subplot(1,3,1)
plt.imshow(cv2.imread(img)[:,:,::-1])
plt.subplot(1,3,2)
w=CIConv2d(inv, k=3, scale=-0.)(i)
plt.imsave(r'D:\Documents\Postgraduate\Project\summary_sdg\edge\Edge_BraTS\BraTS19_2013_7_1_t2_z_74_edge_ciconv2.png', w[0,0,:].detach().numpy(), cmap='gray')

# print(torch.isnan(w).any())
plt.imshow(w.detach()[0,0,:,:], vmin=-r, vmax=r, cmap='gray')
plt.title('scale=0')
plt.subplot(1,3,3)
w=CIConv2d(inv, k=3, scale=-2.5)(i)
print(w.shape)

# print(torch.isnan(w).any())
plt.imshow(w.detach()[0,0,:,:], vmin=-r, vmax=r, cmap='gray')
plt.title('scale=-2.5')
_=plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.show()

# np.savetxt(r'D:\Documents\Postgraduate\Project\summary_sdg\edge\Edge_BraTS\BraTS19_2013_7_1_t2_z_74_edge.txt', w[0,0,:].detach().numpy(), fmt='%d',newline='\n')
