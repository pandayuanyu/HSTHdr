from structure_tensor import Structure_Tensor, image_gradient
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
file_path=r'F:\\Dual\\DSC07472.tif'





image=cv.imread(file_path)
img=cv.cvtColor(image, cv.COLOR_BGR2GRAY)
plt.imshow(img,cmap='gray')
A_elems = structure_tensor(img,sigma=1.5,order='rc')
eigen = structure_tensor_eigenvalues(A_elems)
K=eigen.prod(axis=0)
H=eigen.sum(axis=0)
sd=H<0.01
sdf=sd.astype(float)
plt.figure('structure tensor')
plt.imshow(sdf)
plt.axis('off')
plt.colorbar()
plt.show()