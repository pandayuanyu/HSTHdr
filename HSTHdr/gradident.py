# import cv2
# import numpy as np
#
# img = cv2.imread('F:\\SwinTransHDR\\pic\\3.tif',0)
# img1 = np.power(img/float(np.max(img)), 1/2.2)
# #img2 = np.power(img/float(np.max(img)), 2.2)
# #cv2.imshow('src',img)
# cv2.imshow('gamma=1/2.2',img1)
# #cv2.imshow('gamma=2.2',img2)
# #cv2.waitKey(0)



import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

img = cv2.imread("F:\\SwinTransHDR\\pic\\mertens.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)
# rocg = Recognise.Recognise()
img1 = np.power(gray / 255.0, 1/3)
# image = rocg.gamma_rectify(gray, 0.4)
#cv2.imshow("gray", gray)
cv2.imshow("image", img1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()




def Laplace_suanzi(img):
    r, c = img.shape
    new_image = np.zeros((r, c))
    #L_sunnzi = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    L_sunnzi = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    for i in range(r-2):
        for j in range(c-2):
            new_image[i+1, j+1] = abs(np.sum(img[i:i+3, j:j+3] * L_sunnzi))
    return new_image


#img = cv2.imread('F:\\SwinTransHDR\\pic\\3.tif', cv2.IMREAD_GRAYSCALE)
#cv2.imshow('image', img)
# Laplace算子
out_laplace = Laplace_suanzi(img1)
cv2.imshow('out_laplace_image', out_laplace)
save = "F:\\SwinTransHDR\\1_gradident.png"
cv2.imwrite(save, out_laplace)
cv2.waitKey(0)
cv2.destroyAllWindows()


