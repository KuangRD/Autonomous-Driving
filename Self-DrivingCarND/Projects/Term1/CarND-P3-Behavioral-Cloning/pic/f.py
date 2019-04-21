import cv2
import numpy as np

center_image = cv2.imread('flip.jpg')
img = np.fliplr(center_image)

cv2.imwrite('ff.jpg',img)
