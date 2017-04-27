# Camera Distortion

小孔成像不存在畸变,镜头才存在畸变

1. **Radial Distortion（径向畸变）** 由于相机镜头组中心与边缘的成像效果不均衡造成的失真.
![alt test][image1]
2. **Tangential Distortion（切向畸变）** 相机架设的姿态与被观测物体没有绝对垂直造成的失真.
![alt test][image2]

## Distortion Coefficients and Correction

![alt test][image3]

图中（x,y）是畸变图像中的一个点，（xcorrect, ycorrect）是（x,y）在正常图像上对应的点。畸变图像的中心通常也是正常图像的中心（xc,yc)。

### 径向畸变, K1, K2, K3

![alt test][image4]

>*K3* 反应了镜头径向畸变的主要部分，但是大部分常规镜头的*K3*已经做的很小，可以忽略不计。OpenCV中也可以选择忽略该系数。

### 切向畸变, P1, P2
![alt test][image5]


## 校准
OpenCV指令 [cv2.findChessboardCorners()](http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#cv2.findChessboardCorners)和 [cv2.drawChessboardCorners()](http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#cv2.drawChessboardCorners)

![alt test][image6]

```sh
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# prepare object points
nx = 8#TODO: enter the number of inside corners in x
ny = 6#TODO: enter the number of inside corners in yd corners
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

# Make a list of calibration images
fname = 'calibration_test.png'
img = cv2.imread(fname)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

# If found, draw corners
if ret == True:
    # Draw and display the corners
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    plt.imshow(img)

```
Convert2Grey
读一个系列的图，如img1, img2, img3....
```sh
import glob
images = glob.glob('../some_folder/img*.jpg')
```
读RGB图，
```sh
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
```
读视频中的图，(BGR)
```sh
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
```
相机校准，给定对象点，图像点和灰度图像的形状//**gray.shape[::-1]**和img.shape[0:2](检索前两位)返回图像尺寸,dist: Distortion Coefficients, mtx: Camera Matrix, rvecs,tvecs: rotation vectors, translation vectors 反应相机在真实世界中的位置
```sh
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
```
Undistorting//dst: undistorted image
```sh
dst = cv2.undistort(img, mtx, dist, None, mtx)
```


[//]: # (Image References)

[image1]: ./pic/radial.png
[image2]: ./pic/tangential.png
[image3]: ./pic/dis_undis.png
[image4]: ./pic/rd.png
[image5]: ./pic/td.png
[image6]: ./pic/corners-found3.jpg
