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

### Perspective Transform
**Perspective** 透视


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
#### Tips:
***Note***: If you read in an image using *matplotlib.image.imread()* you will get an **RGB** image, but if you read it in using OpenCV *cv2.imread()* this will give you a **BGR** image.
### 相机校准

相机校准，给定对象点，图像点和灰度图像的形状//**gray.shape[::-1]**和img.shape[0:2](检索前两位)返回图像尺寸,dist: Distortion Coefficients, mtx: Camera Matrix, rvecs,tvecs: rotation vectors, translation vectors 反应相机在真实世界中的位置
```sh
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
```
Undistorting//dst: undistorted image
```sh
dst = cv2.undistort(img, mtx, dist, None, mtx)
```
相机校准前后效果

![alt test][image7]

### Perspective Transform

Compute the perspective transform, M, given source and destination points:
```sh
M = cv2.getPerspectiveTransform(src, dst)
```
Compute the inverse perspective transform:
```sh
Minv = cv2.getPerspectiveTransform(dst, src)
```
Warp an image using the perspective transform, M:
```sh
warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
```
透视变换前后效果
![alt test][image8]
### 校准点的初始化
```#!/bin/sh
objpoints = []
imgpoints = []

objp = np.zeros((6*9,3),np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

```
可以通过glob函数往object points和image points中加多个图的corner点,但值得注意的是一些图中Corners显示不全（部分遮盖）.会导致后面出bug，可以通过ret是True还是False来确认．

###  梯度检测(边缘加测)
**Sobel Operator** Canny 边缘检测算法的核心部分. *Sx*和*Sy*分别用于X方向和Y方向的梯度检测.维度必须为奇数,最小为3.维度越大,求梯度的面积越大,梯度越平滑
![alt test][image9]

(1, 0) 指X方向,(0,1)指Y方向
```sh
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
```

计算导数的绝对值
```sh
abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
```
Convert the absolute value image to 8-bit:
不一定要转换为8位（范围从0到255）,主要是用于规范不同图像像素值范围不同的情况.
```sh
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
```
过滤梯度
```sh
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
```
过滤梯度的三个方面
```sh
# 1
if orient == 'x':
  sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
if orient == 'y':
  sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
abs_sobel =  np.absolute(sobel)
# 2
gradmag = np.sqrt(sobelx**2 + sobely**2)# Magnitude of the Gradient
# 3
dirt = np.arctan2(abs_sobely, abs_sobelx)# Direction of the Gradient

# Apply each of thresholding functions
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(0, 255))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(0, 255))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(0, 255))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2))

# Combine threshold
combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
```
## 色彩空间
**RGB**,三原色
**HSV**,hue 色调, saturation 饱和度, value
**HLS**,hue, lightness 亮度, saturation 饱和度.
Hue: value that represents color independent of any change in brightness
Lightness&Value: represent different ways to measure the relative lightness or darkness of a color. For example, a dark red will have a similar hue but much lower value for lightness than a light red.

采用不同色彩空间,主要是因为表达色彩的方式不同,直观地看就是求梯度时,差异很大.举个例子,黄色车道线处,相对泊油路处R值和G值梯度很大,但是B值梯度不明显;阴影和强光处RGB差值都很大但是HLS中,L变换就不明显.

![alt test][image10]

![alt test][image11]


```sh
hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
H = hls[:,:,0]
L = hls[:,:,1]
S = hls[:,:,2]

# S值过滤
thresh = (90, 255)
binary = np.zeros_like(S)
binary[(S > thresh[0]) & (S <= thresh[1])] = 1
```
## 色彩过滤和梯度过滤的结合
```sh
# Threshold x gradient
thresh_min = 20
thresh_max = 100
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

# Threshold color channel
s_thresh_min = 170
s_thresh_max = 255
s_binary = np.zeros_like(s_channel)
s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

# Stack each channel to view their individual contributions in green and blue respectively
# This returns a stack of the two binary images, whose components you can see as different colors
color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))

# Combine the two binary thresholds
combined_binary = np.zeros_like(sxbinary)
combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
```

Project Steps

Steps we’ve covered so far:

Camera calibration
Distortion correction
Color/gradient threshold
Perspective transform
After doing these steps, you’ll be given two additional steps for the project:

Detect lane lines
Determine the lane curvature


[//]: # (Image References)

[image1]: ./pic/radial.png
[image2]: ./pic/tangential.png
[image3]: ./pic/dis_undis.png
[image4]: ./pic/rd.png
[image5]: ./pic/td.png
[image6]: ./pic/corners-found3.jpg
[image7]: ./pic/orig-and-undist.png
[image8]: ./pic/perspective_trans.png
[image9]: ./pic/soble-operator.png
[image10]:./pic/RGB.png
[image11]:./pic/HSV&HLS.png
