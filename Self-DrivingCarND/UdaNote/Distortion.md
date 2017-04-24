# Camera Distortion

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



[//]: # (Image References)

[image1]: ./pic/radial.png
[image2]: ./pic/tangential.png
[image3]: ./pic/dis_undis.png
[image4]: ./pic/rd.png
[image5]: ./pic/td.png
