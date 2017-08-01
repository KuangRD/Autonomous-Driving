# Tracking

Here is an example of Google Self-Driving Car using a road map localizing itself, RADAR and LIDAR to track other vehicles

![alt test][image1]


| Kalman Filter| Particle Filter | Monte Carlo Localization |
|:-------:|:---:|:-------:|
| Continuous | Continuous | Discrete |
| Uni-Modal | Multi-Modal | Multi-Modal |

## Measurement and Motion

The Kalman Filter represents our distributions by guassians and iterates on two main cycles.

The first cycle is the Measurement Update.
* requires a product
* Uses Bayes rule.

The second cycle is the Motion Update.
* involves a convolution
* uses total probability.

## Process Measurement
![alt test][image3]


[//]: # (Image References)

[image1]: ./pic/track.png
[image2]: ./pic/LIDAR_visual.png
[image3]: ./pic/ProcessMeasurement.png
[image4]: ./pic/properties.png
