## Vehicle Detection and Tracking

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./pic/readin.png
[image2]: ./pic/hogf.png
[image3]: ./pic/svcT.png
[image4]: ./pic/heat.png
[image5]: ./pic/label.png
[image6]: ./pic/sl.png
[image7]: ./pic/ms.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the Jupyter Notebook(VehicleDetect.ipynb).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HLS` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I use the default parameters in the class.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using various combinations of features. As I use a sever with 4 TITAN X and 256GB RAM, it didn't spend too much time.

At first, I tried **Histogram of Color**, **Spatial of Color** and **Histogram of Oriented Gradient**, one by one.

Then, I tried to combine each two of them. I found **Spatial of Color** can't improve the performance to much, while occupying too many space.

At last, I choose a series parameters as follow:

```sh
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off
```

###Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I use a single scale `2.0` for test and got a sample as follow:

![alt text][image3]

From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are heatmaps:

![alt text][image4]

### Here is the output of `scipy.ndimage.measurements.label()`
![alt text][image5]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image6]

I decided to search window positions at scales `1.0`,`1.5` and `0.8` with cross over ROI, because I think the search window should be continuous, but we use discrete scale window to simplify the calculation, to prevent shocking.

and came up with this:

| ROI | Scaler |Window Size|
|:-----:|:-------:|:------:|
|400~450|0.8|51x51 px|
|400~500|1.0|64x64 px|
|400~600|1.5|96x96 px|


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image7]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections of 3 scales windows in each frame of the video.

And I used a global variable to store the `heatmap` of last frame. In this way I can reduce more False Positive sample with little effluence on True Positive.




---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* The most impressive thing is the procession of choosing the `color space` and `window size`.when I begin to build up the pipeline, the accuracy of SVC with different color space didn't performance quite different(except YUV and LUV didn't finish training due to occupy too many space, and I choose a extremely window size for YCrCb), after I built up the whole pipeline with HLS and HSV, there are too many FN windows, I tried a lot of techniques such as adjust the data set, use the heat map of last frame and multi scale search windows, both of them works little after I found when I choose the color space the default window size 96x96 didn't suit for YCrCb. After I change to YCrCb, every thing works.

* I will submit the version combining with Lane Finding in next submit.

* I will try this project on U-net, YOLO and SSD in next week.
