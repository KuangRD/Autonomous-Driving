# Traffic Sign Recognition

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
*  Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./pic/hist.png "Visualization"
[image2]: ./pic/bitbug_favicon.jpg "Traffic Sign 1"
[image3]: ./pic/bitbug_favicon-2.jpg "Traffic Sign 2"
[image4]: ./pic/bitbug_favicon-3.jpg "Traffic Sign 3"
[image5]: ./pic/bitbug_favicon-4.jpg "Traffic Sign 4"
[image6]: ./pic/bitbug_favicon-5.jpg "Traffic Sign 5"
[image7]: ./pic/bitbug_favicon-6.jpg "Traffic Sign 6"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README
 #### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/KuangRD/Autonomous-Driving/blob/master/Self-DrivingCarND/Project2/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Data Set Summary
I made a data set brief summary as follow:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

**From my point of view, the training set is big enough, so we can build a relatively completive CNN to get a higher accuracy.**

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribu

![alt text][image1]

** It would be lucky to have a model that the amount of data in different classes are totally balance. **

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

As a first step, I didn't convert the images to grayscale because I think color is an important feature. color contains basic semantic information convied by traffic designer, many specific color was used to get noticed in the traffic signs.


I normalized the image data because the the range learning rate and gradient of activation function is relatively narrow to different kinds data.  

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

After some test, I found the dataset,**train.p**, **valid.p**, **test.p** , is big enough to use them directly with batch size 100. So I didn't doing data augment

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x10 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x64 				|
| Convolution 5x5	    |  1x1 stride, valid padding, outputs 10x10x20 	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x20 				|
| Convolution 2x2	    |  1x1 stride, valid padding, outputs 4x4x40 	|
| Fully connected		|Input 640,output 120        							|
| Fully connected		|Input 120,output 84        							|
| Fully connected		|Input 84,output 43        							|
| Softmax				| 43       									|




#### 4.The batch size, number of epochs and any other hyperparameters.

The code for training the model is located in the eigth cell of the ipython notebook.

batch size 100
epochs 30
As the whole processing learning rate didn't cause to oscillated since it was been set in LeNet, so I keep it at 0.01.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.938
* test set accuracy of 0.918

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I use the LeNet at first, as it is easy to make it work. But after a few epochs, the accuracy just hang around at 0.89. As there is still huge mount of data to use, so I choose to add depth and conv layer.    


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image2] 
I think the series signs of speed limit is quite simular with each other especially in low resolution situation.

![alt text][image3] 
As it is one of the speed limit sign, what's more ,one part on the upside of sign have been cut off.

![alt text][image4]
Part of te sigh has been cut off.  

![alt text][image5]
There is a mark on the sign, and it just mask part of the sign.

![alt text][image6] 
There is a mark on the sigh whlie it is a little bit complicated.

![alt text][image7]
It's one of the speedlimit sign.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed Limit(30km/h)     		| Speed Limit(30km/h)     								|
| Speed Limit(50km/h)      			  | No entry  								|
| No entry					  | No entry 									|
| Roundabout	    		|  Roundabout		 				|
| Children crossing		| Children crossing	  	|
| Speed Limit(50km/h) | Speed Limit(50km/h)   |


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%. While the accuracy in the original test set is 91.8%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.         			| Speed Limit(30km/h)   								|
| 0.     				| Speed limit (20km/h)										|
| 0.					| Speed limit (50km/h)											|
| 0.	      			| Speed limit (60km/h)				 				|
| 0.				    | Speed limit (70km/h)    							|


For the second image 

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.         			| No entry   								|
| 0.     				| Speed limit (20km/h)										|
| 0.					| Speed limit (30km/h)											|
| 0.	      			| Speed limit (50km/h)				 				|
| 0.				    | Speed limit (60km/h)    							|

For the thrid image 

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.         			| No entry   								|
| 0.     				| Speed limit (20km/h)										|
| 0.					| Speed limit (30km/h)											|
| 0.	      			| Speed limit (50km/h)				 				|
| 0.				    | Speed limit (26km/h)    							|

For the fourth image 

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.         			| Roundabout	   								|
| 0.     				| Speed limit (20km/h)										|
| 0.					| Speed limit (30km/h)											|
| 0.	      			| Speed limit (50km/h)				 				|
| 0.				    | Speed limit (60km/h)    							|

For the fifth image

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.         			| Children crossing   								|
| 0.     				| Speed limit (20km/h)										|
| 0.					| Speed limit (30km/h)											|
| 0.	      			| Speed limit (50km/h)				 				|
| 0.				    | Speed limit (70km/h)    							|

For the sixth image 

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.         			| Speed Limit(50km/h)     								|
| 0.     				| Speed limit (20km/h)										|
| 0.					| Speed limit (30km/h)											|
| 0.	      			| Speed limit (60km/h)				 				|
| 0.				    | Speed limit (70km/h)    							|


** I think it is seems to be overfit, I am tring to change the Architecture and use more pre-train procession **
