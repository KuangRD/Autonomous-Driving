# The Four Main Cases When Using Transfer Learning

Transfer learning involves taking a pre-trained neural network and adapting the neural network to a new, different data set.

Depending on both:

* the **size** of the new data set, and
* the **similarity** of the new data set to the original data set

the approach for using transfer learning will be different. There are four main cases:

 1. new data set is small, new data is similar to original training data
 2. new data set is small, new data is different from original training data
 3. new data set is large, new data is similar to original training data
 4. new data set is large, new data is different from original training data

![alt test][image1]
## Demonstration Network   

![alt test][image2]

## Case 1: Small Data Set, Similar Data

![alt test][image3]   

If the new data set is small and similar to the original training data:

* slice off the end of the neural network
* add a new fully connected layer that matches the number of classes in the new data set
* randomize the weights of the new fully connected layer; freeze all the weights from the pre-trained network
* train the network to update the weights of the new fully connected layer

To avoid overfitting on the small data set, the weights of the original network will be held constant rather than re-training the weights.

Since the data sets are similar, images from each data set will have similar higher level features. Therefore most or all of the pre-trained neural network layers already contain relevant information about the new data set and should be kept.

Here's how to visualize this approach:
![alt test][image4]

## Case 2: Small Data Set, Different Data

![alt test][image5]

If the new data set is small and different from the original training data:

* slice off most of the pre-trained layers near the beginning of the network
* add to the remaining pre-trained layers a new fully connected layer that matches the number of classes in the new data set
* randomize the weights of the new fully connected layer; freeze all the weights from the pre-trained network
* train the network to update the weights of the new fully connected layer
Because the data set is small, overfitting is still a concern. To combat overfitting, the weights of the original neural network will be held constant, like in the first case.

But the original training set and the new data set do not share higher level features. In this case, the new network will only use the layers containing lower level features.

Here is how to visualize this approach:
![alt test][image6]

## Case 3: Large Data Set, Similar Data
![alt test][image7]
If the new data set is large and similar to the original training data:

* remove the last fully connected layer and replace with a layer matching the number of classes in the new data set
* randomly initialize the weights in the new fully connected layer
* initialize the rest of the weights using the pre-trained weights
* re-train the entire neural network
Overfitting is not as much of a concern when training on a large data set; therefore, you can re-train all of the weights.

Because the original training set and the new data set share higher level features, the entire neural network is used as well.

Here is how to visualize this approach:
![alt test][image8]

## Case 4: Large Data Set, Different Data
![alt test][image9]
If the new data set is large and different from the original training data:

* remove the last fully connected layer and replace with a layer matching the number of classes in the new data set
* retrain the network from scratch with randomly initialized weights
* alternatively, you could just use the same strategy as the "large and similar" data case

Even though the data set is different from the training data, initializing the weights from the pre-trained network might make training faster. So this case is exactly the same as the case with a large, similar data set.

If using the pre-trained network as a starting point does not produce a successful model, another option is to randomly initialize the convolutional neural network weights and train the network from scratch.

Here is how to visualize this approach:
![alt test][image10]

## Reference
1, [Udacity Self-Driving Car Nano Degree](https://www.udacity.com/drive)

2, [Transfer Learning - Machine Learning's Next Frontier](http://sebastianruder.com/transfer-learning/index.html)



[//]: # (Image References)

[image1]: ./pic/02-guide-how-transfer-learning-v3-01.png "Visualization"
[image2]: ./pic/02-guide-how-transfer-learning-v3-02.png "Traffic Sign 1"
[image3]: ./pic/02-guide-how-transfer-learning-v3-03.png "Traffic Sign 2"
[image4]: ./pic/02-guide-how-transfer-learning-v3-04.png "Traffic Sign 3"
[image5]: ./pic/02-guide-how-transfer-learning-v3-05.png "Traffic Sign 4"
[image6]: ./pic/02-guide-how-transfer-learning-v3-06.png "Traffic Sign 5"
[image7]: ./pic/02-guide-how-transfer-learning-v3-07.png "Traffic Sign 6"
[image8]: ./pic/02-guide-how-transfer-learning-v3-08.png "Traffic Sign 6"
[image9]: ./pic/02-guide-how-transfer-learning-v3-09.png "Traffic Sign 6"
[image10]: ./pic/02-guide-how-transfer-learning-v3-10.png "Traffic Sign 6"
