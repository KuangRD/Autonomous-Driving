# TensorFlow Cheat Sheet

 Author: [Kuang](https://github.com/KuangRD)

 Description: This cheat sheet acts as a intro to TensorFlow and realize some
 simple samples during my **Udacity Self-Driving Car** classes(TensorFlow 0.12.1) and reading the
 book **TensorFlow实战**.

 ***Found any typos or have a suggestion? Fork, contribute and tune it to your taste!***

## Index
Use the following import convention:
```sh
import tensorflow as tf
```
## Basic
### Tensor
#### constant
```sh
# A is a 0-dimensional int32 tensor
A = tf.constant(1234)
# B is a 1-dimensional int32 tensor
B = tf.constant([123,456,789])
# C is a 2-dimensional int32 tensor
C = tf.constant([ [123,456,789], [222,333,444] ])
# Converting types
D = tf.cast(A,tf.float32)
```
#### Placeholder
```sh
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)
```
#### Variable
>The goal of training a neural network is to modify weights and biases to best predict the labels. In order to use weights and bias, you'll need a Tensor that can be modified. This leaves out **tf.placeholder()** and **tf.constant()**, since those Tensors can't be modified. This is where tf.Variable class comes in.

Tensors in *tf.placeholder()* and *tf.constant()* can't be modified. This is where tf.Variable class comes in.

```sh
x = tf.Variable(5)
```
Variable Initialization
```sh
#Generate random numbers from a normal distribution.
n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
bias = tf.Variable(tf.zeros(n_labels))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
```

## Session
>A "TensorFlow Session" is an environment for running a graph. The session is in charge of allocating the operations to GPU(s) and/or CPU(s), including remote machines.

```sh
with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Test String', y: 123, z: 45.67})
```
## Math
```sh
A = tf.add(5, 2)  # 7
B = tf.subtract(10, 4) # 6
C = tf.multiply(2, 5)  # 10
D = tf.div(10,2) # 5
Y = tf.add(tf.matmul(input,w),b) # matrix multiply
output = tf.nn.softmax(logits)
x = tf.reduce_sum([1,2,3,4,5]) # 15 ,just sum
x = tf.log(100) # 4.60517 Natural Log
cross_entropy = -tf.reduce_mean(tf.mul(tf.log(softmax),one_hot))
```
## Mini-batching
```sh
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])
```
**None** here is the batch size. As the batch size can be variable and different(7 samples divide to 3 batches).

## Epochs
An epoch is a single forward and backward pass of the whole dataset.
```sh
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch_i in range(epochs):

        # Loop over all batches
        for batch_features, batch_labels in train_batches:
            train_feed_dict = {
                features: batch_features,
                labels: batch_labels,
                learning_rate: learn_rate}
            sess.run(optimizer, feed_dict=train_feed_dict)

        # Print cost and validation accuracy of an epoch
        print_epoch_stats(epoch_i, sess, batch_features, batch_labels)

    # Calculate accuracy for test dataset
    test_accuracy = sess.run(
        accuracy,
        feed_dict={features: test_features, labels: test_labels})

```
