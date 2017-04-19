# TensorFlow Cheat Sheet

 Author: [Kuang](https://github.com/KuangRD)

 Description: This cheat sheet acts as a intro to TensorFlow and realize some
 simple samples during my **Udacity self-driving car** classes(TensorFlow 0.12.1) and reading the
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
Tensors in *tf.placeholder()* and *tf.constant()*can't be modified. This is where tf.Variable class comes in.

#### Variable
```sh

```

#### Session
```sh
with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Test String', y: 123, z: 45.67})
```
## Math
###
```sh
A = tf.add(5, 2)  # 7
B = tf.subtract(10, 4) # 6
C = tf.multiply(2, 5)  # 10
D = tf.div(10,2) # 5
```
