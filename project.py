import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
# import MNIST Dataset, which contains 55,000 test, target, and check images
# import this data in one_hot format which saves images as array with 1s and 0s
# 1s represent black space and 0 represents white space


x = tf.placeholder(tf.float32, [None, 784])
# placeholder to flatten an image to 784 dimensional vector

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# Variable is a tensor that lives in TensorFlow's graph of interacting operations
# above tow commands create a tensor system with 0s in the given array sizes

y = tf.nn.softmax(tf.matmul(x, W) + b)
# multiply x with W add b and apply tf.nn.softmax

y_ = tf.placeholder(tf.float32, [None, 10])
# placeholder to create a flat image of a 10 dimensional vector

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# tf.log computes the logarithm of each element of y.
# multiply each element of y_ with tf.log(y)
# tf.reduce_sum adds elements in the second dimension of y due to reduction indices = [1] parameter
# tf.reduce_mean computes the mean over all the examples in the batch

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
# Tensorflow minimizes cross_entropy using the gradient_descent algorithm with
# a learning rate of 0.5
# this optimization algorithm shifts the image in a direction that reduces its cost

sess = tf.InteractiveSession()
# sess is a wrapper used to launch the model

tf.global_variables_initializer().run()
# initializes the variables created

for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
# run the training step 1000 times, every step of the loop gives a batch of
# 100 points from the set
# running train_step feeds in batches data to replace the placeholders used

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# checks if prediction matches the test image by checking the label

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# checks for % accuracy on test data

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
# code outputs an accurace of ~92%

