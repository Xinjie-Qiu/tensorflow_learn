import tensorflow as tf
import numpy as np

def my_image_filiter(image):
    with tf.variable_scope('conv'):
        conv1 = tf.layers.conv2d(image, 32, 3, 2, 'same', activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, 32, 3, 2, 'same', activation=tf.nn.relu)
    return conv2


with tf.variable_scope("image_filters") as scope:
    image = tf.placeholder(tf.float32, [1, 4, 4, 1])
    result1 = my_image_filiter(image)
    scope.reuse_variables()
    result2 = my_image_filiter(image)

n_image = np.reshape(range(16), [1, 4, 4, 1])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
n_result1 = sess.run(result1, {image: n_image})
n_result2 = sess.run(result2, {image: n_image})
print(n_result1)
print(n_result2)