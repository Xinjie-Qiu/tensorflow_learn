import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist', one_hot=True)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y_ = tf.placeholder(tf.float32, [None, 10])

# 28,28,1 -> 14,14,32
c1 = tf.layers.conv2d(x, 32, 3, 1, 'same', activation=tf.nn.relu)
c1_pool = tf.layers.max_pooling2d(c1, 2, 2, 'same')
# 14,14,32 -> 7,7,64
c2 = tf.layers.conv2d(c1_pool, 64, 3, 1, 'same', activation=tf.nn.relu)
c2_pool = tf.layers.max_pooling2d(c2, 2, 2, 'same')
# 7,7,64 -> 7*7*64 -> 1024 -> 10
c2_flat = tf.reshape(c2_pool, [-1, 7 * 7 * 64])
c3 = tf.layers.dense(c2_flat, 1024)
c4 = tf.layers.dense(c3, 10)
# 10 -> 1024 -> 7*7*64 -> 7,7,64
t1 = tf.layers.dense(c4, 1024)
t2 = tf.layers.dense(t1, 7 * 7 * 64)
t3 = tf.reshape(t2, [-1, 7, 7, 64])
# 7,7,64 -> 14,14,32
t4 = tf.image.resize_images(t3, [14, 14])
t5 = tf.layers.conv2d_transpose(t4, 32, 3, 1, 'same', activation=tf.nn.sigmoid)
# t5 = tf.layers.conv2d(t4, 32, 3, 1, 'same', activation=tf.nn.sigmoid)
# 14,14,32 -> 28,28,1
t6 = tf.image.resize_images(t5, [28, 28])
t7 = tf.layers.conv2d_transpose(t6, 1, 3, 1, 'same', activation=tf.nn.sigmoid)
# t7 = tf.layers.conv2d(t6, 1, 3, 1, 'same', activation=tf.nn.sigmoid)

g_loss = tf.losses.mean_squared_error(labels=x, predictions=t7)

g_train = tf.train.AdamOptimizer(0.001).minimize(g_loss)
 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
plt.ion()

for step in range(5000):
    x_in, y_in = mnist.test.next_batch(500)
    x_in = x_in.reshape([500, 28, 28, 1])
    result, D_l = sess.run([t7, g_loss, g_train], {x: x_in})[:2]
    if step % 100 == 0:
        print('step = {}, loss = {}'.format(step, D_l))
        plt.imshow(np.reshape(result[0], (28, 28)), cmap='gray')
        plt.draw()
        plt.pause(0.01)
