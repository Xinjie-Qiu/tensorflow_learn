import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist', one_hot=True)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])

c1 = tf.layers.conv2d(x, 32, 3, 1, 'same', activation=tf.nn.relu)
c1_pool = tf.layers.max_pooling2d(c1, 2, 2, 'same')
c2 = tf.layers.conv2d(c1_pool, 64, 3, 1, 'same', activation=tf.nn.relu)
c2_pool = tf.layers.max_pooling2d(c2, 2, 2, 'same')

t1 = tf.layers.conv2d_transpose(c2_pool, 64, 2, 2, 'same')
t2 = tf.layers.conv2d(t1, 64, 3, 1, 'same', activation=tf.nn.relu)
t3 = tf.layers.conv2d_transpose(t2, 32, 2, 2, 'same')
t4 = tf.layers.conv2d(t3, 1, 3, 1, 'same', activation=tf.nn.relu)

loss = tf.losses.mean_squared_error(labels=x, predictions=t4)

train = tf.train.AdamOptimizer(0.001).minimize(loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
plt.ion()

for step in range(5000):
    x_in = mnist.test.next_batch(500)[0].reshape([500, 28, 28, 1])
    result, D_l = sess.run([t4, loss, train], {x: x_in})[:2]
    if step % 50 == 0:
        print(D_l)
        plt.imshow(np.reshape(result[0], (28, 28)), cmap='gray')
        plt.draw()
        plt.pause(0.01)



