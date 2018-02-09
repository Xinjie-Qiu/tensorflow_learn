import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist', one_hot=True)

x_in = tf.placeholder(tf.float32, [None, 784])
x = tf.reshape(x_in, [-1, 28, 28, 1])

# encoder
e = tf.layers.conv2d(x, 32, 3, 1, 'same', activation=tf.nn.relu)
e = tf.layers.max_pooling2d(e, 2, 2, 'same')
e = tf.layers.conv2d(e, 16, 3, 1, 'same', activation=tf.nn.relu)
e = tf.layers.max_pooling2d(e, 2, 2, 'same')
e = tf.reshape(e, [-1, 16 * 7 * 7])
e = tf.layers.dense(e, 1024)
e = tf.layers.dense(e, 10)

# decoder
d = tf.layers.dense(e, 1024)
d = tf.layers.dense(e, 16 * 7 * 7)
d = tf.reshape(d, [-1, 7, 7, 16])
d = tf.layers.conv2d_transpose(d, 16, 2, 2, 'same')
d = tf.layers.conv2d(d, 16, 3, 1, 'same', activation=tf.nn.relu)
d = tf.layers.conv2d_transpose(d, 32, 2, 2, 'same')
d_out = tf.layers.conv2d(d, 1, 3, 1, 'same', activation=tf.nn.relu)

loss = tf.losses.mean_squared_error(labels=x, predictions=d_out)
train = tf.train.AdamOptimizer(0.002).minimize(loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
plt.ion()

for step in range(5000):
    train_x, _ = mnist.test.next_batch(500)
    result, D_l = sess.run([d_out, loss, train], {x_in: train_x})[:2]
    if step % 50 == 0:
        print(D_l)
        plt.imshow(np.reshape(result[0], (28, 28)), cmap='gray')
        plt.draw()
        plt.pause(0.01)
