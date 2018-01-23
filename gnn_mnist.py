import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist', one_hot=True)

BATCH_SIZE = 500
LR_G = 0.0002
LR_D = 0.0002
G_IDEA = 10

with tf.variable_scope('Generator'):
    g_in = tf.placeholder(tf.float32, [None, G_IDEA])
    g_l1 = tf.layers.dense(g_in, 28 * 28, activation=tf.nn.relu)
    g_l1_2d = tf.reshape(g_l1, [-1, 28, 28, 1])
    g_l2 = tf.layers.conv2d(g_l1_2d, 32, 3, 1, 'same', activation=tf.nn.relu)
    g_l3 = tf.layers.conv2d(g_l2, 64, 3, 1, 'same', activation=tf.nn.relu)
    g_out = tf.layers.conv2d(g_l3, 1, 3, 1, 'same')

with tf.variable_scope('Discriminator'):
    d_in = tf.placeholder(tf.float32, [None, 784])
    d_2d = tf.reshape(d_in, [-1, 28, 28, 1])
    d_l1 = tf.layers.conv2d(d_2d, 32, 3, 1, 'same', activation=tf.nn.relu, name='l1')
    d_l2 = tf.layers.conv2d(d_l1, 64, 3, 1, 'same', activation=tf.nn.relu, name='l2')
    d_flat = tf.reshape(d_l2, [-1, 28 * 28 * 64])
    d_l3 = tf.layers.dense(d_flat, 1024, activation=tf.nn.relu, name='l3')
    d_out = tf.layers.dense(d_l3, 1, name='out')

    d_g_l1 = tf.layers.conv2d(g_out, 32, 3, 1, 'same', activation=tf.nn.relu, name='l1', reuse=True)
    d_g_l2 = tf.layers.conv2d(d_g_l1, 64, 3, 1, 'same', activation=tf.nn.relu, name='l2', reuse=True)
    d_g_flat = tf.reshape(d_g_l2, [-1, 28 * 28 * 64])
    d_g_l3 = tf.layers.dense(d_g_flat, 1024, activation=tf.nn.relu, name='l3', reuse=True)
    d_g_out = tf.layers.dense(d_g_l3, 1, name='out', reuse=True)

D_loss = -tf.reduce_mean(tf.log(d_out) + tf.log(1-d_g_out))
G_loss = tf.reduce_mean(tf.log(1-d_g_out))

D_train = tf.train.AdamOptimizer(LR_D).minimize(
    D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
G_train = tf.train.AdamOptimizer(LR_G).minimize(
    G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

for step in range(50000):
    G_idea = np.random.randn(BATCH_SIZE, G_IDEA)
    d_input, _ = mnist.test.next_batch(BATCH_SIZE)
    result = sess.run([g_out, D_train, G_train], {g_in: G_idea, d_in: d_input})[0]
    if step % 50 == 0:
        plt.imshow(np.reshape(result[0], (28, 28)), cmap='gray')
        plt.draw()
        plt.pause(0.01)
