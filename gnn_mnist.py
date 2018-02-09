import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist', one_hot=True)

BATCH_SIZE = 1000
LR_G = 0.0006
LR_D = 0.0003
G_IDEA = 10
with tf.device('/device:GPU:1'):
    # with tf.variable_scope('auto_encoder'):
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    # 28,28,1 -> 14,14,32
    c1 = tf.layers.conv2d(x, 32, 3, 1, 'same', activation=tf.nn.relu, name='dl1')
    c1_pool = tf.layers.max_pooling2d(c1, 2, 2, 'same')
    # 14,14,32 -> 7,7,64
    c2 = tf.layers.conv2d(c1_pool, 64, 3, 1, 'same', activation=tf.nn.relu, name='dl2')
    c2_pool = tf.layers.max_pooling2d(c2, 2, 2, 'same')
    # 7,7,64 -> 7*7*64 -> 1024 -> 10
    c2_flat = tf.reshape(c2_pool, [-1, 7 * 7 * 64])
    c3 = tf.layers.dense(c2_flat, 1024, name='dl3')
    c4 = tf.layers.dense(c3, 10, name='dl4')

    # 10 -> 1024 -> 7*7*64 -> 7,7,64
    t1 = tf.layers.dense(c4, 1024, name='gl1')
    t2 = tf.layers.dense(t1, 7 * 7 * 64, name='gl2')
    t3 = tf.reshape(t2, [-1, 7, 7, 64])
    # 7,7,64 -> 14,14,32
    t4 = tf.image.resize_images(t3, [14, 14])
    t5 = tf.layers.conv2d_transpose(t4, 32, 3, 1, 'same', activation=tf.nn.sigmoid, name='gl3')
    # t5 = tf.layers.conv2d(t4, 32, 3, 1, 'same', activation=tf.nn.sigmoid)
    # 14,14,32 -> 28,28,1
    t6 = tf.image.resize_images(t5, [28, 28])
    t7 = tf.layers.conv2d_transpose(t6, 1, 3, 1, 'same', activation=tf.nn.sigmoid, name='gl4')
    # t7 = tf.layers.conv2d(t6, 1, 3, 1, 'same', activation=tf.nn.sigmoid)

with tf.device('/device:GPU:0'):
    # with tf.variable_scope('Generator'):
    g_in = tf.placeholder(tf.float32, [None, G_IDEA])
    # 10 -> 1024 -> 7*7*64 -> 7,7,64
    g_l1 = tf.layers.dense(g_in, 1024, name='gl1', reuse=True)
    g_l2 = tf.layers.dense(g_l1, 7 * 7 * 64, name='gl2', reuse=True)
    g_2d = tf.reshape(g_l2, [-1, 7, 7, 64])
    # 7,7,64 -> 14,14,32
    g_l2_resize = tf.image.resize_images(g_2d, [14, 14])
    g_l3 = tf.layers.conv2d_transpose(g_l2_resize, 32, 3, 1, 'same', activation=tf.nn.sigmoid, name='gl3', reuse=True)
    # 14,14,32 -> 28,28,1
    g_l3_resize = tf.image.resize_images(g_l3, [28, 28])
    g_out = tf.layers.conv2d_transpose(g_l3_resize, 1, 3, 1, 'same', activation=tf.nn.sigmoid, name='gl4', reuse=True)

with tf.device('/device:GPU:1'):
    # with tf.variable_scope('Discriminator'):
    d_in = tf.placeholder(tf.float32, [None, 784])
    d_2d = tf.reshape(d_in, [-1, 28, 28, 1])
    d_l1 = tf.layers.conv2d(d_2d, 32, 3, 1, 'same', activation=tf.nn.relu, name='dl1', reuse=True)
    d_l1_pool = tf.layers.max_pooling2d(d_l1, 2, 2, 'same')
    d_l2 = tf.layers.conv2d(d_l1_pool, 64, 3, 1, 'same', activation=tf.nn.relu, name='dl2', reuse=True)
    d_l2_pool = tf.layers.max_pooling2d(d_l2, 2, 2, 'same')
    d_flat = tf.reshape(d_l2_pool, [-1, 7 * 7 * 64])
    d_l3 = tf.layers.dense(d_flat, 1024, name='dl3', reuse=True)
    d_l4 = tf.layers.dense(d_l3, 10, name='dl4', reuse=True)
    d_l5 = tf.layers.dense(d_l4, 1024, name='dl5')
    d_out = tf.layers.dense(d_l5, 1, name='out')

    d_g_l1 = tf.layers.conv2d(g_out, 32, 3, 1, 'same', activation=tf.nn.relu, name='dl1', reuse=True)
    d_g_l1_pool = tf.layers.max_pooling2d(d_g_l1, 2, 2, 'same')
    d_g_l2 = tf.layers.conv2d(d_g_l1_pool, 64, 3, 1, 'same', activation=tf.nn.relu, name='dl2', reuse=True)
    d_g_l2_pool = tf.layers.max_pooling2d(d_g_l2, 2, 2, 'same')
    d_g_flat = tf.reshape(d_g_l2_pool, [-1, 7 * 7 * 64])
    d_g_l3 = tf.layers.dense(d_g_flat, 1024, name='dl3', reuse=True)
    d_g_l4 = tf.layers.dense(d_g_l3, 10, name='dl4', reuse=True)
    d_g_l5 = tf.layers.dense(d_g_l4, 1024, name='dl5', reuse=True)
    d_g_out = tf.layers.dense(d_g_l5, 1, name='out', reuse=True)

tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'd' in var.name]
g_vars = [var for var in tvars if 'g' in var.name]

with tf.device('/device:GPU:1'):
    A_loss = tf.losses.mean_squared_error(labels=x, predictions=t7)
    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_out, labels=tf.ones_like(d_out)))
    D_g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_g_out, labels=tf.zeros_like(d_g_out)))
with tf.device('/device:GPU:0'):
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_g_out, labels=tf.ones_like(d_g_out)))

with tf.device('/device:GPU:1'):
    A_train = tf.train.AdamOptimizer(0.001).minimize(A_loss)
    D_train = tf.train.AdamOptimizer(LR_D).minimize(D_loss, var_list=d_vars)
    D_g_train = tf.train.AdamOptimizer(LR_D).minimize(D_g_loss, var_list=d_vars)
with tf.device('/device:GPU:0'):
    G_train = tf.train.AdamOptimizer(LR_G).minimize(G_loss, var_list=g_vars)

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# """ For setting TensorBoard """

# From this point forward, reuse variables
tf.get_variable_scope().reuse_variables()

tf.summary.scalar('Generator_loss', G_loss)
tf.summary.scalar('Discriminator_loss_real', D_loss)
tf.summary.scalar('Discriminator_loss_fake', D_g_loss)

images_for_tensorboard = g_out
tf.summary.image('Generated_images', images_for_tensorboard, 5)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

sess.run(tf.global_variables_initializer())


plt.ion()

# for step in range(1000):
#     x_in, y_in = mnist.test.next_batch(500)
#     x_in = x_in.reshape([500, 28, 28, 1])
#     result, D_l = sess.run([t7, A_loss, A_train], {x: x_in})[:2]
#     if step % 100 == 0:
#         print('step = {}, loss = {}'.format(step, D_l))
#         plt.imshow(np.reshape(result[0], (28, 28)), cmap='gray')
#         plt.draw()
#         plt.pause(0.01)

for step in range(100000):
    G_idea = np.random.normal(0, 1, size=[BATCH_SIZE, G_IDEA])
    d_input, _ = mnist.test.next_batch(BATCH_SIZE)
    result, _, _, _ = sess.run([g_out, D_train, D_g_train, G_train], {g_in: G_idea, d_in: d_input})
    if step % 1000 == 0:
        plt.imshow(np.reshape(result[0], (28, 28)), cmap='gray')
        plt.draw()
        plt.pause(0.01)
        summary = sess.run(merged, feed_dict={g_in: G_idea, d_in: d_input})
        writer.add_summary(summary, global_step=step)
