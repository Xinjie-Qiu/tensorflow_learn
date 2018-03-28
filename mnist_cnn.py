import tensorflow as tf
import datetime
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist', one_hot=True)


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
# y =[]
#
# for d in ['/device:GPU:0', '/device:GPU:1']:
#     with tf.device(d):
x_flat = tf.reshape(x, [-1, 28, 28, 1])
cnn_1 = tf.layers.conv2d(x_flat, 16, 5, 1, 'same', activation=tf.nn.relu)
pool_1 = tf.layers.max_pooling2d(cnn_1, 2, 2)  # 14, 14, 16
cnn_2 = tf.layers.conv2d(pool_1, 32, 5, 1, 'same', activation=tf.nn.relu)
pool_2 = tf.layers.max_pooling2d(cnn_2, 2, 2)  # 7, 7, 32
cnn_3 = tf.layers.conv2d(pool_2, 64, 5, 1, 'same', activation=tf.nn.relu)
cnn_3_flat = tf.reshape(cnn_3, [-1, 7 * 7 * 64])
fc1 = tf.layers.dense(cnn_3_flat, 1024)
y = tf.layers.dense(fc1, 10)

loss = tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=y)
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
accuracy = tf.metrics.accuracy(labels=tf.arg_max(y_, 1), predictions=tf.arg_max(y, 1))[1]

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

tf.summary.scalar('loss', loss)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, session.graph)

session.run(init_op)

for i in range(6000):
    b_x, b_y = mnist.train.next_batch(500)
    _ = session.run(train_op, {x: b_x, y_: b_y})
    if i % 50 == 0:
        accuracy_print, sumery = session.run([accuracy, merged], {x: b_x, y_: b_y})
        print(accuracy_print)
        writer.add_summary(sumery, i)
test_x, test_y = mnist.test.next_batch(2000)
print('test accuracy = {}', format(session.run(accuracy, {x: test_x, y_: test_y})))
