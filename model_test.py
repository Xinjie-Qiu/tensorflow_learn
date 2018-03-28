import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from varialbe_leran import my_image_filiter

mnist = input_data.read_data_sets('./mnist', one_hot=True)

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

image = tf.placeholder(tf.float32, [None, 784])
label = tf.placeholder(tf.float32, [None, 10])
y = my_image_filiter(image)
accuracy = tf.metrics.accuracy(labels=tf.arg_max(label, 1), predictions=tf.arg_max(y, 1))[1]

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('./mymodel/'))

test_image, test_label = mnist.test.next_batch(50)
print(sess.run(accuracy, {image: test_image, label: test_label}))

