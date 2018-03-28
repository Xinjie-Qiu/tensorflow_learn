import tensorflow as tf
import numpy as np
import threading
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./mnist', one_hot=True)
batch_size = 500

with tf.device('/cpu:0'):
    # images = tf.placeholder(tf.float32, [784])
    # label = tf.placeholder(tf.float32, [10])
    q = tf.FIFOQueue(10000, [tf.float32, tf.float32], [[784], [10]])
    images, label = mnist.train.next_batch(1)
    enqueue_op = q.enqueue([images[0], label[0]])
    images_batch, label_batch = q.dequeue_many(500)

    coord = tf.train.Coordinator()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    qr = tf.train.QueueRunner(q, [enqueue_op] * 4)
    enqueue_op = qr.create_threads(sess, coord, start=True)

    # def load_and_enqueue():
    #     while not coord.should_stop():
    #         n_images, n_label = mnist.train.next_batch(1)
    #         sess.run(enqueue_op, {images: n_images[0], label: n_label[0]})
    #
    # num_threads = 4
    # for i in range(num_threads):
    #     t = threading.Thread(target=load_and_enqueue)
    #     t.setDaemon(True)
    #     t.start()

    plt.ion()
    for i in range(10):
        plt.cla()
        n_images, n_labels = sess.run([images_batch, label_batch])
        plt.imshow(np.reshape(n_images[0], [28, 28]))
        print(n_labels[0])
    coord.request_stop()
    coord.join(enqueue_op)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    i = 0
    try:
        while not coord.should_stop():
            image_batch_v, label_batch_v = sess.run([images_batch, label_batch])
            i += 1
            for j in range(10):
                print(image_batch_v.shape, label_batch_v[j])
    except tf.errors.OutOfRangeError:
        print("done")
    finally:
        coord.request_stop()
    coord.join(threads)