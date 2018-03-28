import tensorflow as tf
import numpy as np
import threading
import time
import sys
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./mnist', one_hot=True)
batch_size = 500


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
    return average_grads


def my_image_filiter(image):
    with tf.variable_scope('conv'):
        x_in = tf.reshape(image, [-1, 28, 28, 1])
        cnn_1 = tf.layers.conv2d(x_in, 16, 5, 1, 'same', activation=tf.nn.relu) #28 * 28 * 16
        pool_1 = tf.layers.max_pooling2d(cnn_1, 2, 2)  # 14, 14, 16
        cnn_2 = tf.layers.conv2d(pool_1, 32, 5, 1, 'same', activation=tf.nn.relu)# 14 * 14 * 32
        pool_2 = tf.layers.max_pooling2d(cnn_2, 2, 2)  # 7, 7, 32
        cnn_3 = tf.layers.conv2d(pool_2, 64, 5, 1, 'same', activation=tf.nn.relu)
        cnn_3_flat = tf.reshape(cnn_3, [-1, 7 * 7 * 64])
        fc1 = tf.layers.dense(cnn_3_flat, 1024)
        y = tf.layers.dense(fc1, 10, name='y')
    return y

def main(argvs):
    with tf.device('/cpu:0'):
        image = tf.placeholder(tf.float32, [784], name='image')
        label = tf.placeholder(tf.float32, [10], name='label')
        q = tf.FIFOQueue(10000, [tf.float32, tf.float32], [[784], [10]])
        enqueue_op = q.enqueue([image, label])
        queue_size = q.size()

        coord = tf.train.Coordinator()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        def load_and_enqueue():
            while not coord.should_stop():
                images, labels = mnist.train.next_batch(1)
                sess.run(enqueue_op, {image: images[0], label: labels[0]})

        num_threads = 4
        for i in range(num_threads):
            t = threading.Thread(target=load_and_enqueue)
            t.setDaemon(True)
            t.start()

        # num_samples_in_queue = 0
        # while num_samples_in_queue < 500:
        #     num_samples_in_queue = sess.run(queue_size)
        #     print("Initializing queue, current size = %i" % num_samples_in_queue)
        #     time.sleep(1)
        plt.ion()
        for i in range(10):
            plt.cla()
            n_images, n_labels = sess.run(q.dequeue_many(batch_size))
            print(sess.run(queue_size))
            plt.imshow(np.reshape(n_images[0], [28, 28]))
            print(n_labels[0])
        coord.should_stop()
        opt = tf.train.AdamOptimizer(0.001)
        tower_gradient = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(2):
                with tf.device('/gpu:%d' % i):
                    # queue_image, queue_label = q.dequeue_many(batch_size)
                    result = my_image_filiter(queue_image)
                    loss = tf.losses.softmax_cross_entropy(onehot_labels=queue_label, logits=result)
                    accuracy = tf.metrics.accuracy(labels=tf.argmax(queue_label, 1), predictions=tf.argmax(result, 1))[1]
                    tf.get_variable_scope().reuse_variables()
                    gradient = opt.compute_gradients(loss)
                    tower_gradient.append(gradient)


        # grads = average_gradients(tower_gradient)
        apply_gradient_op_1 = opt.apply_gradients(tower_gradient[0])
        # apply_gradient_op = opt.apply_gradients(grads)
        apply_gradient_op_2 = opt.apply_gradients(tower_gradient[1])
        apply_gradient_op = tf.group(apply_gradient_op_1, apply_gradient_op_2)
        # with tf.device('/gpu:0'):
        #     result = my_image_filiter(image)
        #     loss = tf.losses.softmax_cross_entropy(onehot_labels=label, logits=result)
        #     accuracy = tf.metrics.accuracy(labels=tf.argmax(result, 1), predictions=tf.argmax(label, 1))[1]

        #     train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()
        for step in range(1000):
            n_image, n_label = mnist.train.next_batch(batch_size * 2)
            plt.imshow(np.reshape(n_image[0], [28, 28]))
            _ = sess.run(apply_gradient_op, {image: n_image, label: n_label})
            if step % 100 == 0:
                print(sess.run(accuracy, {image: n_image, label: n_label}))
        saver.save(sess, './mymodel/mymodel.ckpt')
        coord.request_stop()


if __name__ == '__main__':
    main(sys.argv)