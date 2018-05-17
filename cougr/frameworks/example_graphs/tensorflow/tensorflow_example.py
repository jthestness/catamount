import os
import tensorflow as tf


def main():
    # Graph:
    # a, b, weights, bias inputs
    # out = matmul(a, weights) + b * bias
    m = None
    M = 64
    k = 128
    n = 256
    a_dims = (m, k)
    b_dims = (m, n)
    weights_dims = (k, n)
    bias_dims = (M, n)
    a = tf.placeholder(tf.float32, shape=a_dims, name='a')
    b = tf.placeholder(tf.float32, shape=b_dims, name='b')
    weights = tf.Variable(tf.random_uniform(weights_dims,
                                            minval=-0.1,
                                            maxval=0.1),
                          name='weights')
    bias = tf.Variable(tf.random_uniform(bias_dims,
                                         minval=-0.1,
                                         maxval=0.1),
                       name='bias')
    matmul_out = tf.matmul(a, weights, name='matmul')
    bias_sub = tf.split(bias, [tf.shape(b)[0]], 0)
    mul_out = b * bias_sub
    output = matmul_out + mul_out

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter(os.path.join('.'), sess.graph)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join('.', 'tf_example_graph'))

if __name__ == "__main__":
    main()

