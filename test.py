import tensorflow as tf
import numpy as np

a = np.arange(4).reshape(2, 2)
tfa = tf.constant(value=a, dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tfa))
    print(sess.run(tf.nn.l2_loss(tfa)))