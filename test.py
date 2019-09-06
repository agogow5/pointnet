import tensorflow as tf
import numpy as np

npa = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(2, 3, 2)
point_cloud = tf.constant(value=npa, name='tc', dtype=tf.float32, shape=(2, 3, 2))

print(point_cloud.get_shape().as_list())
print(point_cloud)
point_cloud_t = tf.expand_dims(point_cloud, -1)
print(point_cloud_t.get_shape().as_list())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    stc = sess.run(tf.nn.l2_loss(point_cloud_t))
    print(sess.run(point_cloud))
    print(sess.run(point_cloud_t))

print(stc)