import tensorflow as tf
import numpy as np

# 这是 tf.nn.sparse_softmax_cross_entropy_with_logits 函数的手工验证程序

batch_size = 50
num_class = 40

vc = np.random.randint(1, 100, size=[batch_size, num_class])
vl = np.random.randint(0, num_class, size=[batch_size])
index_np = np.arange(batch_size)

lg = tf.Variable(initial_value=vc, dtype=tf.float32, name='lg')
sl = tf.constant(value=vl, dtype=tf.int32, name='sl')

exp_sl = tf.expand_dims(sl, axis=-1)  # 维度扩展，否则无法用 concat 进行拼接
nsl_index = tf.constant(value=index_np, dtype=tf.int32, name='nsl_ind')
t2 = tf.expand_dims(nsl_index, axis=-1)


nsl_concat = tf.concat([t2, exp_sl], 1)
nsl = tf.sparse_to_dense(nsl_concat, [batch_size, num_class], 1.0, 0.0)

stdres = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lg, labels=sl)
sm_lg = tf.nn.softmax(lg)
crs_etp = -tf.reduce_sum((nsl * tf.log(sm_lg)), 1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run(sl))
    print(sess.run(exp_sl))
    print(sess.run(t2))
    print(sess.run(nsl_concat))
    print(sess.run(nsl))

    print('stdres: ', sess.run(stdres))
    print('manualres: ', sess.run(crs_etp))
