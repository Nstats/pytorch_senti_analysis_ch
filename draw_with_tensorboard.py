import tensorflow as tf
import numpy as np

related_train_loss = np.array([1.8742,1.0335,1.3461,1.0218,1.4572,1.3624,0.9587,0.5821,0.8468,0.4633,0.2056,0.3311,
                               0.3643,0.5077,0.3121,0.4359,0.6133,0.7527,0.4342,0.2296,0.4422,0.5238,0.2314,0.3286,
                               0.3391,0.1223,0.2132,0.4482,0.2235,0.2465,0.1273,0.3362,0.1279,0.1183,0.2381,0.1265,
                               0.2142,0.2364,0.1202,0.1472])
eval_related_f1_scores = np.array([0.351,0.587,0.653,0.797,0.814,0.702,0.814,0.787,0.694,0.746])

loss = tf.placeholder(dtype=tf.float32)
f1 = tf.placeholder(dtype=tf.float32)

loss_summary = tf.summary.scalar('train_loss', loss)
f1_summary = tf.summary.scalar('eval_f1', f1)


with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('./log', sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in np.arange(0, 40):
        loss_summary_ = sess.run(loss_summary, feed_dict={loss: related_train_loss[i]})
        summary_writer.add_summary(loss_summary_, i*100)
        if i%4 == 0:
            f1_summary_ = sess.run(f1_summary, feed_dict={f1: eval_related_f1_scores[int(i/4)]})
            summary_writer.add_summary(f1_summary_, i*100)
    summary_writer.close()