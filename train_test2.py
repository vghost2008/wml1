import tensorflow as tf
from wnn import *
#tf.enable_eager_execution()
global_step = tf.train.get_or_create_global_step()
x0 = tf.Variable(10, dtype=tf.float32, name="x0",trainable=False)
x1 = tf.Variable(20, dtype=tf.float32, name="x1",trainable=False)
y = tf.Variable(30, dtype=tf.float32, name="y")
z = tf.Variable(10, dtype=tf.float32, name="z")
tower_grads = []
opt = get_optimizer(global_step, optimizer="PAdam", learning_rate=1.0,
                    batch_size=1, num_epochs_per_decay=100, example_size=1,
                    learn_rate_decay_factor=0.2, min_learn_rate=1e-5)
xs = [x0,x1]
for i in range(2):
    with tf.device("/cpu:{}".format(0)):
        with tf.name_scope("cpu_{}".format(i)):
            x = xs[i]
            loss = x*y*z
            loss = tf.reduce_sum(loss)
            tf.losses.add_loss(loss)

            grads, _, _ = get_train_opv3(optimizer=opt, loss=loss)
            tower_grads.append(grads)

avg_grads = average_grads(tower_grads, clip_norm=None)
opt0 = apply_gradientsv3(avg_grads, global_step, opt)
g_y = avg_grads[0][0]
g_z = avg_grads[1][0]
opt1 = get_batch_norm_ops()
train_op = tf.group(opt0, opt1)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(60):
    res = sess.run(tower_grads[0]+tower_grads[1])
    print(res)
    r_y, rgz, rgy, l, _ = sess.run([y, g_z, g_y, loss, train_op])
    print(f"step {i}: ", r_y, rgz, rgy, l)
'''
output:
[(100.0, 30.0), (300.0, 10.0), (200.0, 30.0), (600.0, 10.0)]
step 0:  -120.0 450.0 150.0 6000.0
[(-4400.0, -120.0), (-1200.0, -440.0), (-8800.0, -120.0), (-2400.0, -440.0)]
step 1:  6480.0 -1800.0 -6600.0 1056000.0
'''
