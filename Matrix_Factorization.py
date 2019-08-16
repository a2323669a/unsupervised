#%%
import numpy as np
import keras
import tensorflow as tf
tf.enable_eager_execution()
def full_data():
    A = np.array([
        [5, 3, 0, 1],
        [4, 3, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 4, 4],
        [0, 1, 5, 4]
    ])
    k = 2
    u_full, s, vt = np.linalg.svd(A)

    u = u_full[:,:k]
    sigma = np.eye(N=k,M=vt.shape[0]) * s[:k].reshape((-1,1))
    sigma_vt = sigma @ vt
    return u, sigma_vt
#%%

B = np.array([
        [5, 3, None, 1],
        [4, 3, None, 1],
        [1, 1, None, 5],
        [1, None, 4, 4],
        [None,1, 5, 4]
    ])
k = 2

r1 = tf.Variable(tf.random_normal((5,k)), name='r1')
r2 = tf.Variable(tf.random_normal((k,4)), name='r2')

def cal_loss():
    loss = 0
    result = tf.matmul(r1,r2)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            if not B[i][j] == None:
                loss = loss + (tf.square(result[i][j] - B[i][j]))

    return loss

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
for epoch in range(800):
    with tf.GradientTape() as tape:
        loss_value = cal_loss()

        grads = tape.gradient(loss_value, [r1,r2])
        optimizer.apply_gradients(zip(grads, [r1,r2]),
                                global_step=tf.train.get_or_create_global_step())
        print(loss_value)
#%%
r1_n = r1.numpy()
r2_n = r2.numpy()
k = r1_n @ r2_n
