import tensorflow as tf
import numpy as np

def get_data() -> np.ndarray:
    import keras
    _, (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test: np.ndarray = x_test.reshape(-1, 28 * 28).astype('float') / 255.
    x_test = x_test - np.mean(x_test,axis=0).reshape(1,28*28)
    return x_test
x_test = get_data()

def linear_tf(x_test :np.ndarray, reduced_dim = 30):
    dim = x_test.shape[-1]

    x = tf.placeholder(shape=(None,dim),dtype=tf.float32)

    u = tf.Variable(initial_value=tf.random_normal(shape=(dim, reduced_dim), stddev=0.2), name = 'u',)

    y1 = tf.matmul(x, u)
    y2 = tf.matmul(y1, tf.transpose(u))

    loss_fucntion = tf.reduce_mean(tf.losses.mean_squared_error(labels=x, predictions=y2))
    optimizer = tf.train.AdamOptimizer().minimize(loss_fucntion)

    batch_size = 32
    batch_count = int(x_test.shape[0] // batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(50):

            np.random.shuffle(x_test)
            for batch in range(batch_count):
                x_batch = x_test[batch*batch_size : (batch+1)*batch_size]

                sess.run(optimizer, feed_dict={x:x_batch})

            loss = sess.run(loss_fucntion,feed_dict={x:x_test})
            print("loss:{:.5f}".format(loss))