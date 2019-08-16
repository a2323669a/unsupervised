#%%
import numpy as np
import keras
import tensorflow as tf
tf.enable_eager_execution()

def test_CK_equal() -> float:
    '''
    use SVD to solve the PCA. This function is test whether the result of c_k using two ways is equal.
    one way is U.T @ A, another is simga @ V.T

    :return: the result of MSE of results in two ways
    '''

    from PCA_NN import get_data
    x_test = get_data()
    A = x_test.T #A(M :dim, N :size)

    u_full, s, vt = np.linalg.svd(A, full_matrices=False)
    dim = 30
    u = u_full[:,:dim]
    s = s[:dim]

    sigma = np.eye(N=dim,M=784) * s[:dim].reshape((dim,1))
    sigma_vt = sigma @ vt

    return float(np.sum(keras.losses.mean_squared_error((u.T @ A).T,sigma_vt.T)).astype('float'))
