#%%
import keras
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

_, (x_test,y_test) = keras.datasets.mnist.load_data()
x_test :np.ndarray = x_test.reshape(-1,28*28).astype('float')/255.
cov = np.cov(x_test,rowvar=False)
#%%
eigenvalue, featureVector = np.linalg.eig(cov)
eigenvalue = eigenvalue.astype('float')
featureVector = featureVector.astype('float').T #dim=1 is the vector
#%%
sort_index = np.argsort(eigenvalue,axis=0).astype('int')
#%%
new_fv = np.zeros_like(featureVector)
new_ev = np.zeros_like(eigenvalue)
for i in range(new_fv.shape[0]):
    reverse_i = new_fv.shape[0] - i - 1 #from big to low
    k = sort_index[reverse_i]
    new_fv[i] = featureVector[k]
    new_ev[i] = eigenvalue[k]
#%%
show_w = new_fv.reshape((-1,28,28))
from util import plot_image_labels
plot_image_labels(show_w,cmap='binary')
#%%
mean = np.mean(x_test,axis=0).reshape((1,-1))
def simple_view():
    dim = 10
    k = x_test[0].reshape(1, -1) - mean
    c_k = new_fv[:dim] @ k.T
    x_pred: np.ndarray = c_k.T @ new_fv[:dim] + mean
    show = np.array([x_test[0].reshape(-1, 784), x_pred]).reshape((-1, 28, 28))
    plot_image_labels(show, rows=1, cols=2, cmap='binary')
#%%
def loss(x_test :np.ndarray, featureVector :np.ndarray ,reduced_dim):
    x = x_test - np.mean(x_test, axis=0).reshape((1,-1))
    u = featureVector[:reduced_dim].T
    c_k = u.T @ x.T
    pred = (u @ c_k).T

    return keras.losses.mean_squared_error(y_true=x, y_pred=pred)
loss_val = loss(x_test, new_fv, 30)