#%%
import keras
import numpy as np

(x_train,y_train), (x_test,y_test) = keras.datasets.mnist.load_data()
x_test :np.ndarray = x_test.reshape(-1,28*28).astype('float')/255.
x_train :np.ndarray = x_train.reshape(-1,28*28).astype('float')/255.
x = np.concatenate((x_train,x_test), axis=0)
#%%
from keras.layers import Dense

inputs = keras.Input(shape=(784,))

d1 = Dense(1000,activation='relu', name='dense_1')(inputs)
d2 = Dense(500,activation='relu', name='dense_2')(d1)
d3 = Dense(250,activation='relu', name='dense_3')(d2)

encoder = Dense(30,activation='relu', name='encoder')(d3)

d4 = Dense(250,activation='relu', name='dense_4')(encoder)
d5 = Dense(500,activation='relu', name='dense_5')(d4)
d6 = Dense(1000,activation='relu', name='dense_6')(d5)

decoder = Dense(784,activation='sigmoid', name='decoder')(d6)

model = keras.models.Model(inputs = inputs, outputs = decoder)
model.compile(optimizer='adam', loss='mse')
model.summary()
model.fit(x,x,epochs=20,verbose=2,batch_size=32)
#%%

