def custom_loss(y_true, y_pred):
  from keras import backend as K
  mask = K.cast(K.not_equal(y_true, 0), dtype='float32')
  diff = y_pred - y_true
  sqdiff = diff * diff * mask
  sse = K.sum(K.sum(sqdiff))
  n = K.sum(K.sum(mask))
  return sse / n

def generator(A, M,mu,batch_size):
   while True:
    from sklearn.utils import shuffle
    A, M = shuffle(A, M)
    for i in range(A.shape[0] // batch_size + 1):
      upper = min((i+1)*batch_size, A.shape[0])
      a = A[i*batch_size:upper].toarray()
      m = M[i*batch_size:upper].toarray()
      a = a - mu * m # must keep zeros at zero!
      # m2 = (np.random.random(a.shape) > 0.5)
      # noisy = a * m2
      noisy = a # no noise
      yield noisy, a

def test_generator(A, M, A_test, M_test,mu,batch_size):
  # assumes A and A_test are in corresponding order
  # both of size N x M
  while True:
    for i in range(A.shape[0] // batch_size + 1):
      upper = min((i+1)*batch_size, A.shape[0])
      a = A[i*batch_size:upper].toarray()
      m = M[i*batch_size:upper].toarray()
      at = A_test[i*batch_size:upper].toarray()
      mt = M_test[i*batch_size:upper].toarray()
      a = a - mu * m
      at = at - mu * mt
      yield a, at

def fit_autoencoder(A,A_test):
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  from sklearn.utils import shuffle
  from scipy.sparse import save_npz, load_npz

  import keras.backend as K
  from keras.models import Model
  from keras.layers import Input, Dropout, Dense
  from keras.regularizers import l2
  from keras.optimizers import SGD
  
  # config
  batch_size = 256
  epochs = 20
  reg = 0
  # reg = 0

  mask = (A > 0) * 1.0
  mask_test = (A_test > 0) * 1.0
  A.data -=1
  A_test.data-=1
  # make copies since we will shuffle
  A_copy = A.copy()
  mask_copy = mask.copy()
  A_test_copy = A_test.copy()
  mask_test_copy = mask_test.copy()

  N,M = A.shape

  # center the data
  mu = A.sum() / mask.sum()

  # build the model - just a 1 hidden layer autoencoder
  i = Input(shape=(M,))
  # bigger hidden layer size seems to help!
  x = Dropout(0.7)(i)
  x = Dense(700, activation='tanh', kernel_regularizer=l2(reg))(x)
  x = Dropout(0.5)(x)
  x = Dense(M, kernel_regularizer=l2(reg),activation="sigmoid")(x)
  model = Model(i, x)

  model.compile(
    loss=custom_loss,
    optimizer=SGD(lr=0.07, momentum=0.9),
    # optimizer='adam',
    metrics=[custom_loss],
  )

  r = model.fit_generator(
    generator(A, mask,mu,batch_size),
    validation_data=test_generator(A_copy, mask_copy, A_test_copy, mask_test_copy,mu,batch_size),
    epochs=epochs,
    steps_per_epoch=A.shape[1] // batch_size + 1,
    validation_steps=A_test.shape[1] // batch_size + 1,
  )
  return model
