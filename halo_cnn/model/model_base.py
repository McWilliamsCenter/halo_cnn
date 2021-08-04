# This file contains the ML models used for halo_cnn

import os
import pickle
from functools import wraps
import numpy as np

try:
    # Tensorflow backend
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv1D
    from tensorflow.keras.constraints import MaxNorm
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
except ImportError:
    # Theano backend
    from keras.models import Model
    from keras.layers import Input, Dense, Dropout, Flatten, Conv1D
    from keras.constraints import MaxNorm
    from keras.optimizers import Adam
    from keras.regularizers import l2
    
# decorators

def check_predict_input(func):
    wraps(func)

    def wrapper(self, x, *args, **kwargs):
        in_len = len(self.input_shape)
        if x.shape[-in_len:] != self.input_shape:
            raise Exception(f'Input shape {x.shape} not compatible with model '
                            f'input {self.input_shape}.')
        if len(x.shape) == in_len+1:
            return func(self, x, *args, **kwargs)
        elif len(x.shape) == in_len:
            x = np.expand_dims(x, 0)
            return func(self, x, *args, **kwargs)[0, ...]
        else:
            raise Exception(f'Input shape {x.shape} not compatible with model '
                            f'input {self.input_shape}.')
    return wrapper


# base classes


class BaseModel():
    """Base model for halo_cnn regression"""

    def __init__(self, input_shape,
                 batch_size=50, epochs=20,
                 learning_rate=1e-3, decay=1e-4,
                 *args, **kwargs):
        # initialization
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.decay = decay

        self.history = None
        self.trained = False

        self.__init_spec__(*args, **kwargs)

        # build ML model
        self.model = self._build_model()

    def __init_spec__(self):
        return

    def _build_model(self):
        return Model()

    def __str__(self):
        self.model.summary()
        return ''

    def _build_input(self):
        return Input(shape=self.input_shape)

    def fit(self, x_train, y_train, x_val=None, y_val=None, **kwargs):

        self.history = self.model.fit(x_train, y_train,
                                      validation_data=((x_val, y_val)
                                                       if ((x_val is not None) &
                                                           (y_val is not None))
                                                       else None),
                                      batch_size=self.batch_size,
                                      epochs=self.epochs,
                                      **kwargs)
        self.history = self.history.history
        self.trained = True

        return self.history

    def save(self, fileheader, version=False, verbose=False):
        # version-ing
        if version & (os.path.isfile(fileheader+'_model.h5') |
                      os.path.isfile(fileheader+'_modpar.h5')):
            v = 1
            while v < 10e4:
                if os.path.isfile(fileheader+str(v)+'_model.h5') | \
                   os.path.isfile(fileheader+str(v)+'_modpar.h5'):
                    v += 1
                else:
                    fileheader += str(v)
                    break

        # save keras model
        self.model.save_weights(fileheader+'_model.h5')
        if verbose:
            print('Saved model to %s' % (fileheader+'_model.h5'))

        # save model parameters
        model = self.__dict__.pop('model')
        with open(fileheader+'_modpar.p', 'wb') as f:
            pickle.dump(self.__dict__, f)
        self.model = model
        if verbose:
            print('Saved model parameters to %s' % (fileheader+'_modpar.p'))

    def load(self, fileheader, verbose=False):
        # load model parameters
        with open(fileheader+'_modpar.p', 'rb') as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        if verbose:
            print('Loaded model parameters from %s' % (fileheader+'_modpar.p'))

        # load keras model
        self.model = self._build_model()
        self.model.load_weights(fileheader+'_model.h5')
        if verbose:
            print('Loaded model from %s' % (fileheader+'_model.h5'))


class Regressor(BaseModel):
    """Base model for halo_cnn regression"""
    
    def _build_model(self):
        in_layer = self._build_input()
        x = Conv1D(filters=24, kernel_size=5, padding='same',
                   activation='relu', kernel_regularizer=l2(self.decay))(in_layer)
        x = Conv1D(filters=10, kernel_size=3, padding='same',
                   activation='relu', kernel_regularizer=l2(self.decay))(x)
        x = Flatten()(x)
        x = Dense(units=128, activation='relu',
                  kernel_regularizer=l2(self.decay))(x)
        x = Dense(units=64, activation='relu',
                  kernel_regularizer=l2(self.decay))(x)
        x = Dense(units=1, activation='linear',
                  kernel_regularizer=l2(self.decay))(x)

        model = Model(in_layer, x)

        opt = Adam(lr=self.learning_rate)

        model.compile(loss='mean_squared_error', optimizer=opt,
                      metrics=['mean_squared_error'])

        return model

    @check_predict_input
    def predict(self, x):
        return self.model.predict(x)
    
    @check_predict_input
    def evaluate(self, x, y):
        return self.model.evaluate(x, y, 
                                   batch_size=self.batch_size,
                                   verbose=0
                                  )
