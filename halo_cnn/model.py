# This file contains the ML models used for halo_cnn

import os
import pickle
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv1D
from keras.constraints import maxnorm
from keras.optimizers import adam


class BaseHaloCNNModel():
    """Base model for halo_cnn regression"""

    def __str__(self):
        self.model.summary()
        return ''

    def _build_input(self):
        return Input(shape=self.input_shape)

    def fit(self, x_train, y_train, x_val=None, y_val=None, **kwargs):

        self.history = self.model.fit(x_train, y_train,
                                      validation_data=((x_val, y_val)
                                                       if (x_val is not None) &
                                                          (y_val is not None)
                                                       else None),
                                      batch_size=self.batch_size,
                                      epochs=self.epochs,
                                      **kwargs)
        self.trained = True

        return self.history

    def predict(self, x):
        return self.model.predict(x)

    def plot_history(self, ax=None, label=None):
        if ~self.trained:
            raise Exception('Model is untrained.')

        if ax is None:
            f, ax = plt.subplots()

        ax.plot(self.history['loss'],
                label=(str(label)+' ')*(label is not None)+'training')

        if 'val_loss' in self.history.keys():
            ax.plot(self.history['loss'],
                    label=(str(label)+' ')*(label is not None)+'validation')

        return f, ax

    def save(self, fileheader, version=True):
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
        print('Saved model to %s' % (fileheader+'_model.h5'))

        # save model parameters
        model = self.__dict__.pop('model')
        with open(fileheader+'_modpar.p', 'wb') as f:
            pickle.dump(self.__dict__, f)
        self.model = model
        print('Saved model parameters to %s' % (fileheader+'_modpar.p'))

    def load(self, fileheader):
        # load model parameters
        with open(fileheader+'_modpar.p', 'rb') as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        print('Loaded model parameters from %s' % (fileheader+'_modpar.p'))

        # load keras model
        self.model = self._build_model()
        self.model.load_weights(fileheader+'_model.h5')
        print('Loaded model from %s' % (fileheader+'_model.h5'))


class HaloCNNRegressor(BaseHaloCNNModel):
    """Base model for halo_cnn regression"""

    def __init__(self, input_shape,
                 batch_size=50, epochs=20,
                 learning_rate=1e-3, decay=1e-6):
        # initialization
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.decay = decay

        self.history = None
        self.trained = False

        # build ML model
        self.model = self._build_model()

    def _build_model(self):
        in_layer = self._build_input()
        x = Conv1D(filters=10, kernel_size=5, padding='same',
                   activation='relu', kernel_constraint=maxnorm(3))(in_layer)
        x = Conv1D(filters=5, kernel_size=3, padding='same',
                   activation='relu', kernel_constraint=maxnorm(3))(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(units=128, activation='relu',
                  kernel_constraint=maxnorm(3))(x)
        x = Dense(units=64, activation='relu',
                  kernel_constraint=maxnorm(3))(x)
        x = Dense(units=1, activation='linear')(x)

        model = Model(in_layer, x)

        opt = adam(lr=self.learning_rate, decay=self.decay)

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=[])

        return model
