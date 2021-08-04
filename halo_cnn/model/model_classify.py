
try:
    import tensorflow.keras.backend as K
    from tensorflow.keras.layers import Activation
except ImportError:
    import keras.backend as K
    from keras.layers import Activation

from .model_base import np, pickle, os, \
    Model, Dense, Dropout, Flatten, Conv1D, Adam, MaxNorm, \
    BaseModel, check_predict_input, l2


class Classifier(BaseModel):
    def __init_spec__(self, nbins):
        self.nbins = nbins
        
    def _get_bin_edges(self):
        return np.linspace(-1, 1, self.nbins+1)

    def _get_bin_centers(self):
        return K.constant(np.convolve(self._get_bin_edges(),
                                      [0.5, 0.5], mode='valid'),
                          shape=(self.nbins, 1))

    def _bin_to_point(self, Y):
        return K.dot(Y, self._get_bin_centers())
    
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
        x = Dense(units=self.nbins, activation='softmax')(x)

        model = Model(in_layer, x)

        opt = Adam(lr=self.learning_rate)
        
        def mse(y_true, y_pred):
            return K.mean(K.square(self._bin_to_point(y_true) - 
                                   self._bin_to_point(y_pred)))

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt, metrics=[mse])

        return model

    @check_predict_input
    def predict_bins(self, x):
        return self.model.predict(x)


class DropoutClassifier(Classifier):
    def __init_spec__(self, nbins, dropout_p):
        self.nbins = nbins
        self.dropout_p = dropout_p

        self.soft_input = None

    def _build_model(self):
        in_layer = self._build_input()
        x = Dropout(self.dropout_p)(in_layer, training=True)
        x = Conv1D(filters=24, kernel_size=5, padding='same',
                   activation='relu', kernel_regularizer=l2(self.decay))(x)
        x = Dropout(self.dropout_p)(x, training=True)
        x = Conv1D(filters=10, kernel_size=3, padding='same',
                   activation='relu', kernel_regularizer=l2(self.decay))(x)
        x = Dropout(self.dropout_p)(x, training=True)
        x = Flatten()(x)
        x = Dense(units=128, activation='relu',
                  kernel_regularizer=l2(self.decay))(x)
        x = Dropout(self.dropout_p)(x, training=True)
        x = Dense(units=64, activation='relu',
                  kernel_regularizer=l2(self.decay))(x)
        x = Dropout(self.dropout_p)(x, training=True)
        x = Dense(units=self.nbins, activation='linear',
                  kernel_regularizer=l2(self.decay))(x)
        x = Activation('softmax')(x)

        model = Model(in_layer, x)

        opt = Adam(lr=self.learning_rate)
        
        def mse(y_true, y_pred):
            return K.mean(K.square(self._bin_to_point(y_true) - 
                                   self._bin_to_point(y_pred)))

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt, metrics=[mse])

        self.soft_input = K.function([model.input],
                                     [model.layers[-2].output])

        return model
    
#     def _build_model(self):
#         in_layer = self._build_input()
#         x = Dropout(self.dropout_p)(in_layer, training=True)
#         x = Conv1D(filters=10, kernel_size=5, padding='same',
#                    activation='relu', kernel_constraint=MaxNorm(3))(x)
#         x = Dropout(self.dropout_p)(x, training=True)
#         x = Conv1D(filters=5, kernel_size=3, padding='same',
#                    activation='relu', kernel_constraint=MaxNorm(3))(x)
#         x = Dropout(self.dropout_p)(x, training=True)
#         x = Flatten()(x)
#         x = Dense(units=128, activation='relu',
#                   kernel_constraint=MaxNorm(3))(x)
#         x = Dropout(self.dropout_p)(x, training=True)
#         x = Dense(units=64, activation='relu',
#                   kernel_constraint=MaxNorm(3))(x)
#         x = Dropout(self.dropout_p)(x, training=True)
#         x = Dense(units=self.nbins, activation='linear')(x)
#         x = Activation('softmax')(x)

#         model = Model(in_layer, x)

#         opt = Adam(lr=self.learning_rate, decay=self.decay)
        
#         def mse(y_true, y_pred):
#             return K.mean(K.square(self._bin_to_point(y_true) - 
#                                    self._bin_to_point(y_pred)))

#         model.compile(loss='categorical_crossentropy',
#                       optimizer=opt, metrics=[mse])

#         self.soft_input = K.function([model.input],
#                                      [model.layers[-2].output])

#         return model

    @check_predict_input
    def predict_bins(self, x, T, low_memory=False):
        if low_memory:
            return np.reshape(
                np.concatenate([self.soft_input(x)[0] for _ in range(T)], axis=0),
                newshape=(len(x), T, -1), order='F')
        else:
            return np.reshape(self.soft_input(np.repeat(x, T, axis=0))[0],
                              newshape=(len(x), T, -1), order='C')

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
        soft_input = self.__dict__.pop('soft_input')
        with open(fileheader+'_modpar.p', 'wb') as f:
            pickle.dump(self.__dict__, f)
        self.model = model
        self.soft_input = soft_input
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
        self.soft_input = K.function([self.model.input],
                                     [self.model.layers[-2].output])
        if verbose:
            print('Loaded model from %s' % (fileheader+'_model.h5'))
