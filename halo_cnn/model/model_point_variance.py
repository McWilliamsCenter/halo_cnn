
try:
    import tensorflow.keras.backend as K
except ImportError:
    import keras.backend as K

from .model_base import np, \
    Model, Dense, Dropout, Flatten, Conv1D, Adam, MaxNorm, \
    Regressor, check_predict_input, l2

class GaussRegressor(Regressor):
    """Model for halo_cnn point+variance"""

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
        x = Dense(2, activation='linear')(x)

        model = Model(in_layer, x)

        opt = Adam(learning_rate=self.learning_rate)

        def loss(y_true, y_pred):
            return K.mean(K.square(y_true[:, 0] - y_pred[:, 0])*K.exp(-y_pred[:, 1])+
                          y_pred[:, 1])/2.
        
        def mse(y_true, y_pred):
            return K.mean(K.square(y_true[:, 0] - y_pred[:, 0]))

        model.compile(loss=loss, optimizer=opt, metrics=[mse])

        return model

    @check_predict_input
    def predict(self, x):
        pred = np.array(self.model(x))
        pred[:, 1] = np.exp(pred[:, 1])
        return pred



class GaussDropoutRegressor(GaussRegressor):
    def __init_spec__(self, dropout_p):
        self.dropout_p = dropout_p

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
        x = Dense(2, activation='linear')(x)

        model = Model(in_layer, x)

        opt = Adam(learning_rate=self.learning_rate)

        def loss(y_true, y_pred):
            return K.mean(K.square(y_true[:, 0] - y_pred[:, 0])*K.exp(-y_pred[:, 1])+
                          y_pred[:, 1])/2.
        
        def mse(y_true, y_pred):
            return K.mean(K.square(y_true[:,0] - y_pred[:, 0]))

        model.compile(loss=loss, optimizer=opt, metrics=[mse])

        return model

    @check_predict_input
    def predict(self, x, T, low_memory=False):
        if low_memory:
            pred = np.concatenate([self.model.predict(x) for _ in range(T)], axis=0)
            pred[:, 1] = np.exp(pred[:, 1])
            return np.reshape(pred, newshape=(len(x), T, -1), order='F')
        else:
            pred = np.array(self.model.predict(np.repeat(x, T, axis=0)))
            pred[:, 1] = np.exp(pred[:, 1])
            return np.reshape(pred, newshape=(len(x), T, -1), order='C')
