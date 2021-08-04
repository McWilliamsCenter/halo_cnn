
from .model_base import np, Model, Dense, Dropout, Flatten, Conv1D, Adam, \
    BaseModel, Regressor, check_predict_input, l2


class PointRegressor(Regressor):
    def __init_spec__(self):
        self.variance = None

    def fit(self, x_train, y_train, x_val=None, y_val=None, **kwargs):

        self.history = self.model.fit(x_train, y_train,
                                      validation_data=((x_val, y_val)
                                                       if (x_val is not None) &
                                                          (y_val is not None)
                                                       else None),
                                      batch_size=self.batch_size,
                                      epochs=self.epochs,
                                      **kwargs)
        self.history = self.history.history
        self.trained = True
        self.variance = self.model.evaluate(x_train, y_train, verbose=0)[0]  # MSE

        return self.history

    def get_variance(self):
        return self.variance


class PointDropoutRegressor(BaseModel):
    """Base model for halo_cnn regression"""
    def __init_spec__(self, dropout_p, tau):#, N, length_scale=1e-2):
        self.dropout_p = dropout_p
        self.tau = tau
#         self.N = N
#         self.length_scale = length_scale

#         self.decay = ((1 - self.dropout_p)*self.length_scale**2 /
#                       (2*self.N*self.tau))

    def _build_model(self):
        in_layer = self._build_input()
        x = Dropout(self.dropout_p)(in_layer, training=True)
        x = Conv1D(filters=24, kernel_size=5, padding='same',
                   activation='relu',
                   kernel_regularizer=l2(self.decay))(in_layer)
        x = Dropout(self.dropout_p)(x, training=True)
        x = Conv1D(filters=10, kernel_size=3, padding='same',
                   activation='relu',
                   kernel_regularizer=l2(self.decay))(x)
        x = Dropout(self.dropout_p)(x, training=True)
        x = Flatten()(x)
        x = Dense(units=128, activation='relu',
                  kernel_regularizer=l2(self.decay))(x)
        x = Dropout(self.dropout_p)(x, training=True)
        x = Dense(units=64, activation='relu',
                  kernel_regularizer=l2(self.decay))(x)
        x = Dropout(self.dropout_p)(x, training=True)
        x = Dense(units=1, activation='linear',
                  kernel_regularizer=l2(self.decay))(x)

        model = Model(in_layer, x)

        opt = Adam(lr=self.learning_rate)

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])

        return model

    @check_predict_input
    def predict(self, x, T, low_memory=False):
        # low_memory takes longer, but reduces memory usage when x or T is large
        if low_memory:
            return np.reshape(
                np.concatenate([self.model.predict(x) for _ in range(T)], axis=0),
                newshape=(len(x), T), order='F')
        else:
            return np.reshape(self.model.predict(np.repeat(x, T, axis=0)),
                              newshape=(len(x), T), order='C')
            
