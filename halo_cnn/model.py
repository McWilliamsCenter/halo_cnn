# This file contains the ML models used for halo_cnn

import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.constraints import maxnorm
from keras.optimizers import adam

class BaseHaloCNNRegressor():
    """
       Base model for halo_cnn regression
    """
    
    def __init__(self, input_shape,
                       batch_size=50,
                       epochs=20,
                       learning_rate=1e-3,
                       decay=1e-6):
        # initialization
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.history = None
        
        # build ML model
        self.model = self._build_model()
    
    def _build_input(self):
        return Input(shape=self.input_shape)
        
    def _build_model(self):
        model = self._build_input()
        model = Conv1D(filters=10, kernel_size=5, padding='same', 
                       activation='relu', kernel_constraint=maxnorm(3))(model)
        model = Conv1D(filters=5, kernel_size=3, padding='same', 
                       activation='relu', kernel_constraint=maxnorm(3))(model)
        model = Dropout(0.25)(model)
        model = Flatten()(model)
        model = Dense(units=128, activation='relu', kernel_constraint=maxnorm(3))(model)
        model = Dense(units=64, activation='relu', kernel_constraint=maxnorm(3))(model)
        model = Dense(units=1, activation='linear')
        
        opt = adam(lr=self.learning_rate, decay=self.decay)
        
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=[])
        
        return model
        
    
    def fit(self, x_train, y_train, x_val=None, y_val=None):
    
        self.history = self.model.fit(x_train, y_train,
                                      validation_data = (x_val,y_val) \ 
                                                        if (x_val is not None) & \
                                                           (y_val is not None) \
                                                        else None,
                                      batch_size = self.batch_size,
                                      epochs = self.epochs,
                                      shuffle = True,
                                      verbose = 2)
        return self.history
    
    def predict(self, x):
        return model.predict(x)
        
    def plot_history(self, ax=None, label=None):
        if self.history is None: raise Exception('Model is untrained.')
            
        if ax is None:
            f, ax = plt.subplots()
            
        ax.plot(self.history['loss'], 
            label=(str(label)+' ')*(label is not None)+'training')
            
        if 'val_loss' in self.history.keys():
            label=(str(label)+' ')*(label is not None)+'validation')
        
        return f, ax
    
    def save(self, filename):
        if filename[-3:] != '.h5': raise Warning('.h5 filetype is recommended')
        
        self.model.save_weights(filename)
    
    def load(self, filename):
        self.model.load_weights(filename)
        
