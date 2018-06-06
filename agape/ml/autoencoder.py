import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from keras.regularizers import l1

__all__ = ['AutoEncoder']


class AutoEncoder:
    '''Autoencoder class.

    For autoencoder flavours:
     - vanilla
     - sparse
     - deep
     - denoising
    '''
    def __init__(self, x_train, x_test, embedding_size=None,
                 deep=None, sparse=None, denoising=None,
                 epochs=1, batch_size=128, activation='relu',
                 optimizer='adam', loss='binary_crossentropy',
                 verbose=1):
        self.x_train = x_train
        self.x_test = x_test
        self.x_dim = x_train.shape[1]
        self.embedding_size = embedding_size
        self.deep = deep
        self.sparse = sparse
        self.denoising = denoising
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.verbose = verbose
        self._param_check()

    def _param_check(self):
        '''Check validity of function arguments.
        '''
        if all((self.embedding_size is None, self.deep is None)):
            raise ValueError('Cannot both be None: `embdding_size` and `deep`')
        if not any((self.embedding_size is None, self.deep is None)):
            raise ValueError('Cannot specify both: `embdding_size` and `deep`')

        if not (self.deep is None
                or (isinstance(self.deep, list) and len(self.deep) > 1)):
            raise ValueError('`deep` must be a list of ints specifying layers')

        if not (self.sparse is None or isinstance(self.sparse, float)):
            raise ValueError('`sparse` must be a float')

        if not (isinstance(self.epochs, int) and self.epochs > 0):
            raise ValueError('`epochs` must be a positive int')

        if not (isinstance(self.batch_size, int) and self.batch_size > 0):
            raise ValueError('`batch_size` must be a positive int')

        if self.denoising is None or 0 <= self.denoising <= 1:
            if self.denoising is not None:
                self.x_train_in = add_noise(self.x_train, self.denoising)
                self.x_test_in = add_noise(self.x_test, self.denoising)
            else:
                self.x_train_in = self.x_train
                self.x_test_in = self.x_test
            self.x_train_out = self.x_train
            self.x_test_out = self.x_test
        else:
            raise ValueError('`denoising` must be between 0 and 1')

    def _build(self):
        '''Builds the autoencoder.
        '''
        if self.deep:
            self._build_deep()
        else:
            self._build_shallow()

        self.autoencoder = Model(self.input, self.decoded)
        self.encoder = Model(self.input, self.encoded)
        self.autoencoder.compile(self.optimizer, self.loss)

    def _build_shallow(self):
        '''Builds a shallow autoencoder.
        '''
        self.input = Input(shape=(self.x_dim,))
        self.encoded = self._encoder_layer(self.embedding_size, self.input)
        self.decoded = Dense(self.x_dim, activation='sigmoid')(self.encoded)

    def _build_deep(self):
        '''Builds a deep autoencoder.
        '''
        self.input = Input(shape=(self.x_dim,))

        hidden = self.input
        for i in range(len(self.deep) - 1):
            hidden = Dense(self.deep[i], activation=self.activation)(hidden)

        self.encoded = self._encoder_layer(self.deep[-1], hidden)

        hidden = self.encoded
        for i in range(len(self.deep) - 2, -1, -1):
            hidden = Dense(self.deep[i], activation=self.activation)(hidden)

        self.decoded = Dense(self.x_dim, activation='sigmoid')(hidden)

    def train(self):
        '''Train the autoencoder.
        '''
        self._build()
        self.autoencoder.fit(
            self.x_train_in, self.x_train_out,
            validation_data=(self.x_test_in, self.x_test_out),
            epochs=self.epochs, batch_size=self.batch_size, shuffle=True,
            verbose=self.verbose)

    def test(self):
        '''Test using `x_test`.
        '''
        return self.autoencoder.predict(self.x_test_in)

    def encode(self, x):
        '''Encode `x`.
        '''
        return self.encoder.predict(x)

    def _encoder_layer(self, embedding_size, previous_layer):
        '''Generates the middle encoding layer.

        Optionally applies l1 regulation for sparsity.
        '''
        if self.sparse is None:
            return Dense(embedding_size,
                         activation=self.activation)(previous_layer)
        else:
            return Dense(embedding_size, activation=self.activation,
                         activity_regularizer=l1(self.sparse))(previous_layer)


def add_noise(X, noise_factor):
    '''Add noise to training data for denoising.
    '''
    X_n = X + noise_factor * np.random.normal(size=X.shape)
    return np.clip(X_n, 0., 1.)
