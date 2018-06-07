import numpy as np
from keras.layers import Dense, Input, Concatenate
from keras.models import Model
from keras.regularizers import l1
from typing import Union, List

__all__ = ['AutoEncoder']


class AutoEncoder:
    '''Autoencoder class.

    Autoencoder architectures:
     - Shallow
     - Deep
       - Multimodal

    Autoencoder variants (available for all architectures):
     - Sparse
     - Denoising

    # Arguments
        x_train: np.ndarray (List[np.ndarray] if Multimodal), training data
        x_val: np.ndarray (List[np.ndarray] if Multimodal), validation data
        embedding_size: int (optional), size of embedding
        layers: List[int] (optional), layer sizes if Deep
        sparse: float (optional), l1 regularization factor if Sparse
        denoising: float (optional), noise factor if Denoising
        epochs: int, number of epochs to train for
        batch_size: int, batch size
        activation: str, activation function
        optimizer: str, training optimizer
        loss: str, loss function
        verbose: int, logging verbosity
    '''
    def __init__(self,
                 x_train: Union[np.ndarray, List[np.ndarray]],
                 x_val: Union[np.ndarray, List[np.ndarray]],
                 embedding_size: Union[int, None] = None,
                 layers: Union[List[int], None] = None,
                 sparse: Union[float, None] = None,
                 denoising: Union[float, None] = None,
                 epochs: int = 1,
                 batch_size: int = 128,
                 activation: str = 'relu',
                 optimizer: str = 'adam',
                 loss: str = 'binary_crossentropy',
                 verbose: int = 1):
        self.x_train = x_train
        self.x_val = x_val
        self.embedding_size = embedding_size
        self.layers = layers
        self.sparse = sparse
        self.denoising = denoising
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.verbose = verbose
        self._param_check()

    def train(self):
        '''Train the autoencoder.
        '''
        self._build()
        self.autoencoder.fit(
            self.x_train_in, self.x_train_out,
            validation_data=(self.x_val_in, self.x_val_out),
            epochs=self.epochs, batch_size=self.batch_size, shuffle=True,
            verbose=self.verbose)

    def predict(self, x: Union[np.ndarray, List[np.ndarray]]
                ) -> Union[np.ndarray, List[np.ndarray]]:
        '''Predict `x`.
        '''
        return self.autoencoder.predict(x)

    def encode(self, x: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        '''Encode `x`.
        '''
        return self.encoder.predict(x)

    def _param_check(self):
        '''Check validity of function arguments.
        '''
        xs = ('x_train', 'x_val')
        if all(isinstance(getattr(self, x), np.ndarray) for x in xs):
            pass  # Autoencoder
        elif all(isinstance(y, np.ndarray) for x in xs
                 for y in getattr(self, x)):
            pass  # Multimodal
        else:
            raise TypeError(' '.join((
                '`x_train` and `x_val` must be',
                'np.ndarray or List[np.ndarray]')))

        if all((self.embedding_size is None, self.layers is None)):
            raise ValueError(
                'Cannot both be None: `embdding_size` and `layers`')
        if not any((self.embedding_size is None, self.layers is None)):
            raise ValueError(
                'Cannot specify both: `embdding_size` and `layers`')

        if not (self.layers is None or isinstance(self.layers, list)):
            raise TypeError('`layers` must be a list of ints or None')
        if self.layers:
            if not all(isinstance(l, int) for l in self.layers):
                raise TypeError('`layers` must be a list of ints')
            if not len(self.layers) > 1:
                raise ValueError('`len(layers)` must be > 1')

        if not (self.sparse is None or isinstance(self.sparse, float)):
            raise TypeError('`sparse` must be a float or None')

        if not isinstance(self.epochs, int):
            raise TypeError('`epochs` must be an int')
        if not self.epochs > 0:
            raise ValueError('`epochs` must be positive')

        if not isinstance(self.batch_size, int):
            raise TypeError('`batch_size` must be an int')
        if not self.batch_size > 0:
            raise ValueError('`batch_size` must be positive')

        if not (self.denoising is None or isinstance(self.denoising, float)):
            raise TypeError('`denoising` must be a float or None')
        if self.denoising:
            if not 0 <= self.denoising <= 1:
                raise ValueError('`denosing` must be between 0 and 1')
            self.x_train_in = add_noise(self.x_train, self.denoising)
            self.x_val_in = add_noise(self.x_val, self.denoising)
        else:
            self.x_train_in = self.x_train
            self.x_val_in = self.x_val
        self.x_train_out = self.x_train
        self.x_val_out = self.x_val

        if self.denoising is None or 0 <= self.denoising <= 1:
            if self.denoising is not None:
                self.x_train_in = add_noise(self.x_train, self.denoising)
                self.x_val_in = add_noise(self.x_val, self.denoising)
            else:
                self.x_train_in = self.x_train
                self.x_val_in = self.x_val
            self.x_train_out = self.x_train
            self.x_val_out = self.x_val
        else:
            raise ValueError('`denoising` must be between 0 and 1')

    def _build(self):
        '''Builds the autoencoder.
        '''
        if self.layers:
            if isinstance(self.x_train, np.ndarray):
                self._build_deep()
            elif isinstance(self.x_train, list):
                self._build_multimodal()
        else:
            self._build_shallow()

        self.autoencoder = Model(self.input, self.decoded)
        self.encoder = Model(self.input, self.encoded)
        self.autoencoder.compile(self.optimizer, self.loss)

    def _build_shallow(self):
        '''Builds a shallow autoencoder.
        '''
        self.input = Input(shape=(self.x_train.shape[1],))
        self.encoded = self._encoder_layer(self.embedding_size, self.input)
        self.decoded = Dense(self.x_train.shape[1],
                             activation='sigmoid')(self.encoded)

    def _build_deep(self):
        '''Builds a deep autoencoder.
        '''
        self.input = Input(shape=(self.x_train.shape[1],))

        hidden = self.input
        for i in range(len(self.layers) - 1):
            hidden = Dense(self.layers[i], activation=self.activation)(hidden)

        self.encoded = self._encoder_layer(self.layers[-1], hidden)

        hidden = self.encoded
        for i in range(len(self.layers) - 2, -1, -1):
            hidden = Dense(self.layers[i], activation=self.activation)(hidden)

        self.decoded = Dense(self.x_train.shape[1],
                             activation='sigmoid')(hidden)

    def _build_multimodal(self):
        '''Builds a multimodal deep autoencoder.
        '''
        self.input = [Input(shape=(self.x_train[i].shape[1],))
                      for i in range(len(self.x_train))]

        hidden = [Dense(self.layers[0], activation=self.activation)(input)
                  for input in self.input]

        concatenated = Concatenate()(hidden)

        hidden = concatenated
        for i in range(1, len(self.layers) - 1):
            hidden = Dense(self.layers[i], activation=self.activation)(hidden)

        self.encoded = self._encoder_layer(self.layers[-1], hidden)

        hidden = self.encoded
        for i in range(len(self.layers) - 2, 0, -1):
            hidden = Dense(self.layers[i], activation=self.activation)(hidden)

        concatenated = Dense(self.layers[0] * len(self.x_train),
                             activation=self.activation)(hidden)

        hidden = [
            Dense(self.layers[0], activation=self.activation)(concatenated)
            for i in range(len(self.x_train))]

        self.decoded = [
            Dense(self.x_train[i].shape[1], activation='sigmoid')(hidden[i])
            for i in range(len(self.x_train))]

    def _encoder_layer(self, embedding_size: int, previous_layer: Dense):
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
