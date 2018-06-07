import numpy as np
from keras.layers import Dense, Input, Concatenate
from keras.models import Model
from keras.regularizers import l1
from typing import Union, List

__all__ = ['Autoencoder']


class Autoencoder:
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
            self.x_train_in, self.x_train,
            validation_data=(self.x_val_in, self.x_val),
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
        xs = [getattr(self, x) for x in ('x_train', 'x_val')]
        if all(isinstance(x, list) for x in xs):
            if not all(isinstance(x_i, np.ndarray) for x in xs for x_i in x):
                raise TypeError('`x`s must be List[np.ndarray] for Multimodal')
            if not self.layers:
                raise TypeError(
                    'Multimodal is only implemented for Deep autoencoders',
                    '`layer` must be provided')
        elif not all(isinstance(x, np.ndarray) for x in xs):
            raise TypeError('`x`s must be np.ndarray or List[np.ndarray]')

        if all((self.embedding_size is None, self.layers is None)):
            raise ValueError(
                'Cannot both be None: `embdding_size` and `layers`')
        if not any((self.embedding_size is None, self.layers is None)):
            raise ValueError(
                'Cannot specify both: `embdding_size` and `layers`')

        if self.layers and not len(self.layers) > 1:
            raise ValueError('`len(layers)` must be > 1')

        if self.denoising is None:
            self.x_train_in = self.x_train
            self.x_val_in = self.x_val
        elif 0 <= self.denoising <= 1:
            self.x_train_in = add_noise(self.x_train, self.denoising)
            self.x_val_in = add_noise(self.x_val, self.denoising)
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
    def _add_noise(x):
        x_n = x + noise_factor * np.random.normal(size=x.shape)
        return np.clip(x_n, 0., 1.)

    try:
        return _add_noise(X)
    except AttributeError:  # Multimodal
        return [_add_noise(X_i) for X_i in X]
