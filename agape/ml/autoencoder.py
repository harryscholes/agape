'''Flexible classes to build and train a variety of autoencoder architectures.

New autoencoder architectures can be implemented easily by inheriting from the
AbstractAutoencoder class.

# Autoencoder architectures
    Autoencoder: one hidden layer
    DeepAutoencoder: n hidden layers
    MultimodalAutoencoder: m input modes with n hidden layers

# Autoencoder variants
    Sparse: sparse embeddings using l1 regularization
    Denoising: corrupt input data with noise
'''
import numpy as np
from keras.layers import Dense, Input, Concatenate
from keras.models import Model
from keras.regularizers import l1
from typing import Union, List
from abc import ABC, abstractmethod

__all__ = ['Autoencoder', 'DeepAutoencoder', 'MultimodalAutoencoder']


class AbstractAutoencoder(ABC):
    '''AbstractAutoencoder class.

    # Autoencoder architectures that inherit from AbstractAutoencoder
        Autoencoder
        DeepAutoencoder
        MultimodalAutoencoder

    # Autoencoder variants (available for all architectures)
        Sparse
        Denoising

    # Subclasses must implement
        `_compile_autoencoder` for a particular architecture
        `_check_autoencoder_parameters` for the parameters of this architecture

    # Arguments
        x_train: Union[np.ndarray, List[np.ndarray]], training data
        x_val: Union[np.ndarray, List[np.ndarray]], validation data
        embedding_size: int, size of embedding
        layers: List[int], layer sizes
    '''
    generic_arguments = '''
        sparse: float (optional), l1 regularization factor if Sparse
        denoising: float (optional), noise factor if Denoising
        epochs: int, number of epochs to train for
        batch_size: int, batch size
        activation: str, activation function
        optimizer: str, training optimizer
        loss: str, loss function
        verbose: int, logging verbosity
    '''
    __doc__ += generic_arguments[5:]

    def __init__(self,
                 # Subclass-specific arguments
                 embedding_size: Union[int, None] = None,
                 layers: Union[List[int], None] = None,
                 *,
                 # Generic arguments
                 x_train: Union[np.ndarray, List[np.ndarray]],
                 x_val: Union[np.ndarray, List[np.ndarray]],
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
        self._check_parameters()
        self._compile()

    # The API exposes the following methods:
    # `train`   : trains the autoencoder
    # `summary` : prints a summary of the architecture
    # `predict` : returns predictions
    # `encode`  : returns embeddings

    def train(self):
        '''Train the autoencoder.
        '''
        self.history = self.autoencoder.fit(
            self.x_train_in, self.x_train,
            validation_data=(self.x_val_in, self.x_val),
            epochs=self.epochs, batch_size=self.batch_size, shuffle=True,
            verbose=self.verbose)

    def summary(self):
        '''Summary of the autoencoder architecture.
        '''
        return self.autoencoder.summary()

    def predict(self, x: Union[np.ndarray, List[np.ndarray]]) \
            -> Union[np.ndarray, List[np.ndarray]]:
        '''Predict `x`.
        '''
        return self.autoencoder.predict(x)

    def encode(self, x: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        '''Encode `x`.
        '''
        return self.encoder.predict(x)

    # Abstract methods that must be implemented by each subclass

    @abstractmethod
    def _check_autoencoder_parameters(self):
        '''Check architecture-specific parameters.
        '''
        pass

    @abstractmethod
    def _compile_autoencoder(self):
        '''Compile a particular autoencoder architecture.

        Must set the following attributes:
            input
            encoded
            decoded
        '''
        pass

    # Private methods

    def _check_parameters(self):
        '''Check the validity of model parameters (function arguments).
        '''
        try:
            self._check_autoencoder_parameters()

            if all((self.embedding_size is None, self.layers is None)):
                raise ValueError(
                    'Cannot both be None: `embdding_size` and `layers`')
            if not any((self.embedding_size is None, self.layers is None)):
                raise ValueError(
                    'Cannot specify both: `embdding_size` and `layers`')

            if self.denoising is None:
                self.x_train_in = self.x_train
                self.x_val_in = self.x_val
            elif 0 <= self.denoising <= 1:
                self.x_train_in = self._add_noise(self.x_train)
                self.x_val_in = self._add_noise(self.x_val)
            else:
                raise ValueError('`denoising` must be between 0 and 1')
        except Exception as e:
            e.args = (f'Parameter error. {e.args[0]}',)
            raise

    def _compile(self):
        '''Compiles the autoencoder.
        '''
        try:
            self._compile_autoencoder()
            self.autoencoder = Model(self.input, self.decoded)
            self.encoder = Model(self.input, self.encoded)
            self.autoencoder.compile(self.optimizer, self.loss,
                                     metrics=['accuracy'])
        except AttributeError as e:
            e.args = (f'Subclass not implemented correctly. {e.args[0]}',)
            raise

    def _encoder_layer(self, embedding_size: int, previous_layer: Dense):
        '''Generates the middle encoding layer.

        Optionally applies l1 regularisation for sparsity.
        '''
        if self.sparse is None:
            return Dense(embedding_size,
                         activation=self.activation)(previous_layer)
        else:
            return Dense(embedding_size, activation=self.activation,
                         activity_regularizer=l1(self.sparse))(previous_layer)

    def _add_noise(self, x):
        '''Add noise to `x`.

        Noise factor is determined by 0 <= `self.denoising` <= 1.
        '''
        def f(x):
            x_n = x + self.denoising * np.random.normal(size=x.shape)
            return np.clip(x_n, 0., 1.)

        try:
            return f(x)
        except AttributeError:  # Multimodal
            return [f(x_i) for x_i in x]


class Autoencoder(AbstractAutoencoder):
    '''Autoencoder.

    # Arguments
        x_train: np.ndarray, training data
        x_val: np.ndarray, validation data
        embedding_size: int, size of embedding
    '''
    __doc__ += AbstractAutoencoder.generic_arguments[5:]

    def __init__(self,
                 x_train: np.ndarray,
                 x_val: np.ndarray,
                 embedding_size: int,
                 sparse: Union[float, None] = None,
                 denoising: Union[float, None] = None,
                 epochs: int = 1,
                 batch_size: int = 128,
                 activation: str = 'relu',
                 optimizer: str = 'adam',
                 loss: str = 'binary_crossentropy',
                 verbose: int = 1):
        super().__init__(
            embedding_size=embedding_size,
            x_train=x_train, x_val=x_val, sparse=sparse, denoising=denoising,
            epochs=epochs, batch_size=batch_size, activation=activation,
            optimizer=optimizer, loss=loss, verbose=verbose)

    def _check_autoencoder_parameters(self):
        if not all(isinstance(x, np.ndarray)
                   for x in (self.x_train, self.x_val)):
            raise TypeError('`x`s must be np.ndarray or List[np.ndarray]')

    def _compile_autoencoder(self):
        self.input = Input(shape=(self.x_train.shape[1],))
        self.encoded = self._encoder_layer(self.embedding_size, self.input)
        self.decoded = Dense(self.x_train.shape[1],
                             activation='sigmoid')(self.encoded)


class DeepAutoencoder(AbstractAutoencoder):
    '''Deep autoencoder.

    # Arguments
        x_train: np.ndarray, training data
        x_val: np.ndarray, validation data
        layers: List[int], layers sizes
    '''
    __doc__ += AbstractAutoencoder.generic_arguments[5:]

    def __init__(self,
                 x_train: np.ndarray,
                 x_val: np.ndarray,
                 layers: List[int],
                 sparse: Union[float, None] = None,
                 denoising: Union[float, None] = None,
                 epochs: int = 1,
                 batch_size: int = 128,
                 activation: str = 'relu',
                 optimizer: str = 'adam',
                 loss: str = 'binary_crossentropy',
                 verbose: int = 1):
        super().__init__(
            layers=layers,
            x_train=x_train, x_val=x_val, sparse=sparse, denoising=denoising,
            epochs=epochs, batch_size=batch_size, activation=activation,
            optimizer=optimizer, loss=loss, verbose=verbose)

    def _check_autoencoder_parameters(self):
        if not all(isinstance(x, np.ndarray)
                   for x in (self.x_train, self.x_val)):
            raise TypeError('`x_train` and `x_val` must be np.ndarray')

        if not len(self.layers) > 1:
            raise ValueError('`len(layers)` must be > 1')

    def _compile_autoencoder(self):
        self.input = Input(shape=(self.x_train.shape[1],))

        hidden = self.input  # For looping over all hidden layers easily
        for i in range(len(self.layers) - 1):
            hidden = Dense(self.layers[i], activation=self.activation)(hidden)

        self.encoded = self._encoder_layer(self.layers[-1], hidden)

        hidden = self.encoded  # For looping over all hidden layers easily
        for i in range(len(self.layers) - 2, -1, -1):
            hidden = Dense(self.layers[i], activation=self.activation)(hidden)

        self.decoded = Dense(self.x_train.shape[1],
                             activation='sigmoid')(hidden)


class MultimodalAutoencoder(AbstractAutoencoder):
    '''Multimodal deep autoencoder.

    # Arguments
        x_train: List[np.ndarray], training data
        x_val: List[np.ndarray], validation data
        layers: List[int], layers sizes
    '''
    __doc__ += AbstractAutoencoder.generic_arguments[5:]

    def __init__(self,
                 x_train: List[np.ndarray],
                 x_val: List[np.ndarray],
                 layers: List[int],
                 sparse: Union[float, None] = None,
                 denoising: Union[float, None] = None,
                 epochs: int = 1,
                 batch_size: int = 128,
                 activation: str = 'relu',
                 optimizer: str = 'adam',
                 loss: str = 'binary_crossentropy',
                 verbose: int = 1):
        super().__init__(
            layers=layers,
            x_train=x_train, x_val=x_val, sparse=sparse, denoising=denoising,
            epochs=epochs, batch_size=batch_size, activation=activation,
            optimizer=optimizer, loss=loss, verbose=verbose)

    def _check_autoencoder_parameters(self):
        xs = (self.x_train, self.x_val)
        if not (all(isinstance(x, list) for x in xs) and
                all(isinstance(y, np.ndarray) for x in xs for y in x)):
            raise TypeError('`x_train` and `x_val` must be List[np.ndarray]')

        if not len(self.layers) > 1:
            raise ValueError('`len(layers)` must be > 1')

    def _compile_autoencoder(self):
        self.input = [Input(shape=(self.x_train[i].shape[1],))
                      for i in range(len(self.x_train))]
        # Each of the m input modes goes through its own dense layer
        hidden = [Dense(self.layers[0], activation=self.activation)(input)
                  for input in self.input]
        # These m dense layers are then concatenated
        concatenated = Concatenate()(hidden)
        # Standard deep autoencoder architecture
        hidden = concatenated  # For looping over all hidden layers easily
        for i in range(1, len(self.layers) - 1):
            hidden = Dense(self.layers[i], activation=self.activation)(hidden)

        self.encoded = self._encoder_layer(self.layers[-1], hidden)

        hidden = self.encoded  # For looping over all hidden layers easily
        for i in range(len(self.layers) - 2, 0, -1):
            hidden = Dense(self.layers[i], activation=self.activation)(hidden)
        # Regenerate the concatenated layer
        concatenated = Dense(self.layers[0] * len(self.x_train),
                             activation=self.activation)(hidden)
        # Split the concatenated layer into m dense layers
        hidden = [
            Dense(self.layers[0], activation=self.activation)(concatenated)
            for i in range(len(self.x_train))]
        # Regenerate the m input modes
        self.decoded = [
            Dense(self.x_train[i].shape[1], activation='sigmoid')(hidden[i])
            for i in range(len(self.x_train))]
