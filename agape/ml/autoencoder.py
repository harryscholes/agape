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
from keras.layers import Dense, Input, Concatenate, LeakyReLU, Dropout
from keras.models import Model, load_model
from keras.regularizers import l1
from keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Union, List, Tuple
from abc import ABC, abstractmethod
from numba import jit
import functools

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
        `__init__`
        `_compile` to compile the architecture
        `_check_parameters` to check its parameters

    # Arguments
        x_train: Union[np.ndarray, List[np.ndarray]], training data
        x_val: Union[np.ndarray, List[np.ndarray], float], validation data or
            proportion of x_train to use as validation data
        embedding_size: int, size of embedding
        layers: List[int], layer sizes
    '''
    generic_arguments = '''
        sparse: float (optional), l1 regularization factor if Sparse
        dropout: float (optional), dropout probability for regularization
        denoising: float (optional), noise factor if Denoising
        epochs: int, number of epochs to train for
        batch_size: int, batch size
        activation: str, activation function
        optimizer: str, training optimizer
        loss: str, loss function
        early_stopping: Tuple[int, float], of the form (patience, min_delta)
        save_best_model: str, filepath for where to save the best epoch's model
        verbose: int, logging verbosity
    '''[5:]
    __doc__ += generic_arguments

    @abstractmethod
    def __init__(self, *,
                 x_train: Union[np.ndarray, List[np.ndarray]],
                 x_val: Union[np.ndarray, List[np.ndarray], float],
                 sparse: Union[float, None] = None,
                 dropout: Union[float, None] = None,
                 denoising: Union[float, None] = None,
                 epochs: int = 1, batch_size: int = 128,
                 activation: str = 'relu', optimizer: str = 'adam',
                 loss: str = 'binary_crossentropy',
                 early_stopping: Union[Tuple[int, float], None] = None,
                 save_best_model: Union[str, None] = None,
                 verbose: int = 1):
        self.x_train = x_train
        self.x_val = x_val
        self.sparse = sparse
        self.dropout = dropout
        self.denoising = denoising
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.early_stopping = early_stopping
        self.save_best_model = save_best_model
        self.verbose = verbose

        if self.activation == 'leaky_relu':
            self.af = functools.partial(LeakyReLU, alpha=.1)
            self.activation = None

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
        callbacks = []

        if self.early_stopping:
            callbacks.append(EarlyStopping(patience=self.early_stopping[0],
                                           min_delta=self.early_stopping[1]))

        if self.save_best_model:
            callbacks.append(ModelCheckpoint('best_model.h5',
                                             save_best_only=True))

        if isinstance(self.x_val, float):
            validation_split = self.x_val
            validation_data = None
        else:
            validation_split = None
            validation_data = (self.x_val_in, self.x_val)

        self.history = self.autoencoder.fit(
            self.x_train_in, self.x_train,
            validation_data=validation_data, validation_split=validation_split,
            epochs=self.epochs, batch_size=self.batch_size, shuffle=True,
            callbacks=callbacks, verbose=self.verbose)

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
    def _check_parameters(self):
        '''Check the validity of model parameters (function arguments).
        '''
        if self.denoising is None:
            self.x_train_in = self.x_train
            self.x_val_in = self.x_val
        elif 0 <= self.denoising <= 1:
            self.x_train_in = self._add_noise(self.x_train)
            self.x_val_in = self._add_noise(self.x_val)
        else:
            raise ValueError('`denoising` must be between 0 and 1')

        if isinstance(self.early_stopping, tuple):
            if not all((isinstance(self.early_stopping[0], int),
                        isinstance(self.early_stopping[1], float))):
                raise TypeError(
                    '`early_stopping` must be Tuple[int, float]')
        elif self.early_stopping is not None:
            raise TypeError(
                '`early_stopping must be Tuple[int, float] or None')

        if not isinstance(self.dropout, float):
            if self.dropout is not None:
                raise TypeError('`dropout` must be between 0 and 1')

    @abstractmethod
    def _compile(self):
        '''Compiles the autoencoder.
        '''
        try:
            self.autoencoder = Model(self.input, self.decoded)
            self.encoder = Model(self.input, self.encoded)
            self.autoencoder.compile(self.optimizer, self.loss,
                                     metrics=['accuracy'])
        except AttributeError as e:
            e.args = (f'Subclass not implemented correctly. {e.args[0]}',)
            raise

    # Private methods

    def _encoder_layer(self, embedding_size: int, previous_layer: Dense):
        '''Generates the middle encoding layer.

        Optionally applies l1 regularisation for sparsity.
        '''
        if self.sparse is None:
            h = Dense(embedding_size, activation=self.activation,
                      name='encoding')(previous_layer)
        else:
            h = Dense(embedding_size, activation=self.activation,
                      activity_regularizer=l1(self.sparse),
                      name='encoding')(previous_layer)

        if self.activation is None:
            h = self.af()(h)

        if self.dropout:
            h = Dropout(self.dropout)(h)

        return h

    def _add_noise(self, x):
        '''Add noise to `x`.

        Noise factor is determined by 0 <= `self.denoising` <= 1.
        '''
        @jit(nopython=True)
        def jitted_noise(x, nf):
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    x[i, j] += nf * np.random.normal()
            return x

        def f(x, nf):
            x_n = jitted_noise(x, nf)
            return np.clip(x_n, 0, 1)

        try:
            return f(x, self.denoising)
        except AttributeError:  # Multimodal
            return [f(x_i, self.denoising) for x_i in x]


class Autoencoder(AbstractAutoencoder):
    '''Autoencoder.

    # Arguments
        x_train: np.ndarray, training data
        x_val: np.ndarray, validation data
        embedding_size: int, size of embedding
    '''
    __doc__ += AbstractAutoencoder.generic_arguments

    def __init__(self, x_train: np.ndarray, x_val: np.ndarray,
                 embedding_size: int, sparse: Union[float, None] = None,
                 dropout: Union[float, None] = None,
                 denoising: Union[float, None] = None, epochs: int = 1,
                 batch_size: int = 128, activation: str = 'relu',
                 optimizer: str = 'adam', loss: str = 'binary_crossentropy',
                 early_stopping: Union[Tuple[int, float], None] = None,
                 save_best_model: Union[str, None] = None, verbose: int = 1):
        self.embedding_size = embedding_size
        super().__init__(
            x_train=x_train, x_val=x_val, sparse=sparse, denoising=denoising,
            epochs=epochs, batch_size=batch_size, activation=activation,
            optimizer=optimizer, loss=loss, early_stopping=early_stopping,
            save_best_model=save_best_model, verbose=verbose, dropout=dropout)

    def _check_parameters(self):
        if not isinstance(self.x_train, np.ndarray):
            raise TypeError('`x_train` must be np.ndarray')

        if not any((isinstance(self.x_val, np.ndarray),
                   isinstance(self.x_val, float))):
            raise TypeError('`x_val` must be np.ndarray or float')

        if not (isinstance(self.embedding_size, int)
                and self.embedding_size > 0):
            raise TypeError('`embedding_size` must be a positive int')

        super()._check_parameters()

    def _compile(self):
        self.input = Input(shape=(self.x_train.shape[1],))
        self.encoded = self._encoder_layer(self.embedding_size, self.input)
        self.decoded = Dense(self.x_train.shape[1],
                             activation='sigmoid')(self.encoded)

        super()._compile()

    def train(self):
        super().train()

        if self.save_best_model:
            best_model = load_model(self.save_best_model)
            input = best_model.layers[0].input
            self.autoencoder = Model(input,
                                     best_model.layers[-1].output)
            self.encoder = Model(input,
                                 best_model.get_layer('encoding').output)


class DeepAutoencoder(AbstractAutoencoder):
    '''Deep autoencoder.

    # Arguments
        x_train: np.ndarray, training data
        x_val: np.ndarray, validation data
        layers: List[int], layers sizes
    '''
    __doc__ += AbstractAutoencoder.generic_arguments

    def __init__(self, x_train: np.ndarray, x_val: np.ndarray,
                 layers: List[int], sparse: Union[float, None] = None,
                 dropout: Union[float, None] = None,
                 denoising: Union[float, None] = None, epochs: int = 1,
                 batch_size: int = 128, activation: str = 'relu',
                 optimizer: str = 'adam', loss: str = 'binary_crossentropy',
                 early_stopping: Union[Tuple[int, float], None] = None,
                 save_best_model: Union[str, None] = None, verbose: int = 1):
        self.layers = layers
        super().__init__(
            x_train=x_train, x_val=x_val, sparse=sparse, denoising=denoising,
            epochs=epochs, batch_size=batch_size, activation=activation,
            optimizer=optimizer, loss=loss, early_stopping=early_stopping,
            save_best_model=save_best_model, verbose=verbose, dropout=dropout)

    def _check_parameters(self):
        if not isinstance(self.x_train, np.ndarray):
            raise TypeError('`x_train` must be np.ndarray')

        if not any((isinstance(self.x_val, np.ndarray),
                   isinstance(self.x_val, float))):
            raise TypeError('`x_val` must be np.ndarray or float')

        if not len(self.layers) > 1:
            raise ValueError('`len(layers)` must be > 1')

        super()._check_parameters()

    def _compile(self):
        self.input = Input(shape=(self.x_train.shape[1],))

        hidden = self.input  # For looping over all hidden layers easily
        for i in range(len(self.layers) - 1):
            hidden = Dense(self.layers[i], activation=self.activation)(hidden)

            if self.activation is None:
                hidden = self.af()(hidden)

            if self.dropout:
                hidden = Dropout(self.dropout)(hidden)

        self.encoded = self._encoder_layer(self.layers[-1], hidden)

        hidden = self.encoded  # For looping over all hidden layers easily
        for i in range(len(self.layers) - 2, -1, -1):
            hidden = Dense(self.layers[i], activation=self.activation)(hidden)

            if self.activation is None:
                hidden = self.af()(hidden)

            if self.dropout:
                hidden = Dropout(self.dropout)(hidden)

        self.decoded = Dense(self.x_train.shape[1],
                             activation='sigmoid')(hidden)

        super()._compile()

    def train(self):
        super().train()

        if self.save_best_model:
            best_model = load_model(self.save_best_model)
            input = best_model.layers[0].input
            self.autoencoder = Model(input,
                                     best_model.layers[-1].output)
            self.encoder = Model(input,
                                 best_model.get_layer('encoding').output)


class MultimodalAutoencoder(AbstractAutoencoder):
    '''Multimodal deep autoencoder.

    # Arguments
        x_train: List[np.ndarray], training data
        x_val: List[np.ndarray], validation data
        layers: List[int], layers sizes
    '''
    __doc__ += AbstractAutoencoder.generic_arguments

    def __init__(self, x_train: np.ndarray, x_val: np.ndarray,
                 layers: List[int], sparse: Union[float, None] = None,
                 dropout: Union[float, None] = None,
                 denoising: Union[float, None] = None, epochs: int = 1,
                 batch_size: int = 128, activation: str = 'relu',
                 optimizer: str = 'adam', loss: str = 'binary_crossentropy',
                 early_stopping: Union[Tuple[int, float], None] = None,
                 save_best_model: Union[str, None] = None, verbose: int = 1):
        self.layers = layers
        super().__init__(
            x_train=x_train, x_val=x_val, sparse=sparse, denoising=denoising,
            epochs=epochs, batch_size=batch_size, activation=activation,
            optimizer=optimizer, loss=loss, early_stopping=early_stopping,
            save_best_model=save_best_model, verbose=verbose, dropout=dropout)

    def _check_parameters(self):
        if not (isinstance(self.x_train, list)
                and all(isinstance(x, np.ndarray) for x in self.x_train)):
            raise TypeError('`x_train` must be np.ndarray')

        if not any((isinstance(self.x_val, list)
                    and all(isinstance(x, np.ndarray) for x in self.x_val),
                    isinstance(self.x_val, float))):
            raise TypeError('`x_val` must be np.ndarray or float')

        if not len(self.layers) > 1:
            raise ValueError('`len(layers)` must be > 1')

        super()._check_parameters()

    def _compile(self):
        self.input = [Input(shape=(self.x_train[i].shape[1],),
                            name=f'input_{i}')
                      for i in range(len(self.x_train))]
        # Each of the m input modes goes through its own dense layer
        hidden = [Dense(self.layers[0], activation=self.activation)(input)
                  for input in self.input]
        if self.activation is None:
            hidden = [self.af()(h) for h in hidden]
        if self.dropout:
            hidden = [Dropout(self.dropout)(h) for h in hidden]
        # These m dense layers are then concatenated
        concatenated = Concatenate(name='concatenated_0')(hidden)
        # Standard deep autoencoder architecture
        hidden = concatenated  # For looping over all hidden layers easily
        for i in range(1, len(self.layers) - 1):
            hidden = Dense(self.layers[i], activation=self.activation)(hidden)
            if self.activation is None:
                hidden = self.af()(hidden)
            if self.dropout:
                hidden = Dropout(self.dropout)(hidden)

        self.encoded = self._encoder_layer(self.layers[-1], hidden)

        hidden = self.encoded  # For looping over all hidden layers easily
        for i in range(len(self.layers) - 2, 0, -1):
            hidden = Dense(self.layers[i], activation=self.activation)(hidden)
            if self.activation is None:
                hidden = self.af()(hidden)
            if self.dropout:
                hidden = Dropout(self.dropout)(hidden)
        # Regenerate the concatenated layer
        concatenated = Dense(self.layers[0] * len(self.x_train),
                             activation=self.activation,
                             name='concatenated_1')(hidden)
        # Split the concatenated layer into m dense layers
        hidden = [
            Dense(self.layers[0], activation=self.activation)(concatenated)
            for i in range(len(self.x_train))]
        if self.activation is None:
            hidden = [self.af()(h) for h in hidden]
        if self.dropout:
            hidden = [Dropout(self.dropout)(h) for h in hidden]
        # Regenerate the m input modes
        self.decoded = [
            Dense(self.x_train[i].shape[1], activation='sigmoid',
                  name=f'output_{i}')(hidden[i])
            for i in range(len(self.x_train))]

        super()._compile()

    def train(self):
        super().train()

        if self.save_best_model:
            best_model = load_model(self.save_best_model)
            inputs = [best_model.get_layer(f'input_{i}').input
                      for i in range(len(self.x_train))]
            self.autoencoder = Model(inputs,
                                     best_model.layers[-1].output)
            self.encoder = Model(inputs,
                                 best_model.get_layer('encoding').output)
