from keras import backend as K
from keras.layers import Input, Dense, Layer
import numpy as np
from .autoencoder import AbstractAutoencoder
from typing import Union, Tuple


class KSparse(Layer):
    def __init__(self, k, **kwargs):
        self.k = k
        self.uses_learning_phase = True
        super().__init__(**kwargs)

    def call(self, inputs, mask=None):
        return K.in_train_phase(self.k_sparsify(inputs), inputs)

    def k_sparsify(self, inputs):
        kth_smallest = K.tf.contrib.framework.sort(
            inputs)[..., K.shape(inputs)[-1]-1-self.k]
        return inputs * K.cast(K.greater(
            inputs, kth_smallest[:, None]), K.floatx())

    def k_sparsify_top_k(self, inputs):
        top_k = K.tf.nn.top_k(inputs, k=self.k)
        kth_largest = K.tf.reduce_min(top_k.values, axis=-1)
        return inputs * K.cast(
            K.greater_equal(inputs, kth_largest[:, None]), K.floatx())

    def get_config(self):
        config = {'k': self.k}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def sparsity_level_per_epoch(n_epochs):
    # TODO make odd number of epochs work
    return np.hstack((np.linspace(100, 15, n_epochs // 2, dtype=np.int),
                      np.repeat(15, n_epochs // 2)))


class KSparseAutoencoder(AbstractAutoencoder):
    '''Autoencoder.

    # Arguments
        x_train: np.ndarray, training data
        x_val: np.ndarray, validation data
        embedding_size: int, size of embedding
    '''
    __doc__ += AbstractAutoencoder.generic_arguments

    def __init__(self, x_train: np.ndarray, x_val: np.ndarray,
                 embedding_size: int, k: Union[int, None] = None,
                 denoising: Union[float, None] = None, epochs: int = 1,
                 batch_size: int = 128, activation: str = 'relu',
                 optimizer: str = 'adam', loss: str = 'binary_crossentropy',
                 early_stopping: Union[Tuple[int, float], None] = None,
                 verbose: int = 1):
        self.embedding_size = embedding_size
        self.sparsity_level_per_epoch = sparsity_level_per_epoch(epochs)
        self.k = K.variable(self.sparsity_level_per_epoch[0], dtype=K.tf.int32)
        super().__init__(
            x_train=x_train, x_val=x_val, denoising=denoising,
            epochs=epochs, batch_size=batch_size, activation=activation,
            optimizer=optimizer, loss=loss, early_stopping=early_stopping,
            verbose=verbose)

    def _compile(self):
        self.input = Input(shape=(self.x_train.shape[1],))
        h = Dense(64, activation='sigmoid')(self.input)
        self.encoded = KSparse(k=self.k)(h)
        self.decoded = Dense(self.x_train.shape[1],
                             activation='sigmoid')(self.encoded)
        super()._compile()

    def train(self):
        '''Train the autoencoder.
        '''
        # if self.early_stopping:
        #     callbacks = [EarlyStopping(patience=self.early_stopping[0],
        #                                min_delta=self.early_stopping[1])]
        # else:
        #     callbacks = None

        if isinstance(self.x_val, float):
            validation_split = self.x_val
            validation_data = None
        else:
            validation_split = None
            validation_data = (self.x_val_in, self.x_val)

        for i in range(self.epochs):
            K.set_value(self.k, self.sparsity_level_per_epoch[i])
            # TODO make history save each epoch
            self.history = self.autoencoder.fit(
                self.x_train_in, self.x_train,
                validation_data=validation_data, validation_split=validation_split,
                epochs=self.epochs, batch_size=self.batch_size, shuffle=True,
                verbose=self.verbose)  # callbacks=callbacks
