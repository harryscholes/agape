from keras.models import Model
from keras.layers import Input, Dense, concatenate
from keras.optimizers import SGD

__all__ = ["AE", "MDA"]


def AE(input_dim=6400, encoding_dims=3*[256]):
    """Autoencoder.
    """
    # input layer
    input_layer = Input(shape=(input_dim, ))
    hidden_layer = input_layer
    for i in range(0, len(encoding_dims)):
        # generate hidden layer
        if i == len(encoding_dims)//2:
            hidden_layer = Dense(encoding_dims[i],
                                 activation='sigmoid',
                                 name='middle_layer')(hidden_layer)
        else:
            hidden_layer = Dense(encoding_dims[i],
                                 activation='sigmoid',
                                 name='layer_' + str(i+1))(hidden_layer)

    # reconstruction of the input
    decoded = Dense(input_dim,
                    activation='sigmoid')(hidden_layer)

    # autoencoder model
    sgd = SGD(lr=0.2, momentum=0.95, decay=0.0, nesterov=False)
    model = Model(inputs=input_layer, outputs=decoded)
    model.compile(optimizer=sgd, loss='binary_crossentropy')

    return model


def MDA(input_dims, encoding_dims):
    """Multi-modal autoencoder.
    """
    # input layers
    input_layers = []
    for dim in input_dims:
        input_layers.append(Input(shape=(dim, )))

    # hidden layers
    hidden_layers = []
    for j in range(0, len(input_dims)):
        hidden_layers.append(Dense(encoding_dims[0]//len(input_dims),
                                   activation='sigmoid')(input_layers[j]))

    # Concatenate layers
    if len(encoding_dims) == 1:
        hidden_layer = concatenate(hidden_layers, name='middle_layer')
    else:
        hidden_layer = concatenate(hidden_layers)

    # middle layers
    for i in range(1, len(encoding_dims)-1):
        if i == len(encoding_dims)//2:
            hidden_layer = Dense(encoding_dims[i],
                                 name='middle_layer',
                                 activation='sigmoid')(hidden_layer)
        else:
            hidden_layer = Dense(encoding_dims[i],
                                 activation='sigmoid')(hidden_layer)

    if len(encoding_dims) != 1:
        # reconstruction of the concatenated layer
        hidden_layer = Dense(encoding_dims[0],
                             activation='sigmoid')(hidden_layer)

    # hidden layers
    hidden_layers = []
    for j in range(0, len(input_dims)):
        hidden_layers.append(Dense(encoding_dims[-1]//len(input_dims),
                                   activation='sigmoid')(hidden_layer))
    # output layers
    output_layers = []
    for j in range(0, len(input_dims)):
        output_layers.append(Dense(input_dims[j],
                                   activation='sigmoid')(hidden_layers[j]))

    # autoencoder model
    sgd = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
    model = Model(inputs=input_layers, outputs=output_layers)
    model.compile(optimizer=sgd, loss='binary_crossentropy')
    print(model.summary())

    return model
