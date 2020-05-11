from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, ZeroPadding2D
from tensorflow.keras import Model, Input

# %% Networks models

def BuildConvNet(

    state_space,
    action_space,

    filters = [3, 5, 15, 25, 50],
    kernels = [5, 4, 3, 3, 2],
    pooling = [(3,3), (3,3), None, (2,2), None],
    zero_padding = None,
    activations_conv = 'relu',
    strides_conv = (1,1),
    strides_pool = None,

    units = [500, 200, 100],
    activations = 'relu',

):

    '''
    Basic Network for agent using simple Q-Learning

    :param [int] filters: list of number of filters per each convolutional layer
    :param [int] kernels: list of kernel sizes for each convolutional layer
    :param [tuple] pooling: list of None of pooling strides to apply after each convolutional layer
    :param [str] actications_conv: list of activation functions for convolutional layers
    :param [tuple] strides: list of strides for each convolutional layer
    :param [int] units: sizes of dense layers
    :param [str] activations: list of activations for dense layers
    '''
    if isinstance(activations_conv, str):
        activations_conv = [activations_conv]*len(filters)

    if isinstance(activations, str):
        activations = [activations]*len(units)

    if len(strides_conv) != len(filters) and len(strides_conv) == 2:
        strides_conv = [strides_conv]*len(filters)

    if strides_pool is None or (len(strides_pool) != len(filters) and len(strides_pool) == 2):
        strides_pool = [strides_pool]*len(filters)

    if pooling is None or isinstance(pooling, tuple):
        pooling = [pooling]*len(filters)

    if zero_padding is None or isinstance(zero_padding, tuple):
        zero_padding = [zero_padding]*len(filters)

    assert (isinstance(filters, (list, tuple)) and isinstance(kernels, (list, tuple)) and isinstance(activations_conv, (list, tuple)) 
        and isinstance(strides_conv, (list, tuple)) and isinstance(strides_pool, (list, tuple)) 
            and isinstance(pooling, (list, tuple)) and isinstance(zero_padding, (list, tuple)), "Type not understood")
    assert isinstance(units, (list, tuple)) and isinstance(activations, (list, tuple)), "Tupe not understood"
    assert len(filters) > 0, "Must have at least one convolutional layer"
    assert len(filters) == len(kernels), "Number of filter sizes and kernel sizes must be the same"
    assert len(filters) == len(activations_conv), "Number of activation functions and filters must be the same"
    assert len(filters) == len(pooling), "Number of pooling layers and filters must be the same"
    assert len(units) == len(activations), "Number of units and activations for linear layers must coincide"

    # Construct first part of the network

    in_x = Input(batch_shape = (None, *state_space, 3))
    x = in_x

    for i in range(len(filters)):
        x = Conv2D(
            filters = filters[i],
            kernel_size = kernels[i],
            strides = strides_conv[i],
            activation = activations_conv[i],
            data_format = 'channels_last'
        )(x)

        if pooling[i] is not None:
            x = MaxPool2D(
                pool_size = pooling[i],
                strides = strides_pool[i],
                data_format = 'channels_last'
            )(x)

        if zero_padding[i] is not None:
            x = ZeroPadding2D(
                padding=zero_padding[i],
                data_format = 'channels_last'
            )(x)

    x = Flatten()(x)

    # Construct dense layers
    for i in range(len(units)):
        x = Dense(
            units = units[i],
            activation = activations[i],
        )(x)

    # Construct output layer

    return in_x, Dense(
        units = len(action_space),
        activation = 'linear',
    )(x)
