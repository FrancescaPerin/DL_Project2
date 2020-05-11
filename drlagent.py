from agent import QAgent

from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

# %% Networks models

class Predictor:

    def __init__(self, state_space, action_space):
        pass

class ConvNet(Predictor):

    def __init__(

        self,

        state_space,
        action_space,

        filters = [3, 5, 15, 25, 50],
        kernels = [5, 4, 3, 3, 2],
        pooling = [(3,3), (3,3), None, (2,2), None],
        activations_conv = 'relu',
        strides = (1,1),

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

        super().__init__(state_space, action_space)

        if isinstance(activations_conv, str):
            activations_conv = [activations_conv]*len(filters)

        if isinstance(activations, str):
            activations = [activations]*len(units)

        if len(strides) != len(filters) and len(strides) == 2:
            strides = [strides]*len(filters)

        if pooling is None or isinstance(pooling, tuple):
            pooling = [pooling]*len(filters)

        assert isinstance(filters, (list, tuple)) and isinstance(kernels, (list, tuple)) and isinstance(activations_conv, (list, tuple)) and isinstance(strides, (list, tuple)) and isinstance(pooling, (list, tuple)), "Type not understood"
        assert isinstance(units, (list, tuple)) and isinstance(activations, (list, tuple)), "Tupe not understood"
        assert len(filters) > 0, "Must have at least one convolutional layer"
        assert len(filters) == len(kernels), "Number of filter sizes and kernel sizes must be the same"
        assert len(filters) == len(activations_conv), "Number of activation functions and filters must be the same"
        assert len(filters) == len(pooling), "Number of pooling layers and filters must be the same"
        assert len(units) == len(activations), "Number of units and activations for linear layers must coincide"

        # Construct first part of the network

        x = state_space

        for i in range(len(filters)):
            x = Conv2D(
                filters = filters[i],
                kernel_size = kernels[i],
                strides = strides[i],
                activation = activations_conv[i],
                data_format = 'channels_last'
            )(x)

            if pooling[i] is not None:
                x = MaxPool2D(
                    pool_size = pooling[i]
                )(x)

        x = Flatten()(x)

        # Construct dense layers
        for i in range(len(units)):
            x = Dense(
                units = units[i],
                activation = activations[i],
            )(x)

        # Construct output layer

        self.action_values = Dense(
            units = len(action_space),
            activation = 'linear',
        )(x)

# %% Agent Models

class SimpleNeuralAgent(QAgent):

    def __init__(

        self,
        state_space, action_space,

        NetCLS,

        **kwargs
    ):

        '''
        Basic Network for agent using simple Q-Learning

        :param class NetCLS: network to use, must be subclass of Predictor
        :param dict kwargs: parameters to pass to network class
        '''

        super().__init__(state_space, action_space)

        # Construct first part of the network

        self.action_values = NetCLS(self.state_space, self.action_space, **kwargs).action_values
        self.model = None

    def compile(
        self,
        optim_pars = (0.0005, 0.9, 0.999),
        loss = 'mse',
        name = 'naiveQL',
    ):

        # Make model

        self.model = Model(inputs = self.state_space, outputs = self.action_values, name = 'naiveQL')

        self.model.compile(
            loss = loss,
            optimizer = Adam(*optim_pars),
            metrics = ['mae']
        )

        return self
