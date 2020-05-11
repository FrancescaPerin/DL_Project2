from tensorflow.keras import Model

# %% Interface

def getQNetFunc(
        state_space,
        action_space,
        NetCLS, **kwargs
):
    in_x, out = NetCLS(state_space, action_space, **kwargs)
    return Model(in_x, out, name='QNetModel')


