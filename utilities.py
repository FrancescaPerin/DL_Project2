from tensorflow.keras import Model, Input
from tensorflow import stop_gradient, reduce_max, reduce_sum

def getOffPolicyMaxUtil(action_space):

    in_action_vals = Input(batch_shape = (None, len(action_space)), name='action_vals_MaxUtility')

    out = reduce_max(
        in_action_vals,
        axis=-1,
        keepdims = True
    )

    return Model(in_action_vals, out, name='MaxUtility_Model')

def getExpectedSarasUtil(action_space, PolicyModel):

    in_action_vals = Input(batch_shape = (None, len(action_space)), name='action_vals_ExSarsaUtility')
    in_action_probs = stop_gradient(
        PolicyModel(
            in_action_vals
        )
    )

    
    out = reduce_sum(in_action_vals*in_action_probs, axis=1, keepdims = True)

    return Model(in_action_vals, out, name='ExSarsaUtility_Model')
