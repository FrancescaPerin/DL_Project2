import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Lambda, Softmax
import numpy as np

def _get_rand_int(maxval):

    def _get_rand(t):
        return tf.random.uniform(tf.shape([1]), minval=0, maxval=maxval)

    return _get_rand

def getGreedyEAgent(
        action_space,
        e
):

    # Get number of action
    n_actions = len(action_space)

    # Define constants

    # Define all possible discrete returns
    I = tf.constant(np.eye(n_actions))

    # Make input state tensor
    in_action_vals = Input(batch_shape = (None, n_actions), name='action_vals_GPolicy')

    # Sample conditions to determine if acting randomly or not

    sampled = Lambda(
        _get_rand_int(1)
    )(in_action_vals)

    conditions = tf.cast(
        sampled < tf.constant(e),
        "int32"
    )

    # Compute greedy actions to do if condition is false
    greedy_actions = tf.argmax(
        in_action_vals,
        axis = -1,
        output_type = "int32"
    )

    # Sample random actions to do if condition is true
    random_actions = tf.cast(
        Lambda(
            _get_rand_int(n_actions)
        )(in_action_vals),
        "int32"
    )

    # Apply condition
    greedy_e_actions = conditions*random_actions + (tf.constant(1, dtype="int32") - conditions)*greedy_actions

    # Get respective one-hot discrete distributions
    action_dists = tf.cast(
        tf.gather(
            I,
            greedy_e_actions
        ),
        "float32"
    )

    return Model(in_action_vals, action_dists, name='GreedyPolicyModel')

def getSoftmaxAgent(
        action_space,
        temp
):

    # Get number of action
    n_actions = len(action_space)

    # Make input state tensor
    in_action_vals = Input(batch_shape = (None, n_actions), name='action_vals_GPolicy')

    return Model(in_action_vals, Softmax()(in_action_vals/temp), name='SoftmaxPolicyModel')

def getPursuitAgent(
        action_space,
        beta,
        temp
):

    # Get number of action
    n_actions = len(action_space)

    # Define constants

    # Define all possible discrete returns
    I = tf.constant(np.eye(n_actions))

    # Rate of increment/decrement
    beta = tf.constant(beta)

    # Make input state tensor
    in_action_vals = Input(batch_shape = (None, n_actions), name='action_vals_GPolicy')

    # Get action probabilities directly proportional to Q(·)
    p = Softmax()(in_action_vals/temp)

    # Get greedy action
    greedy_actions = tf.argmax(
        in_action_vals,
        axis = -1,
        output_type = "int32"
    )

    # Increment value of greedy action

    increment = tf.cast(
        tf.gather(
            I,
            greedy_actions
        ), 
        "float32"
    ) * (1-p) * beta

    decrement = ( 1 - tf.cast(
        tf.gather(
            I,
            greedy_actions
        ),
        "float32")
    ) * (-p) * beta

    
    return Model(in_action_vals, p+increment+decrement, name='PursuitPolicyModel')
