import tensorflow as tf
from tensorflow.keras import Model, Input
import numpy as np

def getGreedyEAgent(
        action_space,
        e
):

    # Get number of action
    n_actions = len(action_space)

    # Make input state tensor
    in_action_vals = Input(batch_shape = (None, n_actions), name='action_vals_GPolicy')

    # Sample conditions to determine if acting randomly or not

    # batch_size = Input(batch_shape = [1], dtype = "int32")

    batch_size = tf.constant(np.array([1]))
    sampled = tf.random.uniform(batch_size, minval=0, maxval=1)

    conditions = tf.cast(
        sampled < tf.constant(e),
        "int32"
    )

    # Define all possible discrete returns
    I = tf.constant(np.eye(n_actions))

    # Compute greedy actions to do if condition is false
    greedy_actions = tf.argmax(
        in_action_vals,
        axis = -1,
        output_type = "int32"
    )

    # Sample random actions to do if condition is true
    random_actions = tf.cast(
        tf.random.uniform(
            batch_size,
            minval = 0,
            maxval = n_actions
        ),
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
