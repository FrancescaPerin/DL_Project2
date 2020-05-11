import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, Model
from tensorflow.keras.backend import function as KFunction

# %% Custom training

def getSimpleQLOptim(
        gamma,
        alpha,
        QModel,
        UModel,
):

    # Init constants
    gamma = tf.constant(gamma)

    # Init inputs
    in_s_1 = Input(batch_shape = QModel.input.get_shape(), name='current_state_optim')
    in_a = Input(batch_shape = (None, 1), name='action_optim', dtype = "int64")
    in_r = Input(batch_shape = (None, 1), name='reward_optim')
    in_s_2 = Input(batch_shape = QModel.input.get_shape(), name='next_state_optim')

    # Get node for loss
    Q_sa = tf.gather(
        QModel(
            in_s_1
        ),
        in_a,
        batch_dims = 1
    )

    # Get node with U(s_2)
    U_s2 = UModel(
        QModel(
            in_s_2
        )
    )

    # Get approximation of Q_sa
    Q_sa_pred = in_r + gamma*U_s2

    # Get loss
    loss_val = (Q_sa - Q_sa_pred)**2

    loss_model = Model([in_s_1, in_a, in_r, in_s_2], loss_val, name="loss_model")

    # Init optimizer
    optim = Adam(lr = alpha, name='Adam')

    # Get operator for updating weights
    # return KFunction(
    #     inputs = [in_s_1, in_a, in_r, in_s_2],
    #     outputs = [loss_val],
    #     updates = optim.get_updates(
    #         params = loss_model.trainable_weights,
    #         loss = loss_val
    #     )
    # )

    def trainFN(s1, a, r, s2):

        with tf.GradientTape() as tape:

            loss = loss_model([s1, a, r, s2])
            gradients = tape.gradient(loss, loss_model.trainable_weights)

        optim.apply_gradients(zip(gradients, loss_model.trainable_weights))

    return trainFN
