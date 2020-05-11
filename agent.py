from tensorflow.keras import Input, Model

import numpy as np
import random as rand

# %% General Ange Interface

class AgentBase():

    def __init(self):

        self.prev_action = None
        self.prev_state = None

    def __call__(self, state):
        self.prev_action = self.act(state)
        self.prev_state = state

        return self.prev_action

    def act(self, state):
        pass

    def observe(self, new_state, reward):
        pass

class RandomAgent(AgentBase):

    def __init__(self):
        self.last_transition = None

    def act(self, state):
        return rand.randint(0,8)

    def observe(self, new_state, reward):

        self.last_transition = (self.prev_state, self.prev_action, reward, new_state)
        return self.last_transition

# %% RL Agent Interface

class QAgent(AgentBase):

    def __init__(self, state_space, action_space):

        assert isinstance(state_space, (int, tuple)), "State space must be input size"
        assert isinstance(action_space, (int, np.ndarray)), "Action space must be number of actions or array of action identifiers"

        self.state_space = state_space
        self.action_space = np.arange(action_space) if isinstance(action_space, int) else action_space

    def act(self, state):

        # get probability distribution over actions
        probs = np.array(
            self.policy(state)
        )[0]

        probs /= probs.sum() # ensure normalization

        return int(np.random.choice(self.action_space, p=probs)) # sample actions

    def observe(self, new_state, reward):
        self.trainFN(self.prev_state, np.array([self.prev_action])[None], np.array([reward])[None], new_state)
        return self

    def setQ(self, Q):
        self.QModel = Q
        return self

    def setU(self, U):
        self.UModel = U
        return self

    def setPolicy(self, policy):
        self.PolicyModel = policy
        return self

    def compile(self):

        in_state = Input(batch_shape = (None, *self.state_space, 3),name='input_state_agent')

        self.U = Model(
            in_state,
            self.UModel.call(
                self.QModel.call(
                    in_state
                )
            ),
            name='UModel'
        )

        self.policy = Model(
            in_state,
            self.PolicyModel.call(
                self.QModel.call(
                    in_state
                )
            ),
            name='PolicyModel'
        )

    def setTrain(self, trainFN):
        self.trainFN = trainFN
        return self

