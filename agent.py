from tensorflow.keras import Input, Model

import numpy as np
import random as rand

# %% General Ange Interface

class AgentBase():

    def __init__(self, state_space, action_space):

        assert isinstance(state_space, (int, tuple)), "State space must be input size"
        assert isinstance(action_space, (int, np.ndarray)), "Action space must be number of actions or array of action identifiers"

        self.prev_action = None
        self.prev_state = None

        self.state_space = state_space if isinstance(state_space, tuple) else (state_space, state_space)
        self.action_space = np.arange(action_space) if isinstance(action_space, int) else action_space

    def __call__(self, state):
        self.set_prev_action(self.act(state))
        self.set_prev_state(state)

        return self.prev_action

    def act(self, state):
        pass

    def learn(self):
        pass

    def observe(self, new_state, reward):

        self.set_transition((self.prev_state, self.prev_action, reward, new_state))
        return self.last_transition

    # getters
    def get_state_space(self):
        return self.state_space

    def get_action_space(self):
        return self.action_space

    # setters

    def set_transition(self, new_transition):
        self.last_transition = new_transition

    def set_prev_action(self, prev_action):
        self.prev_action = prev_action

    def set_prev_state(self, prev_state):
        self.prev_state = prev_state

class RandomAgent(AgentBase):

    def __init__(self):
        self.last_transition = None

    def act(self, state):
        return rand.randint(0,8)

# %% RL Agent Interface

class QAgent(AgentBase):

    def __init__(self, state_space, action_space):
        super().__init__(state_space, action_space)

    def act(self, state):

        # get probability distribution over actions
        probs = np.array(
            self.policy(state)
        )[0]

        probs /= probs.sum() # ensure normalization

        return int(np.random.choice(self.action_space, p=probs)) # sample actions

    def learn(self, t=None):

        # Get transition
        if t is None:
            t = self.last_transition

        s1, a, r, s2 = t

        # Account for simple int/float used as a and r
        if not isinstance(a, np.ndarray):
            a = np.array([a])[None]

        if not isinstance(r, np.ndarray):
            r = np.array([r])[None]

        # Pass transition to train op
        self.trainFN(
            s1, 
            a, 
            r, 
            s2
        )

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


# %% Decorators Base class

class AgentDecorator(AgentBase):

    def __init__(self, decorated):
        assert isinstance(decorated, AgentBase), "AgentDecorator can only decorate instance of AgentBase"
        self.decorated = decorated

    def __call__(self, state):
        return self.decorated(state)

    def act(self, state):
        return self.decorated.act(state)

    def learn(self, t):
        return self.decorated.learn(t)

    def observe(self, new_state, reward):
        return self.decorated.observe(new_state, reward)

    # getters

    def get_state_space(self):
        return self.decorated.get_state_space()

    def get_action_space(self):
        return self.decorated.get_action_space()

    # setters

    def set_transition(self, new_transition):
        self.decorated.set_transition(new_transition)

    def set_prev_action(self, prev_action):
        self.decorated.set_prev_action(prev_action)

    def set_prev_state(self, prev_state):
        self.decorated.set_prev_state(prev_state)


# %% Decorators Implementations

class RandomReplay(AgentDecorator):

    def __init__(self, decorated, memory_span):

        super().__init__(decorated)

        self.memory_span = memory_span

        # init table of pas transition

        self.s1 = np.zeros([memory_span, *self.get_state_space(), 3])
        self.a = np.zeros([memory_span, 1], "int64")
        self.r = np.zeros([memory_span, 1])
        self.s2 = np.zeros([memory_span, *self.get_state_space(), 3])

        # init counter
        self.idx = 0

        # init index list
        self.order = list(range(memory_span))

    def __shuffle(self):

        self.idx = 0
        rand.shuffle(self.order)

        self.s1 = self.s1[self.order]
        self.a = self.a[self.order]
        self.r = self.r[self.order]
        self.s2 = self.s2[self.order]

    def observe(self, new_state, reward):

        s1, a, r, s2 = super().observe(new_state, reward)

        # store in random spots of table
        self.s1[self.idx % self.memory_span] = s1
        self.a[self.idx % self.memory_span] = a
        self.r[self.idx % self.memory_span] = r
        self.s2[self.idx % self.memory_span] = s2

        self.idx += 1

    def learn(self, _ = None):

        # loop every memory_span iterations through memory 
        if self.idx == self.memory_span:    
            self.__shuffle()
            super().learn((self.s1, self.a, self.r, self.s2))



