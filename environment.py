import numpy as np
import gym
from gym.wrappers import Monitor

import types

from enum import Enum

PUNISHMENT_RESTART = -100.0 # punishment for going outside tracs

class ActionSpace:

    class Dir:
        LEFT = np.array([-1,0,0]) # 0
        RIGHT = np.array([1,0,0]) # 1
        STRAIGHT = np.zeros(3) # 2

    class Acc:
        INCREASE = np.array([0,1,0]) # 0
        BREAK = np.array([0,0,0.5]) # 1
        NONE = np.zeros(3) # 2

    Dirs = [Dir.LEFT, Dir.RIGHT, Dir.STRAIGHT]
    Accs = [Acc.INCREASE, Acc.BREAK, Acc.NONE]

    def __call__(self, direction, acc = None):
        '''
        Produces full action array. Accepts:

        - (Dir, Acc), returns the sum of the respective arrays
        - (int, int), convert each int to Dir and Acc respectively and performs the aforementioned operation
        - (int), converts the int (between 0 and 8) into a specific combination of Dir and Acc (every 3 directions switches accelleration, base 3 2 digit number fs)
        '''

        if acc is None:
            acc = direction//3
            direction = direction%3

        if isinstance(direction, int):
            direction = self.Dirs[direction]

        if isinstance(acc, int):
            acc = self.Accs[acc]

        return direction + acc

    def __len__(self):
        return 9

class Env(gym.Env):

    def __new__(cls, do_render=False):

        # Load Car Racing game
        obj = gym.make('CarRacing-v0')

        if not do_render:
            obj = Monitor(obj, '.env_stats', video_callable=False, force=True)

        # Store parameters
        obj.do_render = do_render
        obj.actionInterpreter = ActionSpace()

        # Set methods (do smth about this later)
        obj.reset_sim = types.MethodType(Env.reset_sim, obj)
        obj.get_action = types.MethodType(Env.get_action, obj)

        return obj

    def reset_sim(self):
        self.reset()

        # Zoom in
        for i in range(50):
            self.step(np.zeros(3))

        self.start_state, *_ = self.step(np.zeros(3))

        return self

    def get_action(self, *action):

        '''
        Performs action in environment (and renders if necessary)
        '''

        # Perform actions
        state, reward, done, info = self.step(self.actionInterpreter(*action))

        # Compute if agent cannot see track anymore

        temp = state.reshape(-1, state.shape[-1]) # flatten first two dims
        info["sees_track"] = ((temp>100) * (temp < 110)).all(-1).any() # check if gray is in the picture

        # If outside tracks then reset game and give very negative reward
        if not info["sees_track"]:
            reward = PUNISHMENT_RESTART
            self.reset_sim()
            state, *_ = self.step(self.actionInterpreter(*action))

        # Render if specified
        if self.do_render:
            self.render()

        return state, reward, info

