from agent import *
from optimizers import *

from policies import *
from utilities import *
from qfunctions import *

from dnnets import *

from environment import Env

from argparse import ArgumentParser

import os
from tensorflow.python.ops import summary_ops_v2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(args):

    # init environment
    env = Env(do_render = args.render).reset_sim()

    # Init tf writer

    writer = tf.summary.create_file_writer('./logs')

    # Compile agent

    with writer.as_default():
        state_space = (96, 96)
        action_space = np.arange(9)

        a = QAgent(
            state_space = state_space,
            action_space = action_space,
        ).setQ(
            getQNetFunc(
                state_space,
                action_space,
                BuildConvNet
            )
        ).setU(
            getOffPolicyMaxUtil(
                action_space
            )
        ).setPolicy(
            getGreedyEAgent(
                action_space,
                0.2
            )
        )

        a.setTrain(
            getSimpleQLOptim(
                0.98,
                0.25,
                a.QModel,
                a.UModel
            )
        )

        a.compile()

        if args.log:
            summary_ops_v2.graph(a.trainFN.outputs[0].graph, step=0)

    # Start action loop

    state = env.start_state[None].astype(np.float32)

    for i in range(args.steps):

        # Act in the environment
        new_state, reward, *_ = env.get_action(a(state))

        new_state = new_state[None].astype(np.float32)

        # Observe transition
        a.observe(new_state, reward)
        a.learn()

        # Update state
        state = new_state

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--render", type = lambda x : x.lower()=="true", default = False)
    parser.add_argument("--log", type = lambda x : x.lower()=="true", default = False)
    parser.add_argument("--pause", type = float, default = 0.1)
    parser.add_argument("--steps", type = int, default = 300)

    main(parser.parse_args())
