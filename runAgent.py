from agent import *
from optimizers import *

from policies import *
from utilities import *
from qfunctions import *

from dnnets import *

from environment import Env

from argparse import ArgumentParser

from tqdm import tqdm

import os
import json
from tensorflow.python.ops import summary_ops_v2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(args):

    # init environment
    env = Env(do_render = args.render).reset_sim()

    # Init tf writer

    writer = tf.summary.create_file_writer('./logs')

    # Compile agent

    # get net config
    with open(args.net_config, "rt") as f:
        net_config = json.load(f)

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
                BuildConvNet,
                **net_config
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
                args.gamma,
                args.alpha,
                a.QModel,
                a.UModel
            )
        )

        a.compile()

        if args.log:
            summary_ops_v2.graph(a.trainFN.outputs[0].graph, step=0)

    # Start action loop

    state = env.start_state[None].astype(np.float32)

    for i in tqdm(range(args.steps), desc="steps"):

        # Act in the environment
        new_state, reward, done, info = env.get_action(a(state))

        new_state = new_state[None].astype(np.float32)

        # Observe transition
        a.observe(new_state, reward)
        a.learn()

        # Update state
        state = new_state

        if done:
            break

if __name__ == "__main__":

    parser = ArgumentParser()

    # Simulation settings
    parser.add_argument(
        "--render", 
        type = lambda x : x.lower()=="true", 
        default = False, 
        help="Set to False to suppress rendering of simulation"
    )

    parser.add_argument(
        "--log", 
        type = lambda x : x.lower()=="true", 
        default = False,
        help = "Set to true to log network graph's (for tensorboard)"
    )

    # Experiments settings

    parser.add_argument(
        "--steps", 
        type = int, 
        default = 300,
        help = "Number of steps for which to run the simulation"
    )

    # Agent settings

    parser.add_argument(
        "--net_config", 
        type = str, 
        default = "BaseNet.json", 
        help="Config file for network topology"
    )

    parser.add_argument(
        "--gamma",
        type = float,
        default = 0.99,
        help = "Discount factor"
    )

    parser.add_argument(
        "--alpha",
        type = float,
        default = 5e-3,
        help = "Learning rate"
    )

    main(parser.parse_args())
