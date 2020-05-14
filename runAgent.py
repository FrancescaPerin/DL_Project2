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

    # get agent config
    with open(args.rl_config, "rt") as f:
        rl_config = json.load(f)

    # get net config
    with open(args.net_config, "rt") as f:
        net_config = json.load(f)

    with writer.as_default():
        state_space = (96, 96)
        action_space = np.arange(9)

        # Construct Q Function

        QFun = getQNetFunc(
            state_space,
            action_space,
            BuildConvNet,
            **net_config
        )

        # Consturct Policy


        for key, val in rl_config["pi"]["kwargs_eval"].copy().items():
            rl_config["pi"]["kwargs_eval"][key] = eval(val)

        policy = eval(rl_config["pi"]["type"])(
            action_space,
            **rl_config["pi"]["kwargs"],
            **rl_config["pi"]["kwargs_eval"]
        )

        # Construct Utility

        for key, val in rl_config["util"]["kwargs_eval"].copy().items():
            rl_config["util"]["kwargs_eval"][key] = eval(val)

        UFun = eval(rl_config["util"]["type"])(
            action_space,
            **rl_config["util"]["kwargs"],
            **rl_config["util"]["kwargs_eval"]
        )


        a = QAgent(
            state_space = state_space,
            action_space = action_space,
        ).setQ(
            QFun
        ).setU(
            UFun
        ).setPolicy(
            policy
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
        "--rl_config",
        type = str,
        default = "BaseAgent.json",
        help = "Config file with utility and policy to use"
    )

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
