# Reinforcement Learning in a Racing Game

This repository was created in order to work for an assignment in the Deep Learning course provided by the University of Groningen. We used the OpenAI Car Racing v.0 environment and implemented a DQN approach in order to make the car(agent) learn the track and  race successfully.

## Set-up
- Anaconda Python prepackaged distribution <br />
- Open AI Gym 0.17.1 v. <br />
- python 3.6.10 v. <br />
- Tensorflow 2.1.0 v. <br />

The available parameteres are --render, --alpha, --gamma, --net_config, --rl_config, --steps, --save_to


## Issues
Regarding the parameters, --render must always be True because there is a bug and if the environment is not rendered, then it will crash when generating a new track.

