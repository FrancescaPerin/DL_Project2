# Reinforcement Learning in a Racing Game

This repository was created in order to work for an assignment in the Deep Learning course provided by the University of Groningen. We used the OpenAI Car Racing v.0 environment and implemented a DQN approach in order to make the car(agent) learn the track and  race successfully.

## Issues
Regarding the parameters, --render must always be True because there is a bug and if the environment is not rendered, then it will crash when generating a new track.

## Set-up
- Anaconda Python prepackaged distribution <br />
- Open AI Gym 0.17.1 v. <br />
- python 3.6.10 v. <br />
- Tensorflow 2.1.0 v. <br />

## Running the code

The experiments can be run by the single script `runAgent.py` with the following command

```
python3 runAgent.py --<argument> <argument value>
```

The available arguments are:
```--render```: Always set to True
```--log```:Set to true to log network graph's (for tensorboard)
```--alpha```:Learning rate
```--gamma```:Discount factor of ANN
```--memory_span```:Memory span for experience replay
```--net_config```:Config file for network topology
```--rl_config```:Config file with utility and policy to use
```--steps```:Number of steps for which to run the simulation
```--save_to```:Name of file in which to save image with results and the json of the data and pars

## Config Files 

`BaseNet.json` and `AlexNet.json` are two given json config files used to define the DQN's topology

## RL Config files

All other json files are used to define configuration of the agent (mainly the policy and the learning algorithm). `BaseAgent.json`, `SoftmaxAgent.json`, and `PursuitAgent.json` use, respectively, e-Greedy, Softmax, and Pursuit exploration, with Q-Learning. `SoftmaxSARSAAgent.json` and `PursuitSARSAAgent.json` use Softmax and Pursuit, respectively, with Expected SARSA as learning algorithm.

## Experiments commands
```
python3 runAgent.py --render True --steps 10000 --net_config AlexNet.json --rl_config BaseAgent.json --alpha 0.01 --memory_span 20 --save_to 'Experiments/Greedy_MaxUtily_lr01_run1';
```
```
python3 runAgent.py --render True --steps 10000 --net_config AlexNet.json --rl_config SoftmaxAgent.json --alpha 0.01 --memory_span 20 --save_to 'Experiments/Softmax_MaxUtil_lr_01_run1';
```
```
python3 runAgent.py --render True --steps 10000 --net_config AlexNet.json --rl_config PursuitAgent.json --alpha 0.01 --memory_span 20 --save_to 'Experiments/Pursuit_MaxUtil_lr_01_run1';
```
```
python3 runAgent.py --render True --steps 10000 --net_config AlexNet.json --rl_config SoftmaxSARSAAgent.json --alpha 0.0001 --memory_span 20 --save_to 'Experiments/Softmax_SARSA_lr_0001_run1';
```
```
python3 runAgent.py --render True --steps 10000 --net_config AlexNet.json --rl_config PursuitSARSAAgent.json --alpha 0.0001 --memory_span 20 --save_to 'Experiments/Pursuit_SARSA_lr_0001_run1';
```