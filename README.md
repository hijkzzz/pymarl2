# MARL Tricks
Our code for [Hyperparameter Tricks in Multi-Agent Reinforcement Learning:An Empirical Study](). We implemented and standardized the hyperparameters of the SOTA MARL algorithms.

## Python MARL framework

PyMARL is [WhiRL](http://whirl.cs.ox.ac.uk)'s framework for deep multi-agent reinforcement learning and includes implementations of the following algorithms:

Value-based Methods:

- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) 
- [**IQL**: Independent Q-Learning](https://arxiv.org/abs/1511.08779)
- [**QTRAN**: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)
- [**MAVEN**: MAVEN: Multi-Agent Variational Exploration](https://arxiv.org/abs/1910.07483)
- [**Qatten**: Qatten: A general framework for cooperative multiagent reinforcement learning](https://arxiv.org/abs/2002.03939)
- [**QPLEX**: Qplex: Duplex dueling multi-agent q-learning](https://arxiv.org/abs/2008.01062)
- [**WQMIX**: Weighted QMIX: Expanding Monotonic Value Function Factorisation](https://arxiv.org/abs/2006.10800)

Actor Critic Methods:

- [**COMA**: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [**VMIX**: Value-Decomposition Multi-Agent Actor-Critics](https://arxiv.org/abs/2007.12306)
- [**FacMADDPG**: Deep Multi-Agent Reinforcement Learning for Decentralized Continuous Cooperative Control](https://arxiv.org/abs/2006.10800)
- [**LICA**: Deep Multi-Agent Reinforcement Learning for Decentralized Continuous Cooperative Control](https://arxiv.org/abs/2007.02529)
- [**DOP**: Off-Policy Multi-Agent Decomposed Policy Gradients](https://arxiv.org/abs/2007.12322)

PyMARL is written in PyTorch and uses [SMAC](https://github.com/oxwhirl/smac) as its environment.

## Installation instructions

Install Python packages
```shell
# require Anaconda 3 or Miniconda 3
bash install_dependecies.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

## Run an experiment 

```shell
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=corridor
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

## Run parallel experiments:
```shell
# bash run.sh config_name map_name_list (threads_num arg_list gpu_list experinments_num)
bash run.sh qmix corridor 2 epsilon_anneal_time=500000 0,1 5
```

`xxx_list` is separated by `,`.

All results will be stored in the `Results` folder and named with `map_name`.

## Force all processes to exit

```shell
# all python and game processes of current user will quit.
bash clean.sh
```

