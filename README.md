# RIIT
Our open-source code for [RIIT: Rethinking the Importance of Implementation Tricks in Multi-AgentReinforcement Learning](https://arxiv.org/abs/2102.03479). 

## Tricks in Multi-agent Reinforcement Learning
There are so many tricks in the RL, such as:
- Value function clipping (clip max Q values for QMIX)
- Reward scaling
- Orthogonal initialization and layer scaling
- **Adam** 
- learning rate annealing
- Reward Clipping
- Observation Normalization
- Gradient Clipping
- **Large Batch Size**
- **N-step Returns(including GAE and Q($\lambda$))**
- **Rollout Process Number**
- **$\epsilon$-greedy annealing steps**
- Death Agent Masking

**Reference**
- Implementation Matters in Deep RL: A Case Study on PPO and TRPO
- The Surprising Effectiveness of MAPPO in Cooperative, Multi-Agent Games

### Our QMIX
Using just a few of them (Bold texts), we enabled QMIX to solve almost all of SMAC's tasks. 


| Senarios       | Difficulty |      QMIX      |               OurQMIX              |
|----------------|:----------:|:--------------:|:----------------------------------:|
| 5m_vs_6m     |    Hard    |      84%      |           **90\%**          |
| 3s_vs_5z     |    Hard    |      96%      |          **100\%**          |
| bane_vs_bane |    Hard    |**100\%**|          **100\%**          |
| 2c_vs_64zg   |    Hard    |**100\%**|          **100\%**          |
| corridor       | Super Hard |       0%      |          **100\%**          |
| MMM2           | Super Hard |      98%      |          **100\%**          |
| 3s5z_vs_3s6z | Super Hard |       3%      |**85\%**(Number of Envs = 4) |
| 27m_vs_30m   | Super Hard |      56%      |          **100\%**          |
| 6h_vs_8z     | Super Hard |       0%      |  **93\%**($\lambda$ = 0.3)  |


## Our Benchmarks
Afterwards, we finetune and standardize the hyperparameters of numerous QMIX variants, and find that QMIX achieve the SOTA.

| Algo.     | Type |  MNS |   5m_vs_6m  | 3s5z_vs_3s6z |    corridor    |   6h_vs_8z  |      MMM2      |      Predator-Prey     |
|-----------|:----:|:----:|:-------------:|:--------------:|:--------------:|:-------------:|:--------------:|:-----------:|
| OurQMIX   |  VB  |  41K | **90%** |  **75%** | **100%** |      84%     | **100%** | **40** |
| OurVDNs   |  VB  |  0K  | **90%** |      43%      |      98%      | **87%** |      96%      |      39     |
| OurQatten |  VB  |  58K | **90%** |      62%      | **100%** |      68%     | **100%** |      -      |
| OurQPLEX  |  VB  | 152K | **90%** |      68%      |      96%      |      78%     | **100%** |      39     |
| OurWQMIX  |  VB  | 247K | **90%** |       6%      |      96%      |      78%     |      23%      |      39     |
| OurLICA   |  PG  | 208K |      53%     |       0%      |       0%      |      4%      |       0%      |      30     |
| OurDOP    |  PG  | 122K |      9%      |       0%      |       0%      |      1%      |       0%      |      32     |
| RIIT      |  PG  |  69K |      67%     |  **75%** | **100%** |      19%     | **100**% |      38     |


## PyMARL

PyMARL is [WhiRL](http://whirl.cs.ox.ac.uk)'s framework for deep multi-agent reinforcement learning and includes implementations of the following algorithms:

Value-based Methods:

- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) 
- [**IQL**: Independent Q-Learning](https://arxiv.org/abs/1511.08779)
- [**QTRAN**: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)
- [**Qatten**: Qatten: A general framework for cooperative multiagent reinforcement learning](https://arxiv.org/abs/2002.03939)
- [**QPLEX**: Qplex: Duplex dueling multi-agent q-learning](https://arxiv.org/abs/2008.01062)
- [**WQMIX**: Weighted QMIX: Expanding Monotonic Value Function Factorisation](https://arxiv.org/abs/2006.10800)

Actor Critic Methods:

- [**COMA**: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [**VMIX**: Value-Decomposition Multi-Agent Actor-Critics](https://arxiv.org/abs/2007.12306)
- [**FacMADDPG**: Deep Multi-Agent Reinforcement Learning for Decentralized Continuous Cooperative Control](https://arxiv.org/abs/2003.06709)
- [**LICA**: Learning Implicit Credit Assignment for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2007.02529)
- [**DOP**: Off-Policy Multi-Agent Decomposed Policy Gradients](https://arxiv.org/abs/2007.12322)
- [**RIIT**: RIIT: Rethinking the Importance of Implementation Tricks in Multi-AgentReinforcement Learning](https://arxiv.org/abs/2102.03479)

### Installation instructions

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

### Command Tools

**Run an experiment**

```shell
# For SMAC
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=corridor
```

```shell
# For Cooperative Predator-Prey
python3 src/main.py --config=qmix_prey --env-config=stag_hunt with env_args.map_name=stag_hunt
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

**Run n parallel experiments**

```shell
# bash run.sh config_name map_name_list (threads_num arg_list gpu_list experinments_num)
bash run.sh qmix corridor 2 epsilon_anneal_time=500000 0,1 5
```

`xxx_list` is separated by `,`.

All results will be stored in the `Results` folder and named with `map_name`.

**Kill all game processes**

```shell
# all python and game processes of current user will quit.
bash clean.sh
```

## Cite
```
@article{hu2021riit,
      title={RIIT: Rethinking the Importance of Implementation Tricks in Multi-Agent Reinforcement Learning}, 
      author={Jian Hu and Siyang Jiang and Seth Austin Harding and Haibin Wu and Shih-wei Liao},
      year={2021},
      eprint={2102.03479},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

