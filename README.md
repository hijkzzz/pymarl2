# RIIT
Open-source code for [Rethinking the Implementation Tricks and Monotonicity Constraint in Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2102.03479). Our goal is to call for a fair comparison of the performance of MARL algorithms.

## Code-level Optimizations
There are so many code-level tricks in the  Multi-agent Reinforcement Learning (MARL), such as:
- Value function clipping (clip max Q values for QMIX)
- Value Normalization
- Reward scaling
- Orthogonal initialization and layer scaling
- **Adam** 
- learning rate annealing
- Reward Clipping
- Observation Normalization
- Gradient Clipping
- **Large Batch Size**
- **N-step Returns(including GAE($\lambda$) and Q($\lambda$))**
- **Rollout Process Number**
- **$\epsilon$-greedy annealing steps**
- Death Agent Masking

**Related Works**
- Implementation Matters in Deep RL: A Case Study on PPO and TRPO
- What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study
- The Surprising Effectiveness of MAPPO in Cooperative, Multi-Agent Games

### Finetuned-QMIX
Using a few of tricks above (Bold texts), we enabled QMIX to solve almost all of SMAC's scenarios (finetuned QMIX for each scenarios).


| Senarios       | Difficulty |      QMIX (batch_size=128)      |               Finetuned-QMIX              |
|----------------|:----------:|:--------------:|:----------------------------------:|
| 8m    |    Easy    |      -      |           **100\%**          |
| 2c_vs_1sc     |    Easy    |      -      |          **100\%**          |
| 2s3z |    Easy    |-|          **100\%**          |
| 1c3s5z   |    Easy    |-|          **100\%**          |
| 3s5z       |  Easy |      -      |          **100\%**          |
| 8m_vs_9m           |  Hard |      84%      |          **100\%**          |
| 5m_vs_6m     |    Hard    |      84%      |           **90\%**          |
| 3s_vs_5z     |    Hard    |      96%      |          **100\%**          |
| bane_vs_bane |    Hard    |**100\%**|          **100\%**          |
| 2c_vs_64zg   |    Hard    |**100\%**|          **100\%**          |
| corridor       | Super Hard |       0%      |          **100\%**          |
| MMM2           | Super Hard |      98%      |          **100\%**          |
| 3s5z_vs_3s6z | Super Hard |       3%      |**85\%**(Number of Envs = 4) |
| 27m_vs_30m   | Super Hard |      56%      |          **100\%**          |
| 6h_vs_8z     | Super Hard |       0%      |  **93\%**($\lambda$ = 0.3)  |


## Re-Evaluation
Afterwards, we re-evaluate numerous QMIX variants with normalized the tricks (a **genaral** set of hyperparameters), and find that QMIX achieves the SOTA. 

| Scenarios      | Difficulty     |   Value-based  |                |                 |                |                |  Policy-based  |        |                |
|----------------|----------------|:--------------:|:--------------:|:---------------:|:--------------:|:--------------:|:--------------:|:------:|:--------------:|
|                |                | QMIX           |      VDNs      |      Qatten     |      QPLEX     |      WQMIX     |      LICA      |   DOP  |       RIIT      |
| 2c\_vs\_64zg   | Hard           | **100\%** | **100\%** |  **100\%** | **100**\% |      93\%      | **100**\% |  56\%  | **100**\% |
| 8m\_vs\_9m     | Hard           | **100\%** | **100\%** |  **100\%** |      95\%      |      90\%      |      48\%      |  18\%  |      95\%      |
| 3s\_vs\_5z     | Hard           | **100\%** | **100\%** | **100** \% | **100**\% | **100**\% |       3\%      |   0\%  |      96\%      |
| 5m\_vs\_6m     | Hard           | **90\%**  |  **90\%** |  **90\%**  |  **90\%** |  **90\%** |      53\%      |   9\%  |      67\%      |
| 3s5z\_vs\_3s6z | Super-Hard         | **75\%**  |      43\%      |       62\%      |      68\%      |       6\%      |       0\%      |   0\%  |  **75\%** |
| corridor       | Super-Hard         | **100\%** |      98\%      |  **100\%** |      96\%      |      96\%      |       0\%      |   0\%  | **100\%** |
| 6h\_vs\_8z     | Super-Hard         | 84\%           |  **87\%** |       82\%      |      78\%      |      78\%      |       4\%      |   1\%  |      19\%      |
| MMM2           | Super-Hard        | **100\%** |      96\%      |  **100\%** | **100\%** |      23\%      |       0\%      |   0\%  | **100**\% |
| 27m\_vs\_30m   | Super-Hard         | **100\%** | **100\%** |  **100\%** | **100\%** |       0\%      |       9\%      |   0\%  |      93\%      |
| Discrete PP    | -              | **40**    |       39       |        -        |       39       |       39       |       30       |   32   |       38       |
| Avg. Score     | Hard+ | **94.9\%**         | 91.2\%         | 92.7\%          | 92.5\%         | 67.4\%         | 29.2\%         | 14.0\% | 84.0\%         |


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
- [**RIIT**: Rethinking the Implementation Tricks and Monotonicity Constraint in Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2102.03479)

### Installation instructions

Install Python packages
```shell
# require Anaconda 3 or Miniconda 3
bash install_dependecies.sh
```

Set up StarCraft II (2.4.10) and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2.4.10 into the 3rdparty folder and copy the maps necessary to run over.

### Command Line Tool

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

**Kill all training processes**

```shell
# all python and game processes of current user will quit.
bash clean.sh
```

## Cite
```
@article{hu2021rethinking,
      title={Rethinking the Implementation Tricks and Monotonicity Constraint in Cooperative Multi-Agent Reinforcement Learning}, 
      author={Jian Hu and Siyang Jiang and Seth Austin Harding and Haibin Wu and Shih-wei Liao},
      year={2021},
      eprint={2102.03479},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

