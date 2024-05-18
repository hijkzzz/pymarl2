```diff
- If you want high sample efficiency, please use qmix_high_sample_efficiency.yaml
- which uses 4 processes for training, slower but higher sample efficiency.
- Performance is *not* comparable of models trained with different number of processes. 
```

# PyMARL2

Open-source code for [Rethinking the Implementation Tricks and Monotonicity Constraint in Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2102.03479).

This repository is fine-tuned for StarCraft Multi-agent Challenge (SMAC). For other multi-agent tasks, we also recommend an optimized implementation of QMIX: https://github.com/marlbenchmark/off-policy.

**StarCraft 2 version: SC2.4.10. difficulty: 7.**

```
2022.10.10 update: add qmix_high_sample_efficiency.yaml, which uses 4 processes for training, slower but higher sample efficiency.

2021.10.28 update: add Google Football Environments [vdn_gfootball.yaml] (use `simple115 features`).

2021.10.4 update: add QMIX with attention (qmix_att.yaml) as a baseline for Communication tasks.
```

## Finetuned-QMIX

There are so many code-level tricks in the  Multi-agent Reinforcement Learning (MARL), such as:

- Value function clipping (clip max Q values for QMIX)
- Value Normalization
- Reward scaling
- Orthogonal initialization and layer scaling
- **Adam**
- **Neural networks hidden size**
- learning rate annealing
- Reward Clipping
- Observation Normalization
- Gradient Clipping
- **Large Batch Size**
- **N-step Returns(including GAE($\lambda$) and Q($\lambda$) ...)**
- **Rollout Process Number**
- **$\epsilon$-greedy annealing steps**
- Death Agent Masking

**Related Works**

- Implementation Matters in Deep RL: A Case Study on PPO and TRPO
- What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study
- The Surprising Effectiveness of MAPPO in Cooperative, Multi-Agent Games

Using a few of tricks above (bold texts), we enabled QMIX (qmix.yaml) to solve almost all hard scenarios of SMAC (Fine-tuned hyperparameters **for each scenarios**).

| Senarios     | Difficulty | QMIX (batch_size=128) |                   Finetuned-QMIX                   |
| ------------ | :--------: | :-------------------: | :------------------------------------------------: |
| 8m           |    Easy    |           -           |                  **100\%**                  |
| 2c_vs_1sc    |    Easy    |           -           |                  **100\%**                  |
| 2s3z         |    Easy    |           -           |                  **100\%**                  |
| 1c3s5z       |    Easy    |           -           |                  **100\%**                  |
| 3s5z         |    Easy    |           -           |                  **100\%**                  |
| 8m_vs_9m     |    Hard    |          84%          |                  **100\%**                  |
| 5m_vs_6m     |    Hard    |          84%          |                   **90\%**                   |
| 3s_vs_5z     |    Hard    |          96%          |                  **100\%**                  |
| bane_vs_bane |    Hard    |    **100\%**    |                  **100\%**                  |
| 2c_vs_64zg   |    Hard    |    **100\%**    |                  **100\%**                  |
| corridor     | Super Hard |          0%          |                  **100\%**                  |
| MMM2         | Super Hard |          98%          |                  **100\%**                  |
| 3s5z_vs_3s6z | Super Hard |          3%          | **93\%**(hidden_size = 256, qmix_large.yaml) |
| 27m_vs_30m   | Super Hard |          56%          |                  **100\%**                  |
| 6h_vs_8z     | Super Hard |          0%          |         **93\%**($\lambda$ = 0.3, epsilon_anneal_time = 500000)         |

## Re-Evaluation

Afterwards, we re-evaluate numerous QMIX variants with normalized the tricks (a **general** set of hyperparameters), and find that QMIX achieves the SOTA.

| Scenarios    | Difficulty |   Value-based   |                |                |                |                |  Policy-based  |       |                |                |
| ------------ | ---------- | :-------------: | :------------: | :------------: | :------------: | :------------: | :------------: | ----- | :------------: | :------------: |
|              |            |      QMIX      |      VDNs      |     Qatten     |     QPLEX     |     WQMIX     |      LICA      | VMIX  |      DOP      |      RIIT      |
| 2c_vs_64zg   | Hard       | **100%** | **100%** | **100%** | **100%** | **100%** | **100%** | 98%   |      84%      | **100%** |
| 8m_vs_9m     | Hard       | **100%** | **100%** | **100%** |      95%      |      95%      |      48%      | 75%   |      96%      |      95%      |
| 3s_vs_5z     | Hard       | **100%** | **100%** | **100%** | **100%** | **100%** |      96%      | 96%   | **100%** |      96%      |
| 5m_vs_6m     | Hard       |  **90%**  | **90%** | **90%** | **90%** | **90%** |      53%      | 9%    |      63%      |      67%      |
| 3s5z_vs_3s6z | S-Hard     |  **75%**  |      43%      |      62%      |      68%      |      56%      |       0%       | 56%   |       0%       | **75%** |
| corridor     | S-Hard     | **100%** |      98%      | **100%** |      96%      |      96%      |       0%       | 0%    |       0%       | **100%** |
| 6h_vs_8z     | S-Hard     |       84%       | **87%** |      82%      |      78%      |      75%      |       4%       | 80%   |       0%       |      19%      |
| MMM2         | S-Hard     | **100%** |      96%      | **100%** | **100%** |      96%      |       0%       | 70%   |       3%       | **100%** |
| 27m_vs_30m   | S-Hard     | **100%** | **100%** | **100%** | **100%** | **100%** |       9%       | 93%   |       0%       |      93%      |
| Discrete PP  | -          |  **40**  |       39       |       -       |       39       |       39       |       30       | 39    |       38       |       38       |
| Avg. Score   | Hard+      | **94.9%** |     91.2%     |     92.7%     |     92.5%     |     90.5%     |     29.2%     | 67.4% |     44.1%     |     84.0%     |

## Communication

We also tested our QMIX-with-attention (qmix_att.yaml, $\lambda=0.3$, attention\_heads=4) on some maps (from [NDQ](https://github.com/TonghanWang/NDQ)) that require communication.

| Senarios (200w steps) | Difficulty | Finetuned-QMIX (No Communication) | QMIX-with-attention ( Communication) |
| --------------------- | :--------: | :-------------------------------: | :----------------------------------: |
| 1o_10b_vs_1r          |     -     |                56%                |            **87\%**            |
| 1o_2r_vs_4r           |     -     |                50%                |            **95\%**            |
| bane_vs_hM            |     -     |                0%                |            **0\%**            |

## Google Football

We also tested VDN (vdn_gfootball.yaml) on some maps (from [Google Football](https://github.com/google-research/football)). Specially, we use `simple115 features` to train the model (The Google Football original paper use complex `CNN features`). We did not test QMIX because this environment does not provide global status information.

| Senarios                   | Difficulty | VDN ($\lambda=1.0$) |
| -------------------------- | :--------: | :-------------------: |
| academy_counterattack_hard |     -     |   0.71 (Test Score)   |
| academy_counterattack_easy |     -     |   0.87 (Test Score)   |

# Usage

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
- [**LICA**: Learning Implicit Credit Assignment for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2007.02529)
- [**DOP**: Off-Policy Multi-Agent Decomposed Policy Gradients](https://arxiv.org/abs/2007.12322)
- [**RIIT**: Rethinking the Implementation Tricks and Monotonicity Constraint in Cooperative Multi-Agent Reinforcement Learning.](https://arxiv.org/abs/2102.03479)

## Installation instructions

Install Python packages

```shell
# require Anaconda 3 or Miniconda 3
conda create -n pymarl python=3.8 -y
conda activate pymarl

bash install_dependecies.sh
```

Set up StarCraft II (2.4.10) and SMAC:

```shell
bash install_sc2.sh
```

This will download SC2.4.10 into the 3rdparty folder and copy the maps necessary to run over.

Set up Google Football:

```shell
bash install_gfootball.sh
```

## Command Line Tool

**Run an experiment**

```shell
# For SMAC
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=corridor
```

```shell
# For Difficulty-Enhanced Predator-Prey
python3 src/main.py --config=qmix_predator_prey --env-config=stag_hunt with env_args.map_name=stag_hunt
```

```shell
# For Communication tasks
python3 src/main.py --config=qmix_att --env-config=sc2 with env_args.map_name=1o_10b_vs_1r
```

```shell
# For Google Football (Insufficient testing)
# map_name: academy_counterattack_easy, academy_counterattack_hard, five_vs_five...
python3 src/main.py --config=vdn_gfootball --env-config=gfootball with env_args.map_name=academy_counterattack_hard env_args.num_agents=4
```

The config files act as defaults for an algorithm or environment.

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

**Run n parallel experiments**

```shell
# bash run.sh config_name env_config_name map_name_list (arg_list threads_num gpu_list experinments_num)
bash run.sh qmix sc2 6h_vs_8z epsilon_anneal_time=500000,td_lambda=0.3 2 0 5
```

`xxx_list` is separated by `,`.

All results will be stored in the `Results` folder and named with `map_name`.

**Kill all training processes**

```shell
# all python and game processes of current user will quit.
bash clean.sh
```

# Citation

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
