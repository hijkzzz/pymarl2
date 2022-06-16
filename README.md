
# Running Hyperparameter Tuning

1. Build the Docker container by running `docker build -t pymarl2:ben_smac -f docker/Dockerfile --build-arg UID=$UID .` from the `pymarl2` directory
2. Set up StarCraft II by running `./install_sc2.sh`. Make sure to install the `32x32_flat.SC2Map` by copying it to the `$SC2PATH/Maps` directory from 
the [smacv2 repo](https://github.com/oxwhirl/smacv2).
3. Make sure to either set `use_wandb=False` or to change the `project` and `entity` in `src/config/default.yaml`.
4. Run the `run_exp.sh` script.

# Citation
This repository has very few changes from previous QMIX work, except that it runs on SMACv2. The original repository is [here](https://github.com/hijkzzz/pymarl2).

```
@article{hu2021revisiting,
      title={Revisiting the Monotonicity Constraint in Cooperative Multi-Agent Reinforcement Learning}, 
      author={Jian Hu and Siyang Jiang and Seth Austin Harding and Haibin Wu and Shih-wei Liao},
      year={2021},
      eprint={2102.03479},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

