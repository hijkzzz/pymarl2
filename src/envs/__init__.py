from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

from smac.env import StarCraft2Env, StarCraftCapabilityEnvWrapper
from .one_step_matrix_game import OneStepMatrixGame
from .stag_hunt import StagHunt

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["sc2wrapped"] = partial(env_fn, env=StarCraftCapabilityEnvWrapper)
REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
