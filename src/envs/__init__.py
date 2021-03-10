from functools import partial
import sys
import os

from smac.env import MultiAgentEnv, StarCraft2Env
from .one_step_matrix_game import OneStepMatrixGame
from .particle import Particle
from .stag_hunt import StagHunt

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)
REGISTRY["particle"] = partial(env_fn, env=Particle)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
