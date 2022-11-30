from .episode_runner import EpisodeRunner
from .parallel_runner import ParallelRunner
from .parallel_runner_x import ParallelRunner_x

REGISTRY = {}
REGISTRY["episode"] = EpisodeRunner
REGISTRY["parallel"] = ParallelRunner
REGISTRY["parallel_x"] = ParallelRunner_x
