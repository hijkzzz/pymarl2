from .run import run as default_run
from .on_off_run import run as on_off_run
from .dop_run import run as dop_run
from .per_run import run as per_run

REGISTRY = {}
REGISTRY["default"] = default_run
REGISTRY["on_off"] = on_off_run
REGISTRY["dop_run"] = dop_run
REGISTRY["per_run"] = per_run