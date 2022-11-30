from .information_bottleneck_comm import IBComm
from .information_bottleneck_comm_full import IBFComm
from .information_bottleneck_comm_not_IB import IBNIBComm
from .tar_comm import TarComm
from .information_bottleneck_pruned_comm import IBPComm

REGISTRY = {}

REGISTRY["information_bottleneck"] = IBComm
REGISTRY["information_bottleneck_full"] = IBFComm
REGISTRY["information_bottleneck_not_IB"] = IBNIBComm
REGISTRY["tar"] = TarComm
REGISTRY["information_bottleneck_pruned"] = IBPComm
