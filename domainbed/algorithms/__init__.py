from .erm import ERM
from .lwf import ERM_DIST
from .ewc import EWC
from .mas import MAS
from .msl_mov import MSL_MOV_DIST
from .mixstyle import Mixstyle, Mixstyle2
from .coral import CORAL
from .triplet_dist_w_proto import TRIPLET_DIST_W_PROTO


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
