from .pot_utils import (
    proj_simplex,
    unif,
    emd
)
from .ot_solver import OptimalTransportSolver
from .losses import (
    WassersteinDistance,
    JointWassersteinDistance,
    MixtureWassersteinDistance,
    JointMixtureWassersteinDistance,
)
from .barycenters import (
    EmpiricalBarycenterSolver,
    GaussianMixtureBarycenterSolver
)


__all__ = [
    proj_simplex, unif, emd,
    OptimalTransportSolver,
    WassersteinDistance, JointWassersteinDistance,
    MixtureWassersteinDistance, JointMixtureWassersteinDistance,
    EmpiricalBarycenterSolver, GaussianMixtureBarycenterSolver
]
