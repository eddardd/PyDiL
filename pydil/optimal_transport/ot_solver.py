import ot
import torch
from pydil.optimal_transport.pot_utils import (
    unif, emd
)


class OptimalTransportSolver(torch.nn.Module):
    """Simple wrapper of optimal transport algorithms. Receives as parameters
    different regularization terms for different transportation problems. Note
    that this is a generic solver that takes as input the ground-cost matrix.
    As such, it handles empirical, as well as Gaussian mixture measures.

    Attributes
    ----------
    reg_e : float, optional
        Entropic regularization. If reg_e > 0, uses the Sinkhorn algorithm
        for computing optimal transport. If reg_e = 0.0, uses linear
        programming. Default is 0.0.
    reg_m : float, optonal
        Unbalanced regularization parameter. Only used if reg_e > 0.
        If reg_m > 0, solves an unbalanced OT problem using the Sinkhorn
        algorithm. Default is 0.0
    n_iter_sinkhorn : int, optional
        Only used if reg_e > 0.0. If that is the case, sets the number
        of Sinkhorn iterations. Default is 1000.
    """
    def __init__(self, reg_e=0.0, reg_m=0.0,
                 n_iter_sinkhorn=1000):
        super(OptimalTransportSolver, self).__init__()

        self.reg_e = reg_e
        self.reg_m = reg_m
        self.n_iter_sinkhorn = n_iter_sinkhorn

    def forward(self, p, q, C):
        """Computes an OT plan.

        Parameters
        ----------
        p : Tensor
            Tensor of shape (n,) containing the importances of
            each element in the measure P.
        q : Tensor
            Tensor of shape (m,) containing the importances of
            each element in the measure Q.
        C : Tensor
            Tensor of shape (n, m) containing the ground-cost
            between elements of the measure P and Q.
        """
        device = C.device
        dtype = C.dtype

        if p is None:
            # If 'p' is not provided, use uniform
            # weights.
            p = unif(len(C), device=device, dtype=dtype)

        if q is None:
            # If 'q' is not provided, use uniform
            # weights.
            q = unif(len(C), device=device, dtype=dtype)

        with torch.no_grad():
            if self.reg_e > 0.0:
                # Sinkhorn algorithm
                if self.reg_m > 0.0:
                    # Unbalanced due to regularization
                    ot_plan = ot.unbalanced.sinkhorn_unbalanced(
                        p, q, C / C.detach().max(),
                        reg=self.reg_e,
                        reg_m=self.reg_m,
                        numItermax=self.n_iter_sinkhorn,
                        reg_type='kl'
                    )
                else:
                    # Balanced Sinkhorn
                    ot_plan = ot.bregman.sinkhorn(
                        p, q, C / C.detach().max(),
                        reg=self.reg_e,
                        numItermax=self.n_iter_sinkhorn,
                        method='sinkhorn_log'
                    )
            else:
                # Standard EMD.
                ot_plan = emd(p, q, C)
        return ot_plan
