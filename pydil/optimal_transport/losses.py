import torch
from pydil.optimal_transport.pot_utils import unif


class WassersteinDistance(torch.nn.Module):
    r"""Wasserstein Distance using the Primal Kantorovich formulation.
    NOTE: We do not backpropagate gradients through the calculation
          of the optimal transport plan.

    Parameters
    ----------
    ot_solver : OptimalTransportSolver
        An object from the OptimalTransportSolver class. It receives
        (p, q, C), where p and q are the vectors with sample importances
        of measures P and Q, and C is the ground-cost matrix.
    """
    def __init__(self, ot_solver):
        super(WassersteinDistance, self).__init__()
        self.ot_solver = ot_solver

    def forward(self, XP, XQ, p=None, q=None):
        r"""Computes the Wasserstien Distance between samples XP ~ P
        and XQ ~ Q,

        Parameters
        ----------
        XP : tensor
            Tensor of shape (n, d) containing samples from distribution P
        XQ : tensor
            Tensor of shape (m, d) containing samples from distribution Q
        p : tensor
            Tensor of shape (n,) containing the importance of samples
            from P. Must be positive and sum to 1.
        q : tensor
            Tensor of shape (m,) containing the importance of samples
            from Q. Must be positive and sum to 1.
        """
        p = unif(XP.shape[0], device=XP.device)
        q = unif(XQ.shape[0], device=XQ.device)

        if self.debias:
            bias = 0.0

            # Loss between P and P
            CPP = torch.cdist(XP, XP, p=2) ** 2
            plan_PP = self.ot_solver(p, p, CPP)
            bias += torch.sum(CPP * plan_PP)

            # Loss between Q and Q
            CQQ = torch.cdist(XQ, XQ, p=2) ** 2
            plan_QQ = self.ot_solver(q, q, CQQ)
            bias += torch.sum(CQQ * plan_QQ)

            # Loss between P and Q
            CPQ = torch.cdist(XP, XQ, p=2) ** 2
            plan_PQ = self.ot_solver(p, q, CPQ)
            loss_val = torch.sum(CPQ * plan_PQ)

            # Debiased Sinkhorn loss
            loss_val = loss_val - 0.5 * bias
        else:
            C = torch.cdist(XP, XQ, p=2) ** 2
            plan = self.ot_solver(p, q, C)
            loss_val = torch.sum(plan * C)
        return loss_val


class JointWassersteinDistance(torch.nn.Module):
    r"""Joint Wasserstein Distance using the Primal Kantorovich formulation.
    NOTE: We do not backpropagate gradients through the calculation
          of the optimal transport plan.

    Parameters
    ----------
    ot_solver : OptimalTransportSolver
        An object from the OptimalTransportSolver class. It receives
        (p, q, C), where p and q are the vectors with sample importances
        of measures P and Q, and C is the ground-cost matrix.
    beta : float (optional, default=None)
        Float indicating the importance of label distances in OT.
        If not given, uses the maximum distance between features as
        the weight factor.
    """
    def __init__(self, ot_solver, beta=None, debias=False):
        super(JointWassersteinDistance, self).__init__()
        self.ot_solver = ot_solver
        self.beta = beta
        self.debias = debias

    def forward(self, XP, XQ, YP=None, YQ=None, p=None, q=None):
        r"""Computes the Joint Wasserstien Distance between
        samples (XP, YP) ~ P and (XQ, YQ) ~ Q,

        Parameters
        ----------
        XP : tensor
            Tensor of shape (n, d) containing the features of samples
            from distribution P
        XQ : tensor
            Tensor of shape (m, d) containing the features of samples
            from distribution Q
        YP : tensor
            Tensor of shape (n, n_c) containing the labels of samples
            from distribution P
        YQ : tensor
            Tensor of shape (m, n_c) containing the labels of samples
            from distribution Q
        p : tensor
            Tensor of shape (n,) containing the importance of samples
            from P. Must be positive and sum to 1.
        q : tensor
            Tensor of shape (m,) containing the importance of samples
            from Q. Must be positive and sum to 1.
        """
        p = unif(XP.shape[0], device=XP.device)
        q = unif(XQ.shape[0], device=XQ.device)

        if self.debias:
            bias = 0.0

            # Loss between P and P
            CPP = torch.cdist(XP, XP, p=2) ** 2
            if YP is not None and YQ is not None:
                if self.beta is None:
                    beta = CPP.detach().max()
                else:
                    beta = self.beta
                CPP += beta * torch.cdist(YP, YP, p=2) ** 2
            plan_PP = self.ot_solver(p, p, CPP)
            bias += torch.sum(CPP * plan_PP)

            # Loss between Q and Q
            CQQ = torch.cdist(XQ, XQ, p=2) ** 2
            if YP is not None and YQ is not None:
                if self.beta is None:
                    beta = CQQ.detach().max()
                else:
                    beta = self.beta
                CQQ += beta * torch.cdist(YQ, YQ, p=2) ** 2
            plan_QQ = self.ot_solver(q, q, CQQ)
            bias += torch.sum(CQQ * plan_QQ)

            # Loss between P and Q
            CPQ = torch.cdist(XP, XQ, p=2) ** 2
            if YP is not None and YQ is not None:
                if self.beta is None:
                    beta = CPQ.detach().max()
                else:
                    beta = self.beta
                CPQ += beta * torch.cdist(YP, YQ, p=2) ** 2
            plan_PQ = self.ot_solver(p, q, CPQ)
            loss_val = torch.sum(CPQ * plan_PQ)

            # Debiased Sinkhorn loss
            loss_val = loss_val - 0.5 * bias
        else:
            C = torch.cdist(XP, XQ, p=2) ** 2
            if YP is not None and YQ is not None:
                if self.beta is None:
                    beta = C.detach().max()
                else:
                    beta = self.beta
                C += beta * torch.cdist(YP, YQ, p=2) ** 2
            plan = self.ot_solver(p, q, C)
            loss_val = torch.sum(plan * C)
        return loss_val


class MixtureWassersteinDistance(torch.nn.Module):
    r"""Mixture Wasserstein Distance using the Primal Kantorovich formulation.
    NOTE: We do not backpropagate gradients through the calculation
          of the optimal transport plan.

    Parameters
    ----------
    ot_solver : OptimalTransportSolver
        An object from the OptimalTransportSolver class. It receives
        (p, q, C), where p and q are the vectors with sample importances
        of measures P and Q, and C is the ground-cost matrix.
    """
    def __init__(self, ot_solver):
        super(MixtureWassersteinDistance, self).__init__()
        self.ot_solver = ot_solver

    def forward(self,
                means_P,
                variances_P,
                means_Q,
                variances_Q,
                p=None,
                q=None):
        r"""Computes the mixture-Wasserstein distance between Gaussian
        mixtures with diagonal covariance matrices.

        Parameters
        ----------
        means_P : tensor
            Tensor of shape (KP, d) containing the mean vectors of GMM P
        variances_P : tensor
            Tensor of shape (KP, d) containing the variance vectors of GMM Q
        means_Q : tensor
            Tensor of shape (KQ, d) containing the mean vectors of GMM P
        variances_Q : tensor
            Tensor of shape (KQ, d) containing the variance vectors of GMM Q
        p : tensor
            Tensor of shape (n,) containing the importance of components
            from P. Must be positive and sum to 1.
        q : tensor
            Tensor of shape (m,) containing the importance of components
            from Q. Must be positive and sum to 1.
        """
        p = unif(means_P.shape[0], device=means_P.device)
        q = unif(means_Q.shape[0], device=means_Q.device)

        if self.debias:
            bias = 0.0

            # Loss between P and P
            CPP = (torch.cdist(means_P, means_P, p=2) ** 2 +
                   torch.cdist(variances_P, variances_P, p=2) ** 2)
            plan_PP = self.ot_solver(p, p, CPP)
            bias += torch.sum(CPP * plan_PP)

            # Loss between Q and Q
            CQQ = (torch.cdist(means_Q, means_Q, p=2) ** 2 +
                   torch.cdist(variances_Q, variances_Q, p=2) ** 2)
            plan_QQ = self.ot_solver(q, q, CQQ)
            bias += torch.sum(CQQ * plan_QQ)

            # Loss between P and Q
            CPQ = (torch.cdist(means_P, means_Q, p=2) ** 2 +
                   torch.cdist(variances_P, variances_Q, p=2) ** 2)
            plan_PQ = self.ot_solver(p, q, CPQ)
            loss_val = torch.sum(CPQ * plan_PQ)

            # Debiased Sinkhorn loss
            loss_val = loss_val - 0.5 * bias
        else:
            C = (torch.cdist(means_P, means_Q, p=2) ** 2 +
                 torch.cdist(variances_P, variances_Q, p=2) ** 2)
            plan = self.ot_solver(p, q, C)
            loss_val = torch.sum(plan * C)
        return loss_val


class JointMixtureWassersteinDistance(torch.nn.Module):
    r"""Joint Wasserstein Distance using the Primal Kantorovich formulation.
    NOTE: We do not backpropagate gradients through the calculation
          of the optimal transport plan.

    Parameters
    ----------
    ot_solver : OptimalTransportSolver
        An object from the OptimalTransportSolver class. It receives
        (p, q, C), where p and q are the vectors with sample importances
        of measures P and Q, and C is the ground-cost matrix.
    debias : boolean (optional, default=False)
        If True, computes additional terms between (P, P) and (Q, Q).
    beta : float (optional, default=None)
        Float indicating the importance of label distances in OT.
        If not given, uses the maximum distance between features as
        the weight factor.
    """
    def __init__(self, ot_solver, beta=None, debias=False):
        super(JointMixtureWassersteinDistance, self).__init__()
        self.ot_solver = ot_solver
        self.debias = debias
        self.beta = beta

    def forward(self,
                means_P,
                variances_P,
                labels_P,
                means_Q,
                variances_Q,
                labels_Q,
                p=None, q=None):
        r"""Computes the Joint Wasserstien Distance between
        samples (XP, YP) ~ P and (XQ, YQ) ~ Q,

        Parameters
        ----------
        means_P : tensor
            Tensor of shape (KP, d) containing the mean vectors
            of the GMM P
        variances_P : tensor
            Tensor of shape (KP, d) containing the variance vectors
            of the GMM P
        labels_P : tensor
            Tensor of shape (KP, d) containing the labels
            of the GMM P
        means_Q : tensor
            Tensor of shape (KQ, d) containing the mean vectors
            of the GMM Q
        variances_Q : tensor
            Tensor of shape (KQ, d) containing the variance vectors
            of the GMM Q
        labels_Q : tensor
            Tensor of shape (KQ, d) containing the labels
            of the GMM Q
        p : tensor
            Tensor of shape (n,) containing the importance of samples
            from P. Must be positive and sum to 1.
        q : tensor
            Tensor of shape (m,) containing the importance of samples
            from Q. Must be positive and sum to 1.
        """
        p = unif(means_P.shape[0], device=means_P.device)
        q = unif(means_Q.shape[0], device=means_Q.device)

        if self.debias:
            bias = 0.0

            # Loss between P and P
            CPP = (torch.cdist(means_P, means_P, p=2) ** 2 +
                   torch.cdist(variances_P, variances_P, p=2) ** 2)
            if labels_P is not None and labels_Q is not None:
                if self.beta is None:
                    beta = CPP.detach().max()
                else:
                    beta = self.beta
                CPP += beta * torch.cdist(labels_P, labels_P, p=2) ** 2
            plan_PP = self.ot_solver(p, p, CPP)
            bias += torch.sum(CPP * plan_PP)

            # Loss between Q and Q
            CQQ = (torch.cdist(means_Q, means_Q, p=2) ** 2 +
                   torch.cdist(variances_Q, variances_Q, p=2) ** 2)
            if labels_P is not None and labels_Q is not None:
                if self.beta is None:
                    beta = CQQ.detach().max()
                else:
                    beta = self.beta
                CQQ += beta * torch.cdist(labels_Q, labels_Q, p=2) ** 2
            plan_QQ = self.ot_solver(q, q, CQQ)
            bias += torch.sum(CQQ * plan_QQ)

            # Loss between P and Q
            CPQ = (torch.cdist(means_P, means_Q, p=2) ** 2 +
                   torch.cdist(variances_P, variances_Q, p=2) ** 2)
            if labels_P is not None and labels_Q is not None:
                if self.beta is None:
                    beta = CPQ.detach().max()
                else:
                    beta = self.beta
                CPQ += beta * torch.cdist(labels_P, labels_Q, p=2) ** 2
            plan_PQ = self.ot_solver(p, q, CPQ)
            loss_val = torch.sum(CPQ * plan_PQ)

            # Debiased Sinkhorn loss
            loss_val = loss_val - 0.5 * bias
        else:
            C = (torch.cdist(means_P, means_Q, p=2) ** 2 +
                 torch.cdist(variances_P, variances_Q, p=2) ** 2)
            if labels_P is not None and labels_Q is not None:
                if self.beta is None:
                    beta = C.detach().max()
                else:
                    beta = self.beta
                C += beta * torch.cdist(labels_P, labels_Q, p=2) ** 2
            plan = self.ot_solver(p, q, C)
            loss_val = torch.sum(plan * C)
        return loss_val
