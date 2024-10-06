import time
import torch
import numpy as np

from pydil.optimal_transport.pot_utils import unif


def get_label_frequencies(Y):
    """Get label frequencies from one-hot encoded labels.

    Parameters
    ----------
    Y : torch Tensor
        Tensor of shape (n, n_c), containing one-hot encoded labels.
    """
    y = Y.argmax(dim=1)
    u, c = torch.unique(y, return_counts=True)
    frequencies = torch.zeros(Y.shape[1])
    frequencies[u] += c
    return frequencies


class EmpiricalBarycenterSolver(torch.nn.Module):
    r"""
    Empirical barycenter solver. This class assumes
    that measures are represented by their samples,
    that is,

    $$\hat{P}(\mathbf{x}) = \dfrac{1}{n}\sum_{i=1}^{n}\delta(\mathbf{x} -
    \mathbf{x}_{i}^{(P)})$$

    As a result, this class computes Wasserstein barycenters,

    $$\mathcal{B}(\lambda,\mathcal{P}) = \text{argmin}_{B}\sum_{k}
    \lambda_{k}\mathcal{W}_{2}(P_{k},B)$$

    Attributes
    ----------
    n_samples : int
        Number of samples in the support of the barycentric measure
    ot_solver : OptimalTransportSolver
        Optimal Transport solver object.
    beta : float, optional
        Weight for the label distances. Default is None
    n_iter_max : int, optional
        Number of iterations for the barycenter algorithm.
        Default is 100
    tol : float, optional
        Criterium for stopping the barycenter algorithm.
        Default is 1e-4
    verbose : bool, optional
        If True, prints info about execution. Default is False
    propagate_labels : bool, optional
        Only used for labeled measures. If True, computes
        barycentric mappings of labels at each iteration.
        Default is False.
    penalize_labels : bool, optional
        Only used for labeled measures. If True, adds a
        distance term between labels of measures.
        Default is False
    log : bool, optional
        If True, keeps info about barycenter algorithm execution.
        Default is False
    init : Literal['random', 'samples'], optional
        If 'random', initializes the barycenter measure randomly.
        If 'samples', initializes the barycenter using points
        from input measures. Default is 'random'.
    """
    def __init__(self,
                 n_samples,
                 ot_solver,
                 beta=None,
                 n_iter_max=100,
                 tol=1e-4,
                 verbose=False,
                 propagate_labels=False,
                 penalize_labels=False,
                 log=False,
                 init='random'):
        super(EmpiricalBarycenterSolver, self).__init__()

        self.n_samples = n_samples
        self.ot_solver = ot_solver
        self.beta = beta
        self.n_iter_max = n_iter_max
        self.tol = tol
        self.verbose = verbose
        self.propagate_labels = propagate_labels
        self.penalize_labels = penalize_labels
        self.log = log

        assert init.lower() in [
            'random',
            'samples'
        ], ("Expected 'init' to be either 'random' or 'samples',"
            f"but got {init.lower()}")
        self.init = init.lower()

    def __init(self, XP, YP, weights):
        """Initializes the support of the barycenter measure."""
        dtype = XP[0].dtype
        device = XP[0].device
        with torch.no_grad():
            if self.init == 'random':
                n_dim = XP[0].shape[1]
                XB = torch.randn(
                    self.n_samples, n_dim,
                    device=XP[0].device,
                    dtype=XP[0].dtype
                )
                if YP is not None:
                    n_classes = YP[0].shape[1]
                    frequencies = [
                        get_label_frequencies(YPc.cpu())
                        for YPc in YP
                    ]
                    probabilities = np.stack([
                        fc / fc.sum() for fc in frequencies
                    ])
                    probabilities = np.einsum(
                        'i,ij->j', weights.to('cpu'), probabilities)
                    unique_classes = np.arange(n_classes)
                    YB = torch.nn.functional.one_hot(
                        torch.from_numpy(
                            np.random.choice(
                                unique_classes,
                                size=self.n_samples,
                                replace=True,
                                p=probabilities)
                        ).long(), num_classes=n_classes
                    )
                else:
                    YB = None
            else:
                all_X = torch.cat(XP, dim=0)
                assert self.n_samples <= len(all_X), (
                    f"Expected at least {self.n_samples} in the input",
                    f" matrices, but got {len(all_X)}. Try using init",
                    " = 'random'."
                )
                ind = np.random.choice(
                    np.arange(len(all_X)), size=self.n_samples)
                XB = all_X[ind]
                if YP is not None:
                    all_Y = torch.cat(YP, dim=0)
                    YB = all_Y[ind]
                else:
                    YB = None
        XB = XB.to(dtype=dtype, device=device)
        if YB is not None:
            YB = YB.to(dtype=dtype, device=device)
        return XB, YB

    def forward(self, XP, YP=None, weights=None):
        """Computes the Wasserstein barycenter of input measures.

        Parameters
        ----------
        XP : list of tensors
            List of tensors with the support of the input measures.
            The k-th element of this list has shape (n_k, d), where
            n_k is the number of samples, and d is the number of
            dimensions.
        YP : list of tensors, optional
            Should be given if input measures are labeled. List of
            tensors where the k-th element has shape (n_k, n_c), where
            n_k is the number of samples, and n_c is the number of
            classes.
        weights : tensor, optional
            Tensor of shape (K,), where K is the number of elements
            in the list XP and YP. Default is None, in which case
            the weight vector is uniform.
        """
        # Number of measures
        n_measures = len(XP)

        # Dtype and device
        device = XP[0].device
        dtype = XP[0].dtype

        # If weights are not given, use uniform
        if weights is None:
            weights = unif(n_measures, device=device, dtype=dtype)

        # Support initialization
        XB, YB = self.__init(XP, YP, weights)

        # Variables for termination
        it = 0
        delta = self.tol + 1
        last_loss = torch.inf
        comp_start = time.time()

        # Sample importances
        u_P = [unif(len(XPc), device=device) for XPc in XP]
        u_B = unif(len(XB), device=device)

        if self.verbose:
            print("-" * (26 * 4 + 1))
            print("|{:^25}|{:^25}|{:^25}|{:^25}|".format('Iteration',
                                                         'Loss',
                                                         'δLoss',
                                                         'Elapsed Time'))
            print("-" * (26 * 4 + 1))

        if self.log:
            log = {'gamma': [], 'delta_loss': [], 'loss': [],
                   'features': [], 'labels': []}

        while (delta > self.tol and it < self.n_iter_max):
            with torch.no_grad():
                tstart = time.time()
                ot_plans = []

                loss = 0.0
                for c in range(n_measures):
                    # Computes the ground cost
                    # between measure Pc and B.
                    C = torch.cdist(XP[c], XB, p=2) ** 2

                    # Determines beta.
                    if self.beta is None:
                        beta = C.detach().max()
                    else:
                        beta = self.beta

                    if self.penalize_labels and YP is not None:
                        # Adds pairwise squared Euclidean distances
                        # between soft-probabilities
                        C += beta * torch.cdist(YP[c], YB, p=2) ** 2

                    # Solves optimal transport
                    plan_c = self.ot_solver(u_P[c], u_B, C)

                    # Accumulates loss
                    loss += (weights[c] * torch.sum(plan_c * C).sum()).item()

                    # Appends plan to list of plans
                    ot_plans.append(plan_c)

            # Updates the support of the barycentric measure
            XB = sum([
                w_c * self.n_samples * torch.mm(plan_c.T, XP_c)
                for w_c, plan_c, XP_c in zip(weights, ot_plans, XP)
            ])

            if self.propagate_labels and YP is not None:
                # Updates the labels of the barycentric measure.
                YB = sum([
                    w_c * self.n_samples * torch.mm(plan_c.T, YP_c)
                    for w_c, plan_c, YP_c in zip(weights, ot_plans, YP)
                ])

            # Updates termination variables
            delta = abs(loss - last_loss)
            last_loss = loss
            it += 1
            tfinish = time.time()

            if self.verbose:
                # Prints execution info
                delta_t = tfinish - tstart
                print("|{:^25}|{:^25}|{:^25}|{:^25}|".format(it,
                                                             loss,
                                                             delta,
                                                             delta_t))

            if self.log:
                # Adds tracking info to log
                log['loss'].append(loss)
                log['d_loss'].append(delta)
                log['gamma'].append(ot_plans)
                log['features'].append(XB.detach().data.clone())
                log['labels'].append(YB.detach().data.clone())

        if self.verbose:
            duration = time.time() - comp_start
            print("-" * (26 * 4 + 1))
            print(f"Barycenter calculation took {duration} seconds")
        # Re-evaluate the support at optimality for calculating the gradients
        # NOTE: now we define the support while holding its gradients w.r.t.
        # the weight vector and eventually the support.
        XB = sum([
            w_c * self.n_samples * torch.mm(plan_c.T, XP_c)
            for w_c, plan_c, XP_c in zip(weights, ot_plans, XP)
        ])
        if self.propagate_labels and YP is not None:
            YB = sum([
                w_c * self.n_samples * torch.mm(plan_c.T, YP_c)
                for w_c, plan_c, YP_c in zip(weights, ot_plans, YP)
            ])
            if self.log:
                return XB, YB, log
            return XB, YB
        if self.log:
            return XB, log
        return XB


class GaussianMixtureBarycenterSolver(torch.nn.Module):
    r"""
    Gaussian mixture barycenter solver. This class computes the
    barycenter of a collection of Gaussian measures represented by
    their parameters, specifically the mean and covariance.

    The barycenter is defined as the solution to the optimization
    problem:

    $$\mathcal{B}(\lambda, \mathcal{N}) = \text{argmin}_{B} \sum_{k}
    \lambda_{k} \mathcal{MW}_{2}(P_{k}, B)$$

    where $\mathcal{MW}_{2}$ is the mixture-Wasserstein distance between
    the Gaussian mixture measures.

    Attributes
    ----------
    ot_solver : OptimalTransportSolver
        Optimal Transport solver object.
    beta : float, optional
        Weight for the label distances. Default is None.
    n_iter_max : int, optional
        Number of iterations for the barycenter algorithm. Default is 100.
    tol : float, optional
        Criterium for stopping the barycenter algorithm. Default is 1e-4.
    verbose : bool, optional
        If True, prints info about execution. Default is False.
    propagate_labels : bool, optional
        If True, computes barycentric mappings of labels at each iteration
        for labeled measures. Default is False.
    penalize_labels : bool, optional
        If True, adds a distance term between labels of measures. Default
        is False.
    log : bool, optional
        If True, keeps info about barycenter algorithm execution. Default
        is False.
    """
    def __init__(self,
                 ot_solver,
                 beta=None,
                 n_iter_max=100,
                 tol=1e-4,
                 verbose=False,
                 propagate_labels=False,
                 penalize_labels=False,
                 log=False):
        super(GaussianMixtureBarycenterSolver, self).__init__()

        self.ot_solver = ot_solver
        self.beta = beta
        self.n_iter_max = n_iter_max
        self.tol = tol
        self.verbose = verbose
        self.propagate_labels = propagate_labels
        self.penalize_labels = penalize_labels
        self.log = log

    def __init(self, means, variances, labels):
        n_components = means[0].shape[0]
        n_dim = variances[0].shape[1]
        device = means[0].device
        dtype = means[0].dtype

        means_barycenter = torch.randn(
            n_components, n_dim, dtype=dtype, device=device)
        variances_barycenter = torch.ones(
            n_components, n_dim, dtype=dtype, device=device)

        if labels is not None:
            n_classes = labels[0].shape[1]
            n_elems = n_components // n_classes
            labels_barycenter = torch.nn.functional.one_hot(
                torch.Tensor([i for i in range(n_classes)] * n_elems).long(),
                num_classes=n_classes
            ).to(dtype=dtype, device=device)
            return means_barycenter, variances_barycenter, labels_barycenter
        return means_barycenter, variances_barycenter, None

    def forward(self, means, variances, labels=None, weights=None):
        """Computes the Wasserstein barycenter of Gaussian measures.

        Parameters
        ----------
        means : list of tensors
            List of tensors representing the means of the input Gaussian
            mixture measures. The k-th element of this list has shape (n_k, d),
            where n_k is the number of Gaussian components, and d is the
            number of dimensions.
        variances : list of tensors
            List of tensors representing the variances along each axis of
            the input Gaussian mixture measures. The k-th element of this list
            has shape (n_k, d), where n_k is the number of Gaussian components,
            and d is the number of dimensions.
        labels : list of tensors, optional
            Should be given if input measures are labeled. List of tensors
            where the k-th element has shape (n_k, n_c), where n_k is the
            number of Gaussian components, and n_c is the number of classes.
        weights : tensor, optional
            Tensor of shape (K,), where K is the number of elements in the list
            means, variances, and labels. Default is None, in which case the
            weight vector is uniform.
        """
        # Number of measures
        n_measures = len(means)
        n_components = means[0].shape[0]

        # Dtype and device
        device = means[0].device
        dtype = means[0].dtype

        # If weights are not given, use uniform
        if weights is None:
            weights = unif(n_measures, device=device, dtype=dtype)

        # Barycentric GMM initialization
        (means_barycenter,
         variances_barycenter,
         labels_barycenter) = self.__init(
            means, variances, labels)

        # Variables for termination
        it = 0
        delta = self.tol + 1
        last_loss = torch.inf
        comp_start = time.time()

        # Sample importances
        u_P = [unif(len(means_c), device=device) for means_c in means]
        u_B = unif(len(means_barycenter), device=device)

        if self.verbose:
            print("-" * (26 * 4 + 1))
            print("|{:^25}|{:^25}|{:^25}|{:^25}|".format('Iteration',
                                                         'Loss',
                                                         'δLoss',
                                                         'Elapsed Time'))
            print("-" * (26 * 4 + 1))

        if self.log:
            log = {'gamma': [], 'delta_loss': [], 'loss': [],
                   'features': [], 'labels': []}

        while (delta > self.tol and it < self.n_iter_max):
            with torch.no_grad():
                tstart = time.time()
                ot_plans = []

                loss = 0.0
                for c in range(n_measures):
                    # Computes the pairwise Wasserstein distance
                    # between components of Pc and B.
                    C_mean = torch.cdist(
                        means[c], means_barycenter, p=2) ** 2
                    C_std = torch.cdist(
                        variances[c] ** 0.5,
                        variances_barycenter ** 0.5, p=2) ** 2
                    C = C_mean + C_std

                    # Determines beta.
                    if self.beta is None:
                        beta = C.detach().max()
                    else:
                        beta = self.beta

                    if self.penalize_labels and labels is not None:
                        # Adds pairwise squared Euclidean distances
                        # between soft-probabilities
                        C += beta * torch.cdist(
                            labels[c], labels_barycenter, p=2) ** 2

                    # Solves optimal transport
                    plan_c = self.ot_solver(u_P[c], u_B, C)

                    # Accumulates loss
                    loss += (weights[c] * torch.sum(plan_c * C).sum()).item()

                    # Appends plan to list of plans
                    ot_plans.append(plan_c)

            # Updates the means of the barycentric GMM
            means_barycenter = sum([
                w_c * n_components * torch.mm(plan_c.T, means_c)
                for w_c, plan_c, means_c in zip(weights, ot_plans, means)
            ])

            # Updates the variances of the barycentric GMM
            variances_barycenter = sum([
                w_c * n_components * torch.mm(plan_c.T, variances_c ** 0.5)
                for w_c, plan_c, variances_c in zip(weights,
                                                    ot_plans,
                                                    variances)
            ]) ** 2

            if self.propagate_labels and labels is not None:
                # Updates the labels of the barycentric GMM.
                labels_barycenter = sum([
                    w_c * n_components * torch.mm(plan_c.T, labels_c)
                    for w_c, plan_c, labels_c in zip(weights, ot_plans, labels)
                ])

            # Updates termination variables
            delta = abs(loss - last_loss)
            last_loss = loss
            it += 1
            tfinish = time.time()

            if self.verbose:
                # Prints execution info
                delta_t = tfinish - tstart
                print("|{:^25}|{:^25}|{:^25}|{:^25}|".format(it,
                                                             loss,
                                                             delta,
                                                             delta_t))

            if self.log:
                # Adds tracking info to log
                log['loss'].append(loss)
                log['d_loss'].append(delta)
                log['gamma'].append(ot_plans)
                log['means'].append(
                    means_barycenter.detach().data.clone())
                log['variances'].append(
                    variances_barycenter.detach().data.clone())
                log['labels'].append(
                    labels_barycenter.detach().data.clone())

        if self.verbose:
            duration = time.time() - comp_start
            print("-" * (26 * 4 + 1))
            print(f"Barycenter calculation took {duration} seconds")

        # Re-evaluate the support at optimality for calculating the gradients
        # NOTE: now we define the support while holding its gradients w.r.t.
        # the weight vector and eventually the support.
        means_barycenter = sum([
            w_c * n_components * torch.mm(plan_c.T, means_c)
            for w_c, plan_c, means_c in zip(weights, ot_plans, means)
        ])

        variances_barycenter = sum([
            w_c * n_components * torch.mm(plan_c.T, variances_c ** 0.5)
            for w_c, plan_c, variances_c in zip(weights,
                                                ot_plans,
                                                variances)
        ]) ** 2

        if self.propagate_labels and labels is not None:
            labels_barycenter = sum([
                w_c * n_components * torch.mm(plan_c.T, labels_c)
                for w_c, plan_c, labels_c in zip(weights, ot_plans, labels)
            ])
            if self.log:
                return (means_barycenter,
                        variances_barycenter,
                        labels_barycenter, log)
            return (means_barycenter,
                    variances_barycenter,
                    labels_barycenter)
        if self.log:
            return means_barycenter, variances_barycenter, log
        return means_barycenter, variances_barycenter
