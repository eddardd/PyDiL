import math
import torch
import numpy as np
from sklearn.mixture import GaussianMixture


def diag_gmm_log_probs(X, weights, means, stds):
    n_dim = X.shape[1]
    log_probs = (torch.log(weights)[:, None] - 0.5 * (
        n_dim * math.log(2 * math.pi) +
        2 * torch.log(stds).sum(dim=1)[:, None] + (
            (1 / stds ** 2) @ X.T ** 2 +
            torch.sum((means / stds) ** 2, 1)[:, None] -
            2 * ((means / (stds ** 2)) @ X.T)
        )
    ))

    return log_probs


def diag_gmm_predict_proba(X, weights, means, stds):
    log_probs = diag_gmm_log_probs(
        X, weights, means, stds)
    log_proba_components = (
        log_probs - log_probs.logsumexp(dim=0)[None, :])
    return log_proba_components.exp()


def diag_gmm_score(X, weights, means, stds):
    log_probs = diag_gmm_log_probs(
        X, weights, means, stds)
    return - log_probs.logsumexp(dim=0).mean()


def diag_gmm_sample(n, weights, means, stds):
    n_dim = means.shape[1]
    n_components = means.shape[0]
    selected_components = np.random.choice(
        np.arange(n_components),
        size=n, replace=True, p=weights)
    _means = means[selected_components]
    _stds = stds[selected_components]
    eps = torch.randn(n, n_dim)

    return _stds * eps + _means


def diag_gmm_predict_class(X, weights, means, stds, labels):
    proba_components = diag_gmm_predict_proba(X, weights, means, stds)
    return torch.mm(labels.T, proba_components).T


class DiagonalGaussianMixture:
    def __init__(self,
                 n_components,
                 random_state=None,
                 dtype=torch.float32,
                 device='cpu'):
        self.n_components = n_components
        self.random_state = random_state
        self.dtype = dtype
        self.device = torch.device(device)

        self.weights = None
        self.means = None
        self.variances = None
        self.labels = None
        self.fitted = False

        self.n_dim = None

    def fit(self, X):
        self.n_dim = X.shape[1]
        gmm = GaussianMixture(
            n_components=self.n_components,
            random_state=self.random_state,
            covariance_type='diag'
        ).fit(X)

        self.weights = torch.from_numpy(
            gmm.weights_).to(dtype=self.dtype, device=self.device)
        self.means = torch.from_numpy(
            gmm.means_).to(dtype=self.dtype, device=self.device)
        self.variances = torch.from_numpy(
            gmm.covariances_).to(dtype=self.dtype, device=self.device)
        self.fitted = True

        return self

    def set_gmm(self, weights, means, variances):
        self.n_dim = means.shape[1]
        self.weights = weights.to(dtype=self.dtype, device=self.device)
        self.means = means.to(dtype=self.dtype, device=self.device)
        self.variances = variances.to(dtype=self.dtype, device=self.device)
        self.fitted = True

    def predict_proba(self, X):
        return diag_gmm_predict_proba(
            X, self.weights, self.means, self.variances ** 0.5)

    def score(self, X):
        return diag_gmm_score(
            X, self.weights, self.means, self.variances ** 0.5)

    def sample(self, n):
        return diag_gmm_sample(
            n, self.weights, self.means, self.variances ** 0.5)


class DiagonalLabeledGaussianMixture:
    def __init__(self,
                 components_per_class,
                 random_state=None,
                 dtype=torch.float32,
                 device='cpu'):
        self.components_per_class = components_per_class
        self.random_state = random_state
        self.dtype = dtype
        self.device = torch.device(device)

        self.weights = None
        self.means = None
        self.variances = None
        self.labels = None
        self.fitted = False

        self.n_dim = None
        self.n_classes = None
        self.n_components = None

    def fit(self, X, Y):
        self.n_dim = X.shape[1]
        self.n_classes = Y.shape[1]
        self.n_components = self.components_per_class * self.n_classes

        weights = []
        means = []
        variances = []
        labels = []

        for c in Y.argmax(dim=1).unique():
            # Gets indices from samples of class c
            ind = torch.where(Y.argmax(dim=1) == c)[0]

            # Fit a GMM on samples from class c exclusively
            gmm_c = GaussianMixture(
                n_components=self.components_per_class,
                random_state=self.random_state,
                covariance_type='diag'
            ).fit(X[ind])

            # Appends to the list of parameters.
            weights.append(torch.from_numpy(
                gmm_c.weights_).to(dtype=self.dtype, device=self.device))
            means.append(torch.from_numpy(
                gmm_c.means_).to(dtype=self.dtype, device=self.device))
            variances.append(torch.from_numpy(
                gmm_c.covariances_).to(dtype=self.dtype, device=self.device))
            labels.append(torch.nn.functional.one_hot(
                torch.tensor([c] * self.components_per_class),
                num_classes=self.n_classes).to(
                    dtype=self.dtype, device=self.device))
        # Concatenates all parameters
        self.weights = torch.cat(weights)
        self.means = torch.cat(means, dim=0)
        self.variances = torch.cat(variances, dim=0)
        self.labels = torch.cat(labels, dim=0)

        # Re-normalizes weights to sum to 1.
        self.weights = self.weights / self.weights.sum()

        # Set fitted flag as True.
        self.fitted = True

        return self

    def set_gmm(self, weights, means, variances, labels):
        self.n_dim = means.shape[1]
        self.n_classes = labels.shape[1]
        self.n_components = means.shape[0]

        self.weights = weights.to(dtype=self.dtype, device=self.device)
        self.means = means.to(dtype=self.dtype, device=self.device)
        self.variances = variances.to(dtype=self.dtype, device=self.device)
        self.labels = labels.to(dtype=self.dtype, device=self.device)
        self.fitted = True

    def predict_proba(self, X):
        return diag_gmm_predict_proba(
            X, self.weights, self.means, self.variances ** 0.5)

    def score(self, X):
        return diag_gmm_score(
            X, self.weights, self.means, self.variances ** 0.5)

    def sample(self, n):
        return diag_gmm_sample(
            n, self.weights, self.means, self.variances ** 0.5)

    def predict_class(self, X):
        return diag_gmm_predict_class(
            X, self.weights, self.means, self.variances ** 0.5,
            self.labels)
