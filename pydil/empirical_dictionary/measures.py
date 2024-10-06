import torch
import warnings
import numpy as np


class EmpiricalMeasure:
    r"""Unsupervised Dataset Measure class.
    Samples elements from datasets with replacement.
    Datasets are assumed unsupervised, i.e., no annotations
    are available for features.

    Parameters
    ----------
    features : np.array
        Numpy array of shape (n, d) containing features.
    transforms : list of functions
        pre-processing steps for data samples.
    batch_size : int, optional (default=64)
        Number of elements in batches.
    device : str
        Either 'cpu' or 'cuda', corresponding to the devie
        of returned batches.
    """
    def __init__(self, features, batch_size=64, device='cpu',
                 dtype=torch.float32):
        if not torch.cuda.is_available() and device == 'cuda':
            warnings.warn(("Trying to use gpu when no"
                           " device is available. Using CPU instead."))
            device = 'cpu'
        self.device = torch.device(device)

        self.features = features
        self.batch_size = batch_size
        self.dtype = dtype
        self.n_dim = features.shape[1]
        self.ind = np.arange(len(features))

    def sample(self, n=None):
        r"""Samples $n$ points from the measure support.

        Parameters
        ----------
        n : int, optional (default=None)
            If given, samples n samples from the support.
            By default samples batch_size elements.
        """
        n = self.batch_size if n is None else n
        minibatch_ind = np.random.choice(self.ind, size=n)
        minibatch_features = self.features[minibatch_ind]

        return minibatch_features.to(self.dtype).to(self.device), None


class LabeledEmpiricalMeasure:
    r"""Supervised Dataset Measure class.
    Samples elements from datasets with replacement.
    Datasets are assumed supervised, i.e., to each feature
    there corresponds a categorical annotation.

    Parameters
    ----------
    features : np.array
        Numpy array of shape (n, d) containing features.
    labels : np.array
        Numpy array of shape (n,) containing categorical labels.
    transforms : list of functions
        pre-processing steps for data samples.
    batch_size : int, optional (default=64)
        Number of elements in batches.
    device : str
        Either 'cpu' or 'cuda', corresponding to the devie
        of returned batches.
    """
    def __init__(self,
                 features,
                 labels,
                 transforms=None,
                 batch_size=64,
                 balanced_sampling=False,
                 device='cpu',
                 dtype=torch.float32):
        if not torch.cuda.is_available() and device == 'cuda':
            warnings.warn(("Trying to use gpu when no"
                           " device is available. Using CPU instead."))
            device = 'cpu'
        self.dtype = dtype
        self.device = torch.device(device)

        self.labels = labels
        self.features = features

        self.n_dim = features.shape[1]
        self.n_classes = labels.shape[1]
        self.n_samples = features.shape[0]

        self.transforms = transforms
        self.batch_size = batch_size

        self.balanced_sampling = balanced_sampling
        self.labels_cat = labels.argmax(dim=1)
        self.samples_per_class = self.n_samples // self.n_classes

    def sample(self, n=None):
        r"""Samples $n$ points from the measure support.

        Parameters
        ----------
        n : int, optional (default=None)
            If given, samples n samples from the support.
            By default samples batch_size elements.
        """
        n = self.batch_size if n is None else n
        if self.balanced_sampling:
            minibatch_ind = []
            for yu in range(self.n_classes):
                ind = np.random.choice(
                    np.where(self.labels_cat == yu)[0],
                    size=self.samples_per_class)
                minibatch_ind.append(ind)
            minibatch_ind = np.concatenate(minibatch_ind, axis=0)
        else:
            minibatch_ind = np.random.choice(
                np.arange(self.n_samples), size=n)
        np.random.shuffle(minibatch_ind)
        minibatch_labels = self.labels[minibatch_ind]
        minibatch_features = self.features[minibatch_ind]

        return (minibatch_features.to(self.dtype).to(self.device),
                minibatch_labels.to(self.dtype).to(self.device))
