import torch
import numpy as np

from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pydil.optimal_transport.pot_utils import proj_simplex
from pydil.optimal_transport.losses import (
    WassersteinDistance,
    JointWassersteinDistance
)


class EmpiricalDictionary(torch.nn.Module):
    def __init__(self,
                 barycenter_solver,
                 atoms_features=None,
                 barycentric_coordinates=None,
                 n_samples=1024,
                 n_dim=None,
                 n_classes=None,
                 n_components=2,
                 weight_initialization='uniform',
                 n_distributions=None,
                 loss_fn=None,
                 learning_rate_features=1e-1,
                 learning_rate_weights=None,
                 domain_names=None,
                 optimizer_name='adam',
                 tensor_dtype=torch.float32,
                 track_atoms=False,
                 schedule_lr=True,
                 device='cpu'):
        super(EmpiricalDictionary, self).__init__()

        self.barycenter_solver = barycenter_solver

        self.n_samples = n_samples
        self.n_dim = n_dim
        self.n_classes = n_classes
        self.n_components = n_components
        self.weight_initialization = weight_initialization
        self.n_distributions = n_distributions
        self.tensor_dtype = tensor_dtype
        self.learning_rate_features = learning_rate_features
        self.device = device

        if learning_rate_weights is None:
            self.learning_rate_weights = self.learning_rate_features
        else:
            self.learning_rate_weights = learning_rate_weights

        if loss_fn is None:
            self.loss_fn = WassersteinDistance()
        else:
            self.loss_fn = loss_fn

        if domain_names is None:
            if n_distributions is None:
                raise ValueError(("If 'domain_names' is not given,"
                                  " 'n_distributions' must be provided."))
            self.domain_names = [
                "Domain {}".format(ℓ) for ℓ in range(n_distributions)]
        else:
            self.domain_names = domain_names

        self.optimizer_name = optimizer_name
        self.track_atoms = track_atoms
        self.schedule_lr = schedule_lr

        self.__initialize_atoms_features(atoms_features)
        self.__initialize_weights(barycentric_coordinates)

        self.history = {
            'loss': [],
            'weights': [self.barycentric_coordinates.cpu().detach().clone()],
            'atoms_features': [
                torch.stack([XPk.cpu().detach().clone() for XPk in self.XP])],
            'loss_per_dataset': {name: [] for name in self.domain_names}
        }

        self.optimizer = self.configure_optimizers()
        if self.schedule_lr:
            self.scheduler = ReduceLROnPlateau(self.optimizer)

    def __initialize_atoms_features(self, atoms_features=None):
        if atoms_features is None:
            if self.n_dim is None:
                raise ValueError(("If 'XP' is not given,"
                                  " you should specify 'n_dim'."))
            XP_data = [
                torch.randn(self.n_samples, self.n_dim,
                            requires_grad=True, device=self.device).to(
                                self.tensor_dtype)
                for _ in range(self.n_components)
            ]
            self.XP = torch.nn.ParameterList(
                [torch.nn.parameter.Parameter(data=xp, requires_grad=True)
                 for xp in XP_data]
            )
        else:
            self.XP = torch.nn.ParameterList(
                [torch.nn.parameter.Parameter(data=xp.to(self.tensor_dtype),
                                              requires_grad=True)
                 for xp in atoms_features]
            )
            self.n_dim = atoms_features[0].shape[1]

    def __initialize_weights(self, barycentric_coordinates=None):
        if barycentric_coordinates is None:
            if self.n_distributions is None:
                raise ValueError(("If 'barycentric_coordinates' is not given"
                                  " you should specify 'n_distributions'"))
            if self.weight_initialization == 'random':
                barycentric_coordinates_data = torch.rand(
                    self.n_distributions, self.n_components,
                    requires_grad=True,
                    device=self.device).to(self.tensor_dtype)
            else:
                barycentric_coordinates_data = torch.ones(
                    self.n_distributions, self.n_components,
                    requires_grad=True,
                    device=self.device).to(self.tensor_dtype)
            with torch.no_grad():
                barycentric_coordinates_data = proj_simplex(
                    barycentric_coordinates.cpu().T).T.to(self.device)
        else:
            barycentric_coordinates_data = barycentric_coordinates
        self.barycentric_coordinates = torch.nn.parameter.Parameter(
            data=barycentric_coordinates_data, requires_grad=True)

    def configure_optimizers(self):
        r"""Returns optimizers for dictionary variables."""
        if self.grad_labels:
            return torch.optim.Adam([
                {'params': self.XP,
                 'lr': self.learning_rate_features},
                {'params': self.barycentric_coordinates,
                 'lr': self.learning_rate_weights}
            ])
        else:
            return torch.optim.Adam([
                {'params': self.XP,
                 'lr': self.learning_rate_features},
                {'params': self.barycentric_coordinates,
                 'lr': self.learning_rate_weights}
            ])

    def get_atoms(self):
        r"""Gets a list containing atoms features."""
        with torch.no_grad():
            XP = torch.stack([XPk.data.cpu().clone() for XPk in self.XP])
        return XP

    def get_barycentric_coordinates(self):
        r"""Gets barycentric coordinates."""
        with torch.no_grad():
            if (self.barycentric_coordinates.sum(dim=1) == 1).all():
                return self.barycentric_coordinates.data.cpu()
            else:
                return proj_simplex(
                    self.barycentric_coordinates.data.cpu().T).T

    def sample_from_atoms(self, n=None, detach=False):
        """Samples (with replacement) $n$ samples from atoms support.

        Parameters
        ----------
        n : int, optional (default=None)
            Number of samples (with replacement) acquired from the atoms
            support. If None, gets all samples from the atoms supports.
        detach : bool, optional (default=False).
            If True, detaches tensors so that gradients are not calculated.
        """
        batch_features = []

        # Sampling
        for XPk in zip(self.XP):
            sampled_indices = np.random.choice(
                np.arange(self.n_samples), size=n)
            features_k = XPk[sampled_indices]

            if detach:
                features_k = features_k.detach()

            batch_features.append(features_k)

        return batch_features

    def batch_generator(self, batch_size):
        r"""Creates a generator of batches from
        atoms without replacement.

        Parameters
        ----------
        batch_size : int
            Number of samples in mini-batches.
        """
        n_batches = self.n_samples // batch_size
        indices = np.arange(self.n_samples)
        np.random.shuffle(indices)

        for i in range(n_batches + 1):
            mb_indices = indices[batch_size * i: batch_size * (i + 1)]
            yield mb_indices

    def fit(self,
            datasets,
            n_iter_max=100,
            batches_per_it=10,
            verbose=True):
        r"""Minimizes DaDiL's objective function by sampling
        mini-batches from the atoms support.

        Parameters
        ----------
        datasets : list of measures
            List of measure objects, which implement sampling from
            datasets.
        n_iter_max : int, optional (default=100)
            Number of epoch in DaDiL's optimization
        batches_per_it : int, optional (default=10)
            Number of batches drawn from the atoms per iteration.
        verbose : bool, optional (default=True)
            If True, prints progress of DaDiL's Optimization loop.
        """
        batch_size = datasets[0].batch_size
        for it in range(n_iter_max):
            # Calculates the loss
            avg_it_loss = 0
            avg_it_loss_per_dataset = {
                self.domain_names[ℓ]: 0 for ℓ in range(len(datasets))}
            if verbose:
                pbar = tqdm(range(batches_per_it))
            else:
                pbar = range(batches_per_it)
            for _ in pbar:
                self.optimizer.zero_grad()

                loss = 0
                for ℓ, (Qℓ, barycentric_coordinates_ℓ) in enumerate(zip(
                     datasets, self.barycentric_coordinates)):
                    # Sample minibatch from dataset
                    XQℓ = Qℓ.sample()

                    # Sample minibatch from atoms
                    XP = self.sample_from_atoms(n=batch_size)

                    # Calculates Wasserstein barycenter
                    XBℓ = self.barycenter_solver(
                        XP,
                        None,
                        weights=barycentric_coordinates_ℓ
                    )

                    # Calculates Loss
                    loss_ℓ = self.loss_fn(XQ=XQℓ, XP=XBℓ)

                    # Accumulates loss
                    loss += loss_ℓ
                    loss_val = loss_ℓ.detach().cpu().item() / batches_per_it
                    avg_it_loss_per_dataset[self.domain_names[ℓ]] += loss_val

                loss.backward()
                self.optimizer.step()

                # Projects the weights into the simplex
                with torch.no_grad():
                    self.barycentric_coordinates.data = proj_simplex(
                        self.barycentric_coordinates.data.cpu().T).T.to(
                            self.device)

                avg_it_loss += loss.item() / batches_per_it
            # Saves history info
            _XP = self.get_atoms()
            self.history['atoms_features'].append(_XP)
            self.history['weights'].append(
                proj_simplex(self.barycentric_coordinates.data.cpu().T).T.to(
                    self.device))
            self.history['loss'].append(avg_it_loss)
            for ℓ in range(len(datasets)):
                self.history['loss_per_dataset'][self.domain_names[ℓ]].append(
                    avg_it_loss_per_dataset[self.domain_names[ℓ]]
                )
            if verbose:
                print('It {}/{}, Loss: {}'.format(it, n_iter_max, avg_it_loss))
            if self.schedule_lr:
                self.scheduler.step(avg_it_loss)
        self.fitted = True


class LabeledEmpiricalDictionary(torch.nn.Module):
    def __init__(self,
                 barycenter_solver,
                 atoms_features=None,
                 atoms_labels=None,
                 barycentric_coordinates=None,
                 n_samples=1024,
                 n_dim=None,
                 n_classes=None,
                 n_components=2,
                 weight_initialization='uniform',
                 n_distributions=None,
                 loss_fn=None,
                 learning_rate_features=1e-1,
                 learning_rate_labels=None,
                 learning_rate_weights=None,
                 domain_names=None,
                 grad_labels=True,
                 optimizer_name='adam',
                 balanced_sampling=True,
                 tensor_dtype=torch.float32,
                 track_atoms=False,
                 schedule_lr=True,
                 device='cpu'):
        super(LabeledEmpiricalDictionary, self).__init__()

        self.barycenter_solver = barycenter_solver

        self.n_samples = n_samples
        self.n_dim = n_dim
        self.n_classes = n_classes
        self.n_components = n_components
        self.weight_initialization = weight_initialization
        self.n_distributions = n_distributions
        self.tensor_dtype = tensor_dtype
        self.learning_rate_features = learning_rate_features
        self.device = device

        if loss_fn is None:
            self.loss_fn = JointWassersteinDistance()
        else:
            self.loss_fn = loss_fn

        if learning_rate_labels is None:
            self.learning_rate_labels = learning_rate_features
        else:
            self.learning_rate_labels = learning_rate_labels

        if learning_rate_weights is None:
            self.learning_rate_weights = self.learning_rate_features
        else:
            self.learning_rate_weights = learning_rate_weights

        if domain_names is None:
            if n_distributions is None:
                raise ValueError(("If 'domain_names' is not given,"
                                  " 'n_distributions' must be provided."))
            self.domain_names = [
                "Domain {}".format(ℓ) for ℓ in range(n_distributions)]
        else:
            self.domain_names = domain_names

        self.grad_labels = grad_labels
        self.optimizer_name = optimizer_name
        self.balanced_sampling = balanced_sampling
        self.track_atoms = track_atoms
        self.schedule_lr = schedule_lr

        self.__initialize_atoms_features(atoms_features)
        self.__initialize_atoms_labels(atoms_labels)
        self.__initialize_weights(barycentric_coordinates)

        self.history = {
            'loss': [],
            'weights': [self.barycentric_coordinates.cpu().detach().clone()],
            'atoms_features': [
                torch.stack([XPk.cpu().detach().clone() for XPk in self.XP])],
            'atoms_labels': [
                torch.stack([YPk.cpu().detach().clone().softmax(dim=-1)
                             for YPk in self.YP])],
            'loss_per_dataset': {name: [] for name in self.domain_names}
        }

        self.optimizer = self.configure_optimizers()
        if self.schedule_lr:
            self.scheduler = ReduceLROnPlateau(self.optimizer)

    def __initialize_atoms_features(self, atoms_features=None):
        if atoms_features is None:
            if self.n_dim is None:
                raise ValueError(("If 'XP' is not given,"
                                  " you should specify 'n_dim'."))
            atoms_features_data = [
                torch.randn(self.n_samples, self.n_dim,
                            requires_grad=True, device=self.device).to(
                                self.tensor_dtype)
                for _ in range(self.n_components)
            ]
            self.XP = torch.nn.ParameterList(
                [torch.nn.parameter.Parameter(data=xp, requires_grad=True)
                 for xp in atoms_features_data]
            )
        else:
            self.XP = torch.nn.ParameterList(
                [torch.nn.parameter.Parameter(data=xp.to(self.tensor_dtype),
                                              requires_grad=True)
                 for xp in atoms_features]
            )
            self.n_dim = atoms_features[0].shape[1]

    def __initialize_atoms_labels(self, atoms_labels=None):
        if atoms_labels is None:
            if self.n_classes is None:
                raise ValueError(("If 'YP' is not given,"
                                  " you should specify 'n_classes'"))
            samples_per_class = self.n_samples // self.n_classes
            if self.n_samples % self.n_classes != 0:
                self.n_samples = self.n_classes * samples_per_class
            atoms_labels_data = []
            for _ in range(self.n_components):
                ypk = torch.cat(
                    [torch.tensor([c] * samples_per_class)
                     for c in range(self.n_classes)]
                ).long()
                YPk = torch.nn.functional.one_hot(
                    ypk, num_classes=self.n_classes).to(
                        self.tensor_dtype).to(self.device)
                atoms_labels_data.append(YPk)
            self.YP = torch.nn.ParameterList(
                [torch.nn.parameter.Parameter(data=yp,
                                              requires_grad=self.grad_labels)
                 for yp in atoms_labels_data]
            )
        else:
            self.YP = torch.nn.ParameterList(
                [torch.nn.parameter.Parameter(data=yp.to(
                    self.tensor_dtype).to(self.device),
                                              requires_grad=self.grad_labels)
                 for yp in atoms_labels]
            )
            self.n_classes = atoms_labels[0].shape[1]

    def __initialize_weights(self, barycentric_coordinates=None):
        if barycentric_coordinates is None:
            if self.n_distributions is None:
                raise ValueError(("If 'A' is not given you"
                                  " should specify 'n_distributions'"))
            if self.weight_initialization == 'random':
                barycentric_coordinates_data = torch.rand(
                    self.n_distributions, self.n_components,
                    requires_grad=True,
                    device=self.device).to(self.tensor_dtype)
            else:
                barycentric_coordinates_data = torch.ones(
                    self.n_distributions, self.n_components,
                    requires_grad=True,
                    device=self.device).to(self.tensor_dtype)
            with torch.no_grad():
                barycentric_coordinates_data = proj_simplex(
                    barycentric_coordinates_data.cpu().T).T.to(self.device)
        else:
            barycentric_coordinates_data = barycentric_coordinates
        self.barycentric_coordinates = torch.nn.parameter.Parameter(
            data=barycentric_coordinates_data, requires_grad=True)

    def configure_optimizers(self):
        r"""Returns optimizers for dictionary variables."""
        if self.grad_labels:
            return torch.optim.Adam([
                {'params': self.XP,
                 'lr': self.learning_rate_features},
                {'params': self.YP,
                 'lr': self.learning_rate_labels},
                {'params': self.barycentric_coordinates,
                 'lr': self.learning_rate_weights}
            ])
        else:
            return torch.optim.Adam([
                {'params': self.XP,
                 'lr': self.learning_rate_features},
                {'params': self.barycentric_coordinates,
                 'lr': self.learning_rate_weights}
            ])

    def get_atoms(self):
        r"""Gets a list containing atoms features and labels."""
        with torch.no_grad():
            XP = torch.stack([XPk.data.cpu().clone() for XPk in self.XP])
            YP = torch.stack([
                YPk.softmax(dim=1).data.cpu().clone() for YPk in self.YP])
        return (XP, YP)

    def get_barycentric_coordinates(self):
        r"""Gets barycentric coordinates."""
        with torch.no_grad():
            if (self.barycentric_coordinates.sum(dim=1) == 1).all():
                return self.barycentric_coordinates.data.cpu()
            else:
                return proj_simplex(
                    self.barycentric_coordinates.data.cpu().T).T

    def sample_from_atoms(self, n=None, detach=False):
        """Samples (with replacement) $n$ samples from atoms support.

        Parameters
        ----------
        n : int, optional (default=None)
            Number of samples (with replacement) acquired from the atoms
            support. If None, gets all samples from the atoms supports.
        detach : bool, optional (default=False).
            If True, detaches tensors so that gradients are not calculated.
        """
        batch_features, batch_labels = [], []

        # Determining the number of samples
        if n is not None:
            samples_per_class = n // self.n_classes
        else:
            samples_per_class = None

        # Sampling
        for XPk, YPk in zip(self.XP, self.YP):
            # If balanced sampling, needs to select sampler_per_class
            # from each class
            if self.balanced_sampling:
                # Gets categorical labels
                yPk = YPk.detach().cpu().argmax(dim=1)
                # Initializes list of sampled indices
                sampled_indices = []
                # Loops over each class
                for yu in yPk.unique():
                    # Gets indices from current class
                    ind = torch.where(yPk == yu)[0]
                    # Randomly permutes labels
                    perm = torch.randperm(len(ind))
                    ind = ind[perm]
                    if samples_per_class is None:
                        # If n was not given, samples all samples
                        # from the said class
                        sampled_indices.append(ind[:])
                    else:
                        # Samples "samples_per_class" from given class
                        sampled_indices.append(ind[:samples_per_class])
                # Concatenates all indices
                sampled_indices = torch.cat(sampled_indices, dim=0)
            else:
                # In this case, we randomly select samples
                sampled_indices = np.random.choice(np.arange(self.n_samples),
                                                   size=n)

            # Creates batch arrays
            features_k, labels_k = XPk[sampled_indices], YPk[sampled_indices]

            if self.grad_labels:
                labels_k = labels_k.softmax(dim=-1)

            if detach:
                features_k, labels_k = features_k.detach(), labels_k.detach()

            batch_features.append(features_k)
            batch_labels.append(labels_k)

        return batch_features, batch_labels

    def fit(self,
            datasets,
            n_iter_max=100,
            batches_per_it=10,
            verbose=True):
        r"""Minimizes DaDiL's objective function by sampling
        mini-batches from the atoms support with replacement.

        Parameters
        ----------
        datasets : list of measures
            List of measure objects, which implement sampling from
            datasets.
        n_iter_max : int, optional (default=100)
            Number of epoch in DaDiL's optimization
        batches_per_it : int, optional (default=10)
            Number of batches drawn from the atoms per iteration.
        verbose : bool, optional (default=True)
            If True, prints progress of DaDiL's Optimization loop.
        """
        batch_size = datasets[0].batch_size
        for it in range(n_iter_max):
            # Calculates the loss
            avg_it_loss = 0
            avg_it_loss_per_dataset = {
                self.domain_names[ℓ]: 0 for ℓ in range(len(datasets))}
            if verbose:
                pbar = tqdm(range(batches_per_it))
            else:
                pbar = range(batches_per_it)
            for _ in pbar:
                self.optimizer.zero_grad()

                loss = 0
                for ℓ, (Qℓ, barycentric_coordinates_ℓ) in enumerate(zip(
                     datasets, self.barycentric_coordinates)):
                    # Sample minibatch from dataset
                    XQℓ, YQℓ = Qℓ.sample()

                    # Sample minibatch from atoms
                    XP, YP = self.sample_from_atoms(n=batch_size)

                    # Calculates Wasserstein barycenter
                    XBℓ, YBℓ = self.barycenter_solver(
                        XP, YP, barycentric_coordinates_ℓ
                    )

                    # Calculates Loss
                    loss_ℓ = self.loss_fn(XQ=XQℓ, YQ=YQℓ, XP=XBℓ, YP=YBℓ)

                    # Accumulates loss
                    loss += loss_ℓ
                    loss_val = loss_ℓ.detach().cpu().item() / batches_per_it
                    avg_it_loss_per_dataset[self.domain_names[ℓ]] += loss_val

                loss.backward()
                self.optimizer.step()

                # Projects the weights into the simplex
                with torch.no_grad():
                    self.barycentric_coordinates.data = proj_simplex(
                        self.barycentric_coordinates.data.cpu().T).T.to(
                            self.device)

                avg_it_loss += loss.item() / batches_per_it
            # Saves history info
            _XP, _YP = self.get_atoms()
            self.history['atoms_features'].append(_XP)
            self.history['atoms_labels'].append(_YP)
            self.history['weights'].append(
                proj_simplex(self.barycentric_coordinates.data.cpu().T).T.to(
                    self.device))
            self.history['loss'].append(avg_it_loss)
            for ℓ in range(len(datasets)):
                self.history['loss_per_dataset'][self.domain_names[ℓ]].append(
                    avg_it_loss_per_dataset[self.domain_names[ℓ]]
                )
            if verbose:
                print('It {}/{}, Loss: {}'.format(it, n_iter_max, avg_it_loss))
            if self.schedule_lr:
                self.scheduler.step(avg_it_loss)
        self.fitted = True
