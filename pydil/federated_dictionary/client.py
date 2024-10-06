import torch
import numpy as np
from pydil.optimal_transport.pot_utils import proj_simplex
from pydil.optimal_transport.losses import JointWassersteinDistance


class DictionaryClient(torch.nn.Module):
    def __init__(self,
                 barycenter_solver,
                 client_features,
                 client_labels=None,
                 atoms_features=None,
                 atoms_labels=None,
                 n_samples=1024,
                 n_dim=None,
                 n_classes=None,
                 n_components=2,
                 loss_fn=None,
                 learning_rate_features=1e-1,
                 learning_rate_labels=None,
                 learning_rate_weights=None,
                 tensor_dtype=torch.float32,
                 device='cpu'):
        super(DictionaryClient, self).__init__()

        self.client_features = client_features
        self.client_labels = client_labels

        self.n_samples = n_samples
        self.n_dim = n_dim
        self.n_classes = n_classes
        self.n_components = n_components

        self.tensor_dtype = tensor_dtype
        self.device = torch.device(device)

        self.client_labels_cat = client_labels.argmax(dim=1)
        self.samples_per_class = self.n_samples // self.n_classes

        self.barycenter_solver = barycenter_solver
        self.barycentric_coordinates = torch.ones(
            self.n_components) / self.n_components
        self.barycentric_coordinates.requies_grad = True

        if learning_rate_labels is None:
            self.learning_rate_labels = learning_rate_features
        else:
            self.learning_rate_labels = learning_rate_labels

        if learning_rate_weights is None:
            self.learning_rate_weights = self.learning_rate_features
        else:
            self.learning_rate_weights = learning_rate_weights

        if loss_fn is None:
            self.loss_fn = JointWassersteinDistance()
        else:
            self.loss_fn = loss_fn

        self.__initialize_atoms_features(atoms_features)
        self.__initialize_atoms_labels(atoms_labels)

        # Initializes optimizer
        self.optimizer = self.get_optimizer()

        # History
        self.history = {
            'barycentric_coordinates': [
                self.barycentric_coordinates.data.clone()],
            'loss': []
        }

    def get_optimizer(self):
        if self.optimizer_name == 'sgd':
            optimizer = torch.optim.SGD([
                {'params': self.XP, 'lr': self.lr, 'momentum': self.momentum},
                {'params': self.YP, 'lr': self.lr, 'momentum': self.momentum},
                {'params': self.weights, 'lr': self.lr_weights,
                 'momentum': self.momentum}
            ])
        else:
            optimizer = torch.optim.Adam([
                {'params': self.XP, 'lr': self.lr},
                {'params': self.YP, 'lr': self.lr},
                {'params': self.weights, 'lr': self.lr_weights}
            ])
        return optimizer

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

    def sample_from_data(self, n=None):
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

    def client_update(self,
                      XP,
                      YP,
                      batch_size,
                      n_iter,
                      verbose=False):
        # Clones global versions into local ones
        with torch.no_grad():
            for k in range(self.n_components):
                self.XP[k].data = XP[k].data.clone()
                self.YP[k].data = YP[k].data.clone()

        for it in range(n_iter):
            # Define mini-batch indices for atoms
            n_batches = len(XP[0]) // batch_size
            indices = np.arange(len(XP[0]))
            np.random.shuffle(indices)
            batch_indices = [
                indices[i * batch_size: (i + 1) * batch_size]
                for i in range(n_batches)
            ]

            it_loss = 0.0
            for batch_idx in batch_indices:
                self.optimizer.zero_grad()

                # Samples from data
                XQ, YQ = self.sample_from_data(n=batch_size)

                # Samples from atoms
                minibatch_XP = [
                    XPk[batch_idx_k]
                    for XPk, batch_idx_k in zip(self.XP, batch_idx)
                ]
                minibatch_YP = [
                    YPk[batch_idx_k].softmax(dim=1)
                    for YPk, batch_idx_k in zip(self.YP, batch_idx)
                ]

                # Computes barycenter
                XB, YB = self.barycenter_solver(
                    minibatch_XP, minibatch_YP, self.barycentric_coordinates
                )

                # Computes loss
                loss = self.loss_fn(XB, YB, XQ, YQ)

                # Backprop
                loss.backward()

                # Optimizer step
                self.optimizer.step()

                # Projection into the simplex
                with torch.no_grad():
                    self.barycentric_coordinates.data = proj_simplex(
                        self.barycentric_coordinates.data)

                # Accumulates it loss
                it_loss += loss.item()
            it_loss /= len(batch_indices)
            self.history['loss'].append(it_loss)
            with torch.no_grad():
                w = proj_simplex(
                    self.barycentric_coordinates.data.clone())
                self.history['weights'].append(w)
            if verbose:
                print(f"Local iter {it}, loss {it_loss}")
        return ([XPk.detach().data for XPk in self.XP],
                [YPk.detach().data for YPk in self.YP])
