import torch

from sklearn.metrics import accuracy_score
from pydil.gmm_dictionary.measures import diag_gmm_log_probs
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pydil.optimal_transport import OptimalTransportSolver
from pydil.optimal_transport.pot_utils import proj_simplex
from pydil.optimal_transport.losses import MixtureWassersteinDistance
from pydil.optimal_transport.losses import JointMixtureWassersteinDistance


class GaussianMixtureDictionary(torch.nn.Module):
    def __init__(self,
                 barycenter_solver,
                 atoms_means=None,
                 atoms_variances=None,
                 atoms_labels=None,
                 barycentric_coordinates=None,
                 n_components=20,
                 n_dim=None,
                 n_classes=None,
                 n_atoms=2,
                 weight_initialization='uniform',
                 n_measures=None,
                 loss_fn=None,
                 learning_rate=1e-1,
                 learning_rate_weights=None,
                 momentum=0.9,
                 domain_names=None,
                 grad_labels=True,
                 optimizer_name='adam',
                 tensor_dtype=torch.float32,
                 track_atoms=False,
                 track_nll=False,
                 schedule_lr=True,
                 device='cuda',
                 min_var=0.05):
        super(GaussianMixtureDictionary, self).__init__()

        self.barycenter_solver = barycenter_solver

        self.n_components = n_components
        self.n_dim = n_dim
        self.n_classes = n_classes
        self.n_atoms = n_atoms
        self.weight_initialization = weight_initialization
        self.n_measures = n_measures
        self.tensor_dtype = tensor_dtype
        self.learning_rate = learning_rate
        if learning_rate_weights is None:
            self.learning_rate_weights = learning_rate
        else:
            self.learning_rate_weights = learning_rate_weights
        self.device = torch.device(device)
        self.min_var = min_var

        if loss_fn is None:
            self.loss_fn = MixtureWassersteinDistance(
                ot_solver=OptimalTransportSolver())
        else:
            self.loss_fn = loss_fn

        if domain_names is None:
            if n_measures is None:
                raise ValueError(("If 'domain_names' is not given,"
                                  " 'n_measures' must be provided."))
            self.domain_names = [
                "Domain {}".format(ℓ) for ℓ in range(n_measures)]
        else:
            self.domain_names = domain_names

        self.grad_labels = grad_labels
        self.optimizer_name = optimizer_name
        self.track_atoms = track_atoms
        self.track_nll = track_nll
        self.schedule_lr = schedule_lr
        self.momentum = momentum

        self.__initialize_atoms_means(atoms_means)
        self.__initialize_atoms_variances(atoms_variances)
        self.__initialize_atoms_labels(atoms_labels)
        self.__initialize_barycentric_coordinates(barycentric_coordinates)

        self.history = {
            'loss': [],
            'nll': [],
            'acc': [],
            'weights': [],
            'atoms_means': [
                torch.stack([atoms_means_k.detach().clone()
                             for atoms_means_k in self.atoms_means])
            ],
            'atoms_variances': [
                torch.stack([atoms_variances_k.detach().clone()
                             for atoms_variances_k in self.atoms_variances])
            ],
            'atoms_labels': [
                torch.stack([atoms_labels_k.detach().clone()
                             for atoms_labels_k in self.atoms_labels])
            ],
            'loss_per_dataset': {name: [] for name in self.domain_names},
            'nll_per_dataset': {name: [] for name in self.domain_names},
            'acc_per_dataset': {name: [] for name in self.domain_names}
        }

        self.optimizer = self.configure_optimizers()
        if self.schedule_lr:
            self.scheduler = ReduceLROnPlateau(self.optimizer)

    def __initialize_atoms_means(self, atoms_means=None):
        if atoms_means is None:
            if self.n_dim is None:
                raise ValueError(("If 'atoms_means' is not given,"
                                  " you should specify 'n_dim'."))
            atoms_means_data = [
                torch.randn(self.n_components, self.n_dim,
                            requires_grad=True,
                            dtype=self.tensor_dtype,
                            device=self.device)
                for _ in range(self.n_atoms)
            ]
            self.atoms_means = torch.nn.ParameterList(
                [torch.nn.parameter.Parameter(data=atoms_means,
                                              requires_grad=True)
                 for atoms_means in atoms_means_data]
            )
        else:
            self.atoms_means = torch.nn.ParameterList(
                [torch.nn.parameter.Parameter(data=atoms_means_k.to(
                                                self.device).to(
                                                    self.tensor_dtype),
                                              requires_grad=True)
                    for atoms_means_k in atoms_means]
            )
            self.n_dim = atoms_means[0].shape[1]

    def __initialize_atoms_variances(self, atoms_variances=None):
        if atoms_variances is None:
            if self.n_dim is None:
                raise ValueError(("If 'UP' is not given,"
                                  " you should specify 'n_dim'."))
            atoms_variances_data = [
                torch.ones(self.n_components, self.n_dim,
                           requires_grad=True,
                           dtype=self.tensor_dtype,
                           device=self.device)
                for _ in range(self.n_atoms)
            ]
            self.atoms_variances = torch.nn.ParameterList(
                [torch.nn.parameter.Parameter(data=atoms_variances,
                                              requires_grad=True)
                 for atoms_variances in atoms_variances_data]
            )
        else:
            self.atoms_variances = torch.nn.ParameterList(
                [torch.nn.parameter.Parameter(
                    data=atoms_variances_k.to(self.tensor_dtype),
                    requires_grad=True)
                    for atoms_variances_k in atoms_variances]
            )
            self.n_dim = atoms_variances[0].shape[1]

    def __initialize_atoms_labels(self, YP=None):
        if YP is None:
            torch.stack([
                torch.ones(
                    self.n_components, self.n_classes,
                    dtype=self.tensor_dtype, device=self.device)
                for _ in range(self.n_atoms)
            ])
        else:
            self.atoms_labels = torch.nn.ParameterList(
                [torch.nn.parameter.Parameter(data=yp.to(self.tensor_dtype),
                                              requires_grad=self.grad_labels)
                 for yp in YP]
            )
            self.n_classes = YP[0].shape[1]

    def __initialize_barycentric_coordinates(
            self, barycentric_coordinates=None):
        if barycentric_coordinates is None:
            if self.n_measures is None:
                raise ValueError(("If 'A' is not given you"
                                  " should specify 'n_measures'"))
            if self.weight_initialization == 'random':
                barycentric_coordinates_data = torch.rand(
                    self.n_measures, self.n_atoms,
                    requires_grad=True, dtype=self.tensor_dtype,
                    device=self.device)
            else:
                barycentric_coordinates_data = torch.ones(
                    self.n_measures, self.n_atoms,
                    requires_grad=True, dtype=self.tensor_dtype,
                    device=self.device)
            with torch.no_grad():
                barycentric_coordinates_data = proj_simplex(
                    barycentric_coordinates_data.cpu().T).T
        else:
            barycentric_coordinates_data = barycentric_coordinates
        self.barycentric_coordinates = torch.nn.parameter.Parameter(
            data=barycentric_coordinates_data, requires_grad=True)

    def configure_optimizers(self):
        r"""Returns optimizers for dictionary variables."""
        if self.optimizer_name == 'adam':
            if self.grad_labels:
                return torch.optim.Adam([
                    {'params': self.atoms_means, 'lr': self.learning_rate},
                    {'params': self.atoms_variances, 'lr': self.learning_rate},
                    {'params': self.atoms_labels, 'lr': self.learning_rate},
                    {'params': self.barycentric_coordinates,
                     'lr': self.learning_rate_weights}
                ])
            else:
                return torch.optim.Adam([
                    {'params': self.atoms_means, 'lr': self.learning_rate},
                    {'params': self.atoms_variances, 'lr': self.learning_rate},
                    {'params': self.barycentric_coordinates,
                     'lr': self.learning_rate_weights}
                ])
        else:
            if self.grad_labels:
                return torch.optim.SGD([
                    {'params': self.atoms_means,
                     'lr': self.learning_rate,
                     'momentum': self.momentum},
                    {'params': self.atoms_variances,
                     'lr': self.learning_rate,
                     'momentum': self.momentum},
                    {'params': self.atoms_labels,
                     'lr': self.learning_rate,
                     'momentum': self.momentum},
                    {'params': self.barycentric_coordinates,
                     'lr': self.learning_rate_weights,
                     'momentum': self.momentum}
                ])
            else:
                return torch.optim.SGD([
                    {'params': self.atoms_means,
                     'lr': self.learning_rate,
                     'momentum': self.momentum},
                    {'params': self.atoms_variances,
                     'lr': self.learning_rate,
                     'momentum': self.momentum},
                    {'params': self.barycentric_coordinates,
                     'lr': self.learning_rate_weights,
                     'momentum': self.momentum}
                ])

    def get_atoms(self):
        r"""Gets a list containing atoms means, variances and labels."""
        with torch.no_grad():
            means = torch.stack([
                atoms_means_k.data.cpu()
                for atoms_means_k in self.atoms_means])
            variances = torch.stack([
                atoms_variances_k.data.cpu()
                for atoms_variances_k in self.atoms_variances])
            labels = torch.stack([
                atoms_labels_k.data.cpu()
                for atoms_labels_k in self.atoms_labels])
        return means, variances, labels

    def get_weights(self):
        r"""Gets barycentric coordinates."""
        with torch.no_grad():
            if (self.barycentric_coordinates.sum(dim=1) == 1).all():
                return self.barycentric_coordinates.data.cpu()
            else:
                return proj_simplex(
                    self.barycentric_coordinates.data.cpu().T).T

    def fit(self,
            gmms,
            n_iter_max=100,
            verbose=True,
            validation_data=None):
        for it in range(n_iter_max):
            # Calculates the loss
            loss_per_dataset = {
                self.domain_names[ℓ]: 0 for ℓ in range(len(gmms))}
            nll_per_dataset = {
                self.domain_names[ℓ]: 0 for ℓ in range(len(gmms))}
            acc_per_dataset = {
                self.domain_names[ℓ]: 0 for ℓ in range(len(gmms))}

            self.optimizer.zero_grad()

            loss = 0
            avg_nll = 0
            avg_acc = 0
            for ℓ, (gmm_ℓ, coords_ℓ) in enumerate(
                    zip(gmms, self.barycentric_coordinates)):
                weights_ℓ = gmm_ℓ.weights.to(
                    self.tensor_dtype).to(self.device)
                means_ℓ = gmm_ℓ.means.to(
                    self.tensor_dtype).to(self.device)
                variances_ℓ = gmm_ℓ.variances.to(
                    self.tensor_dtype).to(self.device)
                if gmm_ℓ.labels is not None:
                    labels_ℓ = gmm_ℓ.labels.to(
                        self.tensor_dtype).to(self.device)
                else:
                    labels_ℓ = None

                if self.grad_labels:
                    atoms_labels = [
                        atoms_labels_k.softmax(dim=1)
                        for atoms_labels_k in self.atoms_labels]
                else:
                    atoms_labels = self.atoms_labels

                params_barycenter = self.barycenter_solver(
                    means=self.atoms_means,
                    variances=self.atoms_variances,
                    labels=atoms_labels,
                    weights=coords_ℓ
                )

                # Calculates Loss
                weights_barycenter = torch.ones(
                    self.n_components) / self.n_components
                weights_barycenter.to(self.tensor_dtype).to(self.device)
                loss_ℓ = self.loss_fn(
                    means_P=params_barycenter[0],
                    variances_P=params_barycenter[1],
                    labels_P=params_barycenter[2],
                    means_Q=means_ℓ,
                    variances_Q=variances_ℓ,
                    labels_Q=labels_ℓ,
                    p=weights_barycenter,
                    q=weights_ℓ)

                # Accumulates loss
                loss += loss_ℓ
                loss_val = loss_ℓ.detach().cpu().item()
                loss_per_dataset[self.domain_names[ℓ]] = loss_val

                if self.track_nll and validation_data is not None:
                    X, Y = validation_data[ℓ]
                    with torch.no_grad():
                        stds = torch.sqrt(params_barycenter[1])
                        log_probs = diag_gmm_log_probs(
                            X.to(self.tensor_dtype),
                            weights_barycenter,
                            params_barycenter[0],
                            stds)
                        nll = - log_probs.logsumexp(dim=0).mean().item()
                        proba_components = (
                            log_probs - log_probs.logsumexp(dim=0)[None, :]
                        ).exp()
                        cluster_labels = torch.mm(
                            params_barycenter[2].T, proba_components).T
                        acc = accuracy_score(
                            cluster_labels.argmax(dim=1),
                            Y.to(self.tensor_dtype).argmax(dim=1))

                        avg_nll += nll / len(gmms)
                        avg_acc += acc / len(gmms)

                        nll_per_dataset[self.domain_names[ℓ]] = nll
                        acc_per_dataset[self.domain_names[ℓ]] = acc
                        print(
                            f"[{self.domain_names[ℓ]}] nll: {nll}, acc: {acc}")
            loss.backward()
            self.optimizer.step()

            # Projects the weights into the simplex
            with torch.no_grad():
                self.barycentric_coordinates.data = proj_simplex(
                    self.barycentric_coordinates.data.cpu().T).T
                for VPk in self.atoms_variances:
                    VPk.data[VPk.data < self.min_var] = self.min_var
            # Saves history info
            _CP, _VP, _YP = self.get_atoms()
            if self.track_atoms:
                self.history['atoms_means'].append(_CP)
                self.history['atoms_variances'].append(_VP)
                self.history['atoms_labels'].append(_YP)
            if self.track_nll:
                self.history['nll'].append(avg_nll)
                self.history['acc'].append(avg_acc)
            self.history['weights'].append(proj_simplex(
                self.barycentric_coordinates.data.cpu().T).T)
            self.history['loss'].append(loss.detach().cpu().item())
            for ℓ in range(len(gmms)):
                self.history['loss_per_dataset'][self.domain_names[ℓ]].append(
                    loss_per_dataset[self.domain_names[ℓ]]
                )
                self.history['nll_per_dataset'][self.domain_names[ℓ]].append(
                    nll_per_dataset[self.domain_names[ℓ]]
                )
                self.history['acc_per_dataset'][self.domain_names[ℓ]].append(
                    acc_per_dataset[self.domain_names[ℓ]]
                )
            if verbose:
                print('It {}/{}, Loss: {}'.format(
                    it, n_iter_max, loss.detach().cpu().item()))
            if self.schedule_lr:
                self.scheduler.step(loss.detach().cpu().item())
        self.fitted = True


class LabeledGaussianMixtureDictionary(torch.nn.Module):
    def __init__(self,
                 barycenter_solver,
                 atoms_means=None,
                 atoms_variances=None,
                 atoms_labels=None,
                 barycentric_coordinates=None,
                 n_components=20,
                 n_dim=None,
                 n_classes=None,
                 n_atoms=2,
                 weight_initialization='uniform',
                 n_measures=None,
                 loss_fn=None,
                 learning_rate=1e-1,
                 learning_rate_weights=None,
                 momentum=0.9,
                 domain_names=None,
                 grad_labels=True,
                 optimizer_name='adam',
                 tensor_dtype=torch.float32,
                 track_atoms=False,
                 track_nll=False,
                 schedule_lr=True,
                 device='cuda',
                 min_var=0.05):
        super(LabeledGaussianMixtureDictionary, self).__init__()

        self.barycenter_solver = barycenter_solver

        self.n_components = n_components
        self.n_dim = n_dim
        self.n_classes = n_classes
        self.n_atoms = n_atoms
        self.weight_initialization = weight_initialization
        self.n_measures = n_measures
        self.tensor_dtype = tensor_dtype
        self.learning_rate = learning_rate
        if learning_rate_weights is None:
            self.learning_rate_weights = learning_rate
        else:
            self.learning_rate_weights = learning_rate_weights
        self.device = torch.device(device)
        self.min_var = min_var

        if loss_fn is None:
            self.loss_fn = JointMixtureWassersteinDistance(
                ot_solver=OptimalTransportSolver())
        else:
            self.loss_fn = loss_fn

        if domain_names is None:
            if n_measures is None:
                raise ValueError(("If 'domain_names' is not given,"
                                  " 'n_measures' must be provided."))
            self.domain_names = [
                "Domain {}".format(ℓ) for ℓ in range(n_measures)]
        else:
            self.domain_names = domain_names

        self.grad_labels = grad_labels
        self.optimizer_name = optimizer_name
        self.track_atoms = track_atoms
        self.track_nll = track_nll
        self.schedule_lr = schedule_lr
        self.momentum = momentum

        self.__initialize_atoms_means(atoms_means)
        self.__initialize_atoms_variances(atoms_variances)
        self.__initialize_atoms_labels(atoms_labels)
        self.__initialize_barycentric_coordinates(barycentric_coordinates)

        self.history = {
            'loss': [],
            'nll': [],
            'acc': [],
            'weights': [],
            'atoms_means': [
                torch.stack([atoms_means_k.detach().clone()
                             for atoms_means_k in self.atoms_means])
            ],
            'atoms_variances': [
                torch.stack([atoms_variances_k.detach().clone()
                             for atoms_variances_k in self.atoms_variances])
            ],
            'atoms_labels': [
                torch.stack([atoms_labels_k.detach().clone()
                             for atoms_labels_k in self.atoms_labels])
            ],
            'loss_per_dataset': {name: [] for name in self.domain_names},
            'nll_per_dataset': {name: [] for name in self.domain_names},
            'acc_per_dataset': {name: [] for name in self.domain_names}
        }

        self.optimizer = self.configure_optimizers()
        if self.schedule_lr:
            self.scheduler = ReduceLROnPlateau(self.optimizer)

    def __initialize_atoms_means(self, atoms_means=None):
        if atoms_means is None:
            if self.n_dim is None:
                raise ValueError(("If 'atoms_means' is not given,"
                                  " you should specify 'n_dim'."))
            atoms_means_data = [
                torch.randn(self.n_components, self.n_dim,
                            requires_grad=True,
                            dtype=self.tensor_dtype,
                            device=self.device)
                for _ in range(self.n_atoms)
            ]
            self.atoms_means = torch.nn.ParameterList(
                [torch.nn.parameter.Parameter(data=atoms_means,
                                              requires_grad=True)
                 for atoms_means in atoms_means_data]
            )
        else:
            self.atoms_means = torch.nn.ParameterList(
                [torch.nn.parameter.Parameter(data=atoms_means_k.to(
                                                self.device).to(
                                                    self.tensor_dtype),
                                              requires_grad=True)
                    for atoms_means_k in atoms_means]
            )
            self.n_dim = atoms_means[0].shape[1]

    def __initialize_atoms_variances(self, atoms_variances=None):
        if atoms_variances is None:
            if self.n_dim is None:
                raise ValueError(("If 'UP' is not given,"
                                  " you should specify 'n_dim'."))
            atoms_variances_data = [
                torch.ones(self.n_components, self.n_dim,
                           requires_grad=True,
                           dtype=self.tensor_dtype,
                           device=self.device)
                for _ in range(self.n_atoms)
            ]
            self.atoms_variances = torch.nn.ParameterList(
                [torch.nn.parameter.Parameter(data=atoms_variances,
                                              requires_grad=True)
                 for atoms_variances in atoms_variances_data]
            )
        else:
            self.atoms_variances = torch.nn.ParameterList(
                [torch.nn.parameter.Parameter(
                    data=atoms_variances_k.to(self.tensor_dtype),
                    requires_grad=True)
                    for atoms_variances_k in atoms_variances]
            )
            self.n_dim = atoms_variances[0].shape[1]

    def __initialize_atoms_labels(self, atoms_labels=None):
        if atoms_labels is None:
            atoms_labels_data = torch.stack([
                torch.ones(
                    self.n_components, self.n_classes,
                    dtype=self.tensor_dtype, device=self.device)
                for _ in range(self.n_atoms)
            ])
        else:
            atoms_labels_data = atoms_labels
        self.atoms_labels = torch.nn.ParameterList(
            [torch.nn.parameter.Parameter(
                data=atoms_labels_data_k.to(self.tensor_dtype),
                requires_grad=self.grad_labels)
                for atoms_labels_data_k in atoms_labels_data]
        )
        self.n_classes = atoms_labels_data[0].shape[1]

    def __initialize_barycentric_coordinates(
            self, barycentric_coordinates=None):
        if barycentric_coordinates is None:
            if self.n_measures is None:
                raise ValueError(("If 'A' is not given you"
                                  " should specify 'n_measures'"))
            if self.weight_initialization == 'random':
                barycentric_coordinates_data = torch.rand(
                    self.n_measures, self.n_atoms,
                    requires_grad=True, dtype=self.tensor_dtype,
                    device=self.device)
            else:
                barycentric_coordinates_data = torch.ones(
                    self.n_measures, self.n_atoms,
                    requires_grad=True, dtype=self.tensor_dtype,
                    device=self.device)
            with torch.no_grad():
                barycentric_coordinates_data = proj_simplex(
                    barycentric_coordinates_data.cpu().T).T
        else:
            barycentric_coordinates_data = barycentric_coordinates
        self.barycentric_coordinates = torch.nn.parameter.Parameter(
            data=barycentric_coordinates_data, requires_grad=True)

    def configure_optimizers(self):
        r"""Returns optimizers for dictionary variables."""
        if self.optimizer_name == 'adam':
            if self.grad_labels:
                return torch.optim.Adam([
                    {'params': self.atoms_means, 'lr': self.learning_rate},
                    {'params': self.atoms_variances, 'lr': self.learning_rate},
                    {'params': self.atoms_labels, 'lr': self.learning_rate},
                    {'params': self.barycentric_coordinates,
                     'lr': self.learning_rate_weights}
                ])
            else:
                return torch.optim.Adam([
                    {'params': self.atoms_means, 'lr': self.learning_rate},
                    {'params': self.atoms_variances, 'lr': self.learning_rate},
                    {'params': self.barycentric_coordinates,
                     'lr': self.learning_rate_weights}
                ])
        else:
            if self.grad_labels:
                return torch.optim.SGD([
                    {'params': self.atoms_means,
                     'lr': self.learning_rate,
                     'momentum': self.momentum},
                    {'params': self.atoms_variances,
                     'lr': self.learning_rate,
                     'momentum': self.momentum},
                    {'params': self.atoms_labels,
                     'lr': self.learning_rate,
                     'momentum': self.momentum},
                    {'params': self.barycentric_coordinates,
                     'lr': self.learning_rate_weights,
                     'momentum': self.momentum}
                ])
            else:
                return torch.optim.SGD([
                    {'params': self.atoms_means,
                     'lr': self.learning_rate,
                     'momentum': self.momentum},
                    {'params': self.atoms_variances,
                     'lr': self.learning_rate,
                     'momentum': self.momentum},
                    {'params': self.barycentric_coordinates,
                     'lr': self.learning_rate_weights,
                     'momentum': self.momentum}
                ])

    def get_atoms(self):
        r"""Gets a list containing atoms means, variances and labels."""
        with torch.no_grad():
            means = torch.stack([
                atoms_means_k.data.cpu()
                for atoms_means_k in self.atoms_means])
            variances = torch.stack([
                atoms_variances_k.data.cpu()
                for atoms_variances_k in self.atoms_variances])
            labels = torch.stack([
                atoms_labels_k.data.cpu()
                for atoms_labels_k in self.atoms_labels])
        return means, variances, labels

    def get_weights(self):
        r"""Gets barycentric coordinates."""
        with torch.no_grad():
            if (self.barycentric_coordinates.sum(dim=1) == 1).all():
                return self.barycentric_coordinates.data.cpu()
            else:
                return proj_simplex(
                    self.barycentric_coordinates.data.cpu().T).T

    def fit(self,
            gmms,
            n_iter_max=100,
            verbose=True,
            validation_data=None):
        for it in range(n_iter_max):
            # Calculates the loss
            loss_per_dataset = {
                self.domain_names[ℓ]: 0 for ℓ in range(len(gmms))}
            nll_per_dataset = {
                self.domain_names[ℓ]: 0 for ℓ in range(len(gmms))}
            acc_per_dataset = {
                self.domain_names[ℓ]: 0 for ℓ in range(len(gmms))}

            self.optimizer.zero_grad()

            loss = 0
            avg_nll = 0
            avg_acc = 0
            for ℓ, (gmm_ℓ, coords_ℓ) in enumerate(
                    zip(gmms, self.barycentric_coordinates)):
                weights_ℓ = gmm_ℓ.weights.to(
                    self.tensor_dtype).to(self.device)
                means_ℓ = gmm_ℓ.means.to(
                    self.tensor_dtype).to(self.device)
                variances_ℓ = gmm_ℓ.variances.to(
                    self.tensor_dtype).to(self.device)
                if gmm_ℓ.labels is not None:
                    labels_ℓ = gmm_ℓ.labels.to(
                        self.tensor_dtype).to(self.device)
                else:
                    labels_ℓ = None

                if self.grad_labels:
                    atoms_labels = [
                        atoms_labels_k.softmax(dim=1)
                        for atoms_labels_k in self.atoms_labels]
                else:
                    atoms_labels = self.atoms_labels

                params_barycenter = self.barycenter_solver(
                    means=self.atoms_means,
                    variances=self.atoms_variances,
                    labels=atoms_labels,
                    weights=coords_ℓ
                )

                # Calculates Loss
                weights_barycenter = torch.ones(
                    self.n_components) / self.n_components
                weights_barycenter.to(self.tensor_dtype).to(self.device)
                loss_ℓ = self.loss_fn(
                    means_P=params_barycenter[0],
                    variances_P=params_barycenter[1],
                    labels_P=params_barycenter[2],
                    means_Q=means_ℓ,
                    variances_Q=variances_ℓ,
                    labels_Q=labels_ℓ,
                    p=weights_barycenter,
                    q=weights_ℓ)

                # Accumulates loss
                loss += loss_ℓ
                loss_val = loss_ℓ.detach().cpu().item()
                loss_per_dataset[self.domain_names[ℓ]] = loss_val

                if self.track_nll and validation_data is not None:
                    X, Y = validation_data[ℓ]
                    with torch.no_grad():
                        stds = torch.sqrt(params_barycenter[1])
                        log_probs = diag_gmm_log_probs(
                            X.to(self.tensor_dtype),
                            weights_barycenter,
                            params_barycenter[0],
                            stds)
                        nll = - log_probs.logsumexp(dim=0).mean().item()
                        proba_components = (
                            log_probs - log_probs.logsumexp(dim=0)[None, :]
                        ).exp()
                        cluster_labels = torch.mm(
                            params_barycenter[2].T, proba_components).T
                        acc = accuracy_score(
                            cluster_labels.argmax(dim=1),
                            Y.to(self.tensor_dtype).argmax(dim=1))

                        avg_nll += nll / len(gmms)
                        avg_acc += acc / len(gmms)

                        nll_per_dataset[self.domain_names[ℓ]] = nll
                        acc_per_dataset[self.domain_names[ℓ]] = acc
                        print(
                            f"[{self.domain_names[ℓ]}] nll: {nll}, acc: {acc}")
            loss.backward()
            self.optimizer.step()

            # Projects the weights into the simplex
            with torch.no_grad():
                self.barycentric_coordinates.data = proj_simplex(
                    self.barycentric_coordinates.data.cpu().T).T
                for VPk in self.atoms_variances:
                    VPk.data[VPk.data < self.min_var] = self.min_var
            # Saves history info
            _CP, _VP, _YP = self.get_atoms()
            if self.track_atoms:
                self.history['atoms_means'].append(_CP)
                self.history['atoms_variances'].append(_VP)
                self.history['atoms_labels'].append(_YP)
            if self.track_nll:
                self.history['nll'].append(avg_nll)
                self.history['acc'].append(avg_acc)
            self.history['weights'].append(proj_simplex(
                self.barycentric_coordinates.data.cpu().T).T)
            self.history['loss'].append(loss.detach().cpu().item())
            for ℓ in range(len(gmms)):
                self.history['loss_per_dataset'][self.domain_names[ℓ]].append(
                    loss_per_dataset[self.domain_names[ℓ]]
                )
                self.history['nll_per_dataset'][self.domain_names[ℓ]].append(
                    nll_per_dataset[self.domain_names[ℓ]]
                )
                self.history['acc_per_dataset'][self.domain_names[ℓ]].append(
                    acc_per_dataset[self.domain_names[ℓ]]
                )
            if verbose:
                print('It {}/{}, Loss: {}'.format(
                    it, n_iter_max, loss.detach().cpu().item()))
            if self.schedule_lr:
                self.scheduler.step(loss.detach().cpu().item())
        self.fitted = True
