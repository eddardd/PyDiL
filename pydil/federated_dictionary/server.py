import torch
import numpy as np


class DictionaryServer:
    def __init__(self,
                 atoms_features=None,
                 atoms_labels=None,
                 n_components=2,
                 n_samples=None,
                 n_dim=None,
                 n_classes=None,
                 aggregation='random',
                 track_atoms=False):
        assert aggregation.lower() in ['random', 'avg'], (
            "Expected 'aggregation' to be either 'random' or"
            f" 'avg', but got {aggregation.lower}"
        )

        self.n_components = n_components
        self.n_samples = n_samples
        self.n_dim = n_dim
        self.n_classes = n_classes
        self.aggregation = aggregation.lower()
        self.track_atoms = track_atoms
        self.__initialize_atoms_features(atoms_features)
        self.__initialize_atoms_labels(atoms_labels)

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

    def avg_aggregation(self, XP_versions, YP_versions):
        """Aggregates different versions of atoms via linear interpolation."""
        XPv, YPv = torch.stack(XP_versions), torch.stack(YP_versions)
        new_XP = [
            XPv.mean(dim=0)[k, ...]
            for k in range(self.n_components)]
        new_YP = [
            YPv.mean(dim=0)[k, ...]
            for k in range(self.n_components)]
        return new_XP, new_YP

    def random_aggregation(self, XP_versions, YP_versions):
        """Aggregates different versions of atoms by randomly choosing
        one version."""
        available_versions = np.arange(len(XP_versions))
        selected_client = np.random.choice(available_versions)
        new_XP, new_YP = [], []
        for k in range(self.n_components):
            new_XP.append(XP_versions[selected_client][k])
            new_YP.append(YP_versions[selected_client][k])
        return new_XP, new_YP

    def aggregate(self, XP_versions, YP_versions):
        """Aggregates different versions of atoms by different rules."""
        if len(XP_versions) > 1:
            """Multiple versions of atoms exist. We need to aggregate
            them."""
            if self.aggregation == 'avg':
                return self.avg_aggregation(XP_versions, YP_versions)
            elif self.aggregation == 'random':
                return self.random_aggregation(XP_versions, YP_versions)
        else:
            """Only one version of atoms exist. Return as is."""
            return XP_versions[0], YP_versions[0]

    def fit(self,
            clients,
            batch_size,
            n_iter,
            n_client_iter,
            C,
            verbose=False):
        client_list = np.arange(len(clients))
        n_sampled_clients = max([
            np.round(len(clients) * C).astype(int), 1
        ])
        for it in range(n_iter):
            if verbose:
                message = f"Round {it}"
                print(message)
                print('-' * len(message))

            XP_versions = []
            YP_versions = []

            selected_clients = np.random.choice(client_list,
                                                size=n_sampled_clients,
                                                replace=False)

            for selected_client in selected_clients:
                if verbose:
                    message = f'Client {selected_client}'
                    print(message)
                    print('-' * len(message))

                client = clients[selected_client]

                # Creates local copies of atom variables
                local_XP = [XPk.clone() for XPk in self.XP]
                local_YP = [YPk.clone() for YPk in self.YP]

                _XP, _YP = client.client_update(
                    local_XP,
                    local_YP,
                    batch_size=batch_size,
                    n_iter=n_client_iter,
                    verbose=verbose)

                XP_versions.append(torch.stack(_XP))
                YP_versions.append(torch.stack(_YP))

                if verbose:
                    print('\n')

            if self.track_atom_versions:
                self.history['versions'].append(
                    [[XPv.clone() for XPv in XP_versions],
                     [YPv.clone() for YPv in YP_versions]]
                )

            # Aggregates multiple versions
            new_XP, new_YP = self.aggregate(XP_versions, YP_versions)
            for k in range(self.n_components):
                self.XP[k].data = new_XP[k].data.clone()
                self.YP[k].data = new_YP[k].data.clone()

            if self.track_atoms:
                self.history['atoms'].append(
                    [[XPk.clone() for XPk in self.XP],
                     [YPk.clone() for YPk in self.YP]]
                )
