<p align="center">
  <img src="assets/pydil.svg" width="300"/>
</p>

# PyDiL

A Python package for distribution learning through dictionary learning. In this repository, we implement several methods for learning a set of atoms, that interpolate a set of probability measures given as input. These probability measures can have different kinds of encodings, such as,

- Empirical measures, which are represented through a mixture of diracs positioned at the samples from a given probability measure.
- Gaussian measure, which are represented through a mean and a covariance matrix. These parameters can be estimated from samples
- Gaussian mixture models, which are represented through a set of weights, means and covariances. These parameters can be estimated from data through the celebrated Expectation-Maximization algorithm. In our methods, we use axis-aligned Gaussian mixtures (diagonal covariances) which are easier and faster to estimate.

Our methods can be applied, for instance, for multi-source domain adaptation.

## Associated Publications

[1] Montesuma, E. F., & Mboula, F. M. N. (2021). Wasserstein barycenter for multi-source domain adaptation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 16785-16793).

[2] Montesuma, E. F., & Mboula, F. M. N. (2021, June). Wasserstein Barycenter Transport for Acoustic Adaptation. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 3405-3409). IEEE.

[3] Eduardo Fernandes Montesuma, Fred Maurice Ngole Mboula, and Antoine Souloumiac, ‘Multi-source domain adaptation through dataset dictionary learning in wasserstein space’, in Submitted to the 26th European Conference on Artificial Intelligence, (2023).

[4] Castellon, F. E., Montesuma, E. F., Mboula, F. N., Mayoue, A., Souloumiac, A., & Gouy-Pailler, C. (2024, April). Federated dataset dictionary learning for multi-source domain adaptation. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 5610-5614). IEEE.

[5] Montesuma, E. F., Mboula, F. N., & Souloumiac, A. (2024, August). Lighter, Better, Faster Multi-source Domain Adaptation with Gaussian Mixture Models and Optimal Transport. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 21-38). Cham: Springer Nature Switzerland.

[6] Fernandes Montesuma, E., Espinoza Castellon, F., Ngolè Mboula, F., Mayoue, A., Souloumiac, A., & Gouy-Pailler, C. (2024). Dataset Dictionary Learning in a Wasserstein Space for Federated Domain Adaptation. arXiv e-prints, arXiv-2407.

[7] Montesuma, E. F., Stanc, S. L., & Mboula, F. N. (2024). Online Multi-Source Domain Adaptation through Gaussian Mixtures and Dataset Dictionary Learning. arXiv preprint arXiv:2407.19853.
