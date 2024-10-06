import ot
import torch


def proj_simplex(v, z=1):
    r"""Re-implements ot.utils.proj_simplex using torch.
    This was necessary due to strange behavior of POT w.r.t. GPU memory."""
    n = v.shape[0]
    if v.ndim == 1:
        d1 = 1
        v = v[:, None]
    else:
        d1 = 0
    d = v.shape[1]

    # sort u in ascending order
    u, _ = torch.sort(v, dim=0)
    # take the descending order
    u = torch.flip(u, dims=[0])
    cssv = torch.cumsum(u, dim=0) - z
    ind = torch.arange(n, dtype=v.dtype)[:, None] + 1
    cond = u - cssv / ind > 0
    rho = torch.sum(cond, 0)
    theta = cssv[rho - 1, torch.arange(d)] / rho
    w = torch.maximum(v - theta[None, :], torch.zeros(v.shape, dtype=v.dtype))
    if d1:
        return w[:, 0]
    else:
        return w


def unif(n, device='cpu', dtype=torch.float32):
    r"""Returns uniform sample weights for a number of samples $n > 0$.

    Parameters
    ----------
    n : int
        Number of samples
    device : str, optional (default='cpu')
        Whether the returned tensor is on 'cpu' or 'gpu'.
    dtype : torch dtype, optional (default=torch.float32)
        Data type for the returned vector.
    """
    return torch.ones(n, device=device, dtype=dtype) / n


def emd(a, b, C, n_iter_max=100000):
    r"""Wrapper function for ```ot.emd```.

    NOTE: This function receives torch tensors and converts
    them to numpy, before calling the Earth Mover Distance
    function of POT.

    Parameters
    ----------
    a : tensor
        Tensor of shape (n,) containing the importance of each sample in the
        distribution P. Must be positive and sum to one.
    b : tensor
        Tensor of shape (m,) containing the importance of each sample in the
        distribution Q. Must be positive and sum to one.
    C : tensor
        Tensor of shape (n, m) containing the pairwise distance between samples
        of P and Q.
    n_iter_max : int, optional (default=1000000)
        Number of iterations of Linear Programming.
    """
    _a = a.detach().cpu().numpy()
    _b = b.detach().cpu().numpy()
    _C = C.detach().cpu().numpy()

    np_ot_plan = ot.emd(_a, _b, _C, numItermax=n_iter_max)

    return torch.from_numpy(np_ot_plan).to(C.dtype).to(C.device)
