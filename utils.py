import os
import torch
import shutil
import numpy as np

from skimage.util.dtype import dtype_range
from scipy.ndimage import uniform_filter, gaussian_filter


def remove_folder(folder):
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath) or os.path.islink(filepath):
            os.unlink(filepath)
        elif os.path.isdir(filepath):
            shutil.rmtree(filepath)


def load_checkpoint(model, optimizer, file):
    if os.path.isfile(file):
        print("[*] Loading checkpoint '" + file + "'.")
        checkpoint = torch.load(file)
        epoch = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        model.load_state_dict(checkpoint["model"])
        print("[*] Loaded checkpoint '" + file +
              "' (epoch " + str(epoch) + ").")

        return epoch
    else:
        print("[!] No checkpoint found at '" + file + "'.")

        return 0


def structural_sim(X, Y, win_size=None, data_range=None, multichannel=False, gaussian_weights=False,
                   full=False, **kwargs):
    """Compute the individual componentes of the structural similarity index between two images.

    Parameters
    ----------
    X, Y : ndarray
        Image. Any dimensionality.
    win_size : int or None
        The side-length of the sliding window used in comparison. Must be an
        odd value. If 'gaussian_weights' is True, this is ignored and the
        window size will depend on 'sigma'.
    data_range : int, optional
        The data range of the input image (distance between minimum and
        maximum possible values). By default, this is estimated from the image
        data-type.
    multichannel : bool, optional
        If True, treat the last dimension of the array as channels. Similarity
        calculations are done independently for each channel then averaged.
    gaussian_weights : bool, optional
        If True, each patch has its mean and variance spatially weighted by a
        normalized Gaussian kernel of width sigma=1.5.
    full : bool, optional
        If True, return the full structural similarity image instead of the
        mean value.

    Other Parameters
    ----------------
    use_sample_covariance : bool
        If True, normalize covariances by N-1 rather than, N where N is the
        number of pixels within the sliding window.
    K1 : float
        algorithm parameter, K1 (small constant, see [1]_)
    K2 : float
        algorithm parameter, K2 (small constant, see [1]_)
    sigma : float
        sigma for the Gaussian when 'gaussian_weights' is True.

    Returns
    -------
    L : ndarray
        The full map for the luminance component of SSIM.
    C : ndarray
        The full map for the contrast component of SSIM.
    S : ndarray
        The full map for the structural component of SSIM.

    Notes
    -----
    -> The particular value of each component should be computed as the mean value of the corresponding map.
    -> The full SSIM map should be computed as the pixel-wise product of the three individual maps.
       i.e., SSIM_map = L_map * C_map * S_map.

    Notes 2
    -----
    To match the implementation of Wang et. al. [1]_, set 'gaussian_weights'
    to True, 'sigma' to 1.5, and 'use_sample_covariance' to False.
    References
    ----------
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
       (2004). Image quality assessment: From error visibility to
       structural similarity. IEEE Transactions on Image Processing,
       13, 600-612.
       https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
       DOI:10.1.1.11.2477
    .. [2] Avanaki, A. N. (2009). Exact global histogram specification
       optimized for structural similarity. Optical Review, 16, 613-621.
       http://arxiv.org/abs/0901.0065,
       DOI:10.1007/s10043-009-0119-z
    """

    if not X.dtype == Y.dtype:
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if multichannel:
        # Loop over channels
        args = dict(win_size=win_size, data_range=data_range,
                    multichannel=False, gaussian_weights=gaussian_weights, full=full)
        args.update(kwargs)
        nch = X.shape[-1]
        L = np.empty(X.shape)
        C = np.empty(X.shape)
        S = np.empty(X.shape)
        for ch in range(nch):
            ch_result = structural_sim(X[..., ch], Y[..., ch], **args)
            L[..., ch], C[..., ch], S[..., ch] = ch_result

        return L, C, S

    K1 = kwargs.pop('K1', 0.01)
    K2 = kwargs.pop('K2', 0.03)
    sigma = kwargs.pop('sigma', 1.5)

    if K1 < 0:
        raise ValueError("K1 must be positive")
    if K2 < 0:
        raise ValueError("K2 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be positive")

    use_sample_covariance = kwargs.pop('use_sample_covariance', True)

    if win_size is None:
        if gaussian_weights:
            win_size = 11  # 11 to match Wang et. al. 2004
        else:
            win_size = 7   # Backwards compatibility

    if np.any((np.asarray(X.shape) - win_size) < 0):
        raise ValueError(
            "win_size exceeds image extent.  If the input is a multichannel (color) image, set multichannel=True.")

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if data_range is None:
        dmin, dmax = dtype_range[X.dtype.type]
        data_range = dmax - dmin

    ndim = X.ndim

    if gaussian_weights:
        # sigma = 1.5 to approximately match filter in Wang et. al. 2004
        # this ends up giving a 13-tap rather than 11-tap Gaussian
        filter_func = gaussian_filter
        filter_args = {'sigma': sigma}
    else:
        filter_func = uniform_filter
        filter_args = {'size': win_size}

    # ndimage filters need floating point data
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    NP = win_size ** ndim

    # Filter has already normalized by NP
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)  # Sample covariance
    else:
        cov_norm = 1.0  # Population covariance to match Wang et. al. 2004

    # Compute (weighted) means
    ux = filter_func(X, **filter_args)
    uy = filter_func(Y, **filter_args)

    # Compute (weighted) variances and covariances
    uxx = filter_func(X * X, **filter_args)
    uyy = filter_func(Y * Y, **filter_args)
    uxy = filter_func(X * Y, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vx[vx < 0] = 0
    vy = cov_norm * (uyy - uy * uy)
    vy[vy < 0] = 0
    vxy = cov_norm * (uxy - ux * uy)
    vxy[vxy < 0] = 0
    stdx = np.sqrt(vx)
    stdy = np.sqrt(vy)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    C3 = C2 / 2
    l1, l2 = (2 * ux * uy + C1, ux ** 2 + uy ** 2 + C1)
    c1, c2 = (2 * stdx * stdy + C2, vx + vy + C2)
    # Before: s1 = vxy + C3. But vxx = vx (Var(x) = Covar(x,x))
    s1, s2 = (vxy + C3, stdx * stdy + C3)

    L = l1 / l2
    C = c1 / c2
    S = s1 / s2

    return L, C, S
