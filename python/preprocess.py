import numpy as np
from scipy.linalg import svd
from scipy.signal import butter, filtfilt, lfilter


def p(x):
    print("shape", x.shape, "mean", "%5e" % x.mean())
    print(x[:2, :2])
    print()
    print(x[-2:, -2:])


def gpufilter(buff, channel_map=None, fs=None, fslow=None, fshigh=None, car=True):
    # filter this batch of data after common average referencing with the
    # median
    # buff is timepoints by channels
    # chanMap are indices of the channels to be kep
    # ops.fs and ops.fshigh are sampling and high-pass frequencies respectively
    # if ops.fslow is present, it is used as low-pass frequency (discouraged)

    # set up the parameters of the filter
    if fslow and fslow < fs / 2:
        b1, a1 = butter(3, (2 * fshigh / fs, 2 * fslow / fs), 'bandpass')  # butterworth filter with only 3 nodes (otherwise it's unstable for float32)
    else:
        b1, a1 = butter(3, fshigh / fs * 2, 'high')  # butterworth filter with only 3 nodes (otherwise it's unstable for float32)

    dataRAW = buff.T
    if channel_map is not None:
        dataRAW = dataRAW[:, channel_map]  # subsample only good channels

    # subtract the mean from each channel
    dataRAW -= np.mean(dataRAW, axis=0)  # subtract mean of each channel

    # CAR, common average referencing by median
    if car:
        dataRAW -= np.median(dataRAW, axis=1)[:, np.newaxis]  # subtract median across channels

    # next four lines should be equivalent to filtfilt (which cannot be used because it requires float64)
    datr = lfilter(b1, a1, dataRAW, axis=0)  # causal forward filter
    datr = lfilter(b1, a1, datr[::-1, :], axis=0)[::-1, :]

    return datr


def _is_vect(x):
    return hasattr(x, '__len__') and len(x) > 1

def _make_vect(x):
    if not hasattr(x, '__len__'):
        x = np.array([x])
    return x


def my_min(S1, sig, varargin=None):
    # returns a running minimum applied sequentially across a choice of dimensions and bin sizes
    # S1 is the matrix to be filtered
    # sig is either a scalar or a sequence of scalars, one for each axis to be filtered.
    #  it's the plus/minus bin length for the minimum filter
    # varargin can be the dimensions to do filtering, if len(sig) != x.shape
    # if sig is scalar and no axes are provided, the default axis is 2
    idims = 1
    if varargin is not None:
        idims = varargin
    idims = _make_vect(idims)
    if _is_vect(idims) and _is_vect(sig):
        sigall = sig
    else:
        sigall = np.tile(sig, len(idims))

    for sig, idim in zip(sigall, idims):
        Nd = S1.ndim
        S1 = np.transpose(S1, [idim] + list(range(0, idim)) + list(range(idim + 1, Nd)))
        dsnew = S1.shape
        S1 = np.reshape(S1, (S1.shape[0], -1))
        dsnew2 = S1.shape
        S1 = np.concatenate(
                (np.full((sig, dsnew2[1]), np.inf),
                S1,
                np.full((sig, dsnew2[1]), np.inf)), axis=0)
        Smax = S1[:dsnew2[0], :]
        for j in range(1, 2*sig + 1):
            Smax = np.minimum(Smax, S1[j:j + dsnew2[0], :])
        S1 = np.reshape(Smax, dsnew)
    S1 = np.transpose(S1, list(range(1, idim + 1)) + [0] + list(range(idim + 1, Nd)))
    return S1


def whiteningFromCovariance(CC):
    # function Wrot = whiteningFromCovariance(CC)
    # takes as input the matrix CC of channel pairwise correlations
    # outputs a symmetric rotation matrix (also Nchan by Nchan) that rotates
    # the data onto uncorrelated, unit-norm axes

    E, D, _ = svd(CC)  # covariance eigendecomposition (same as svd for positive-definite matrix)
    eps = 1e-6
    Di = np.diag(1. / (D + eps) ** .5)
    Wrot = E @ Di @ E.T  # this is the symmetric whitening matrix (ZCA transform)
    return Wrot


def whiteningLocal(CC, yc, xc, nRange):
    # function to perform local whitening of channels
    # CC is a matrix of Nchan by Nchan correlations
    # yc and xc are vector of Y and X positions of each channel
    # nRange is the number of nearest channels to consider
    Wrot = np.zeros((CC.shape[0], CC.shape[0]))

    for j in range(CC.shape[0]):
        ds = (xc - xc[j]) ** 2 + (yc - yc[j]) ** 2
        ilocal = np.argsort(ds)
        ilocal = ilocal[:nRange]  # take the closest channels to the primary channel. First channel in this list will always be the primary channel.

        wrot0 = whiteningFromCovariance(CC[np.ix_(ilocal, ilocal)])
        Wrot[ilocal, j] = wrot0[:, 0]  # the first column of wrot0 is the whitening filter for the primary channel

    return Wrot


if __name__ == '__main__':
    arr = np.load('../data/arr1000x16.npy')
    CC = arr[:16, :16]
    yc = arr[1, :16]
    xc = arr[2, :16]
    nRange = 4
    p(whiteningLocal(CC, yc, xc, nRange))
