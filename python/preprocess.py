import os
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import svd
from scipy.signal import butter, filtfilt, lfilter
import cupy as cp
import matplotlib.pyplot as plt


class Bunch(dict):
    """A subclass of dictionary with an additional dot syntax."""
    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self


def p(x):
    print("shape", x.shape, "mean", "%5e" % x.mean())
    print(x[:2, :2])
    print()
    print(x[-2:, -2:])


def get_filter_params(fs, fshigh=None, fslow=None):
    if fslow and fslow < fs / 2:
        # butterworth filter with only 3 nodes (otherwise it's unstable for float32)
        return butter(3, (2 * fshigh / fs, 2 * fslow / fs), 'bandpass')
    else:
        # butterworth filter with only 3 nodes (otherwise it's unstable for float32)
        return butter(3, fshigh / fs * 2, 'high')


def gpufilter(buff, channel_map=None, fs=None, fslow=None, fshigh=None, car=True):
    # filter this batch of data after common average referencing with the
    # median
    # buff is timepoints by channels
    # chanMap are indices of the channels to be kep
    # ops.fs and ops.fshigh are sampling and high-pass frequencies respectively
    # if ops.fslow is present, it is used as low-pass frequency (discouraged)

    # set up the parameters of the filter
    b1, a1 = get_filter_params(fs, fshigh=fshigh, fslow=fslow)

    dataRAW = buff.T
    if channel_map is not None:
        dataRAW = dataRAW[:, channel_map]  # subsample only good channels

    # subtract the mean from each channel
    dataRAW = dataRAW - np.mean(dataRAW, axis=0)  # subtract mean of each channel

    # CAR, common average referencing by median
    if car:
        dataRAW = dataRAW - np.median(dataRAW, axis=1)[:, np.newaxis]  # subtract median across channels

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


def get_whitening_matrix(dat_path=None, **kwargs):
    ops = Bunch(kwargs)

    b1, a1 = get_filter_params(ops.fs, fshigh=ops.fshigh, fslow=ops.fslow)

    Nchan = ops.Nchan
    NchanTOT = ops.NchanTOT
    Nbatch = ops.Nbatch
    ntbuff = ops.ntbuff
    NTbuff = ops.NTbuff
    chanMap = ops.chanMap
    whiteningRange = ops.whiteningRange
    scaleproc = ops.scaleproc
    xc = ops.xc
    yc = ops.yc
    chanMap = ops.chanMap
    twind = ops.twind
    NT = ops.NT
    fs = ops.fs
    fshigh = ops.fshigh
    fslow = ops.fslow
    nSkipCov = ops.nSkipCov

    CC = cp.zeros((Nchan, Nchan))

    ibatch = 1
    while ibatch <= Nbatch:
        offset = max(0, twind + 2 * NchanTOT * ((NT - ntbuff) * (ibatch - 1) - 2 * ntbuff))

        buff = np.fromfile(dat_path, dtype=np.int16, count=NchanTOT * NTbuff, offset=offset)
        buff = buff.reshape((NchanTOT, NTbuff), order='F')

        nsampcurr = buff.shape[1]
        if nsampcurr < NTbuff:
            buff[:, nsampcurr:NTbuff] = np.tile(
                buff[:, nsampcurr - 1], (1, NTbuff - nsampcurr))

        # apply filters and median subtraction
        # TODO: gpufilter on the GPU
        datr = gpufilter(buff, fs=fs, fshigh=fshigh, channel_map=chanMap)

        datr_g = cp.asarray(datr)

        CC += cp.dot(datr_g.T, datr_g) / NT  # sample covariance

        ibatch += nSkipCov  # skip this many batches

    CC /= np.ceil((Nbatch - 1) / nSkipCov)

    if whiteningRange < np.inf:
        #  if there are too many channels, a finite whiteningRange is more robust to noise in the estimation of the covariance
        whiteningRange = min(whiteningRange, Nchan)
        Wrot = whiteningLocal(cp.asnumpy(CC), yc, xc, whiteningRange)  # this function performs the same matrix inversions as below, just on subsets of channels around each channel
    else:
        Wrot = whiteningFromCovariance(cp.asnumpy(CC))

    Wrot *= scaleproc

    return Wrot


def get_good_channels(dat_path=None, **kwargs):
    ops = Bunch(kwargs)

    b1, a1 = get_filter_params(ops.fs, fshigh=ops.fshigh, fslow=ops.fslow)

    Nchan = ops.Nchan
    NchanTOT = ops.NchanTOT
    Nbatch = ops.Nbatch
    ntbuff = ops.ntbuff
    NTbuff = ops.NTbuff
    chanMap = ops.chanMap
    whiteningRange = ops.whiteningRange
    scaleproc = ops.scaleproc
    xc = ops.xc
    yc = ops.yc
    chanMap = ops.chanMap
    twind = ops.twind
    NT = ops.NT
    fs = ops.fs
    fshigh = ops.fshigh
    fslow = ops.fslow
    nSkipCov = ops.nSkipCov
    spkTh = ops.spkTh
    nt0 = ops.nt0
    minfr_goodchannels = ops.minfr_goodchannels

    b1, a1 = get_filter_params(fs, fshigh=fshigh)

    ibatch = 1
    ich = np.zeros((50000,), dtype=np.int16)
    k = 0
    ttime = 0

    while ibatch <= Nbatch:
        offset = twind + 2 * NchanTOT * NT * (ibatch - 1)

        buff = np.fromfile(
            dat_path, dtype=np.int16,
            count=NchanTOT * NT,
            offset=offset)
        buff = buff.reshape((NchanTOT, -1), order='F')
        if buff.size == 0:
            break

        datr = gpufilter(buff, channel_map=chanMap, fs=fs, fshigh=fshigh, fslow=fslow)
        # very basic threshold crossings calculation
        datr /= np.std(datr, axis=0)  # standardize each channel ( but don't whiten)
        mdat = my_min(datr, 30, 0)  # get local minima as min value in +/- 30-sample range
        # take local minima that cross the negative threshold
        xi, xj = np.nonzero((datr < mdat + 1e-3) & (datr < spkTh))

        # filtering may create transients at beginning or end. Remove those.
        xj = xj[(xi >= nt0) & (xi <= NT - nt0)]

        # if necessary, extend the variable which holds the spikes
        if k + xj.size > ich.size:
            ich = np.concatenate((ich, np.zeros_like(ich)))

        # collect the channel identities for the detected spikes
        ich[k:k + xj.size] = xj
        k += xj.size

        # skip every 100 batches
        ibatch += int(np.ceil(Nbatch / 100))

        # keep track of total time where we took spikes from
        ttime += datr.shape[0] / fs

    ich = ich[:k]

    # count how many spikes each channel got
    nc, _ = np.histogram(ich, np.arange(Nchan + 1))

    # divide by total time to get firing rate
    nc = nc / ttime

    # keep only those channels above the preset mean firing rate
    igood = nc >= minfr_goodchannels

    print('found %d threshold crossings in %2.2f seconds of data \n' % (k, ttime))
    print('found %d bad channels \n' % sum(~igood))

    return igood


if __name__ == '__main__':
    # arr = np.load('../data/arr1000x16.npy')
    # CC = arr[:16, :16]
    # yc = arr[1, :16]
    # xc = arr[2, :16]
    # nRange = 4
    # p(whiteningLocal(CC, yc, xc, nRange))

    ops = Bunch()
    ops.fs = 30000.
    ops.fshigh = 150.
    ops.fslow = None
    ops.Nbatch = 46
    ops.twind = 0
    ops.NchanTOT = 385
    ops.NT = 65600
    ops.NTbuff = 65856
    ops.Nchan = 293
    ops.ntbuff = 64
    ops.nSkipCov = 25
    ops.whiteningRange = 32
    ops.scaleproc = 200
    ops.twind = 0
    ops.spkTh = -6
    ops.nt0 = 61
    ops.minfr_goodchannels = .1

    dat_path = '/home/cyrille/git/Kilosort2/experimental/imec_385_100s.bin'

    ops.chanMap = np.load('../data/chanMap.npy').squeeze().astype(np.int64)
    # WARNING
    ops.chanMap -= 1

    ops.xc = np.load('../data/xc.npy').squeeze()
    ops.yc = np.load('../data/yc.npy').squeeze()

    # Wrot = get_whitening_matrix(dat_path, **ops)
    # p(Wrot)

    get_good_channels(dat_path, **ops)
