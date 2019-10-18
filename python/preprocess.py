import logging

import numpy as np
from scipy.signal import butter
import cupy as cp
from tqdm import tqdm

from .cptools import lfilter, median
from .utils import Bunch

logger = logging.getLogger(__name__)


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
    dataRAW -= cp.mean(dataRAW, axis=0)  # subtract mean of each channel

    # CAR, common average referencing by median
    if car:
        # subtract median across channels
        dataRAW = dataRAW - median(dataRAW, axis=1)[:, np.newaxis]

    # next four lines should be equivalent to filtfilt (which cannot be
    # used because it requires float64)
    datr = lfilter(b1, a1, dataRAW, axis=0)  # causal forward filter
    datr = lfilter(b1, a1, datr, axis=0, reverse=True)  # backward
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
        S1 = cp.transpose(S1, [idim] + list(range(0, idim)) + list(range(idim + 1, Nd)))
        dsnew = S1.shape
        S1 = cp.reshape(S1, (S1.shape[0], -1))
        dsnew2 = S1.shape
        S1 = cp.concatenate(
                (cp.full((sig, dsnew2[1]), np.inf), S1, cp.full((sig, dsnew2[1]), np.inf)), axis=0)
        Smax = S1[:dsnew2[0], :]
        for j in range(1, 2*sig + 1):
            Smax = cp.minimum(Smax, S1[j:j + dsnew2[0], :])
        S1 = cp.reshape(Smax, dsnew)
    S1 = cp.transpose(S1, list(range(1, idim + 1)) + [0] + list(range(idim + 1, Nd)))
    return S1


def whiteningFromCovariance(CC):
    # function Wrot = whiteningFromCovariance(CC)
    # takes as input the matrix CC of channel pairwise correlations
    # outputs a symmetric rotation matrix (also Nchan by Nchan) that rotates
    # the data onto uncorrelated, unit-norm axes

    # covariance eigendecomposition (same as svd for positive-definite matrix)
    E, D, _ = cp.linalg.svd(CC)
    eps = 1e-6
    Di = cp.diag(1. / (D + eps) ** .5)
    Wrot = cp.dot(cp.dot(E, Di), E.T)  # this is the symmetric whitening matrix (ZCA transform)
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
        # take the closest channels to the primary channel.
        # First channel in this list will always be the primary channel.
        ilocal = ilocal[:nRange]

        wrot0 = cp.asnumpy(whiteningFromCovariance(CC[np.ix_(ilocal, ilocal)]))
        # the first column of wrot0 is the whitening filter for the primary channel
        Wrot[ilocal, j] = wrot0[:, 0]

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
    nSkipCov = ops.nSkipCov

    CC = cp.zeros((Nchan, Nchan))

    for ibatch in tqdm(range(0, Nbatch, nSkipCov), desc="Computing the whitening matrix"):
        offset = max(0, twind + 2 * NchanTOT * ((NT - ntbuff) * ibatch - 2 * ntbuff))

        buff = np.fromfile(dat_path, dtype=np.int16, count=NchanTOT * NTbuff, offset=offset)
        buff = buff.reshape((NchanTOT, NTbuff), order='F')

        nsampcurr = buff.shape[1]
        if nsampcurr < NTbuff:
            buff[:, nsampcurr:NTbuff] = np.tile(
                buff[:, nsampcurr - 1], (1, NTbuff - nsampcurr))

        buff_g = cp.asarray(buff, dtype=np.float32)
        assert cp.isfortran(buff_g)
        assert buff_g.dtype == np.float32

        # apply filters and median subtraction
        datr = gpufilter(buff_g, fs=fs, fshigh=fshigh, channel_map=chanMap)

        CC += cp.dot(datr.T, datr) / NT  # sample covariance

    CC /= np.ceil((Nbatch - 1) / nSkipCov)

    if whiteningRange < np.inf:
        #  if there are too many channels, a finite whiteningRange is more robust to noise
        # in the estimation of the covariance
        whiteningRange = min(whiteningRange, Nchan)
        # this function performs the same matrix inversions as below, just on subsets of
        # channels around each channel
        Wrot = whiteningLocal(CC, yc, xc, whiteningRange)
    else:
        Wrot = whiteningFromCovariance(CC)

    Wrot *= scaleproc

    logger.info("Computed the whitening matrix.")

    return cp.asnumpy(Wrot)


def get_good_channels(dat_path=None, **kwargs):
    ops = Bunch(kwargs)

    b1, a1 = get_filter_params(ops.fs, fshigh=ops.fshigh, fslow=ops.fslow)

    fs = ops.fs
    fshigh = ops.fshigh
    fslow = ops.fslow
    Nchan = ops.Nchan
    NchanTOT = ops.NchanTOT
    Nbatch = ops.Nbatch
    NT = ops.NT
    chanMap = ops.chanMap
    twind = ops.twind
    spkTh = ops.spkTh
    nt0 = ops.nt0
    minfr_goodchannels = ops.minfr_goodchannels

    b1, a1 = get_filter_params(fs, fshigh=fshigh)

    ich = []
    k = 0
    ttime = 0

    # skip every 100 batches
    for ibatch in tqdm(range(0, Nbatch, int(np.ceil(Nbatch / 100))), desc="Finding good channels"):
        offset = twind + 2 * NchanTOT * NT * ibatch

        buff = np.fromfile(
            dat_path, dtype=np.int16,
            count=NchanTOT * NT,
            offset=offset)
        buff = buff.reshape((NchanTOT, -1), order='F')
        if buff.size == 0:
            break

        # Put on GPU.
        buff = cp.asarray(buff, dtype=np.float32)
        assert cp.isfortran(buff)
        datr = gpufilter(buff, channel_map=chanMap, fs=fs, fshigh=fshigh, fslow=fslow)

        # very basic threshold crossings calculation
        s = cp.std(datr, axis=0)
        datr /= s  # standardize each channel ( but don't whiten)
        mdat = my_min(datr, 30, 0)  # get local minima as min value in +/- 30-sample range

        # take local minima that cross the negative threshold
        xi, xj = cp.nonzero((datr < mdat + 1e-3) & (datr < spkTh))

        # filtering may create transients at beginning or end. Remove those.
        xj = xj[(xi >= nt0) & (xi <= NT - nt0)]

        # collect the channel identities for the detected spikes
        ich.append(xj)
        k += xj.size

        # keep track of total time where we took spikes from
        ttime += datr.shape[0] / fs

    ich = cp.concatenate(ich)

    # count how many spikes each channel got
    nc, _ = cp.histogram(ich, cp.arange(Nchan + 1))

    # divide by total time to get firing rate
    nc = nc / ttime

    # keep only those channels above the preset mean firing rate
    igood = nc >= minfr_goodchannels

    logger.info('Found %d threshold crossings in %2.2f seconds of data.' % (k, ttime))
    logger.info('Found %d bad channels.' % np.sum(~igood))

    return igood
