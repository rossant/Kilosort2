import logging
from math import ceil

import numpy as np
import cupy as cp
from tqdm import tqdm

from preprocess import my_min, my_sum, get_Nbatch
from cptools import svdecon, zscore
from cluster import isolated_peaks_new, get_SpikeSample
from utils import Bunch

logger = logging.getLogger(__name__)


def extractTemplatesfromSnippets(proc=None, probe=None, params=None, Nbatch=None):
    # this function is very similar to extractPCfromSnippets.
    # outputs not just the PC waveforms, but also the template "prototype",
    # basically k-means clustering of 1D waveforms.

    NT = params.NT
    # skip every this many batches
    nskip = params.nskip
    nPCs = params.nPCs
    nt0min = params.nt0min
    Nchan = probe.Nchan
    batchstart = np.arange(0, NT * Nbatch + 1, NT)

    k = 0
    # preallocate matrix to hold 1D spike snippets
    dd = cp.zeros((params.nt0, int(5e4)), dtype=np.float32, order='F')

    for ibatch in tqdm(range(0, Nbatch, nskip), desc="Extracting templates"):
        offset = Nchan * batchstart[ibatch]
        dat = proc.flat[offset:offset + NT * Nchan].reshape((-1, Nchan), order='F')

        # move data to GPU and scale it back to unit variance
        dataRAW = cp.asarray(dat, dtype=np.float32) / params.scaleproc

        # find isolated spikes from each batch
        row, col, mu = isolated_peaks_new(dataRAW, params)

        # for each peak, get the voltage snippet from that channel
        c = get_SpikeSample(dataRAW, row, col, params)

        if k + c.shape[1] > dd.shape[1]:
            dd[:, 2 * dd.shape[1]] = 0

        dd[:, k:k + c.shape[1]] = c
        k = k + c.shape[1]
        if k > 1e5:
            break

    # discard empty samples
    dd = dd[:, :k]

    # initialize the template clustering with random waveforms
    wTEMP = dd[:, cp.random.permutation(dd.shape[1])[:nPCs]]
    wTEMP = wTEMP / cp.sum(wTEMP ** 2, axis=0) ** .5  # normalize them

    for i in range(10):
        # at each iteration, assign the waveform to its most correlated cluster
        cc = cp.dot(wTEMP.T, dd)
        imax = cp.argmax(cc, axis=0)
        amax = cc[imax, np.arange(cc.shape[1])]
        for j in range(nPCs):
            # weighted average to get new cluster means
            wTEMP[:, j] = cp.dot(dd[:, imax == j], amax[imax == j].T)
        wTEMP = wTEMP / cp.sum(wTEMP ** 2, axis=0) ** .5  # unit normalize

    # the PCs are just the left singular vectors of the waveforms
    U, Sv, V = svdecon(dd)

    # take as many as needed
    wPCA = U[:, :nPCs]

    # adjust the arbitrary sign of the first PC so its negativity is downward
    wPCA[:, 0] = -wPCA[:, 0] * cp.sign(wPCA[nt0min, 0])

    return wTEMP, wPCA


def getKernels(params):
    # this function makes upsampling kernels for the temporal components.
    # those are used for interpolating the biggest negative peak,
    # and aligning the template to that peak with sub-sample resolution
    # needs nup, the interpolation factor (default = 10)
    # also needs sig, the interpolation smoothness (default = 1)

    nup = params.nup
    sig = params.sig

    nt0min = params.nt0min
    nt0 = params.nt0

    xs = cp.arange(1, nt0 + 1)
    ys = cp.linspace(.5, nt0 + .5, nt0 * nup + 1)[:-1]

    # these kernels are just standard kriging interpolators

    # first compute distances between the sample coordinates
    # for some reason, this seems to be circular, although the waveforms are not circular
    # I think the reason had to do with some constant offsets in some channels?
    d = cp.mod(xs[:, np.newaxis] - xs[np.newaxis, :] + nt0, nt0)
    d = cp.minimum(d, nt0 - d)
    # the kernel covariance uses a squared exponential of spatial scale sig
    Kxx = cp.exp(-d ** 2 / sig ** 2)

    # do the same for the kernel similarities between upsampled "test" timepoints and
    # the original coordinates
    d = cp.mod(ys[:, np.newaxis] - xs[np.newaxis, :] + nt0, nt0)
    d = cp.minimum(d, nt0 - d)
    Kyx = cp.exp(-d ** 2 / sig ** 2)

    # the upsampling matrix is given by the following formula,
    # with some light diagonal regularization of the matrix inversion
    B = cp.dot(Kyx, cp.linalg.inv(Kxx + .01 * cp.eye(nt0)))
    B = B.reshape((nup, nt0, nt0), order='F')

    # A is just a slice through this upsampling matrix corresponding to the most negative point
    # this is used to compute the biggest negative deflection (after upsampling)
    A = cp.squeeze(B[:, nt0min - 1, :])
    B = cp.transpose(B, [1, 2, 0])

    return A.astype(np.float64), B.astype(np.float64)
