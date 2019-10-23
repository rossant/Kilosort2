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
