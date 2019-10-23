import logging
from math import sqrt

import numpy as np
import cupy as cp
from tqdm import tqdm

from cptools import svdecon
from cluster import isolated_peaks_new, get_SpikeSample
from utils import get_cuda

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


def memorizeW(W, U, mu):
    Wraw = cp.zeros((U.shape[0], W.shape[0], U.shape[1]), dtype=np.float64, order='F')

    for n in range(U.shape[1]):
        # temporarily use U rather Urot until I have a chance to test it
        Wraw[:, :, n] = mu[n] * cp.dot(U[:, n, :], W[:, n, :].T)

    return Wraw


def getMeUtU(iU, iC, mask, Nnearest, Nchan):
    # function [UtU, maskU, iList] = getMeUtU(iU, iC, mask, Nnearest, Nchan)
    # this function determines if two templates share any channels
    # iU are the channels that each template is assigned to, one main channel per template
    # iC has as column K the list of neigboring channels for channel K
    # mask are the weights assigned for the corresponding neighboring channels
    # in iC (gaussian-decaying)

    Nfilt = iU.size

    # create a sparse matrix with ones if a channel K belongs to a template
    U = cp.zeros((Nchan, Nfilt), dtype=np.float32, order='F')

    # use the template primary channel to obtain its neighboring channels from iC
    ix = iC[:, iU] + cp.arange(0, Nchan * Nfilt, Nchan).astype(np.int32)
    U[ix] = 0  # use this as an awkward index into U

    UtU = cp.dot(U.T, U) > 0  # if this is 0, the templates had not pair of channels in common

    # we also return the masks for each template, picked from the corresponding mask of
    # their primary channel
    maskU = mask[:, iU]

    # sort template pairs in order of how many channels they share
    isort = cp.argsort(UtU, axis=0)[::-1]
    iList = isort[:Nnearest, :]  # take the Nnearest templates for each template

    return UtU, maskU, iList


def getMeWtW2(W, U0, Nnearest):
    # this function compute the correlation between any two pairs of templates
    # it relies on the fact that the W and U0 are unit normalized, so that the product of a
    # template with itself is 1, as it should be if we're trying to calculate correlations

    # takes input the temporal and spatial factors of the low-rank template, as
    # well as the number of most similar template pairs desired to be output in
    # iList

    nt0, Nfilt, Nrank = W.size
    WtW = cp.zeros((Nfilt, Nfilt), dtype=np.float32, order='F')

    # since the templates are factorized into orthonormal components, we can compute dot products
    # one dimension at a time
    for i in range(Nrank):
        for j in range(Nrank):
            #  this computes the spatial dot product
            utu0 = cp.dot(U0[:, :, i].T, U0[:, :, j])
            #  this computes the temporal dot product
            wtw0 = cp.dot(W[:, :, i].T, W[:, :, j])

            # the element-wise product of these is added to the matrix of correlatioons
            WtW = WtW + wtw0 * utu0

    # also return a list of most correlated template pairs
    isort = cp.argsort(WtW, axis=0)[::-1]

    # if we don't have enough templates yet, just wrap the indices around the range 1:Nfilt
    iNear = cp.mod(cp.arange(Nnearest), Nfilt)
    iList = isort[iNear, :]  # return the list of pairs for each template

    return WtW, iList


def mexGetSpikes2(Params, drez, wTEMP, iC):
    code, constants = get_cuda('mexGetSpikes2')

    NT = int(Params[0])
    Nchan = int(Params[9])
    nt0 = int(Params[4])
    # Nnearest = int(Params[5])
    Nrank = int(Params[14])

    maxFR = constants.maxFR
    Nthreads = constants.Nthreads

    # tpB = (8, 2 * nt0 - 1)
    # tpF = (16, Nnearest)
    tpS = (nt0, 16)

    d_Params = cp.asarray(Params, dtype=np.float32, order='F')
    d_data = cp.asarray(drez, dtype=np.float32, order='F')
    d_W = cp.asarray(wTEMP, dtype=np.float32, order='F')
    d_iC = cp.asarray(iC, dtype=np.int32, order='F')

    d_dout = cp.zeros((NT, Nchan), dtype=np.float32, order='F')
    d_counter = cp.zeros(2, dtype=np.int32, order='F')
    d_dfilt = cp.zeros((Nrank, NT, Nchan), dtype=np.float32, order='F')
    d_err = cp.zeros(NT, dtype=np.float32, order='F')
    d_kkmax = cp.zeros((NT, Nchan), dtype=np.int32, order='F')
    d_kk = cp.zeros(NT, dtype=np.int32, order='F')
    d_ftype = cp.zeros(NT, dtype=np.int32, order='F')
    d_st = cp.zeros(maxFR, dtype=np.int32, order='F')
    d_id = cp.zeros(maxFR, dtype=np.int32, order='F')
    d_x = cp.zeros(maxFR, dtype=np.float32, order='F')
    d_st1 = cp.zeros(maxFR, dtype=np.int32, order='F')
    d_id1 = cp.zeros(maxFR, dtype=np.int32, order='F')

    counter = np.zeros(2, dtype=np.int32, order='F')

    # filter the data with the temporal templates
    Conv1D = cp.RawKernel(code, 'Conv1D')
    Conv1D((Nchan,), (Nthreads,), (d_Params, d_data, d_W, d_dfilt))

    # sum each template across channels, square, take max
    sumChannels = cp.RawKernel(code, 'sumChannels')
    sumChannels((int(NT / Nthreads),), (Nthreads,), (d_Params, d_dfilt, d_dout, d_kkmax, d_iC))

    # compute the best filter
    bestFilter = cp.RawKernel(code, 'bestFilter')
    bestFilter(
        (int(NT / Nthreads),), (Nthreads,), (d_Params, d_dout, d_err, d_ftype, d_kkmax, d_kk))

    # ignore peaks that are smaller than another nearby peak
    cleanup_spikes = cp.RawKernel(code, 'cleanup_spikes')
    cleanup_spikes(
        (int(NT / Nthreads),), (Nthreads,), (d_Params, d_err, d_ftype, d_x, d_st, d_id, d_counter))

    # ignore peaks that are smaller than another nearby peak
    cleanup_heights = cp.RawKernel(code, 'cleanup_heights')
    cleanup_heights(
        (1 + int(maxFR // 32),), (32,), (d_Params, d_x, d_st, d_id, d_st1, d_id1, d_counter))

    # add new spikes to 2nd counter
    counter[0] = d_counter[1]
    counter[0] = min(maxFR, counter[0])

    d_WU = cp.zeros((counter[0], nt0, Nchan), dtype=np.float32, order='F')
    d_WU1 = cp.zeros((nt0, Nchan, counter[0]), dtype=np.float32, order='F')

    # update dWU here by adding back to subbed spikes
    extract_snips = cp.RawKernel(code, 'extract_snips')
    extract_snips((Nchan,), tpS, (d_Params, d_st1, d_id1, d_counter, d_data, d_WU))

    del (
        d_ftype, d_kkmax, d_err, d_st, d_id, d_st1, d_x, d_kk, d_id1, d_counter,
        d_Params, d_dfilt, d_WU)
    return d_WU1, d_dout


def mexSVDsmall2(Params, dWU, W, iC, iW, Ka, Kb):
    code, constants = get_cuda('mexSVDsmall2')

    Nthreads = constants.Nthreads

    Nfilt = Params[1]
    nt0 = Params[4]
    Nrank = Params[6]
    Nchan = Params[9]

    d_Params = cp.asarray(Params, dtype=np.float32, order='F')

    d_dWU = cp.asarray(dWU, dtype=np.float64, order='F')
    d_iC = cp.asarray(iC, dtype=np.int32, order='F')
    d_iW = cp.asarray(iW, dtype=np.int32, order='F')

    d_A = cp.asarray(Ka, dtype=np.float64, order='F')
    d_B = cp.asarray(Kb, dtype=np.float64, order='F')

    d_U = cp.zeros((Nchan, Nfilt, Nrank), dtype=np.float64, order='F')
    d_mu = cp.zeros(Nfilt, dtype=np.float64, order='F')

    d_W = cp.asarray(W, dtype=np.float64, order='F')

    d_wtw = cp.zeros((nt0, nt0, Nfilt), dtype=np.float64, order='F')
    d_dWUb = cp.zeros((nt0, Nchan, Nfilt), dtype=np.float64, order='F')

    tpS = (nt0, int(Nthreads // nt0))
    tpK = (Nrank, int(Nthreads // Nrank))

    blankdWU = cp.RawKernel(code, 'blankdWU')
    blankdWU((Nfilt,), tpS, (d_Params, d_dWU, d_iC, d_iW, d_dWUb))

    # compute dWU * dWU'
    getwtw = cp.RawKernel(code, 'getwtw')
    getwtw((Nfilt,), tpS, (d_Params, d_dWUb, d_wtw))

    # get W by power svd iterations
    getW = cp.RawKernel(code, 'getW')
    getW((Nfilt,), (nt0,), (d_Params, d_wtw, d_W))

    # compute U by W' * dWU
    getU = cp.RawKernel(code, 'getU')
    getU((Nfilt,), tpK, (d_Params, d_dWUb, d_W, d_U))

    # normalize U, get S, get mu, renormalize W
    reNormalize = cp.RawKernel(code, 'reNormalize')
    reNormalize((Nfilt,), (nt0,), (d_Params, d_A, d_B, d_W, d_U, d_mu))

    del d_wtw, d_Params, d_dWUb

    return d_W, d_U, d_mu


def mexMPnu8(Params, dataRAW, U, W, mu, iC, iW, UtU, iList, wPCA):
    code, constants = get_cuda('mexMPnu8')
    maxFR = constants.maxFR
    nmaxiter = constants.nmaxiter
    Nthreads = constants.Nthreads

    NT = int(Params[0])
    Nfilt = int(Params[1])
    nt0 = int(Params[4])
    Nnearest = int(Params[5])
    Nrank = int(Params[6])
    NchanU = int(Params[10])
    Nchan = int(Params[9])

    d_Params = cp.asarray(Params, dtype=np.float32, order='F')

    d_draw = cp.asarray(dataRAW, dtype=np.float32, order='F')
    d_U = cp.asarray(U, dtype=np.float32, order='F')
    d_W = cp.asarray(W, dtype=np.float32, order='F')
    d_mu = cp.asarray(mu, dtype=np.float32, order='F')
    d_iC = cp.asarray(iC, dtype=np.int32, order='F')
    d_iW = cp.asarray(iW, dtype=np.int32, order='F')
    d_UtU = cp.asarray(UtU, dtype=np.bool, order='F')
    d_iList = cp.asarray(iList, dtype=np.int32, order='F')
    d_wPCA = cp.asarray(wPCA, dtype=np.float32, order='F')

    d_nsp = cp.zeros(Nfilt, dtype=np.int32, order='F')
    d_dWU = cp.zeros((nt0, Nchan, Nfilt), dtype=np.float64, order='F')

    d_dout = cp.zeros((2 * NT, Nfilt), dtype=np.float32, order='F')
    d_data = cp.zeros((NT, Nfilt, Nrank), dtype=np.float32, order='F')
    d_err = cp.zeros(NT, dtype=np.float32, order='F')
    d_ftype = cp.zeros(NT, dtype=np.int32, order='F')
    d_eloss = cp.zeros(NT, dtype=np.float32, order='F')
    d_st = cp.zeros(maxFR, dtype=np.int32, order='F')
    d_id = cp.zeros(maxFR, dtype=np.int32, order='F')
    d_x = cp.zeros(maxFR, dtype=np.float32, order='F')
    d_y = cp.zeros(maxFR, dtype=np.float32, order='F')
    d_z = cp.zeros(maxFR, dtype=np.float32, order='F')

    d_counter = cp.zeros(2, dtype=np.int32, order='F')
    d_count = cp.zeros(nmaxiter, dtype=np.int32, order='F')
    d_feat = cp.zeros((Nnearest, maxFR), dtype=np.float32, order='F')
    d_featPC = cp.zeros((NchanU, Nrank, maxFR), dtype=np.float32, order='F')

    counter = np.zeros(2, dtype=np.int32, order='F')

    # tpB = (8, 2 * nt0 - 1)
    tpF = (16, Nnearest)
    tpS = (nt0, 16)
    # tpW = (Nnearest, Nrank)
    tpPC = (NchanU, Nrank)

    # filter the data with the spatial templates
    spaceFilter = cp.RawKernel(code, 'spaceFilter')
    spaceFilter((Nfilt,), (Nthreads,), (d_Params, d_draw, d_U, d_iC, d_iW, d_data))

    # filter the data with the temporal templates
    timeFilter = cp.RawKernel(code, 'timeFilter')
    timeFilter((Nfilt,), (Nthreads,), (d_Params, d_data, d_W, d_dout))

    # compute the best filter
    bestFilter = cp.RawKernel(code, 'bestFilter')
    bestFilter(
        (int(NT // Nthreads),), (Nthreads,), (d_Params, d_dout, d_mu, d_err, d_eloss, d_ftype))

    # loop to find and subtract spikes
    for k in range(Params[3]):
        # ignore peaks that are smaller than another nearby peak
        cleanup_spikes = cp.RawKernel(code, 'cleanup_spikes')
        cleanup_spikes(
            (int(NT // Nthreads),), (Nthreads,),
            (d_Params, d_dout, d_mu, d_err, d_eloss,
             d_ftype, d_st, d_id, d_x, d_y, d_z, d_counter))

        # add new spikes to 2nd counter
        counter[:] = d_counter[:]
        if counter[0] > maxFR:
            counter[0] = maxFR
            d_counter[0] = counter[0]

        # extract template features before subtraction
        if Params[12] > 1:
            extractFEAT = cp.RawKernel(code, 'extractFEAT')
            extractFEAT(
                (64,), (tpF,), (d_Params, d_st, d_id, d_counter, d_dout, d_iList, d_mu, d_feat))

        # subtract spikes from raw data here
        subtract_spikes = cp.RawKernel(code, 'subtract_spikes')
        subtract_spikes((Nfilt,), tpS, (d_Params,  d_st, d_id, d_y, d_counter, d_draw, d_W, d_U))

        # filter the data with the spatial templates
        spaceFilterUpdate = cp.RawKernel(code, 'spaceFilterUpdate')
        spaceFilterUpdate(
            (Nfilt,), (2 * nt0 - 1,),
            (d_Params, d_draw, d_U, d_UtU, d_iC, d_iW, d_data, d_st, d_id, d_counter))

        # filter the data with the temporal templates
        timeFilterUpdate = cp.RawKernel(code, 'timeFilterUpdate')
        timeFilterUpdate(
            (Nfilt,), (2 * nt0 - 1,),
            (d_Params, d_data, d_W, d_UtU, d_dout, d_st, d_id, d_counter))

        if counter[0] - counter[1] > 0:
            bestFilterUpdate = cp.RawKernel(code, 'bestFilterUpdate')
            bestFilterUpdate(
                (counter[0] - counter[1],), (2 * nt0 - 1,),
                (d_Params, d_dout, d_mu, d_err, d_eloss, d_ftype, d_st, d_id, d_counter))

        d_count[k + 1] = d_counter[0]

        # update 1st counter from 2nd counter
        d_counter[1] = d_counter[0]

    # compute PC features from reziduals + subtractions
    if Params[12] > 0:
        computePCfeatures = cp.RawKernel(code, 'computePCfeatures')
        computePCfeatures(
            (Nfilt,), tpPC,
            (d_Params, d_counter, d_draw, d_st, d_id, d_y,
             d_W, d_U, d_mu, d_iW, d_iC, d_wPCA, d_featPC))

    # update dWU here by adding back to subbed spikes.
    # additional parameter d_idx = array of time sorted indicies
    average_snips = cp.RawKernel(code, 'average_snips')
    average_snips(
        (Nfilt,), tpS,
        (d_Params, d_st, d_id, d_x, d_y, d_counter, d_draw, d_W, d_U, d_dWU, d_nsp, d_mu, d_z))

    if counter[0] < maxFR:
        minSize = counter[0]
    else:
        minSize = maxFR

    del d_counter, d_Params, d_ftype, d_err, d_eloss, d_z, d_dout, d_data

    return (
        d_st[:minSize], d_id[:minSize], d_y[:minSize], d_feat[..., :minSize],
        d_dWU, d_draw, d_nsp, d_featPC[..., :minSize], d_x[:minSize])


def mexWtW2(Params, W1, W2, UtU):
    code, constants = get_cuda('mexWtW2')

    nblock = constants.nblock

    Nfilt = int(Params[1])
    nt0 = int(Params[9])

    d_Params = cp.asarray(Params, dtype=np.float32, order='F')

    d_W1 = cp.asarray(W1, dtype=np.float32, order='F')
    d_W2 = cp.asarray(W2, dtype=np.float32, order='F')
    d_UtU = cp.asarray(UtU, dtype=np.float32, order='F')

    d_WtW = cp.zeros((Nfilt, Nfilt, 2 * nt0 - 1), dtype=np.float32, order='F')

    grid = (1 + int(Nfilt // nblock), 1 + int(Nfilt // nblock))
    block = (nblock, nblock)

    crossFilter = cp.RawKernel(code, 'crossFilter')
    crossFilter(grid, block, (d_Params, d_W1, d_W2, d_UtU, d_WtW))

    del d_Params, d_W1, d_W2, d_UtU

    return d_WtW


def getMeWtW(W, U0, Nnearest):
    # this function computes correlation between templates at ALL timelags from each other
    # takes the max over timelags to obtain a similarity score
    # also returns lists of most similar templates to each template
    # takes as input the low-rank factorization of templates (W for time and U0
    # for space)

    # W is timesamples (default = 61 ), by number of templates, by rank (default = 3)
    nt0, Nfilt, Nrank = W.shape

    Params = [1, Nfilt, 0, 0, 0, 0, 0, 0, 0, nt0]

    # initialize correlation matrix for all timelags
    WtW = cp.zeros((Nfilt, Nfilt, 2 * nt0 - 1), dtype=np.float32, order='F')
    for i in range(Nrank):
        for j in range(Nrank):
            # the dot product factorizes into separable products for each spatio-temporal component
            utu0 = cp.dot(U0[:, :, i].T, U0[:, :, j])  # spatial products
            # temporal convolutions get multiplied wit hthe spatial products
            wtw0 = mexWtW2(Params, W[:, :, i], W[:, :, j], utu0)
            # add it to the full correlation array
            WtW = WtW + wtw0

    # the maximum across timelags accounts for sample alignment mismatch
    cc = cp.max(WtW, axis=2)

    isort = cp.argsort(cc, axis=0)[::-1]
    # if we don't have enough templates yet, just wrap the indices around the range 1:Nfilt
    iNear = cp.mod(np.arange(Nnearest), Nfilt)
    iList = isort[iNear, :]  # return the list of pairs for each template

    return WtW, iList


def triageTemplates2(params, iW, C2C, W, U, dWU, mu, nsp, ndrop):

    # This function checks if some templates should be dropped
    # either because they are very similar to another template,
    # or because they are not catching any spikes, (low mean firing rate).
    # Takes as inputs almost all the information that determines templates, and
    # outputs the same variables back after removing some clusters.

    # this is the firing rate threshold
    m0 = params.minFR * params.NT / params.fs
    idrop = nsp < m0  # drop any templates with firing rate below this

    # remove those templates everywhere
    W = W[:, ~idrop, :]
    U = U[:, ~idrop, :]
    dWU = dWU[:, :, ~idrop]
    mu = mu[~idrop]
    nsp = nsp[~idrop]
    # keep track of how many templates have been removed this way
    ndrop[0] = .9 * ndrop[0] + .1 * idrop.sum()

    # compute pairwise correlations between templates
    cc = getMeWtW2(W, U)
    cc = cc - cp.diag(cp.diag(cc))  # exclude the diagonal

    sd = sqrt(10)  # this is hard-coded here

    # compute a score for the separation of the means
    r0 = 4 * sd / cp.abs(mu[:, np.newaxis] - mu[np.newaxis, :])
    # determine which template has more spikes (that one survives)
    rdir = (nsp[:, np.newaxis] - nsp[np.newaxis, :]) < 0
    # for each pair of template, score their similarity by their template correlation,
    # and amplitude separation
    ipair = (cc > 0.9 & r0 > 1 & rdir)
    # for each template, find its most similar other template
    amax = cp.max(ipair, axis=1)
    # if this score is 1, then all the criteria have bene met for dropping this template
    idrop = amax > 0

    # remove these templates everywhere like before
    W = W[:, ~idrop, :]
    U = U[:, ~idrop, :]
    dWU = dWU[:, :, ~idrop]
    mu = mu[~idrop]
    nsp = nsp[~idrop]
    # keep track of how many templates have been removed this way
    ndrop[1] = .9 * ndrop[1] + .1 * idrop.sum()

    return W, U, dWU, mu, nsp, ndrop
