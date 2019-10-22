import numpy as np
import cupy as cp
from preprocess import my_min, my_sum
from cptools import svdecon
from utils import p


def getClosestChannels(probe, sigma, NchanClosest):
    # this function outputs the closest channels to each channel,
    # as well as a Gaussian-decaying mask as a function of pairwise distances
    # sigma is the standard deviation of this Gaussian-mask

    # compute distances between all pairs of channels
    C2C = (probe.xc[:, np.newaxis] - probe.xc) ** 2 + (probe.yc[:, np.newaxis] - probe.yc) ** 2
    C2C = np.sqrt(C2C)
    Nchan = C2C.shape[0]

    # sort distances
    isort = np.argsort(C2C, kind='stable', axis=0)

    # take NchanCLosest neighbors for each primary channel
    iC = isort[:NchanClosest, :]

    # in some cases we want a mask that decays as a function of distance between pairs of channels
    # this is an awkward indexing to get the corresponding distances
    ix = iC + np.arange(0, Nchan ** 2, Nchan)
    mask = np.exp(-C2C.T.flat[ix] ** 2 / (2 * sigma ** 2))

    # masks should be unit norm for each channel
    mask = mask / np.sqrt(1e-3 + np.sum(mask ** 2, axis=0))

    return iC, mask, C2C


def isolated_peaks_new(S1, params):
    S1 = cp.asarray(S1)
    smin = my_min(S1, params.loc_range, [0, 1])

    peaks = (S1 < smin + 1e-3) & (S1 < params.spkTh)

    sum_peaks = my_sum(peaks, params.long_range, [0, 1])
    peaks = peaks * (sum_peaks < 1.2) * S1

    peaks[:params.nt0, :] = 0
    peaks[-params.nt0:, :] = 0

    col, row = cp.nonzero(peaks.T)

    mu = -peaks[row, col]

    return row, col, mu


def get_SpikeSample(dataRAW, row, col, params):
    nT, nChan = dataRAW.shape

    # times around the peak to consider
    dt = cp.arange(params.nt0)

    # the negativity is expected at nt0min, so we align the detected peaks there
    dt = -params.nt0min + dt

    # temporal indices (awkward way to index into full matrix of data)
    indsT = row + dt[:, np.newaxis] + 1  # broadcasting
    indsC = col

    indsC[indsC < 0] = 0  # anything that's out of bounds just gets set to the limit
    indsC[indsC >= nChan] = nChan - 1  # only needed for channels not time (due to time buffer)

    indsT = cp.transpose(cp.atleast_3d(indsT), [0, 2, 1])
    indsC = cp.transpose(cp.atleast_3d(indsC), [2, 0, 1])

    # believe it or not, these indices grab just the right timesamples forour spikes
    ix = indsT + indsC * nT

    # grab the data and reshape it appropriately (time samples  by channels by num spikes)
    clips = dataRAW.T.ravel()[ix[:, 0, :]].reshape((dt.size, row.size))
    return clips


def extractPCfromSnippets(proc, probe=None, params=None, Nbatch=None):
    # extracts principal components for 1D snippets of spikes from all channels
    # loads a subset of batches to find these snippets

    NT = params.NT
    nPCs = params.nPCs
    Nchan = probe.Nchan

    batchstart = np.arange(0, NT * Nbatch + 1, NT)

    # extract the PCA projections
    # initialize the covariance of single-channel spike waveforms
    CC = cp.zeros(params.nt0, dtype=np.float32)

    # from every 100th batch
    for ibatch in range(0, Nbatch, 100):
        offset = Nchan * batchstart[ibatch]
        dat = proc[offset:offset + NT * Nchan].reshape((-1, Nchan), order='F')

        # move data to GPU and scale it back to unit variance
        dataRAW = cp.asarray(dat, dtype=np.float32) / params.scaleproc

        # find isolated spikes from each batch
        row, col, mu = isolated_peaks_new(dataRAW, params)

        # for each peak, get the voltage snippet from that channel
        c = get_SpikeSample(dataRAW, row, col, params)

        # scale covariance down by 1,000 to maintain a good dynamic range
        CC = CC + cp.dot(c, c.T) / 1e3

    # the singular vectors of the covariance matrix are the PCs of the waveforms
    U, Sv, V = svdecon(CC)

    wPCA = U[:, :nPCs]  # take as many as needed

    # adjust the arbitrary sign of the first PC so its negativity is downward
    wPCA[:, 0] = -wPCA[:, 0] * cp.sign(wPCA[20, 0])

    return wPCA


def sortBatches2(ccb0):
    # takes as input a matrix of nBatches by nBatches containing
    # dissimilarities.
    # outputs a matrix of sorted batches, and the sorting order, such that
    # ccb1 = ccb0(isort, isort)

    # put this matrix on the GPU
    ccb0 = cp.asarray(ccb0)

    # compute its svd on the GPU (this might also be fast enough on CPU)
    u, s, v = svdecon(ccb0)

    # initialize the positions xs of the batch embeddings to be very small but proportional to the first PC
    xs = .01 * u[:, 0] / cp.std(u[:, 0], ddof=1)

    # 200 iterations of gradient descent should be enough
    niB = 200

    # this learning rate should usually work fine, since it scales with the average gradient
    # and ccb0 is z-scored
    eta = 1
    for k in range(niB):
        # euclidian distances between 1D embedding positions
        ds = (xs - xs[:, np.newaxis]) ** 2
        # the transformed distances go through this function
        W = cp.log(1 + ds)

        # the error is the difference between ccb0 and W
        err = ccb0 - W

        # ignore the mean value of ccb0
        err = err - cp.mean(err)

        # backpropagate the gradients
        err = err / (1 + ds)
        err2 = err * (xs[:, np.newaxis] - xs)
        D = cp.mean(err2, axis=1)  # one half of the gradients is along this direction
        E = cp.mean(err2, axis=0)  # the other half is along this direction
        # we don't need to worry about the gradients for the diagonal because those are 0

        # final gradients for the embedding variable
        dx = -D + E.T

        # take a gradient step
        xs = xs - eta * dx

    # sort the embedding positions xs
    isort = cp.argsort(xs, axis=0)

    # sort the matrix of dissimilarities
    ccb1 = ccb0[isort, :][:, isort]

    return ccb1, isort


def initializeWdata2(call, uprojDAT, Nchan, nPCs, Nfilt, iC):
    # this function initializes cluster means for the fast kmeans per batch
    # call are time indices for the spikes
    # uprojDAT are features projections (Nfeatures by Nspikes)
    # some more parameters need to be passed in from the main workspace

    # pick random spikes from the sample
    irand = np.ceil(np.random.rand(Nfilt, 1) * uprojDAT.shape[1]).astype(np.int32)

    W = cp.zeros((nPCs, Nchan, Nfilt), dtype=np.float32)

    for t in range(Nfilt):
        ich = iC[:, call[irand[t]]]  # the channels on which this spike lives
        # for each selected spike, get its features
        W[:, ich, t] = uprojDAT[:, irand[t]].reshape((nPCs, -1))

    W = W.reshape((-1, Nfilt))
    # add small amount of noise in case we accidentally picked the same spike twice
    W = W + .001 * cp.random.normal(size=W.shape).astype(np.float32)
    mu = cp.sqrt(cp.sum(W ** 2, axis=0))  # get the mean of the template
    W = W / (1e-5 + mu)  # and normalize the template
    W = W.reshape((nPCs, Nchan, Nfilt))
    nW = (W[0, ...] ** 2)  # squared amplitude of the first PC feture
    W = W.reshape((nPCs * Nchan, Nfilt))
    # determine biggest channel according to the amplitude of the first PC
    Wheights = cp.argmax(nW, axis=0)

    return W, mu, Wheights, irand
