import numpy as np
import cupy as cp
from preprocess import my_min, my_sum
from cptools import svdecon
from utils import p


def get_SpikeSample(dataRAW, row, col, params):
    nT, nChan = dataRAW.shape

    # times around the peak to consider
    dt = np.arange(params.nt0)

    # the negativity is expected at nt0min, so we align the detected peaks there
    dt = -params.nt0min + dt

    # temporal indices (awkward way to index into full matrix of data)
    indsT = row + dt[:, np.newaxis] + 1  # broadcasting

    indsC = col

    indsC[indsC < 0] = 0  # anything that's out of bounds just gets set to the limit
    indsC[indsC >= nChan] = nChan - 1  # only needed for channels not time (due to time buffer)

    indsT = np.transpose(np.atleast_3d(indsT), [0, 2, 1])

    indsC = np.transpose(np.atleast_3d(indsC), [2, 0, 1])

    indsT.shape, indsC.shape

    # believe it or not, these indices grab just the right timesamples forour spikes
    ix = indsT + indsC * nT

    # grab the data and reshape it appropriately (time samples  by channels by num spikes)
    clips = dataRAW.T.flat[ix[:, 0, :]].reshape((dt.size, row.size))
    return clips


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

    col, row = np.nonzero(peaks.T)

    mu = -peaks[row, col]

    return row, col, mu


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
