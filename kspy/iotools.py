import numpy as np


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
