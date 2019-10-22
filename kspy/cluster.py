from math import ceil

import numpy as np
import cupy as cp

from preprocess import my_min, my_sum
from cptools import svdecon, free_gpu_memory
from utils import Bunch, p


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

    # initialize the positions xs of the batch embeddings to be very small but proportional to
    # the first PC
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


CUDA_KERNELS = {
    'Conv1D': '''
__global__ void Conv1D(const float *Params, const float *data, const float *W, float *conv_sig){
  volatile __shared__ float  sW[81*NrankMax], sdata[Nthreads+81];
  float x, y;
  int tid, tid0, bid, i, nid, Nrank, NT, nt0;

  tid         = threadIdx.x;
  bid         = blockIdx.x;

  NT          =   (int) Params[0];
  nt0       = (int) Params[3];
  Nrank     = (int) Params[6];

  if(tid<nt0*Nrank)
      sW[tid]= W[tid];
  __syncthreads();

  tid0 = 0;

  while (tid0<NT-Nthreads-nt0+1){
      if (tid<nt0)
          sdata[tid] = data[tid0 + tid+ NT*bid];

      sdata[tid + nt0] = data[tid0 + tid + nt0 + NT*bid];
      __syncthreads();

      x = 0.0f;
      for(nid=0;nid<Nrank;nid++){
          y = 0.0f;
          #pragma unroll 4
          for(i=0;i<nt0;i++)
              y    += sW[i + nid*nt0] * sdata[i+tid];

           x += y*y;
      }
      conv_sig[tid0  + tid + NT*bid] = sqrt(x);

      tid0 += Nthreads;
      __syncthreads();
  }
}
''',
    'computeProjections': '''
__global__ void computeProjections(const float *Params, const float *dataraw,
        const int *iC, const int *st, const int *id, const float *W, float *feat){

    float x;
    int tidx, nt0min, tidy, my_chan, this_chan, tid, bid, nt0, NchanNear, j, t, NT, NrankPC;
    volatile __shared__ float sW[nt0max*NrankMax], sD[nt0max*NchanMax];

    NT         = (int) Params[0];
    NchanNear = (int) Params[2];
    nt0       = (int) Params[3];
    NrankPC  = (int) Params[6];
    nt0min    = (int) Params[4];

    tidx = threadIdx.x;
    tidy = threadIdx.y;
    bid = blockIdx.x;

    // move wPCA to shared memory
    while (tidx<nt0){
        sW[tidx + tidy*nt0] = W[tidx + tidy*nt0];
        tidx+=blockDim.x;
    }
    tidx = threadIdx.x;

    tid = tidx + tidy*blockDim.x;
    // move raw data to shared memory
    while (tid<nt0){
        my_chan = id[bid];
        for (j=0;j<NchanNear;j++){
            this_chan = iC[j + NchanNear*my_chan];
            sD[tid + nt0*j] = dataraw[tid + st[bid]+nt0min-1 + NT * this_chan];
        }
        tid+=blockDim.x*blockDim.y;
    }
    __syncthreads();

    x = 0.0f;
    for (t=0;t<nt0;t++)
        x += sD[t + nt0*tidx] * sW[t + nt0*tidy];

    feat[tidy + tidx*NrankPC + NrankPC*NchanNear*bid] = x;

    }
''',

    'maxChannels': '''
__global__ void maxChannels(const float *Params, const float *dataraw, const float *data,
    const int *iC, int *st, int *id, int *counter){

  int nt0, indx, tid, tid0, i, bid, NT, Nchan, NchanNear,j,iChan, nt0min;
  double Cf, d;
  float spkTh;
  bool flag;

  NT             = (int) Params[0];
  Nchan     = (int) Params[1];
  NchanNear = (int) Params[2];
  nt0       = (int) Params[3];
  nt0min    = (int) Params[4];
  spkTh    = (float) Params[5];

  tid         = threadIdx.x;
  bid         = blockIdx.x;

  tid0 = tid + bid * blockDim.x;
  while (tid0<NT-nt0-nt0min){
      for (i=0; i<Nchan;i++){
          iChan = iC[0 + NchanNear * i];
          Cf    = (double) data[tid0 + NT * iChan];
          flag = true;
          for(j=1; j<NchanNear; j++){
              iChan = iC[j+ NchanNear * i];
              if (data[tid0 + NT * iChan] > Cf){
                flag = false;
                break;
              }
          }

          if (flag){
              iChan = iC[NchanNear * i];
              if (Cf>spkTh){
                  d = (double) dataraw[tid0+nt0min-1 + NT*iChan]; //
                  if (d > Cf-1e-6){
                      // this is a hit, atomicAdd and return spikes
                      indx = atomicAdd(&counter[0], 1);
                      if (indx<maxFR){
                          st[indx] = tid0;
                          id[indx] = iChan;
                      }
                  }
              }
          }
      }
      tid0 += blockDim.x * gridDim.x;
  }
}
''',

    'max1D': '''
__global__ void max1D(const float *Params, const float *data, float *conv_sig){

    volatile __shared__ float  sdata[Nthreads+81];
    float y, spkTh;
    int tid, tid0, bid, i, NT, nt0;

    NT         = (int) Params[0];
    nt0       = (int) Params[3];
    spkTh    = (float) Params[5];
    tid         = threadIdx.x;
    bid         = blockIdx.x;

    tid0 = 0;
    while (tid0<NT-Nthreads-nt0+1){
        if (tid<nt0)
            sdata[tid]   = data[tid0 + tid + NT*bid];
        sdata[tid + nt0] = data[nt0+tid0 + tid+ NT*bid];
        __syncthreads();

        y = 0.0f;
        #pragma unroll 4
        for(i=0;i<nt0;i++)
            y    = max(y, sdata[tid+i]);

        if (y>spkTh)
            conv_sig[tid0  + tid + NT*bid]   = y;

        tid0+=Nthreads;
        __syncthreads();
    }
}
'''
}


CONSTANTS = Bunch(
    Nthreads=1024,
    maxFR=10000,
    NrankMax=3,
    nt0max=81,
    NchanMax=17,
)
CUDA_CONSTANTS = 'const int ' + ', '.join('%s = %d' % (n, v) for n, v in CONSTANTS.items()) + ';'


def _get_kernel(name):
    return cp.RawKernel(CUDA_CONSTANTS + '\n\n' + 'extern "C" ' + CUDA_KERNELS[name].strip(), name)


def mexThSpkPC(Params, dataRAW, wPCA, iC):
    Nthreads = CONSTANTS.Nthreads
    maxFR = CONSTANTS.maxFR
    NT, Nchan, NchanNear, nt0, nt0min, spkTh, NrankPC = Params
    NT = int(NT)
    Nchan = int(Nchan)

    # Input GPU arrays.
    d_Params = cp.asarray(Params, dtype=np.float32, order='F')
    d_data = cp.asarray(dataRAW, dtype=np.float32, order='F')
    d_W = cp.asarray(wPCA, dtype=np.float32, order='F')
    d_iC = cp.asarray(iC, dtype=np.int32, order='F')

    # New GPU arrays.
    d_dout = cp.zeros(NT * Nchan, dtype=np.float32, order='F')
    d_dmax = cp.zeros(NT * Nchan, dtype=np.float32, order='F')
    d_st = cp.zeros(maxFR, dtype=np.int32, order='F')
    d_id = cp.zeros(maxFR, dtype=np.int32, order='F')
    d_counter = cp.zeros(1, dtype=np.int32, order='F')

    # filter the data with the temporal templates
    Conv1D = _get_kernel('Conv1D')
    Conv1D((Nchan,), (Nthreads,), (d_Params, d_data, d_W, d_dout))

    # get the max of the data
    max1D = _get_kernel('max1D')
    max1D((Nchan,), (Nthreads,), (d_Params, d_dout, d_dmax))

    # take max across nearby channels
    maxChannels = _get_kernel('maxChannels')
    maxChannels(
        (int(NT // Nthreads),), (Nthreads,),
        (d_Params, d_dout, d_dmax, d_iC, d_st, d_id, d_counter))

    # move d_x to the CPU
    minSize = 1
    minSize = min(maxFR, int(d_counter[0]))

    d_featPC = cp.zeros((NrankPC * NchanNear, minSize), dtype=np.float32, order='F')

    d_id2 = cp.zeros(minSize, dtype=np.int32, order='F')

    if (minSize > 0):
        computeProjections = _get_kernel('computeProjections')
        computeProjections(
            (minSize,), (NchanNear, NrankPC), (d_Params, d_data, d_iC, d_st, d_id, d_W, d_featPC))

    # TODO: check that the copy occurs on the GPU only
    d_id2[:] = d_id[:minSize]

    # Free memory.
    del d_st, d_id, d_counter, d_Params, d_dmax, d_dout
    # free_gpu_memory()

    return d_featPC, d_id2


def extractPCbatch2(proc, params, probe, wPCA, ibatch, iC, Nbatch):
    # this function finds threshold crossings in the data using
    # projections onto the pre-determined principal components
    # wPCA is number of time samples by number of PCs
    # ibatch is a scalar indicating which batch to analyze
    # iC is NchanNear by Nchan, indicating for each channel the nearest
    # channels to it

    nt0min = params.nt0min
    spkTh = params.ThPre
    nt0, NrankPC = wPCA.shape
    NT, Nchan = params.NT, probe.Nchan

    # starts with predefined PCA waveforms
    wPCA = wPCA[:, :3]

    NchanNear = iC.shape[0]

    batchstart = np.arange(0, NT * Nbatch + 1, NT)  # batches start at these timepoints

    offset = Nchan * batchstart[ibatch]
    dat = proc[offset:offset + NT * Nchan].reshape((-1, Nchan), order='F')
    dataRAW = cp.asarray(dat, dtype=np.float32) / params.scaleproc

    # another Params variable to take all our parameters into the C++ code
    Params = [NT, Nchan, NchanNear, nt0, nt0min, spkTh, NrankPC]

    # call a CUDA function to do the hard work
    # returns a matrix of features uS, as well as the center channels for each spike
    uS, idchan = mexThSpkPC(Params, dataRAW, wPCA, iC)

    return uS, idchan
