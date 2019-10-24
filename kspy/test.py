
from pathlib import Path
import os.path as op
import sys
from math import ceil
from importlib import reload

import numpy as np
import cupy as cp
from scipy.linalg import svd
from scipy.signal import butter, filtfilt, lfilter
import matplotlib.pyplot as plt
from importlib import reload
np.set_printoptions(precision=4)

sys.path.append("..")
from kspy import preprocess
reload(preprocess)
from kspy import preprocess, utils, cptools
from kspy.utils import Bunch

from kspy import preprocess
reload(preprocess)
from kspy.utils import *
from kspy.cptools import *
from kspy.preprocess import *
from kspy.cluster import *
from kspy.learn import *
from kspy import preprocess

def p(x):
    print("shape", x.shape, "mean", "%5e" % x.mean())
    print(x[:2, :2])
    print()
    print(x[-2:, -2:])

def h(x):
    plt.hist(cp.asnumpy(x).flat, 50, log=True);

def check(x):
    x = cp.atleast_2d(x)
    print((x!=0).mean())
    plt.subplot(121);
    h(x.mean(axis=0))
    plt.subplot(122);
    h(x.mean(axis=1))

data_path = Path('../data/')

probe = utils.Bunch()
probe.Nchan = 293
probe.NchanTOT = 385
probe.NchanNear = min(probe.Nchan, 2 * 8 + 1);

# WARNING: indexing mismatch with MATLAB hence the -1
probe.chanMap = np.load(data_path / 'chanMap.npy').squeeze().astype(np.int64) - 1
probe.xc = np.load(data_path / 'xc.npy').squeeze()
probe.yc = np.load(data_path / 'yc.npy').squeeze()
probe.kcoords = np.load(data_path / 'kcoords.npy').squeeze()

dat_path = data_path / 'imec_385_100s.bin'
raw_data = utils.read_data(dat_path, shape=(385, -1), dtype=np.int16)

params = Bunch()

params.fs = 30000.
params.fshigh = 150.
params.fslow = None
params.ntbuff = 64
params.NT = 65600
params.NTbuff = params.NT + 4 * params.ntbuff  # we need buffers on both sides for filtering
params.nskip = 25
params.nSkipCov = 25
params.whiteningRange = 32
params.scaleproc = 200
params.spkTh = -6
params.nt0 = 61
params.minfr_goodchannels = .1
params.nfilt_factor = 4
params.nt0 = 61
params.nt0min = ceil(20 * params.nt0 / 61)

params.loc_range = [5, 4]
params.long_range = [30, 6]
params.nPCs = 3
params.Nfilt = params.nfilt_factor * probe.Nchan
params.sigmaMask = 30
params.ThPre = 8
params.reorder = True

params.nup = 10  # for getKernels
params.sig = 1

igood = np.load(data_path / 'igood.npy').squeeze()
probe.Nchan = np.sum(igood)
probe.NchanNear = min(probe.Nchan, 2 * 8 + 1);
probe.chanMap = probe.chanMap[igood]
probe.xc = probe.xc[igood]
probe.yc = probe.yc[igood]
probe.kcoords = probe.kcoords[igood]

ns = op.getsize('../data/temp_wh.dat') // (2 * probe.Nchan)
proc = np.memmap(
    '../data/temp_wh.dat', shape=(probe.Nchan, ns), dtype=np.int16)

from kspy import cptools
reload(cptools)
from kspy import cluster
reload(cluster)

iorig, ccb, ccbsort = cluster.clusterSingleBatches(raw_data=raw_data, proc=proc, probe=probe, params=params)
