from pathlib import Path
import os.path as op
from pytest import fixture

import numpy as np

from ..utils import Bunch, read_data
from .. import add_default_handler

np.random.seed(0)


add_default_handler(level='DEBUG')


@fixture
def data_path():
    path = Path(__file__).parent / '../../data/'
    assert path.exists()
    return path


@fixture
def dat_path(data_path):
    return data_path / 'imec_385_100s.bin'


@fixture
def raw_data(dat_path):
    # WARNING: Fortran order
    return read_data(dat_path, shape=(385, -1), dtype=np.int16)


@fixture
def params():
    params = Bunch()

    params.fs = 30000.
    params.fshigh = 150.
    params.fslow = None
    params.NT = 65600
    params.NTbuff = 65856
    params.ntbuff = 64
    params.nSkipCov = 25
    params.whiteningRange = 32
    params.scaleproc = 200
    params.spkTh = -6
    params.nt0 = 61
    params.minfr_goodchannels = .1

    return params


@fixture
def probe(data_path):
    probe = Bunch()
    probe.NchanTOT = 385
    probe.Nchan = 293
    # WARNING: indexing mismatch with MATLAB hence the -1
    probe.chanMap = np.load(data_path / 'chanMap.npy').squeeze().astype(np.int64) - 1
    probe.xc = np.load(data_path / 'xc.npy').squeeze()
    probe.yc = np.load(data_path / 'yc.npy').squeeze()
    return probe


@fixture(params=[np.float64, np.float32])
def dtype(request):
    return np.dtype(request.param)


@fixture(params=[0, 1])
def axis(request):
    return request.param
