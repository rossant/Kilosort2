from pathlib import Path
import numpy as np
from pytest import fixture

from ..utils import Bunch
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
def params(data_path):
    ops = Bunch()
    ops.fs = 30000.
    ops.fshigh = 150.
    ops.fslow = None
    ops.Nbatch = 46
    ops.twind = 0
    ops.NchanTOT = 385
    ops.NT = 65600
    ops.NTbuff = 65856
    ops.Nchan = 293
    ops.ntbuff = 64
    ops.nSkipCov = 25
    ops.whiteningRange = 32
    ops.scaleproc = 200
    ops.twind = 0
    ops.spkTh = -6
    ops.nt0 = 61
    ops.minfr_goodchannels = .1

    ops.chanMap = np.load(data_path / 'chanMap.npy').squeeze().astype(np.int64)
    # WARNING: indexing mismatch with MATLAB
    ops.chanMap -= 1

    ops.xc = np.load(data_path / 'xc.npy').squeeze()
    ops.yc = np.load(data_path / 'yc.npy').squeeze()

    return ops


@fixture(params=[np.float64, np.float32])
def dtype(request):
    return np.dtype(request.param)


@fixture(params=[0, 1])
def axis(request):
    return request.param
