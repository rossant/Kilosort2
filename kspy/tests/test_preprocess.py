import numpy as np
from ..preprocess import get_good_channels, get_whitening_matrix
# from ..utils import p


def test_good_channels(raw_data, probe, params):
    igood = get_good_channels(raw_data, probe, params)
    assert np.sum(~igood) == 4


def test_whitening_matrix(data_path, raw_data, probe, params):
    Wrot = get_whitening_matrix(raw_data, probe, params)
    Wrot_mat = np.load(data_path / 'whitening_matrix.npy')
    assert np.allclose(Wrot, Wrot_mat, atol=5)
