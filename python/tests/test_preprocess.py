import numpy as np
from ..preprocess import get_good_channels, get_whitening_matrix
# from ..utils import p


def test_good_channels(dat_path, params):
    igood = get_good_channels(dat_path, **params)
    assert np.sum(~igood) == 4


def test_whitening_matrix(dat_path, params):
    Wrot = get_whitening_matrix(dat_path, **params)
    Wrot_mat = np.load(dat_path.parent / 'whitening_matrix.npy')
    assert np.allclose(Wrot, Wrot_mat, atol=5)
