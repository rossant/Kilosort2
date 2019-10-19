import numpy as np


class Bunch(dict):
    """A subclass of dictionary with an additional dot syntax."""
    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self


def p(x):
    print("shape", x.shape, "mean", "%5e" % x.mean())
    print(x[:2, :2])
    print()
    print(x[-2:, -2:])


def is_fortran(x):
    if isinstance(x, np.ndarray):
        return x.flags.f_contiguous


def read_data(dat_path, offset=0, shape=None, dtype=None, axis=0):
    count = shape[0] * shape[1] if shape and -1 not in shape else -1
    buff = np.fromfile(dat_path, dtype=dtype, count=count, offset=offset)
    if -1 not in shape:
        shape = (-1, shape[1]) if axis == 0 else (shape[0], -1)
    buff = buff.reshape(shape, order='F')
    return buff
