from pyrdw.lib.math import *


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def sign(x):
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0


def clamp(v, min_v, max_v):
    if v <= min_v:
        return min_v
    elif v <= max_v:
        return v
    else:
        return max_v


def l1_norm(v):
    return np.sum(np.abs(v))


def l2_norm(v):
    return np.sqrt(np.sum(v ** 2))


def l2_norm_square(v):
    return np.sum(v ** 2)


def cross(v1, v2):
    return np.cross(v1, v2)


def dot(v1, v2):
    return np.dot(v1, v2)
