import numpy as np
import sympy as sym


def signif(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(np.round(x * mags) / mags, 10)


def printtex(F: np.ndarray, p: int = 6):
    """_summary_

    :param F: _description_
    :type F: np.ndarray
    :param p: _description_, defaults to 6
    :type p: int, optional
    :return: _description_
    :rtype: _type_
    """
    return sym.latex(sym.Matrix(signif(F, p)))
