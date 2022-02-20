import stanpy as stp
import numpy as np

# todo: define classes, parametrization


def test_gamma_K_function(self):
    EI = 32000  # kNm²
    GA = 20000  # kNm²
    l = 6  # m
    H = 10  # kN
    q = 4  # kN/m
    N = -1500  # kN
    w_0 = 0.03  # m

    s = {
        "EI": EI,
        "GA": GA,
        "l": l,
        "q": q,
        "P": (H, l / 2),
        "N": N,
        "w_0": w_0,
        "bc_i": {"w": 0, "phi": 0},
        "bc_k": {"w": 0, "M": 0, "H": 0},
    }

    gamma, K = stp.gamma_K_function(**s)

    np.testing.assert_allclose(gamma, 108.108e-2, atol=1e-5)
    np.testing.assert_allclose(K, -506.757e-4, atol=1e-5)


def test_bj_constant_function():
    pass


if __name__ == "__main__":
    test_bj_constant_function()
