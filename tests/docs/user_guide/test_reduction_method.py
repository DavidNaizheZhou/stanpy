import numpy as np
import stanpy as stp
import matplotlib.pyplot as plt


def test_ex01():
    EI = 32000  # kN/m2
    l = 6  # m

    hinged_support = {"w": 0, "M": 0}
    roller_support = {"w": 0, "M": 0, "H": 0}
    fixed_support = {"w": 0, "phi": 0}

    s1 = {"EI": EI, "l": l, "bc_i": hinged_support, "bc_k": {"w": 0}, "q": 10}
    s2 = {"EI": EI, "l": l, "bc_k": roller_support}

    s = [s1, s2]

    x_annotate = np.array([l, 2 * l])
    x = np.sort(np.append(np.linspace(0, 2 * l, 1000), x_annotate))
    x, Z_x = stp.solve_system(s1, s2, x=x)

    w_x = Z_x[:, 0]
    phi_x = Z_x[:, 1]
    M_x = Z_x[:, 2]
    V_x = Z_x[:, 3]

    # test boundary conditions
    np.testing.assert_allclose(w_x[0], 0)
    np.testing.assert_allclose(w_x[x == l], 0)
    np.testing.assert_allclose(w_x[-1], 0)
    np.testing.assert_allclose(M_x[0], 0)
    np.testing.assert_allclose(M_x[-1], 0)

    np.testing.assert_allclose(M_x[x == l], -22.5)
    np.testing.assert_allclose(V_x[0], 26.25)
    np.testing.assert_allclose(np.max(V_x[x == l]), 3.75)
    np.testing.assert_allclose(np.min(V_x[x == l]), -33.75)
    np.testing.assert_allclose(V_x[-1], 3.75)

if __name__ == "__main__":
    test_ex01()
