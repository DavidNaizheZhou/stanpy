import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import stanpy as stp
import copy
from scipy.special import factorial


def gamma_K_function(**s):
    N = -s.get("N", 0)
    EI, GA = load_material_parameters(**s)

    gamma = 1 / (1 - (N / GA))
    K = -gamma * N / EI
    return gamma, K


def convert_poly(function):
    if isinstance(function, sp.core.add.Add) or isinstance(function, sp.core.mul.Mul):
        deg = sp.degree(function)
        a_poly = 0
        q = sp.symbols("q0:{}".format(deg + 1))
        for i in range(deg + 1):
            a_poly += sp.Symbol("x") ** i / np.math.factorial(i) * q[i]
        a_poly = sp.poly(a_poly)

        function_poly = sp.poly(function)

        dict_sol = sp.solve(a_poly - function_poly, q)
        sol = np.array([dict_sol[key] for key in q])
    elif isinstance(function, np.ndarray):
        sol = function * factorial(np.arange(function.size, 0, -1))
    elif isinstance(function, np.poly1d):
        sol = function.coefficients * factorial(np.arange(function.coefficients.size, 0, -1))
    else:
        sol = np.array([function]).flatten()

    return sol


def convert_poly_wv(function):
    if isinstance(function, (sp.core.mul.Mul, sp.core.add.Add)):
        sol = np.zeros(5)
        deg = sp.degree(function)
        factor = factorial(np.arange(0, deg + 1, dtype=int))
        coeffs = np.flip(function.as_poly().all_coeffs())
        sol[: coeffs.size] = coeffs * factor

    else:
        sol = np.array([function]).flatten()

    return sol


def convert_psi0_w0_to_wv(**s):
    x = sp.Symbol("x")
    if ("w_0" in s.keys() or "psi_0" in s.keys()) and "l" in s.keys():
        l = s.get("l")
        w0 = s.get("w_0", 0)
        psi_0 = s.get("psi_0", 0)
        phivi = psi_0 + 4 * w0 / l
        kappav = 8 * w0 / l**2
        # phiv = phivi-x*kappav
        wv = x * phivi - x**2 / 2 * kappav
        return wv
    else:
        return np.zeros(4)


def bj_p89(K: float, x: float, j: int):
    s = j
    aj = x**j / np.math.factorial(j)
    bj = aj
    beta = aj
    num_iterations = 0
    while True:
        s = s + 2
        beta = beta * K * x**2 / s / (s - 1)
        bj = bj + beta
        num_iterations += 1
        if beta <= np.abs(bj) * 10**-9:
            break
    return bj


def bj_struktur_p89(x, n: int = 5, **s):
    gamma, K = gamma_K_function(**s)
    b_j = np.empty((x.size, n + 1))
    for i, xi in enumerate(x):
        for j in range(n + 1):
            b_j[i, j] = bj_p89(K, xi, j)
    return b_j


def bj_opt2_p89(
    x: float = np.array([]),
    n: int = 5,
    n_iterations: int = 30,
    return_aj: bool = False,
    **s,
):
    if isinstance(x, int) or isinstance(x, float):
        x = np.array([x])
    if x.size == 0:
        x = np.array([s.get("l")])
    gamma, K = gamma_K_function(**s)
    t = np.arange(0, n_iterations + 1)
    j = np.arange(n - 1, n + 1).reshape(-1, 1)
    aj = aj_function_x(x, n)
    beta = K / (j + 2 * t) / (j + 2 * t - 1) * x[:, None, None] ** 2
    beta[:, :, 0] = aj[:, -2:]
    bn_end = np.sum(np.multiply.accumulate(beta, axis=2), axis=2)  # todo create a Warning if not convergent
    bn = bj_recursion_p89(K, aj, bn_end)

    bn[x < 0, :] = 0

    if return_aj:
        return aj, bn
    else:
        return bn


def bj_recursion_p89(K: float, aj: np.ndarray, bn_end: np.ndarray):
    n = aj.shape[1] - 1
    bn = np.zeros(aj.shape)
    bn[:, -2:] = bn_end
    bn[:, :-2] = aj[:, :-2]
    bn = np.fliplr(bn)

    for i in range(n - 1):
        bn[:, i + 2] = bn[:, i] * K + bn[:, i + 2]
    bn = np.fliplr(bn)
    return bn


def aj_function_x(x, n):
    if isinstance(x, int) or isinstance(x, float):
        x = np.array([x])

    jn = np.arange(n + 1)
    jn_fact = jn
    jn_fact[0] = 1
    jn_fact = jn_fact.cumprod()
    an = x.reshape(-1, 1) ** jn / jn_fact
    an[:, 0] = 1

    an[x < 0, :] = 0

    return an


def load_material_parameters(**s):
    keys = s.keys()

    # todo: dict tree
    GA = s.get("GA", np.inf)
    if "EI" in keys or "GA" in keys:
        if "EI" in keys:
            EI = s.get("EI")
        elif "E" in keys and "I" in keys:
            E = s.get("E")
            I = s.get("I")
            EI = E * I
        if "GA" in keys:
            GA = s.get("GA")
        elif "G" in keys and "A~" in keys:
            G = s.get("G")
            A_tilde = s.get("A~")
            GA = G * A_tilde
        else:
            GA = np.inf

    elif "cs" in keys and "E" in keys:
        E = s.get("E")
        cs = s.get("cs")
        EI = cs["I_y"] * E
        # todo GA~
    return EI, GA


def flatten_dict(o):
    return [s for i in o for s in flatten_dict(i)] if isinstance(o, (list, tuple)) else [o]


def extract_load_length_index_dict(x: np.ndarray, **s):
    load_dict = {key: s[key] for key in ["P", "q_delta", "M_e", "phi_e", "W_e"] if key in s.keys()}

    xj = np.unique(np.array(flatten_dict([load_dict[key][1:] for key in load_dict.keys()])).astype(float))
    index_dict = {
        key: np.in1d(xj, np.asarray(load_dict[key][1:]).astype(float)).nonzero()[0].astype(int)
        for key in load_dict.keys()
    }
    # assert xj[index_dict["P"]] == s["P"][1]
    x_for_bj = np.zeros(x.size * (xj.size + 1))
    x_for_bj[: x.size] = x
    x_for_bj[x.size :] = (x.reshape(-1, 1) - xj).flatten()
    x_for_bj[x_for_bj < 0] = 0

    return xj, x_for_bj, index_dict


def load_q_hat(q_j: np.ndarray = np.array([]), wv_j: np.ndarray = np.array([]), **s):

    l = s.get("l")
    N = -s.get("N", 0)
    if q_j.size == 0:
        q = s.get("q", 0)
        q_j = convert_poly(q)

    if wv_j.size == 0:
        wv = convert_psi0_w0_to_wv(**s)
        wv_j = convert_poly(wv)

    q_j_hat = q_j - N * wv_j[2 : 2 + q_j.size]

    return q_j_hat.astype(float)


def calc_load_integral_R(
    x: np.ndarray = np.array([]),
    return_all=False,
    wv_j: object = None,
    **s,
):
    gamma, K = gamma_K_function(**s)
    N = -s.get("N", 0)
    P_array = stp.extract_P_from_beam(**s)
    l = s.get("l")
    if x.size == 0:
        x = np.array([l])
    q = s.get("q", 0)
    q_delta = s.get("q_delta", (0, 0, 0))
    EI, GA = load_material_parameters(**s)

    if wv_j == None:
        wv = convert_psi0_w0_to_wv(**s)
        wv_j = convert_poly_wv(wv)

    q_j = convert_poly(q)

    load_j_arrays = calc_loadj_arrays(q_j, wv_j, **s)

    # x_j, x_for_bj, index_dict = extract_load_length_index_dict(x, **s)
    aj, bj, x_loads, x_P, P_array, load_integrals_Q = calc_load_integral_Q(x, return_all=True, **s)

    mask = _load_bj_x_mask(x_loads, x)
    d_R = np.zeros((x.size, 5))
    d_R[:, 0] = -gamma * (bj[mask, 3] / EI - bj[mask, 1] / GA) * N * wv_j[1]
    d_R[:, 1] = -gamma * bj[mask, 2] / EI * N * wv_j[1]
    d_R[:, 2] = gamma * bj[mask, 1] * N * wv_j[1]

    load_integrals_R = load_integrals_Q + d_R

    load_integrals_R[:, 3] = -np.sum(aj[mask, 1 : 1 + q_j.size] * q_j, axis=1)
    if x_P.shape[0] > 0:
        for i in range(P_array.shape[0]):
            mask = _load_bj_x_mask(x_loads, x_P[:, i])
            load_integrals_R[:, 3] += -aj[mask, 0] * P_array[i, 0]

    # if "q_delta" in index_dict.keys():
    #     index_b_s = index_dict["q_delta"][0] + x.size
    #     index_b_ss = index_dict["q_delta"][1] + x.size
    #     load_integrals_R[:, 3] += -(aj[index_b_s :: x_j.size, 1] - aj[index_b_ss :: x_j.size, 1]) * q_delta[0]

    if return_all:
        return aj, bj, x_loads, x_P, P_array, load_integrals_R
    else:
        return load_integrals_R


def calc_load_integral_R_poly(
    x: np.ndarray = np.array([]),
    eta: np.ndarray = np.array([]),
    gamma: np.ndarray = np.array([]),
    load_j_arrays=None,
    return_aj: bool = False,
    return_bj: bool = False,
    return_all: bool = False,
    wv_j: object = None,
    **s,
):
    l = s.get("l")
    q = s.get("q", 0)
    N = -s.get("N", 0)
    _, K = gamma_K_function(**s)
    EI, GA = load_material_parameters(**s)

    eta, gamma = check_and_convert_eta_gamma(eta, gamma, **s)
    x = check_and_convert_input_array(x, **s)

    if isinstance(EI, sp.polys.polytools.Poly):
        EI_poly = np.poly1d(EI.coeffs())
        EI0 = EI_poly(0)

    elif isinstance(EI, (float, int)):
        EI_poly = np.poly1d(np.array([EI]))
        EI0 = EI_poly(0)

    elif isinstance(EI, np.poly1d):
        EI_poly = EI
        EI0 = EI_poly(0)

    # load all loads
    q_delta = s.get("q_delta", (0, 0, 0))
    P = s.get("P", 0)

    M_e = s.get("M_e", (0, 0))
    phi_e = s.get("phi_e", (0, 0))
    W_e = s.get("W_e", (0, 0))

    if wv_j == None:
        wv = convert_psi0_w0_to_wv(**s)
        wv_j = convert_poly_wv(wv)

    if load_j_arrays == None:
        q_j = convert_poly(q)
        load_j_arrays = calc_loadj_arrays(q_j, wv_j, **s)

    q_hat_j = load_j_arrays["q_hat_j"]
    m_j = load_j_arrays["m_j"]
    kappa_j = load_j_arrays["kappa_j"]

    max_bj_index = np.max([m_j.size + 3, q_hat_j.size + 4, kappa_j.size + 2]) - 1
    x_j, x_for_bj, index_dict = extract_load_length_index_dict(x, **s)

    aj, bj, x_loads, x_P, P_array, load_integrals_Q = calc_load_integral_Q_poly(
        x, return_all=True, load_j_arrays=load_j_arrays, **s
    )

    mask = _load_bj_x_mask(x_loads, x)
    d_R = np.zeros((x.size, 5))
    d_R[:, 0] = -bj[mask, 0, 3] / EI0 * N * wv_j[1]
    d_R[:, 1] = -bj[mask, 1, 3] / EI0 * N * wv_j[1]
    d_R[:, 2] = +bj[mask, 0, 1] * N * wv_j[1]

    load_integrals_R = load_integrals_Q + d_R

    load_integrals_R[:, 3] = -np.sum(aj[mask, 1 : 1 + q_j.size] * q_j, axis=1)

    if P_array.shape[0] > 0:
        for i in range(P_array.shape[0]):
            maskP = _load_bj_x_mask(x_loads, x_P[:, i])
            load_integrals_R[:, 3] += -aj[maskP, 0] * P[0]

    if "q_delta" in index_dict.keys():
        index_b_s = index_dict["q_delta"][0] + x.size
        index_b_ss = index_dict["q_delta"][1] + x.size
        load_integrals_R[:, 3] += -(aj[index_b_s :: x_j.size, 1] - aj[index_b_ss :: x_j.size, 1]) * q_delta[0]

    if return_all:
        return aj, bj, load_integrals_R, mask
    if return_bj and return_aj:
        return aj, bj, load_integrals_R
    elif return_bj:
        return bj, load_integrals_R
    elif return_aj:
        return aj, load_integrals_R
    else:
        return load_integrals_R


def calc_loadj_arrays(q_j: np.ndarray = np.array([]), wv_j: np.ndarray = np.array([]), **s):
    m_0 = s.get("m_0", 0)
    kappa_0 = s.get("kappa_0", 0)

    if q_j.size == 0:
        q = s.get("q", 0)
        q_j = convert_poly(q)

    if wv_j.size == 0:
        wv = convert_psi0_w0_to_wv(**s)
        wv_j = convert_poly(wv)

    q_hat_j = load_q_hat(q_j=q_j, wv_j=wv_j, **s)

    m_j = convert_poly(m_0)
    kappa_j = convert_poly(kappa_0)

    return {"q_hat_j": q_hat_j, "m_j": m_j, "kappa_j": kappa_j}


def extract_P_from_beam(**s):
    Px_array = np.array([value for key, value in s.items() if 'P' in key])
    return Px_array  # column 0: magnitude, column 1: position


def extract_N_from_beam(**s):
    Nx_array = np.array([value for key, value in s.items() if 'N' in key])
    return Nx_array  # column 0: magnitude, column 1: position


def _load_bj_x_mask(x, y):
    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, y)

    yindex = np.take(index, sorted_index, mode="clip")
    mask = x[yindex] != y

    result = np.ma.array(yindex, mask=mask)

    return result


def calc_load_integral_Q(x: np.ndarray = np.array([]), return_all=False, **s):

    gamma, K = gamma_K_function(**s)
    l = s.get("l")
    q = s.get("q", 0)
    N = -s.get("N", 0)

    if isinstance(x, (int, float)):
        x = np.array([x])
    if x.size == 0:
        x = np.array([l])

    EI, GA = load_material_parameters(**s)

    q_delta = s.get("q_delta", (0, 0, 0))
    P = s.get("P", 0)

    M_e = s.get("M_e", (0, 0))
    phi_e = s.get("phi_e", (0, 0))
    W_e = s.get("W_e", (0, 0))

    wv = convert_psi0_w0_to_wv(**s)
    wv_j = convert_poly_wv(wv)

    q_j = convert_poly(q)
    load_j_arrays = calc_loadj_arrays(q_j, wv_j, **s)

    q_hat_j = load_j_arrays["q_hat_j"]
    m_j = load_j_arrays["m_j"]
    kappa_j = load_j_arrays["kappa_j"]
    max_bj_index = np.max([m_j.size + 3, q_hat_j.size + 4, kappa_j.size + 2]) - 1

    x_loads = np.copy(x)
    P_array = extract_P_from_beam(**s)
    x_P = np.array([])
    if P_array.shape[0] > 0:
        x_P = x[:, None] - P_array[:, 1]  # every col is one P
        x_P[x_P < 0] = -1
        x_loads = np.append(x_loads, x_P.flatten())
    x_loads[x_loads < 0] = -1
    x_loads = np.unique(x_loads)

    aj, bj = bj_opt2_p89(x=x_loads, n=max_bj_index, return_aj=True, **s)

    q_hat_vec = np.zeros((x.size, 5))
    m_0_vec = np.zeros((x.size, 5))
    kappe_0_vec = np.zeros((x.size, 5))
    q_delta_vec = np.zeros((x.size, 5))
    P_vec = np.zeros((x.size, 5))
    M_e_vec = np.zeros((x.size, 5))
    phi_e_vec = np.zeros((x.size, 5))
    W_e_vec = np.zeros((x.size, 5))

    mask = _load_bj_x_mask(x_loads, x)
    if "q" in s.keys() or "w_0" in s.keys():
        q_hat_vec[:, 0] = gamma * np.sum(
            (bj[mask, 4 : 4 + q_hat_j.size] / EI - bj[mask, 2 : 2 + q_hat_j.size] / GA) * q_hat_j,
            axis=1,
        )
        q_hat_vec[:, 1] = gamma / EI * np.sum(bj[mask, 3 : 3 + q_hat_j.size] * q_hat_j, axis=1)
        q_hat_vec[:, 2] = -gamma * np.sum(bj[mask, 2 : 2 + q_hat_j.size] * q_hat_j, axis=1)
        q_hat_vec[:, 3] = -gamma * np.sum(bj[mask, 1 : 1 + q_hat_j.size] * q_hat_j, axis=1)
        q_hat_vec[:, 4] = 0.0

    if "m_0" in s.keys():
        m_0_vec[:, 0] = -gamma / EI * np.sum((bj[mask, 3 : 3 + m_j.size]) * m_j, axis=1)
        m_0_vec[:, 1] = -1 / EI * np.sum(bj[mask, 2 : 2 + m_j.size] * m_j, axis=1)
        m_0_vec[:, 2] = +np.sum(bj[mask, 1 : 1 + m_j.size] * m_j, axis=1)
        m_0_vec[:, 3] = +K * np.sum(bj[mask, 2 : 2 + m_j.size] * m_j, axis=1)
        m_0_vec[:, 4] = 0.0

    if "kappe_0" in s.keys():
        kappe_0_vec[:, 0] = -gamma * np.sum(bj[mask, 2 : 2 + kappa_j.size] * kappa_j, axis=1)
        kappe_0_vec[:, 1] = -gamma * np.sum(bj[mask, 1 : 1 + kappa_j.size] * kappa_j, axis=1)
        kappe_0_vec[:, 2] = -gamma * N * np.sum(bj[mask, 2 : 2 + kappa_j.size] * kappa_j, axis=1)
        kappe_0_vec[:, 3] = -gamma * N * np.sum(bj[mask, 1 : 1 + kappa_j.size] * kappa_j, axis=1)
        kappe_0_vec[:, 4] = 0.0

    # if "q_delta" in index_dict.keys():
    #     q_delta_vec[:, 0] = (
    #         gamma
    #         * (
    #             (bj[index_b_s :: x_j.size, 4] - bj[index_b_ss :: x_j.size, 4]) / EI
    #             - (bj[index_b_s :: x_j.size, 2] - bj[index_b_ss :: x_j.size, 2]) / GA
    #         )
    #         * q_delta[0]
    #     )
    #     q_delta_vec[:, 1] = +gamma * (bj[index_b_s :: x_j.size, 3] - bj[index_b_ss :: x_j.size, 3]) / EI * q_delta[0]
    #     q_delta_vec[:, 2] = -gamma * (bj[index_b_s :: x_j.size, 2] - bj[index_b_ss :: x_j.size, 2]) * q_delta[0]
    #     q_delta_vec[:, 3] = -gamma * (bj[index_b_s :: x_j.size, 1] - bj[index_b_ss :: x_j.size, 1]) * q_delta[0]
    #     q_delta_vec[:, 4] = 0.0

    if P_array.shape[0] > 0:
        for i in range(P_array.shape[0]):
            mask = _load_bj_x_mask(x_loads, x_P[:, i])
            P_vec[:, 0] += gamma * (bj[mask, 3] / EI - bj[mask, 1] / GA) * P_array[i, 0]
            P_vec[:, 1] += gamma * bj[mask, 2] / EI * P_array[i, 0]
            P_vec[:, 2] += -gamma * bj[mask, 1] * P_array[i, 0]
            P_vec[:, 3] += -gamma * bj[mask, 0] * P_array[i, 0]
            P_vec[:, 4] += 0.0

    # if "M_e" in index_dict.keys():

    #     index_b_s = index_dict["M_e"][0]
    #     M_e_vec[:] = np.array(
    #         [
    #             -gamma * (bj[index_b_s, 2] / EI) * M_e[0],
    #             -bj[index_b_s, 1] / EI * M_e[0],
    #             +bj[index_b_s, 0] * M_e[0],
    #             +K * bj[index_b_s, 1] * M_e[0],
    #             0.0,
    #         ]
    #     )
    # if "phi_e" in index_dict.keys():
    #     index_b_s = index_dict["phi_e"][0]
    #     phi_e_vec[:] = np.array(
    #         [
    #             -gamma * bj[index_b_s, 2] * phi_e[0],
    #             -bj[index_b_s, 0] * phi_e[0],
    #             -gamma * N * bj[index_b_s, 1] * phi_e[0],
    #             -gamma * N * bj[index_b_s, 0] * phi_e[0],
    #             0.0,
    #         ]
    #     )
    # if "W_e" in index_dict.keys():
    #     index_b_s = index_dict["W_e"][0]
    #     W_e_vec[:] = np.array(
    #         [
    #             +bj[index_b_s, 0] * W_e[0],
    #             +K / gamma * bj[index_b_s, 1] * W_e[0],
    #             +N * bj[index_b_s, 0] * W_e[0],
    #             +N * K * bj[index_b_s, 1] * W_e[0],
    #             0.0,
    #         ]
    #     )
    # print(q_hat_vec)
    # print(q_delta_vec)

    load_integrals_Q = q_hat_vec + P_vec + q_delta_vec
    load_integrals_Q[:, -1] = 1.0

    if return_all:
        return aj, bj, x_loads, x_P, P_array, load_integrals_Q
    else:
        return load_integrals_Q


def calc_load_integral_Q_old(
    x: np.ndarray = np.array([]),
    return_bj: bool = False,
    return_aj: bool = False,
    return_all: bool = False,
    wv_j=None,
    load_j_arrays=None,
    side="+",
    **s,
):
    gamma, K = gamma_K_function(**s)
    l = s.get("l")
    q = s.get("q", 0)
    N = -s.get("N", 0)
    if x.size == 0:
        x = np.array([l])

    EI, GA = load_material_parameters(**s)

    q_delta = s.get("q_delta", (0, 0, 0))
    P = s.get("P", 0)

    M_e = s.get("M_e", (0, 0))
    phi_e = s.get("phi_e", (0, 0))
    W_e = s.get("W_e", (0, 0))

    if wv_j == None:
        wv = convert_psi0_w0_to_wv(**s)
        wv_j = convert_poly_wv(wv)

    if load_j_arrays == None:
        q_j = convert_poly(q)
        load_j_arrays = calc_loadj_arrays(q_j, wv_j, **s)

    q_hat_j = load_j_arrays["q_hat_j"]
    m_j = load_j_arrays["m_j"]
    kappa_j = load_j_arrays["kappa_j"]

    max_bj_index = np.max([m_j.size + 3, q_hat_j.size + 4, kappa_j.size + 2]) - 1
    x_j, x_for_bj, index_dict = extract_load_length_index_dict(x, **s)

    # aj, bj = bj_opt2_p89(x_for_bj, max_bj_index, return_aj=True, side=side, **s)
    aj, bj = bj_opt2_p89(x, max_bj_index, return_aj=True, side=side, **s)

    # aj, bj for Forces P
    P_array = extract_P_from_beam(**s)

    q_hat_vec = np.zeros((x.size, 5))
    m_0_vec = np.zeros((x.size, 5))
    kappe_0_vec = np.zeros((x.size, 5))
    q_delta_vec = np.zeros((x.size, 5))
    P_vec = np.zeros((x.size, 5))
    M_e_vec = np.zeros((x.size, 5))
    phi_e_vec = np.zeros((x.size, 5))
    W_e_vec = np.zeros((x.size, 5))

    if "q" in s.keys() or "w_0" in s.keys():
        q_hat_vec[:, 0] = gamma * np.sum(
            (bj[: x.size, 4 : 4 + q_hat_j.size] / EI - bj[: x.size, 2 : 2 + q_hat_j.size] / GA) * q_hat_j,
            axis=1,
        )
        q_hat_vec[:, 1] = gamma / EI * np.sum(bj[: x.size, 3 : 3 + q_hat_j.size] * q_hat_j, axis=1)
        q_hat_vec[:, 2] = -gamma * np.sum(bj[: x.size, 2 : 2 + q_hat_j.size] * q_hat_j, axis=1)
        q_hat_vec[:, 3] = -gamma * np.sum(bj[: x.size, 1 : 1 + q_hat_j.size] * q_hat_j, axis=1)
        q_hat_vec[:, 4] = 0.0

    if "m_0" in s.keys():
        m_0_vec[:, 0] = -gamma / EI * np.sum((bj[: x.size, 3 : 3 + m_j.size]) * m_j, axis=1)
        m_0_vec[:, 1] = -1 / EI * np.sum(bj[: x.size, 2 : 2 + m_j.size] * m_j, axis=1)
        m_0_vec[:, 2] = +np.sum(bj[: x.size, 1 : 1 + m_j.size] * m_j, axis=1)
        m_0_vec[:, 3] = +K * np.sum(bj[: x.size, 2 : 2 + m_j.size] * m_j, axis=1)
        m_0_vec[:, 4] = 0.0

    if "kappe_0" in s.keys():
        kappe_0_vec[:, 0] = -gamma * np.sum(bj[: x.size, 2 : 2 + kappa_j.size] * kappa_j, axis=1)
        kappe_0_vec[:, 1] = -gamma * np.sum(bj[: x.size, 1 : 1 + kappa_j.size] * kappa_j, axis=1)
        kappe_0_vec[:, 2] = -gamma * N * np.sum(bj[: x.size, 2 : 2 + kappa_j.size] * kappa_j, axis=1)
        kappe_0_vec[:, 3] = -gamma * N * np.sum(bj[: x.size, 1 : 1 + kappa_j.size] * kappa_j, axis=1)
        kappe_0_vec[:, 4] = 0.0

    if "q_delta" in index_dict.keys():
        index_b_s = index_dict["q_delta"][0] + x.size
        index_b_ss = index_dict["q_delta"][1] + x.size

        q_delta_vec[:, 0] = (
            gamma
            * (
                (bj[index_b_s :: x_j.size, 4] - bj[index_b_ss :: x_j.size, 4]) / EI
                - (bj[index_b_s :: x_j.size, 2] - bj[index_b_ss :: x_j.size, 2]) / GA
            )
            * q_delta[0]
        )
        q_delta_vec[:, 1] = +gamma * (bj[index_b_s :: x_j.size, 3] - bj[index_b_ss :: x_j.size, 3]) / EI * q_delta[0]
        q_delta_vec[:, 2] = -gamma * (bj[index_b_s :: x_j.size, 2] - bj[index_b_ss :: x_j.size, 2]) * q_delta[0]
        q_delta_vec[:, 3] = -gamma * (bj[index_b_s :: x_j.size, 1] - bj[index_b_ss :: x_j.size, 1]) * q_delta[0]
        q_delta_vec[:, 4] = 0.0

    if "P" in index_dict.keys():
        for i in range(P_array.shape[0]):
            x_P = x - P_array[i, 1]
            x_P[x_P <= 0] = 0
            aj_P, bj_P = bj_opt2_p89(x_P, max_bj_index, return_aj=True, side=side, **s)

            P_vec[:, 0] += gamma * (bj_P[i, 3] / EI - bj_P[i, 1] / GA) * P_array[i, 0]
            P_vec[:, 1] += +gamma * bj_P[i, 2] / EI * P_array[i, 0]
            P_vec[:, 2] += -gamma * bj_P[i, 1] * P_array[i, 0]
            P_vec[:, 3] += -gamma * bj_P[i, 0] * P_array[i, 0]
            P_vec[:, 4] += 0.0
    if "M_e" in index_dict.keys():

        index_b_s = index_dict["M_e"][0]
        M_e_vec[:] = np.array(
            [
                -gamma * (bj[index_b_s, 2] / EI) * M_e[0],
                -bj[index_b_s, 1] / EI * M_e[0],
                +bj[index_b_s, 0] * M_e[0],
                +K * bj[index_b_s, 1] * M_e[0],
                0.0,
            ]
        )
    if "phi_e" in index_dict.keys():
        index_b_s = index_dict["phi_e"][0]
        phi_e_vec[:] = np.array(
            [
                -gamma * bj[index_b_s, 2] * phi_e[0],
                -bj[index_b_s, 0] * phi_e[0],
                -gamma * N * bj[index_b_s, 1] * phi_e[0],
                -gamma * N * bj[index_b_s, 0] * phi_e[0],
                0.0,
            ]
        )
    if "W_e" in index_dict.keys():
        index_b_s = index_dict["W_e"][0]
        W_e_vec[:] = np.array(
            [
                +bj[index_b_s, 0] * W_e[0],
                +K / gamma * bj[index_b_s, 1] * W_e[0],
                +N * bj[index_b_s, 0] * W_e[0],
                +N * K * bj[index_b_s, 1] * W_e[0],
                0.0,
            ]
        )

    load_integrals_Q = q_hat_vec + P_vec + q_delta_vec
    load_integrals_Q[:, -1] = 1.0

    if return_all:
        return aj, bj, aj_P, bj_p, load_integrals_Q
    elif return_bj and return_aj:
        return aj, bj, load_integrals_Q
    elif return_bj:
        return bj, load_integrals_Q
    elif return_aj:
        return aj, load_integrals_Q
    else:
        return load_integrals_Q


def tr_Q(x: np.ndarray = np.array([]), **s):
    l = s.get("l")
    if x.size == 0:
        x = np.array([l])

    gamma, K = gamma_K_function(**s)
    EI, GA = load_material_parameters(**s)

    bj, load_integrals_Q = calc_load_integral_Q(x, return_bj=True, **s)

    tr = np.zeros((x.size, 5, 5))
    tr[:, :, :] = np.eye(5, 5)
    tr[:, 0, 1] = x
    tr[:, 0, 2] = -gamma * bj[: x.size, 2] / EI
    tr[:, 0, 3] = -bj[: x.size, 3] / EI - bj[: x.size, 1] / GA
    tr[:, 1, 2] = -gamma * bj[: x.size, 2] / EI
    tr[:, 1, 3] = -bj[: x.size, 3] / EI - bj[: x.size, 1] / GA
    tr[:, 2, 2] = bj[: x.size, 0]
    tr[:, 2, 3] = bj[: x.size, 1]
    tr[:, 3, 2] = K * bj[: x.size, 1]
    tr[:, 3, 3] = bj[: x.size, 0]

    tr[:, :, 4] = load_integrals_Q

    return tr


def calc_x_system(s_list, x: np.ndarray = np.array([])):
    if isinstance(s_list, dict):
        s_list = [s_list]
    if x.size == 0:
        x = np.zeros(len(s_list))
        l0 = 0
        for i, s in enumerate(s_list):
            l0 += s["l"]
            x[i] = l0
        return x
    else:
        pass  # todo? are there other cases?


def calc_x_mask(s_list: list, x: np.ndarray):
    lengths = np.zeros(len(s_list) + 1)
    lengths[1:] = np.cumsum(np.array([s["l"] for s in s_list]))
    mask = (x >= lengths[:-1].reshape(-1, 1)) & (x <= lengths[1:].reshape(-1, 1))
    return lengths, mask


# def check_constant_or_polynomial(**s):
#     if isinstance(s.get("EI",0), sp.


def tr(
    *args,
    x: np.ndarray = np.array([]),
    side="+",
):

    if isinstance(args, dict):
        args = [args]

    if isinstance(x, (int, float)):
        x = np.array([x])

    if x.size == 0:
        x = stp.calc_x_system(*args)

    tr_R_ends = np.zeros((len(args) + 1, 5, 5))
    tr_R_ends[0, :, :] = np.eye(5, 5)
    tr_R_x = np.zeros((x.size, 5, 5))

    lengths, x_mask = calc_x_mask(args, x)
    for i, s in enumerate(args):
        EI, GA = load_material_parameters(**s)
        if isinstance(EI, (float, int)):
            tr_R_ends[i + 1, :, :] = tr_R(**s).dot(tr_R_ends[i, :, :])
            tr_R_x[x_mask[i], :, :] = tr_R(x=x[x_mask[i]] - lengths[i], **s).dot(tr_R_ends[i, :, :])
        elif isinstance(EI, np.poly1d):
            tr_R_ends[i + 1, :, :] = tr_R_poly(**s).dot(tr_R_ends[i, :, :])
            tr_R_x[x_mask[i], :, :] = tr_R_poly(x=x[x_mask[i]] - lengths[i], **s).dot(tr_R_ends[i, :, :])
    if x.size == 1:
        tr_R_x = tr_R_x.reshape((5, 5))

    return tr_R_x


def R_to_Q(x: np.ndarray = np.array([]), solution_vector: np.ndarray = np.array([]), *args):

    if isinstance(args, dict):
        args = [args]

    if x.size == 0:
        x = calc_x_system(args)

    lengths, x_mask = calc_x_mask(args, x)
    Q = np.zeros(x.size)
    aj = aj_function_x(x, 1)
    for i, s in enumerate(args):
        psi_0 = s.get("psi_0", 0)
        w_0 = s.get("w_0", 0)
        N = -s.get("N", 0)
        l = s.get("l")
        w1v = psi_0 + 4 * w_0 / l
        w2v = -8 * w_0 / l**2
        diff_wv = aj[x_mask[i], 0] * w1v + aj[x_mask[i], 1] * w2v
        gamma, K = gamma_K_function(**s)
        Q[x_mask[i]] = gamma * (solution_vector[x_mask[i], 3] + N * (solution_vector[x_mask[i], 1] + diff_wv))

    return Q


def tr_R(x: np.ndarray = np.array([]), **s):

    if isinstance(x, (int, float, list)):
        x = np.array([x]).flatten()

    N = -s.get("N", 0)
    l = s.get("l")
    if x.size == 0:
        x = np.array([l])
    x_shape = x.shape
    x = x.flatten()

    gamma, K = gamma_K_function(**s)
    EI, GA = load_material_parameters(**s)

    aj, bj, x_loads, x_P, P_array, load_integrals_R = calc_load_integral_R(x, return_all=True, **s)

    tr = np.zeros((x.size, 5, 5))
    mask = _load_bj_x_mask(x_loads, x)

    tr[:, :, :] = np.eye(5, 5)
    tr[:, 0, 1] = gamma * bj[mask, 1]
    tr[:, 0, 2] = -gamma * bj[mask, 2] / EI
    tr[:, 0, 3] = -gamma * (bj[mask, 3] / EI - bj[mask, 1] / GA)
    tr[:, 1, 1] = bj[mask, 0]
    tr[:, 1, 2] = -bj[mask, 1] / EI
    tr[:, 1, 3] = -gamma * bj[mask, 2] / EI
    tr[:, 2, 1] = gamma * N * bj[mask, 1]
    tr[:, 2, 2] = bj[mask, 0]
    tr[:, 2, 3] = gamma * bj[mask, 1]
    tr[:, :, 4] = load_integrals_R

    if x.size == 1:
        return tr.reshape((5, 5))
    else:
        return tr.reshape((*x_shape, 5, 5))


def tr_R_poly(
    x: np.ndarray = np.array([]),
    eta: np.ndarray = np.array([]),
    gamma: np.ndarray = np.array([]),
    **s,
):

    l = s.get("l")
    N = -s.get("N", 0)
    x = check_and_convert_input_array(x, **s)

    _, K = gamma_K_function(**s)
    EI, GA = load_material_parameters(**s)

    if isinstance(EI, sp.polys.polytools.Poly):
        EI_poly = EI
        EI0 = EI(0)
    elif isinstance(EI, (float, int)):
        EI_poly = sp.Poly(EI, sp.Symbol("x"))
        EI0 = EI
    elif isinstance(EI, np.poly1d):
        EI_poly = EI
        EI0 = EI(0)

    eta, gamma = check_and_convert_eta_gamma(eta, gamma, **s)

    # todo repair eta gamma scheme
    # aj, bj = bj_opt2_p119_forloop(K,x,eta=eta, return_aj=True, **s)
    aj, bj, load_integrals_R, mask = calc_load_integral_R_poly(x, eta=eta, gamma=gamma, return_all=True, **s)
    # bj, load_integrals_Q = calc_load_integral_Q(x, return_bj=True,**s)

    tr = np.zeros((x.size, 5, 5))
    tr[:, :, :] = np.eye(5, 5)
    tr[:, 0, 1] = bj[mask, 0, 1]
    tr[:, 0, 2] = -bj[mask, 0, 2] / EI0
    tr[:, 0, 3] = -bj[mask, 0, 3] / EI0
    tr[:, 1, 1] = bj[mask, 1, 1]
    tr[:, 1, 2] = -bj[mask, 1, 2] / EI0
    tr[:, 1, 3] = -bj[mask, 1, 3] / EI0
    tr[:, 2, 1] = N * bj[mask, 0, 1]
    tr[:, 2, 2] = bj[mask, 0, 0]
    tr[:, 2, 3] = bj[mask, 0, 1]

    tr[:, :, 4] = load_integrals_R

    if x.size == 1:
        return tr.reshape((5, 5))
    else:
        return tr.reshape((*x.shape, 5, 5))


def load_boundary_conditions(**s):
    bc_i = copy.copy(s.get("bc_i", {}))
    bc_k = copy.copy(s.get("bc_k", {}))
    if bc_i == "roller_support" or bc_i == "hinged_support":
        bc_i = {"w": 0, "M": 0}
    if bc_k == "roller_support" or bc_k == "hinged_support":
        bc_k = {"w": 0, "M": 0}
    if bc_i == "fixed_support":
        bc_i = {"w": 0, "phi": 0}
    if bc_k == "fixed_support":
        bc_k = {"w": 0, "phi": 0}

    bc_i.setdefault("w", 1)
    bc_i.setdefault("phi", 1)
    bc_i.setdefault("M", 1)
    bc_i.setdefault("V", 1)

    bc_k.setdefault("w", 1)
    bc_k.setdefault("phi", 1)
    bc_k.setdefault("M", 1)
    bc_k.setdefault("V", 1)

    bc_i_vec = np.array([bc_i["w"], bc_i["phi"], bc_i["M"], bc_i["V"]])
    bc_k_vec = np.array([bc_k["w"], bc_k["phi"], bc_k["M"], bc_k["V"]])
    return bc_i_vec, bc_k_vec


def solve_tr(Fki, **s):
    indices = np.arange(4)
    bc_i_vec, bc_k_vec = load_boundary_conditions(**s)
    A = Fki[:, indices[bc_i_vec == 1]][indices[bc_k_vec == 0]]
    b = -Fki[indices[bc_k_vec == 0], -1]
    zi = np.zeros(5)
    zi[-1] = 1
    zk = np.zeros(5)
    zk[-1] = 1
    zi[indices[bc_i_vec == 1]] = np.linalg.solve(A, b).round(15)
    zk = Fki.dot(zi).round(10)
    return zi, zk


def calc_load_integral_Q_poly(
    x: np.ndarray = np.array([]),
    bj=np.array([]),
    aj=np.array([]),
    eta=np.array([]),
    gamma=np.array([]),
    return_bj: bool = False,
    return_aj: bool = False,
    return_all: bool = False,
    wv_j=None,
    load_j_arrays=None,
    **s,
):

    l = s.get("l")
    q = s.get("q", 0)
    N = -s.get("N", 0)
    _, K = gamma_K_function(**s)
    EI, GA = load_material_parameters(**s)

    eta, _ = check_and_convert_eta_gamma(eta, gamma, **s)
    x = check_and_convert_input_array(x, **s)

    if isinstance(EI, sp.polys.polytools.Poly):
        EI_poly = np.poly1d(EI.coeffs())
        EI0 = EI_poly(0)

    elif isinstance(EI, (float, int)):
        EI_poly = np.poly1d(np.array([EI]))
        EI0 = EI_poly(0)
    elif isinstance(EI, (np.poly1d)):
        EI_poly = EI
        EI0 = EI(0)
    # load all loads
    q_delta = s.get("q_delta", (0, 0, 0))
    P = s.get("P", 0)

    M_e = s.get("M_e", (0, 0))
    phi_e = s.get("phi_e", (0, 0))
    W_e = s.get("W_e", (0, 0))

    if wv_j == None:
        wv = convert_psi0_w0_to_wv(**s)
        wv_j = convert_poly_wv(wv)

    if load_j_arrays == None:
        q_j = convert_poly(q)
        load_j_arrays = calc_loadj_arrays(q_j, wv_j, **s)

    q_hat_j = load_j_arrays["q_hat_j"]
    m_j = load_j_arrays["m_j"]
    kappa_j = load_j_arrays["kappa_j"]

    max_bj_index = np.max([m_j.size + 3, q_hat_j.size + 4, kappa_j.size + 2]) - 1

    x_loads = np.copy(x)
    P_array = extract_P_from_beam(**s)
    x_P = np.array([])
    if P_array.shape[0] > 0:
        x_P = x[:, None] - P_array[:, 1]  # every col is one P
        x_P[x_P < 0] = -1
        x_loads = np.append(x_loads, x_P.flatten())
    x_loads[x_loads < 0] = -1
    x_loads = np.unique(x_loads)

    x_j, x_for_bj, index_dict = extract_load_length_index_dict(x, **s)

    aj, bj = bj_opt2_p119_forloop(K, x_loads, eta, max_bj_index + 1, return_aj=True, **s)

    q_hat_vec = np.zeros((x.size, 5))
    m_0_vec = np.zeros((x.size, 5))
    kappe_0_vec = np.zeros((x.size, 5))
    q_delta_vec = np.zeros((x.size, 5))
    P_vec = np.zeros((x.size, 5))
    M_e_vec = np.zeros((x.size, 5))
    phi_e_vec = np.zeros((x.size, 5))
    W_e_vec = np.zeros((x.size, 5))
    N_vec = np.zeros((x.size, 5))

    mask = _load_bj_x_mask(x_loads, x)
    if "q" in s.keys() or "w_0" in s.keys():
        q_hat_vec[:, 0] = 1 / EI0 * np.sum(bj[mask, 0, 4 : 4 + q_hat_j.size] * q_hat_j, axis=1)
        q_hat_vec[:, 1] = 1 / EI0 * np.sum(bj[mask, 1, 4 : 4 + q_hat_j.size] * q_hat_j, axis=1)
        q_hat_vec[:, 2] = -np.sum(aj[mask, 2 : 2 + q_hat_j.size] * q_hat_j, axis=1)
        q_hat_vec[:, 3] = -np.sum(aj[mask, 1 : 1 + q_hat_j.size] * q_hat_j, axis=1)
        q_hat_vec[:, 4] = 0.0

    if "m_0" in s.keys():
        m_0_vec[:, 0] = -1 / EI0 * np.sum(bj[mask, 0, 3 : 3 + m_j.size] * m_j, axis=1)
        m_0_vec[:, 1] = -1 / EI0 * np.sum(bj[mask, 1, 3 : 3 + m_j.size] * m_j, axis=1)
        m_0_vec[:, 2] = +np.sum(aj[mask, 1 : 1 + q_hat_j.size] * q_hat_j, axis=1)
        m_0_vec[:, 3:5] = 0.0

    if "kappa_0" in s.keys():
        kappe_0_vec[:, 0] = -kappa_j * np.sum(bj[mask, 0, 2 : 2 + gamma.size] * gamma, axis=1)
        kappe_0_vec[:, 1] = -kappa_j * np.sum(bj[mask, 1, 2 : 2 + gamma.size] * gamma, axis=1)
        kappe_0_vec[:, 2:5] = 0.0

    if "q_delta" in index_dict.keys():
        # todo rewrite with mask
        index_b_s = index_dict["q_delta"][0] + x.size
        index_b_ss = index_dict["q_delta"][1] + x.size
        EI_star = EI_poly(l - x_for_bj[index_b_s])
        EI_2star = EI_poly(l - x_for_bj[index_b_ss])

        q_delta_vec[:, 0] = (
            bj[index_b_s :: x_j.size, 0, 4] / EI_star - bj[index_b_ss :: x_j.size, 0, 4] / EI_2star
        ) * q_delta[0]
        q_delta_vec[:, 1] = (
            bj[index_b_s :: x_j.size, 1, 4] / EI_star - bj[index_b_ss :: x_j.size, 1, 4] / EI_2star
        ) * q_delta[0]
        q_delta_vec[:, 2] = (aj[index_b_s :: x_j.size, 2] - aj[index_b_ss, 2]) * q_delta[0]
        q_delta_vec[:, 3] = (aj[index_b_s :: x_j.size, 1] - aj[index_b_ss, 1]) * q_delta[0]
        q_delta_vec[:, 4] = 0.0

    if P_array.shape[0] > 0:
        for i in range(P_array.shape[0]):
            mask = _load_bj_x_mask(x_loads, x_P[:, i])
            # index_b_s = index_dict["P"][0] + x.size
            # EI_star = EI_poly(l - x_j[index_dict["P"][0]])
            EI_star = EI_poly(x)
            P_vec[:, 0] = bj[mask, 0, 3] / EI_star * P[0]
            P_vec[:, 1] = bj[mask, 1, 3] / EI_star * P[0]
            P_vec[:, 2] = -aj[mask, 1] * P[0]
            P_vec[:, 3] = -aj[mask, 0] * P[0]
            P_vec[:, 4] = 0.0

    if "M_e" in index_dict.keys():
        # todo rewrite with mask
        index_b_s = index_dict["M_e"][0] + x.size
        EI_star = EI_poly(l - x_for_bj[index_b_s])

        M_e_vec[:, 0] = -bj[index_b_s :: x_j.size, 0, 2] / EI_star * M_e[0]
        M_e_vec[:, 1] = -bj[index_b_s :: x_j.size, 1, 2] / EI_star * M_e[0]
        M_e_vec[:, 2] = aj[index_b_s :: x_j.size, 0] * M_e[0]
        M_e_vec[:, 3:5] = 0.0

    if "phi_e" in index_dict.keys():
        # todo rewrite with mask
        index_b_s = index_dict["phi_e"][0]
        phi_e_vec[:, 0] = -bj[index_b_s :: x_j.size, 0, 1] * phi_e[0]
        phi_e_vec[:, 1] = -bj[index_b_s :: x_j.size, 1, 1] * phi_e[0]
        phi_e_vec[:, 2:5] = 0.0

    if "W_e" in index_dict.keys():
        # todo rewrite with mask
        index_b_s = index_dict["W_e"][0]
        W_e_vec[:, 0] = -bj[index_b_s :: x_j.size, 0, 0] * W_e[0]
        W_e_vec[:, 1] = -bj[index_b_s :: x_j.size, 1, 0] * W_e[0]
        W_e_vec[:, 2:5] = 0.0

    load_integrals_Q = q_hat_vec + m_0_vec + kappe_0_vec + q_delta_vec + P_vec + M_e_vec + phi_e_vec + W_e_vec

    N_vec[:, 2:4] = N * load_integrals_Q[:, :2]

    load_integrals_Q += N_vec
    load_integrals_Q[:, -1] = 1.0

    if return_all:
        return aj, bj, x_loads, x_P, P_array, load_integrals_Q
    elif return_bj and return_aj:
        return aj, bj, load_integrals_Q
    elif return_bj:
        return bj, load_integrals_Q
    elif return_aj:
        return aj, load_integrals_Q
    else:
        return load_integrals_Q


def check_and_convert_input_array(x: np.ndarray = np.array([]), **s):
    l = s.get("l")

    if isinstance(x, list):
        x = np.array(x)

    if isinstance(x, list):
        x = np.array(x)

    if isinstance(x, float) or isinstance(x, int):
        x = np.array([x])

    if x.size == 0:
        x = np.array([l])

    return x


def tr_Q_poly(
    x: np.ndarray = np.array([]),
    eta: np.ndarray = np.array([]),
    gamma: np.ndarray = np.array([]),
    rotation_axis="y",
    **s,
):
    l = s.get("l")

    x = check_and_convert_input_array(x, **s)

    gamma, K = gamma_K_function(**s)
    EI, GA = load_material_parameters(**s)

    if isinstance(EI, sp.polys.polytools.Poly):
        EI_poly = EI
        EI0 = EI(0)
    elif isinstance(EI, float) or isinstance(EI, int):
        EI_poly = sp.Poly(EI, sp.Symbol("x"))
        EI0 = EI

    eta, gamma = check_and_convert_eta_gamma(eta, gamma, **s)

    aj, bj = bj_opt2_p119_forloop(K, x, eta=eta, gamma=gamma, return_aj=True)

    # bj, load_integrals_Q = calc_load_integral_Q(x, return_bj=True,**s)

    tr = np.zeros((x.size, 5, 5))
    tr[:, :, :] = np.eye(5, 5)
    tr[:, 0, 1] = x
    tr[:, 0, 2] = -bj[:, 0, 2] / EI0
    tr[:, 0, 3] = -bj[:, 0, 3] / EI0
    tr[:, 1, 2] = -bj[:, 1, 2] / EI0
    tr[:, 1, 3] = -bj[:, 1, 3] / EI0
    tr[:, 2, 2] = bj[:, 0, 0]
    tr[:, 2, 3] = bj[:, 0, 1]
    tr[:, 3, 2] = bj[:, 1, 0]
    tr[:, 3, 3] = bj[:, 1, 1]

    tr[:, :, 4] = calc_load_integral_Q_poly(x, bj=bj, aj=aj, eta=eta, gamma=gamma, **s)

    if x.size == 1:
        return tr.reshape((5, 5))
    else:
        return tr.reshape((*x.shape, 5, 5))


def bj_struktur_p119(Ka, x, j, n, eta):
    p = int(eta.size)
    beta = np.zeros(p)
    s = j
    f = beta[0] = h = 1
    while True:
        s += 1
        d = 0
        e = x / (s - n)
        for r in np.arange(p - 1, 0, -1):
            beta[r] = e * beta[r - 1]
            d = (d + beta[r] * eta[r]) * (s - r - 1)
        beta[0] = beta[2] * Ka - d
        f = f + beta[0]
        h = h / 10 + np.abs(beta[0])
        if h < 10e-9 * np.abs(f):
            break
        # elif s>2000:
        #   print("Warning!! - no convergence")
        #   break
    return f * x ** (j - n) / np.math.factorial(j - n)


def bj_recursion_p119(K: float, aj: np.array, bn: np.array):
    # from j = 0 to 1
    return aj[0] + K * bn[2], aj[1] + K * bn[3]


def check_and_convert_eta_gamma(eta: np.ndarray = np.array([]), gamma: np.ndarray = np.array([]), **s):
    if len(eta) == 0:
        if "cs" in s.keys():
            if "eta_y" in s["cs"].keys():
                eta = s["cs"]["eta_y"]
            else:
                eta = np.zeros(3)
                eta[0] = 1
        else:
            eta = np.zeros(3)
            eta[0] = 1
    if len([gamma]) == 0:
        if "cs" in s.keys():
            if "gamma_y" in s["cs"].keys():
                gamma = s["cs"]["gamma_y"]
            else:
                gamma = np.zeros(3)
                gamma[0] = 1
        else:
            gamma = np.zeros(3)
            gamma[0] = 1
    return eta, gamma


def bj_opt1_p119_forloop(
    Ka: float,
    x: np.ndarray,
    eta: np.ndarray = np.array([]),
    gamma: np.ndarray = np.array([]),
    n: int = 6,
    n_iterations: int = 50,
    return_aj=False,
    **s,
):
    if "eta_y" in s.keys():
        eta = s["eta_y"]
    eta, gamma = check_and_convert_eta_gamma(eta=eta, gamma=gamma, **s)
    x = check_and_convert_input_array(x, **s)

    j = np.arange(2, n).reshape(-1, 1)
    t = np.arange(1, n_iterations + eta.size)
    s = t + j

    n_array = np.arange(2)
    r = np.arange(1, eta.size)
    beta = np.zeros((x.size, n_array.size, j.size, n_iterations + eta.size, eta.size))
    beta[:, :, :, 0, 0] = 1
    e = x[:, None, None, None] / (
        s - n_array[:, None, None]
    )  # 0 Index = x | 1 index = n | 2 index = j | 3 Index = n_iterations
    beta_diag = np.multiply.accumulate(e, axis=3)[:, :, :, : r.size]
    beta[:, :, :, r, r] = beta_diag
    nom = factorial(s.flatten() - 2)
    denom = factorial(s.flatten() - 2 - r.reshape(-1, 1))
    denom[denom == 0] = -1
    factor = nom / denom
    factor[factor < 0] = 0
    if isinstance(Ka, np.poly1d):
        Ka = Ka(0)
    else:
        Ka = float(Ka)
    factor = np.array([factor.T[i * t.size : i * t.size + t.size] for i in range(j.size)])
    for i in range(t.size - eta.size + 1):
        beta[:, :, :, i + 1, 0] = Ka * beta[:, :, :, i + 1, 2] - np.sum(
            beta[:, :, :, i + 1, 1:] * factor[:, i, :][None, None, :, :] * eta[1:][None, None, None, :],
            axis=3,
        )
        beta_diag = beta[:, :, :, i + 1, 0, None] * np.multiply.accumulate(e[:, :, :, i + 1 : i + 1 + r.size], axis=3)
        beta[:, :, :, i + 1 + r, r] = beta_diag

    f = np.sum(beta[:, :, :, :, 0], axis=3)  # 0 index = x | 1 index = n | 2 Index = j
    j_reshape = j[None, :, :]
    n_reshape = n_array[None, :, None]
    bj = np.zeros((x.size, n_array.size, j.size + 2))
    bj[:, :, 2:] = f * x[:, None, None] ** ((j - n_array).T[None, :, :]) / factorial((j - n_array).T[None, :, :])

    aj = aj_function_x(x, n - 1)

    bj[:, 0, :2] = aj[:, :2] + Ka * bj[:, 0, 2:4]
    bj[:, 1, 1] = aj[:, 0] + Ka * bj[:, 1, 3]

    if return_aj:
        return aj, bj
    else:
        return bj


def bj(x: np.ndarray = np.array([]), n: int = 5, n_iterations=50, **s):
    l = s.get("l")
    if isinstance(x, (int, float)):
        x = np.array([x])
    elif x.size == 0:
        x = np.array([l])
    EI, GA = load_material_parameters(**s)

    if isinstance(EI, (int, float)):
        bj = bj_opt2_p89(x=x, n=n, n_iterations=n_iterations, **s)
    elif isinstance(EI, np.poly1d):
        Iy = s["cs"]["I_y"]
        eta = np.flip((Iy / Iy(0)).c)
        eta, _ = check_and_convert_eta_gamma(eta=eta, **s)
        _, K = gamma_K_function(**s)
        bj = bj_opt2_p119_forloop(Ka=K, x=x, eta=eta, n=n, n_iterations=n_iterations)
    return bj


def bj_opt2_p119_forloop(
    Ka: float,
    x: np.ndarray,
    eta: np.ndarray = np.array([]),
    gamma: np.ndarray = np.array([]),
    n: int = 6,
    n_iterations: int = 50,
    return_aj=False,
    **s,
):
    eta, _ = check_and_convert_eta_gamma(eta, gamma, **s)
    x = check_and_convert_input_array(x, **s)

    j = np.arange(2, n).reshape(-1, 1)
    t = np.arange(1, n_iterations + eta.size)
    s = t + j

    n_array = np.arange(2)
    r = np.arange(1, eta.size)
    beta = np.zeros((x.size, n_array.size, j.size, n_iterations + eta.size, eta.size))
    beta[:, :, :, 0, 0] = 1
    e = x[:, None, None, None] / (
        s - n_array[:, None, None]
    )  # 0 Index = x | 1 index = n | 2 index = j | 3 Index = n_iterations
    beta_diag = np.multiply.accumulate(e, axis=3)[:, :, :, : r.size]
    beta[:, :, :, r, r] = beta_diag
    nom = factorial(s.flatten() - 2)
    denom = factorial(s.flatten() - 2 - r.reshape(-1, 1))
    denom[denom == 0] = -1
    factor = nom / denom
    factor[factor < 0] = 0
    if isinstance(Ka, np.poly1d):
        Ka = Ka(0)
    factor = np.array([factor.T[i * t.size : i * t.size + t.size] for i in range(j.size)])
    eta_prod = np.empty((x.size, 2, j.size, eta.size - 1))
    eta_prod[:, :, :] = eta[1:]
    for i in range(t.size - eta.size + 1):
        prod = beta[:, :, :, i + 1, 1:] * factor[:, i, :] * eta_prod
        beta[:, :, :, i + 1, 0] = Ka * beta[:, :, :, i + 1, 2] - np.sum(prod, axis=3)
        beta_diag = beta[:, :, :, i + 1, 0, None] * np.multiply.accumulate(e[:, :, :, i + 1 : i + 1 + r.size], axis=3)
        beta[:, :, :, i + 1 + r, r] = beta_diag

    f = np.sum(beta[:, :, :, :, 0], axis=3)  # 0 index = x | 1 index = n | 2 Index = j
    bj = np.zeros((x.size, n_array.size, j.size + 2))
    bj[:, :, 2:] = f * x[:, None, None] ** ((j - n_array).T[None, :, :]) / factorial((j - n_array).T[None, :, :])

    aj = aj_function_x(x, n - 1)

    bj[:, 0, :2] = aj[:, :2] + Ka * bj[:, 0, 2:4]
    bj[:, 1, 1] = aj[:, 0] + Ka * bj[:, 1, 3]

    if return_aj:
        return aj, bj
    else:
        return bj


if __name__ == "__main__":

    import numpy as np
    import sympy as sym
    import matplotlib.pyplot as plt
    import stanpy as stp

    x_sym = sym.Symbol("x")
    E = 3e7  # kN/m2
    b = 0.2  # m
    ha = hb = 0.3  # m
    hc = 0.4  # m
    l1 = 4  # m
    l2 = 3  # m
    hx = hb + (hc - hb) / l2 * x_sym

    cs_props1 = stp.cs(b=b, h=ha)
    s1 = {"E": E, "cs": cs_props1, "q": 10, "l": l1, "bc_i": {"w": 0, "M": 0, "H": 0}}

    cs_props2 = stp.cs(b=b, h=hx)
    s2 = {"E": E, "cs": cs_props2, "q": 10, "l": l2, "bc_k": {"w": 0, "phi": 0}}

    s = [s1, s2]

    bj(**s1)
    bj(**s2)

    # fig, ax = plt.subplots(figsize=(12, 5))
    # stp.plot_system(ax, *s)
    # stp.plot_load(ax, *s)
    # ax.grid(linestyle=":")
    # ax.set_axisbelow(True)
    # ax.set_ylim(-0.75, 1.0)
    # plt.show()
