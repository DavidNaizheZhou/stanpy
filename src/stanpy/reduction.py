import stanpy as stp
import numpy as np


def F_roller_support_reduced(Fxi_minus, bc_i, w_e=0, theta=0):
    """ """
    if bc_i == {"w": 0, "M": 0}:
        alpha_index = 1
        beta_index = 3
        delta_index = -1

    alpha = Fxi_minus[:, alpha_index]
    beta = Fxi_minus[:, beta_index]
    delta = Fxi_minus[:, -1]

    Fxi_plus = np.zeros((5, 3))
    # Fxi_plus[:,0] = alpha-alpha_hat/gamma_hat*gamma
    Fxi_plus[:, 0] = alpha - alpha[0] * beta / beta[0]
    Fxi_plus[:, -1] = delta - delta[0] * beta / beta[0]
    Fxi_plus[-2, -2] = 1
    Fxi_plus[-1, -1] = 1
    return Fxi_plus


def F_roller_support(Fxi_minus, bc_i):
    delta_index = -1

    if bc_i == {"w": 0, "M": 0}:
        alpha_index = 1
        beta_index = 3
    elif bc_i == {"w": 0, "phi": 0}:
        alpha_index = 2
        beta_index = 3

    alpha = Fxi_minus[:, alpha_index]
    beta = Fxi_minus[:, beta_index]
    delta = Fxi_minus[:, -1]

    Fxi_delta = np.zeros((5, 5))
    Fxi_delta[:, alpha_index] = -alpha[0] * beta / beta[0]
    Fxi_delta[:, delta_index] = -delta[0] * beta / beta[0]

    Fxi_plus = np.zeros((5, 5))
    Fxi_plus[:, [alpha_index, delta_index]] = (
        Fxi_minus[:, [alpha_index, delta_index]] + Fxi_delta[:, [alpha_index, delta_index]]
    )

    Ax = np.array([0, 0, 0, 1, 0])
    Fxi_plus[:, beta_index] = Ax
    # np.set_printoptions(precision=5)
    return Ax, Fxi_plus


def F_hinge(Fxi_minus, bc_i):
    delta_index = -1

    if bc_i == {"w": 0, "M": 0}:
        alpha_index = 1
        beta_index = 3
    elif bc_i == {"w": 0, "phi": 0}:
        alpha_index = 2
        beta_index = 3

    alpha = Fxi_minus[:, alpha_index]
    beta = Fxi_minus[:, beta_index]
    delta = Fxi_minus[:, -1]

    Fxi_delta = np.zeros((5, 5))
    Fxi_delta[:, alpha_index] = -alpha[2] * beta / beta[2]
    Fxi_delta[:, delta_index] = -delta[2] * beta / beta[2]

    Fxi_plus = np.zeros((5, 5))
    Fxi_plus[:, [alpha_index, delta_index]] = (
        Fxi_minus[:, [alpha_index, delta_index]] + Fxi_delta[:, [alpha_index, delta_index]]
    )

    Ax = np.array([0, 1, 0, 0, 0])
    Fxi_plus[:, 3] = Ax

    return Ax, Fxi_plus


def get_local_coordinates(*slabs, x: np.ndarray):
    l = np.array([s.get("l") for s in slabs])
    l_global = np.cumsum(l) - l
    x_local = np.zeros(x.shape)
    mask_list = []
    for lengths in zip(l_global[:-1], l_global[1:]):
        mask = (x > lengths[0]) & (x <= lengths[1])
        mask_list.append(mask)
        x_local[mask] = x[mask] - lengths[0]
    mask = x > lengths[1]
    x_local[mask] = x[mask] - lengths[1]
    mask_list.append(mask)
    return x_local, mask_list, l_global


def fill_bc_dictionary_slab(*slabs):
    bc_i = [s.get("bc_i") for s in slabs]
    bc_k = [s.get("bc_k") for s in slabs]
    bc = np.array(list(zip(bc_i, bc_k))).flatten()

    if (bc[1:-1:2] != None).any():
        bc[2:-1:2] = bc[1:-1:2]
    elif (bc[2:-1:2] != None).any():
        bc[1:-1:2] = bc[2:-1:2]


def get_bc_interfaces(*slabs):
    bc_i = [s.get("bc_i") for s in slabs]
    bc_k = [s.get("bc_k") for s in slabs]
    bc = np.array(list(zip(bc_i, bc_k))).flatten()

    return bc[1:-1:2]


def solve_system(*slabs, x: np.ndarray = np.array([])):

    fill_bc_dictionary_slab(*slabs)

    bc_interace = get_bc_interfaces(*slabs)

    number_slabs = len(slabs)
    l = np.array([s.get("l") for s in slabs])
    bc_i = [s.get("bc_i") for s in slabs]
    bc_k = [s.get("bc_k") for s in slabs]

    x_local, mask, l_global = get_local_coordinates(*slabs, x=x)

    Fxa_plus = np.zeros((number_slabs + 1, 5, 5))
    Fxa_plus[0] = np.eye(5, 5)
    Fxx = np.zeros((x.size, 5, 5))
    for i, slab in enumerate(slabs):
        Fxx[mask[i]] = stp.tr(slab, x=x_local[mask[i]])

    Axk = np.zeros((len(bc_interace), 5))

    for i, bci_interface in enumerate(bc_interace):
        if bci_interface == {"w": 0}:
            Axk[i], Fxa_plus[i + 1] = F_roller_support(Fxx[mask[i]][-1].dot(Fxa_plus[i]), bc_i=bc_i[0])
        elif bci_interface == {"M": 0}:
            Axk[i], Fxa_plus[i + 1] = F_hinge(Fxx[mask[i]][-1].dot(Fxa_plus[i]), bc_i=bc_i[0])

    Fxx[0] = np.eye(5, 5)

    Fxa_plus[0] = np.zeros((5, 5))

    if ~mask[-1].any():
        raise IndexError("length of system = {} < sum beam list {}".format(l_global[-1], l))

    Fxa_plus[-1] = Fxx[mask[-1]][-1].dot(Fxa_plus[-2])

    za_fiktiv, z_end = stp.solve_tr(Fxa_plus[-1], bc_i=bc_i[0], bc_k=bc_k[-1])

    dza_fiktiv = np.zeros((Fxa_plus.shape[0], 5))
    dza_fiktiv[1:-1, :] = -Axk * za_fiktiv
    zx_minus = Fxa_plus.dot(za_fiktiv) + dza_fiktiv

    zx_minus[-1, :] = z_end

    zx_plus = np.zeros(zx_minus.shape)
    zx_plus[0, :] = za_fiktiv - Axk[0] * (Fxx[mask[0]][-1].dot(za_fiktiv) - zx_minus[1])
    for i in range(1, zx_minus.shape[0] - 1):
        zx_plus[i, :] = zx_minus[i] - Axk[i - 1] * (np.dot(Fxx[mask[i]][-1], zx_minus[i]) - zx_minus[i + 1])

    zx = np.zeros((x.size, 5))
    for i in range(number_slabs):
        zx[mask[i], :] = Fxx[mask[i]].dot(zx_plus[i, :])

    zx[x == 0] = zx_plus[0, :]
    x = np.append(x, l_global[1:])
    zx = np.append(zx, zx_plus[1:-1, :], axis=0)

    arr1inds = x.argsort()
    x = x[arr1inds]
    zx = zx[arr1inds]

    return x, zx.round(9)


if __name__ == "__main__":
    EI = 32000  # kN/m2
    l = 6  # m

    festes_auflager = {"w": 0, "M": 0}
    einspannung = {"w": 0, "phi": 0}

    s1 = {"EI": EI, "l": l, "bc_i": festes_auflager, "bc_k": {"w": 0}, "q": 10}
    s2 = {"EI": EI, "l": l, "bc_k": festes_auflager}

    x_injection = np.array([l, 2 * l])
    x = np.sort(np.append(np.linspace(0, 2 * l, 100), x_injection))
    x, Z_x = solve_system(s1, s2, x=x)

    # np.set_printoptions(precision=5)
    # print(zx[x == 2 * l])
    print(zx[0])
    print(zx[-1])
    print(zx[x == l])
    quit()

    x = np.linspace(0, l, 100)

    Fba = tr(s1, x=l)
    Fcb = tr(s2, x=l)
    Fdc = tr(s3, x=l)
    Fed = tr(s4, x=l)

    Fba_plus = F_roller_support(Fba, bc_i=festes_auflager)
    Fca_minus = Fcb.dot(Fba_plus)
    Fca_plus = F_roller_support(Fca_minus, bc_i=festes_auflager)
    Fda_minus = Fdc.dot(Fca_plus)
    Fda_plus = F_roller_support(Fda_minus, bc_i=festes_auflager)

    Fea = Fed.dot(Fda_plus)
    print(Fea)

    quit()

    za_fiktiv, ze = solve_tr(Fea, bc_i=festes_auflager, bc_k=einspannung)
    zd_plus = Fda_plus.dot(za_fiktiv)
    zd_minus = Fda_plus.dot(za_fiktiv) - np.array([0, 0, 0, 1, 0]) * za_fiktiv
    zc_minus = Fca_plus.dot(za_fiktiv) - np.array([0, 0, 0, 1, 0]) * za_fiktiv
    zb_minus = Fba_plus.dot(za_fiktiv) - np.array([0, 0, 0, 1, 0]) * za_fiktiv

    Fxx = np.zeros((4, 5, 5))
    Fxx[0, :, :] = Fba
    Fxx[1, :, :] = Fcb
    Fxx[2, :, :] = Fdc
    Fxx[3, :, :] = Fed

    nx = 3 + 2
    Fxa_plus = np.zeros((nx, 5, 5))
    Fxa_plus[1, :, :] = Fba_plus
    Fxa_plus[2, :, :] = Fca_plus
    Fxa_plus[3, :, :] = Fda_plus

    dza_fiktiv = np.zeros((Fxa_plus.shape[0], 5))
    dza_fiktiv[1:-1, :] = -np.array([0, 0, 0, 1, 0]) * za_fiktiv
    zx_minus = Fxa_plus.dot(za_fiktiv) + dza_fiktiv
    zx_minus[-1, :] = ze

    zx_plus = np.empty_like(zx_minus)
    zx_plus[0, :] = za_fiktiv - np.array([0, 0, 0, 1, 0]) * (Fxx[0].dot(za_fiktiv) - zx_minus[1])
    for i in range(1, zx_minus.shape[0] - 1):
        zx_plus[i, :] = zx_minus[i] - np.array([0, 0, 0, 1, 0]) * (np.dot(Fxx[i], zx_minus[i]) - zx_minus[i + 1])

    print(zx_minus)
    print(zx_plus)
    quit()

    zc_plus = zc_minus - np.array([0, 0, 0, 1, 0]) * (Fdc.dot(zc_minus) - zd_minus)
    zb_plus = zb_minus - np.array([0, 0, 0, 1, 0]) * (Fcb.dot(zb_minus) - zc_minus)
    za_plus = za_fiktiv - np.array([0, 0, 0, 1, 0]) * (Fba.dot(za_fiktiv) - zb_minus)

    print(zd_plus)
    print(zc_plus)
    print(zb_plus)
    print(za_plus)
    quit()
    # print(ze)
    zd_minus = zd_plus - Fda_minus.dot(za_fiktiv)
    # print(zd_plus)
    # print(zd_minus)
    quit()

    Fca = Fcb.dot(Fba)
    A = np.zeros((2, 2))
    A = Fca_hinge[[0, 1], :-1]
    b = -Fca_hinge[[0, 1], -1].T

    fa = np.zeros(3)
    fa[:-1] = np.linalg.solve(A, b).round(15)
    fa[-1] = 1

    fc = Fca_hinge.dot(fa).round(10)
    fb_r = Fba_plus.dot(fa).round(10)
    fb_l = fb_r - Fba_minus.dot(fa).round(10)
    fa_r = fa + Fba_minus.dot(fb_r[[1, 3, -1]] - Fba[:, [1, 3, -1]].dot(fa)[[1, 3, -1]] - fa)[[1, 3, -1]]

    print(fa_r)
    print(fb_l)
    print(fb_r)
    print(fc)

    quit()