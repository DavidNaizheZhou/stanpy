"""Main module."""
import sympy as sp
import numpy as np
from scipy.integrate import quad

# sp.init_printing(use_unicode=False, wrap_line=False)
###### Functions ############
x = sp.Symbol('x')
r = sp.Symbol('r')
f = sp.Function('f')(x)


def ulg(fx, q, ei, pos):
    return fx, q, ei, pos


def fx_rect(x, hi, hk, l):
    r = -hi * l / (hk-hi)
    return ((r-x)/r)**3


def fx_I(x, b, hi, hk, s, t, l):
    Ag = b * t - t*s/2
    r1 = -(hi*l)/(hk-hi)
    r2 = -l/s * (s*hi+6*Ag)/(hk-hi)
    return (r1-x)**2 * (r2-x) / (r1**2 * r2)


def I_vec(vb, vh):
    vzsi = np.matmul(np.array([[1/2, 0, 0], [1, 1/2, 0], [1, 1, 1/2]]), vh)

    zs = sp.factor(sp.cancel(np.dot(vb*vh, vzsi)/np.dot(vb, vh)))
    I = np.matmul(vb, vh**3)/12 + np.matmul(vb*vh, vzsi**2) - \
        zs*np.matmul(vb * vh, vzsi)
    return I


def I_vec_beliebig(vb, vh, vzsi):
    zs = sp.factor(sp.cancel(np.dot(vb*vh, vzsi)/np.dot(vb, vh)))
    I = np.matmul(vb, vh**3)/12 + np.matmul(vb*vh, vzsi**2) - \
        zs*np.matmul(vb * vh, vzsi)
    return I


def fx_I_exact(x, vb, vh):
    # zs = np.dot(vb*vh, vzsi)/np.dot(vb, vh)
    # I = np.matmul(vb, vh**3)/12 + np.matmul(vb*vh, vzsi**2) - \
    #     zs*np.matmul(vb * vh, vzsi)
    I = I_vec(vb, vh)
    Ix = sp.lambdify(x, I)
    # print(I.cancel().simplify())
    fx = sp.simplify(Ix(x)/Ix(0))
    return fx


def fx_I_beliebig(x, vb, vh, vzsi):
    I = I_vec_beliebig(vb, vh, vzsi)
    Ix = sp.lambdify(x, I)
    fx = sp.simplify(Ix(x)/Ix(0))
    return fx


def r_I(b, hi, hk, s, t, l):
    Ag = b * t - t*s/2
    r1 = -(hi*l)/(hk-hi)
    r2 = -l/s * (s*hi+6*Ag)/(hk-hi)
    return r1, r2


def fx_I_sym(x, r1, r2):
    return (r1-x)**2 * (r2-x) / (r1**2 * r2)


def fx_rect_sym(x, r):
    # r = -hi * l / (hk-hi)
    return ((r-x)/r)**3


def u_base_I(r1, r2, Ei, pos):
    b2 = b2_I(pos, r1, r2)
    b2s = b2s_I(pos, r1, r2)
    b3 = b3_I(pos, r1, r2)
    b3s = b3s_I(pos, r1, r2)
    return np.matrix([[1, pos, -b2/Ei, -b3/Ei, 0],
                      [0, 1, -b2s/Ei, -b3s/Ei, 0],
                      [0, 0, 1, pos, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1]
                      ])


def one_by_fx(fx):
    def onefx(x): return 1/fx
    return onefx

# def one_by_fx(x, r1, r2):
#     return 1/((r1-x)**2 * (r2-x) / (r1**2 * r2))


def u_base_num(Ei, pos, geom="R", **kwargs):
    if geom == "R":
        def one_by_fx(x):
            return 1/((kwargs["r"]-x)/kwargs["r"])**3

        def x_by_fx(x):
            return x/((kwargs["r"]-x)/kwargs["r"])**3
    elif geom == "I":
        def one_by_fx(x):
            return 1/((kwargs["r1"]-x)**2 * (kwargs["r2"]-x) / (kwargs["r1"]**2 * kwargs["r2"]))

        def x_by_fx(x):
            return x/((kwargs["r1"]-x)**2 * (kwargs["r2"]-x) / (kwargs["r1"]**2 * kwargs["r2"]))

    bb2s = quad(one_by_fx, 0, pos)[0]
    bb2 = quad(lambda x: quad(one_by_fx, 0, x)[0], 0, pos)[0]
    bb3s = quad(x_by_fx, 0, pos)[0]
    bb3 = quad(lambda x: quad(x_by_fx, 0, x)[0], 0, pos)[0]

    U = np.array([[1, x.subs(x, pos), -bb2/(Ei), -bb3/(Ei), 0],
                  [0, 1, -bb2s/(Ei), -bb3s/(Ei), 0],
                  [0, 0, 1, x.subs(x, pos), 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]
                  ]).astype(float)
    return U


def solver(fi, fk, Fki, index, folder):
    rhs = np.dot(Fki, fi)

    mask = fk == 0
    # print(mask)
    # if np.sum(mask) != 0:
    fi_solve = sp.solve(rhs[mask])

    try:
        coef_11 = float(str(rhs[mask][0]).split("*phi_a")[0].replace(" ", ""))
        coef_12 = float(str(rhs[mask][0]).split("*phi_a")
                        [1].split("*v_a")[0].replace(" ", ""))
        coef_13 = float(str(rhs[mask][0]).split("*phi_a")
                        [1].split("*v_a")[1].replace(" ", ""))
        coef_21 = float(str(rhs[mask][1]).split("*phi_a")[0].replace(" ", ""))
        coef_22 = float(str(rhs[mask][1]).split("*phi_a")
                        [1].split("*v_a")[0].replace(" ", ""))
        coef_23 = float(str(rhs[mask][1]).split("*phi_a")
                        [1].split("*v_a")[1].replace(" ", ""))

        np.savez("variables/{}/{}_coeff.npz".format(folder, index),
                 coef_11=convert_value(coef_11),
                 coef_12=convert_value(coef_12),
                 coef_13=convert_value(coef_13),
                 coef_21=convert_value(coef_21),
                 coef_22=convert_value(coef_22),
                 coef_23=convert_value(coef_23)
                 )
    except IndexError:
        pass

    for key in fi_solve:
        fi[fi == key] = fi_solve[key]

    fk_solve = sp.solve(np.dot(Fki, fi)[~mask]-fk[~mask])
    for key in fk_solve:
        fk[fk == key] = fk_solve[key]

    return fi.astype(float), fk.astype(float)


def convert_value(value):
    return np.format_float_scientific(value, unique=False, precision=4)


def convert_matrix(matrix):
    copy = np.empty_like(matrix)
    for i in np.arange(matrix[0, :].size):
        for j in np.arange(matrix[:, 0].size):
            copy[j, i] = np.format_float_scientific(
                matrix[j, i], unique=False, precision=4)
    return copy


def u_base(fx, Ei, pos, index=None, print_b=False, folder="ver"):
    func_b2 = sp.lambdify(x, sp.cancel(1/fx))
    bb2s = quad(func_b2, 0, pos)[0]
    bb2 = quad(lambda x: quad(func_b2, 0, x)[0], 0, pos)[0]

    func_b3 = sp.lambdify(x, sp.cancel(x/fx))
    bb3s = quad(func_b3, 0, pos)[0]
    bb3 = quad(lambda x: quad(func_b3, 0, x)[0], 0, pos)[0]

    func_b4 = sp.lambdify(x, sp.cancel(x**2/2/fx))
    bb4s = quad(func_b4, 0, pos)[0]
    bb4 = quad(lambda x: quad(func_b4, 0, x)[0], 0, pos)[0]

    func_b5 = sp.lambdify(x, sp.cancel(x**3/6/fx))
    bb5s = quad(func_b5, 0, pos)[0]
    bb5 = quad(lambda x: quad(func_b5, 0, x)[0], 0, pos)[0]

    if print_b:
        print("bb2s", np.format_float_scientific(
            bb2s, unique=False, precision=4))
        print("bb2", np.format_float_scientific(
            bb2, unique=False, precision=4))
        print("bb3s", np.format_float_scientific(
            bb3s, unique=False, precision=4))
        print("bb3", np.format_float_scientific(
            bb3, unique=False, precision=4))
        print("bb4s", np.format_float_scientific(
            bb4s, unique=False, precision=4))
        print("bb4", np.format_float_scientific(
            bb4, unique=False, precision=4))
        print("bb5s", np.format_float_scientific(
            bb5s, unique=False, precision=4))
        print("bb5", np.format_float_scientific(
            bb5, unique=False, precision=4))

    if index != None:
        np.savez("variables/{}/{}_b.npz".format(folder, index),
                 b2=convert_value(bb2),
                 b2s=convert_value(bb2s),
                 b3=convert_value(bb3),
                 b3s=convert_value(bb3s),
                 b4=convert_value(bb4),
                 b4s=convert_value(bb4s),
                 b5=convert_value(bb5),
                 b5s=convert_value(bb5s)
                 )

    U = np.array([[1, x.subs(x, pos), -bb2/(Ei), -bb3/(Ei), 0],
                  [0, 1, -bb2s/(Ei), -bb3s/(Ei), 0],
                  [0, 0, 1, x.subs(x, pos), 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]
                  ]).astype(float)
    return U


def r_rect(hi, hk, l):
    return -hi * l / (hk-hi)


def u_lastglied_gleichlast(fx, q, Ei, pos, index=None, print_l=False, folder="ver"):
    vl = -sp.integrate(q, (x, 0, x))
    ml = sp.integrate(sp.nsimplify(vl), (x, 0, x))

    func_phil = sp.lambdify(x, sp.cancel(ml/Ei/fx))
    phil = -quad(func_phil, 0, pos)[0]
    wl = -quad(lambda x: quad(func_phil, 0, x)[0], 0, pos)[0]

    vl = sp.re(vl.subs(x, pos).evalf())
    ml = sp.re(ml.subs(x, pos).evalf())
    if print_l:
        print("VL", np.format_float_scientific(vl, unique=False, precision=4))
        print("ML", np.format_float_scientific(ml, unique=False, precision=4))
        print("phiL", np.format_float_scientific(
            phil, unique=False, precision=4))
        print("wL", np.format_float_scientific(wl, unique=False, precision=4))

    if index != None:
        np.savez("variables/{}/{}_L.npz".format(folder, index),
                 vl=convert_value(vl), ml=convert_value(ml),
                 phil=convert_value(phil), wl=convert_value(wl))

    U = np.array([[0, 0, 0, 0, wl],
                  [0, 0, 0, 0, phil],
                  [0, 0, 0, 0, ml],
                  [0, 0, 0, 0, vl],
                  [0, 0, 0, 0, 0]]).astype(float)
    return U


def b2_I(x, r1, r2):
    return r1*r2*(r1*r2*np.log(-r1) - r1*r2*np.log(-r2) - r1*r2*np.log(-r1 + x) + r1*r2*np.log(-r2 + x) - r1*x*np.log(-r1) + r1*x*np.log(-r2) + r1*x*np.log(-r1 + x) - r1*x*np.log(-r2 + x) + r1*x - r2*x)/(r1**2 - 2*r1*r2 + r2**2)


def b2_I_skript(x, r1, r2):
    return -r1**2*r2*(r2-x)/(r1-r2)**2*np.log(r2*(r1-x)/(r1*(r2-x)))+x*r1*r2/(r1-r2)


def b2s_I_skript(x, r1, r2):
    return r1**2*r2/(r1-r2)**2*np.log(r2*(r1-x)/(r1*(r2-x))) - x*r1*r2/((r1-r2)*(r1-x))


def b2s_I(x, r1, r2):
    return r1*r2*(r1*((r1 - r2)**2 + (np.log(-r1 + x) - np.log(-r2 + x))*(-r1**2 + r1*r2 + x*(r1 - r2))) + (-r1**2 + r1*r2 + x*(r1 - r2))*(r1*(-np.log(-r1) + np.log(-r2)) + r1 - r2))/((r1 - r2)**2*(-r1**2 + r1*r2 + x*(r1 - r2)))


def b3_I(x, r1, r2):
    return r1**2*r2*(x*(r1 - r2)**2*(r1 - r2*np.log(-r1) + r2*np.log(-r2) + r2*np.log(-r1 + x) - r2*np.log(-r2 + x) - r2) + (r1**2 - 2*r1*r2 + r2**2)*(-r1*(r1 - 2*r2)*np.log(-r1) + r1*(r1 - 2*r2)*np.log(-r1 + x) - r2**2*np.log(-r2) + r2**2*np.log(-r2 + x)))/((r1 - r2)**2*(r1**2 - 2*r1*r2 + r2**2))


def b3s_I(x, r1, r2):
    return r1**2*r2*(r1*(r1 - r2)**2 + r2*(np.log(-r1 + x) - np.log(-r2 + x))*(-r1**2 + r1*r2 + x*(r1 - r2)) + (r1 + r2*(-np.log(-r1) + np.log(-r2)) - r2)*(-r1**2 + r1*r2 + x*(r1 - r2)))/((r1 - r2)**2*(-r1**2 + r1*r2 + x*(r1 - r2)))


def u_lastglied_einzellast_rect(p, Ej, xj, pos, r_val):
    def f(x, r): return ((r-x)/r)**3
    mo = 1/2*(x-xj+sp.Abs(x-xj))  # maculay - operator

    bb3s = sp.integrate(x/f(x, r), (x, 0, x))
    bb3 = sp.integrate(bb3s, (x, 0, x))

    bb3s = bb3s.subs(x, mo).subs(r, r-xj)
    bb3 = bb3.subs(x, mo).subs(r, r-xj)

    vl = -p*mo/sp.Abs(x-xj)
    ml = -p*mo
    phil = p/Ej*bb3s
    wl = p/Ej*bb3

    vl = vl.subs(x, pos)
    ml = ml.subs(x, pos)
    phil = phil.subs(x, pos)
    wl = wl.subs(x, pos)

    U = sp.Matrix([[0, 0, 0, 0, wl],
                   [0, 0, 0, 0, phil],
                   [0, 0, 0, 0, ml],
                   [0, 0, 0, 0, vl],
                   [0, 0, 0, 0, 0]])
    return U
