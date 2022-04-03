import stanpy as stp
import numpy as np

def test_multiple_w0():

    EI = 32000  # kN/m2
    l = 3  # m

    hinged_support = {"w": 0, "M": 0}
    roller_support = {"w": 0, "M": 0, "H": 0}
    fixed_support = {"w": 0, "phi": 0}

    s1 = {"EI": EI, "l": l, "bc_i": hinged_support, "bc_k": {"w": 0}, "q": 10}
    s2 = {"EI": EI, "l": l, "bc_k": {"w": 0}, "q": 10}
    # s2 = {"EI": EI, "l": l, "q": 10}
    s3 = {"EI": EI, "l": l, "bc_k": {"w": 0}, "q": 10}
    s4 = {"EI": EI, "l": l, "bc_k": roller_support}

    s = [s1, s2, s3, s4]

    x = np.linspace(0,4*l,5)

    Zi, Zk = stp.tr_solver(*s)
    Fx = stp.tr(*s, x=x)
    Zx = Fx.dot(Zi).round(10)

    path = os.path.join("reduction_method_npz", "test_multiple_w0.npz")
    # np.savez_compressed(path, Fx=Fx, Zx=Zx)
    npz = np.load(path)
    Zx_test = npz["Zx"]
    Fx_test = npz["Fx"]

    np.testing.assert_allclose(Fx, Fx_test,rtol=1e-5)
    np.testing.assert_allclose(Zx, Zx_test,rtol=1e-5)

def test_multiple_w0_M0():

    import numpy as np
    np.set_printoptions(precision=6, threshold=5000)
    import matplotlib.pyplot as plt
    import stanpy as stp

    EI = 32000  # kN/m2
    l = 3  # m

    hinged_support = {"w": 0, "M": 0}
    roller_support = {"w": 0, "M": 0, "H": 0}
    fixed_support = {"w": 0, "phi": 0}

    s1 = {"EI": EI, "l": l, "bc_i": hinged_support, "bc_k": {"w": 0}, "q": 10}
    s2 = {"EI": EI, "l": l, "bc_k": {"M": 0}, "q": 10}
    # s2 = {"EI": EI, "l": l, "q": 10}
    s3 = {"EI": EI, "l": l, "bc_k": {"w": 0}, "q": 10}
    s4 = {"EI": EI, "l": l, "bc_k": roller_support}

    s = [s1, s2, s3, s4]

    fig, ax = plt.subplots()
    stp.plot_system(ax, *s)
    plt.show()

    x = np.sort(np.append(np.linspace(0,4*l,4000), [l,2*l, 3*l, 4*l]))

    Zi, Zk = stp.tr_solver(*s)
    Fx = stp.tr(*s, x=x)
    Zx = Fx.dot(Zi).round(10)
  
    scale = 0.5
    fig, ax = plt.subplots()
    stp.plot_system(ax, *s, watermark=False)
    stp.plot_M(
        ax,
        x=x,
        Mx=Zx[:, 2],
        annotate_x=[0,l,2*l, 3*l, 4*l],
        fill_p="red",
        fill_n="blue",
        scale=scale,
        alpha=0.2,
    )

    ax.set_ylim(-1, 1)
    ax.axis('off')

    plt.show()

    # path = os.path.join("reduction_method_npz", "test_multiple_w0.npz")
    # np.savez_compressed(path, Fx=Fx, Zx=Zx)
    # npz = np.load(path)
    # Zx_test = npz["Zx"]
    # Fx_test = npz["Fx"]

    # np.testing.assert_allclose(Fx, Fx_test,rtol=1e-5)
    # np.testing.assert_allclose(Zx, Zx_test,rtol=1e-5)

# if __name__=="__main__":
#     test_multiple_w0_M0()