****************************************************
Reduktionsverfahren (Übertragungsmatrizenverfahren)
****************************************************

.. jupyter-execute::
    :hide-code:

    import numpy as np
    np.set_printoptions(precision=5)


EX01 Stabkonstruktion - R-QS
============================
EX02-01 (problem) 
-----------------

.. jupyter-execute::

    import numpy as np
    import matplotlib.pyplot as plt
    import stanpy as stp

    EI = 32000  # kN/m2
    l = 6  # m

    hinged_support = {"w": 0, "M": 0}
    roller_support = {"w": 0, "M": 0, "H": 0}
    fixed_support = {"w": 0, "phi": 0}

    s1 = {"EI": EI, "l": l, "bc_i": hinged_support, "bc_k": {"w": 0}, "q": 10}
    s2 = {"EI": EI, "l": l, "bc_k": roller_support}

    s = [s1, s2]

    fig, ax = plt.subplots(figsize=(12, 5))
    stp.plot_system(ax, *s)
    plt.show()


EX02-02 (solution) 
------------------
.. jupyter-execute::

    x_annotate = np.array([l, 2 * l])
    x = np.sort(np.append(np.linspace(0, 2 * l, 1000), x_annotate))
    x, Z_x = stp.solve_system(s1, s2, x=x)

EX02-03 (plotting) 
------------------

.. jupyter-execute::

    w_x = Z_x[:, 0]
    phi_x = Z_x[:, 1]
    M_x = Z_x[:, 2]
    V_x = Z_x[:, 3]

    scale = 0.5

    stp.plot_M(
        ax,
        x=x,
        Mx=M_x,
        annotate_x=[0, x[M_x == np.max(M_x)], l, 2 * l],
        fill_p="red",
        fill_n="blue",
        scale=scale,
        alpha=0.2,
    )
    
    ax.grid(linestyle=":")
    ax.set_axisbelow(True)
    ax.set_ylim(-1.0, 0.8)
    ax.set_ylabel("M/Mmax*{}".format(scale))
    ax.set_title("[M] = kNm")
    plt.show()

.. jupyter-execute::

    w_x = Z_x[:, 0]
    phi_x = Z_x[:, 1]
    M_x = Z_x[:, 2]
    V_x = Z_x[:, 3]

    scale = 0.5

    stp.plot_V(
        ax,
        x=x,
        Vx=V_x,
        annotate_x=[0, l, 2 * l],
        fill_p="red",
        fill_n="blue",
        scale=scale,
        alpha=0.2,
    )
    
    ax.grid(linestyle=":")
    ax.set_axisbelow(True)
    ax.set_ylim(-1.0, 0.8)
    ax.set_ylabel("V/Vmax*{}".format(scale))
    ax.set_title("[V] = kN")
    plt.show()


















