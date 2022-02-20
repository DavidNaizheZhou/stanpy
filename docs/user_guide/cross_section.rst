
*****************
Querschnittswerte
*****************
.. Note:: 
    todos: Abbildungen, Definition der Vektoren h und b, Funktion für I, H, HEB, HEA, ...

In diesem Abschnitt werden die Querschnittswerte für verschieden zusammengesetzte Rechtecksquerschnitte berechnet.

EX01 Rechtecksquerschnitt
=========================

.. jupyter-execute::

    import numpy as np
    import stanpy as stp

    b = 0.2 
    h = 0.4
    
    cs_props = stp.cs(b=b,h=h)

    print(cs_props)

    

EX02 Zusammengesetzter Rechtecksquerschnitt
===========================================

.. jupyter-execute::

    import numpy as np
    import stanpy as stp

    s, t = 0.012, 0.02  # m
    b, h = 0.5, 39.4
    b_v = np.array([b, s, b])
    h_v = np.array([t, h - 2 * t, t])
    zsi_v = np.array([t / 2, t + (h - 2 * t) / 2, t + (h - 2 * t) + t / 2])  # von OK

    cs_props = stp.cs(b=b_v, h=h_v, zsi=zsi_v)

    print(cs_props)

EX03 I-Querschnitt
==================

.. jupyter-execute::

    import numpy as np
    import stanpy as stp

    s, t = 0.012, 0.02  # m
    b, h = 0.5, 0.39
    b_v = np.array([b, s, b])
    h_v = np.array([t, h - 2 * t, t])

    cs_props = stp.cs(b=b_v, h=h_v)

    print(cs_props)

EX04 H-Querschnitt
==================

.. jupyter-execute::

    import numpy as np
    import stanpy as stp
    import matplotlib.pyplot as plt

    s, t = 0.012, 0.02  # m
    b, h = 0.5, 0.39

    b_v = np.array([b, s, b])
    h_v = np.array([t, h - 2 * t, t])

    Ay = np.array(
        [
            [1 / 2, 0, 0],
            [1, 1 / 2, 0],
            [1, 1, 1 / 2],
        ]
    )

    y_si = Ay.dot(b_v)

    cs_props = stp.cs(b=b_v, h=h_v, y_si=y_si)

    print(cs_props)

EX05 Kasten-Querschnitt
=======================

.. jupyter-execute::

    import numpy as np
    import stanpy as stp

    s, t = 0.012, 0.02  # m
    b, h = 0.5, 39.4

    b_v = np.array([b, s, s, b])
    h_v = np.array([t, h - 2 * t, h - 2 * t, t])

    Az = np.array(
        [
            [1 / 2, 0, 0, 0],
            [1, 1 / 2, 0, 0],
            [1, 0, 1 / 2, 0],
            [1, 0, 1, 1 / 2],
        ]
    )

    z_si = Az.dot(h_v)

    Ay = np.array(
        [
            [1 / 2, 0, 0, 0],
            [0, 1 / 2, 0, 0],
            [1, 0, -1 / 2, 0],
            [0, 0, 0, 1 / 2],
        ]
    )

    y_si = Ay.dot(b_v)

    cs_props = stp.cs(b=b_v, h=h_v, z_si=z_si, y_si=y_si)

    print(cs_props)

EX06 - Verstärkter I Querschnitt
================================

.. jupyter-execute::

    import numpy as np
    import stanpy as stp

    s, t = 0.012, 0.02  # m
    b, h = 0.5, 0.39
    h_i = 0.05

    b_v = np.array([b, s, b, s, s, s, s])
    h_v = np.array([t, h - 2 * t, t, h_i, h_i, h_i, h_i])

    Az = np.array(
        [
            [1 / 2, 0, 0, 0, 0, 0, 0],
            [1, 1 / 2, 0, 0, 0, 0, 0],
            [1, 1, 1 / 2, 0, 0, 0, 0],
            [1, 0, 0, 0, 1 / 2, 0, 0],
            [1, 0, 0, 0, 0, 1 / 2, 0],
            [1, 1, 0, 0, 0, -1 / 2, 0],
            [1, 1, 0, 0, 0, 0, -1 / 2],
        ]
    )

    z_si = Az.dot(h_v)

    cs_props = stp.cs(b=b_v, h=h_v, z_si=z_si)
    
    print(cs_props)


.. meta::
    :description lang=de:
        Examples of document structure features in pydata-sphinx-theme.