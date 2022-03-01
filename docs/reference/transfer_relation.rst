========================
stanpy.transfer_relation
========================

utils
=====
gamma_K
-------
.. automodule:: stanpy
    :members: gamma_K

.. math::
    :label: gamma_and_K

    \gamma = \dfrac{1}{1 - N / G\tilde{A}}\qquad
    K = -\gamma\dfrac{N}{EI}

.. jupyter-execute::

    import stanpy as stp

    s = {"EI":32000, "GA":20000, "N":-1000}
    gamma, K = stp.gamma_K(**s)
    print(gamma, K)

aj(x)
-----
.. automodule:: stanpy
    :members: aj

.. math::
    :label: aj_coeffs

    x \geq 0&:\quad a_0 = 1,\quad a_j = \dfrac{x^j}{j!} \qquad\text{for}\qquad j=1,~2,~3,~...\\
    x < 0&:\quad \text{all}\quad a_j=0

.. jupyter-execute::

    import stanpy as stp

    aj = stp.aj(x=[-1,0,2,3],n=5)
    print(aj)

bj(x)
-----
.. automodule:: stanpy
    :members: bj

for beams with constant crossections

.. math::
    :label: bj_recursion_constant

    b_j = \sum_{i=0}^{\infty}\beta_i \qquad \beta_i = \dfrac{K x^2}{(j+2i)(j+2i-1)}\beta_{i-1} \qquad \text{with}\qquad i=1,~2,~3,~...,~t

.. jupyter-execute::

    import stanpy as stp

    s = {"EI":32000, "GA":20000, "N":-1000}
    bj = stp.bj(x=[-1,0,2,3],**s)
    print(bj)

for beams with non-constant crossections

.. math::
    :label: bj_recursion_non_constant

    b_j^{(n)} = a_{j-n} \sum_{i=0}^{\infty}\beta_{t,0} \text{todo P118 (rubin)}



Citations
=========
.. bibliography::