
***************************************************
Übertragungsbeziehungen der Stabtheorie II. Ordnung
***************************************************
.. Important:: 
    (todo: Referenzen!!!)

Grundlagen
==========
Ausgangspunkt der Herleitung der Übertragungsbeziehungen für Stäbe nach Theorie II. Ordnung sind die die Differentialgleichungen :eq:`differential_equations` des Biegeproblems 
der Stabtheorie II. Ordnung sowie die statische Äquivalenzbetrachtung von Spannungsresultanten und Schnittgrößen nach Theorie II. Ordnung. 
Aufgrund der Kopplungen von Normalkraft- und Biegeproblem können die Differentialgleichungen nicht Schritt für Schritt aufintegriert werden. 
Daher werden die 

.. math::
    :label: differential_equations_II

    \frac{dR(x)}{dx} &= -q(x) \\[1em] 
    \frac{dM(x)}{dx} &= V(x) + m(x)\\[1em]            
    \frac{d\varphi(x)}{dx} &= -\left[\frac{M(x)}{EI(x)}+\kappa(x)\right]\\[1em] 
    \frac{dw(x)}{dx} &= \varphi (x) + \frac{V(x)}{G\tilde{A}(x)}\\[1em]
    V(x) &= R(x) - N^{II}(x)\left[\frac{dw_v(x)}{dx}+\frac{dw(x)}{dx}\right]

.. admonition:: (?) Question

    Anschreiben der Linearisierung der statischen Äquivalenz Betrachgtung (wie Eva Binder 2011)?

Eliminationsverfahren
---------------------

Lösen der Differentialgleichung
-------------------------------


.. meta::
    :description lang=de:
        Examples of document structure features in pydata-sphinx-theme.
