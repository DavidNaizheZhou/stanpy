
******************************************************************************
Der gerade Stab, konstanter Querschnitt, Stabtheorie I. Ordnung un II. Ordnung
******************************************************************************
.. Important:: 
    (todo: Referenzen!!!)

Grundlagen
==========
Ausgangspunkt der Herleitung der Übertragungsbeziehungen für Stäbe mit konstantem Querschnitt nach Theorie I. Ordnung und II. Ordnung sind 
die Gleichgewichtsbeindungunen sowie die konstitutiven und kinematischen Beziehungen. 
Als zusätzliche Gleichung ergibt sich die Umrechnungsformel von Querkraft auf Transversalkraft aus der statischen 
Äquivalenzbetrachtung von Spannungsresultanten und Schnittgrößen der Theorie II. Ordnung.

.. math::
    :label: differential_equations_II

    \frac{dR(x)}{dx} &= -q(x) \\[1em] 
    \frac{dM(x)}{dx} &= V(x) + m(x)\\[1em]            
    \frac{d\varphi(x)}{dx} &= -\left[\frac{M(x)}{EI(x)}+\kappa(x)\right]\\[1em] 
    \frac{dw(x)}{dx} &= \varphi (x) + \frac{V(x)}{G\tilde{A}(x)}\\[1em]
    V(x) &= R(x) - N^{II}(x)\left[\frac{dw_v(x)}{dx}+\frac{dw(x)}{dx}\right]

.. admonition:: (?) Frage

    Anschreiben der Linearisierung der statischen Äquivalenz Betrachgtung (wie Eva Binder 2011)?

Herleitungen
(todo)


Beispiele
=========
.. code-block:: python
   :linenos:
   :emphasize-lines: 9,14

    roller_support = {"w":0, "M":0} # Gleitlager
    hinge = {"M":0}
    fixed_support = {"w":0, "phi":0} # Einspannung

    s = {"EI": EI, "GA": GA, "l": l, "N": P, 
            "q_delta": (q,0,l), "w_0": w_0, "F": H,"bc_i":fixed_support, "bc_k":roller_support}
    s1 = {"k_i":node_1, "k_k": node_2, # (!) 
          "EI":EI, 
          "q":q, "F":(P,l/2), 
          "bc_i":roller_support, "bc_k":hinge} # (!)

    s2 = {"k_i":node_2, "k_k": node_3, # (!) 
          "EI":EI, 
          "q":q, "F":(P,l/2), 
          "bc_i":{}, "bc_k":fixed_support} # (!)

    s_list = [s1,s2] # oder s_list = [s2,s1] 

    s_list = stp.tr_solve(s_list)
