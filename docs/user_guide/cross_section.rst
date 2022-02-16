
*****************
Querschnittswerte
*****************
Grundlagen
==========
Flächenmomente und der Schwerpunkt
----------------------------------
Der Flächeninhalt A einer beliebige Fläche (Flächtenmoment 0. Ordnung) kann berechnet werden nach

.. math::
    :label: flaeche

    A = \int_A~dA

die statischen Momente (Flächenmomente 1. Ordnung) bezogen auf einen beliebigen Punkt uf einen beliebigen Punkt P nach 

.. math::
    :label: statischen_momente

    S_{p,y} = \int_A (y-y_p)~dA \quad S_{p,z} = \int_A (z-z_p)~dA

(todo)
Herleitung Schwerpunkt

und die Flächenträgheitsmomente bzw. Deviationsmomemnte (Flächenmomente 2. Ordnung) bezogen auf einen beliebigen Punkt P nach

.. math::
    :label: flaechenträgheitsmomente

    I_{p,yy} = \int_A (y-y_p)^2~dA \quad I_{p,zz} = \int_A (z-z_p)^2~dA \quad I_{p,yz} = I_{p,zy} = \int_A (y-y_p)(z-z_p)~dA

Die Flächenträgheitsmomente lassen sich in einer symmetrischen Matrix anordnen, wobei das resultierende Eigenwertproblem zu den Hauptträgheitsmomenten führt.

.. math::
    :label: eigenwert

    \mathbf{I_p}&=\left[\begin{array}{cc}
                            I_{p,yy}&I_{p,yz}\\
                            I_{p,zy}&I_{p,zz}
                            \end{array}
                            \right] \qquad \det{\mathbf{I_p}} - \text{diag}(\lambda) &= 0 

Diskretisierung für zusammengesetzte R-Querschnitte
---------------------------------------------------
Zur Berechnung der Querschnittswerte in einer Vektorschreibweise werden die Höhen, Breiten sowie Abstände der Schwerpunkte zu einem beliebigen Bezugspunkt in y- und z-Richtung 
in eigene Vektoren :math:`\vec{h}`, :math:`\vec{b}`, :math:`\vec{z_{si}}` und :math:`\vec{y_{si}}` zusammengefasst.

.. math::
    :label: querschnittswerte

    \vec{A} &= \vec{b} \odot \vec{h}\qquad
    A = \vec{b} \cdot \vec{h}\\[1em]
    z_s &= \frac{\vec{A} \cdot \vec{z_{si}}}{\vec{b} \cdot \vec{h}}=\frac{\vec{A} \cdot \vec{z_{si}}}{A}\qquad
    y_s = \frac{\vec{A} \cdot \vec{y_{si}}}{\vec{b} \cdot \vec{h}}=\frac{\vec{A} \cdot \vec{y_{si}}}{A}\\[1em]
    I_y &= \frac{\vec{b}\cdot\vec{h}^{\circ 3}}{12}+\vec{A}\cdot\vec{z_{si}}-z_s\cdot\vec{A}\cdot\vec{z_{si}}\qquad
    I_z = \frac{\vec{b}^{\circ 3}\cdot\vec{h}}{12}+\vec{A}\cdot\vec{y_{si}}-z_s\cdot\vec{A}\cdot\vec{y_{si}}\\[1em]

.. Note:: 
    todo: Abbildungen, Definition der Vektoren h und b, erkläre Hadamard-Produkt, Deviationsmomente, Hauptträgheitsmomente

Querschnittswerte
=================

(todo: Abbildungen)

R - Querschnitt
---------------
::

    import numpy as np
    import stanpy as stp

    b = 0.2 
    h = 0.4
    
    cs_props = stp.cs(b=b,h=h)

I - Querschnitt
---------------
(todo)

H - Querschnitt
---------------
(todo)

U - Querschnitt
---------------
(todo)

Kasten - Querschnitt
--------------------
(todo)

Kreis - Querschnitt
-------------------
(todo)

Querschnitte nach Norm
----------------------
(todo: HEB, HEA, IPE, ...)

Zusammengesetzter R-Querschnitt Allgemein
=========================================
Für zusammengesetzte Rechtecksquerschnitte müssen die jeweiligen Breiten, Höhen 
sowie Schwerpunktsabstände in y- und z-Richtung in Listen oder Arrays zusammengefasst werden. (todo: Abbildung)
::

    import numpy as np
    import stanpy as stp

    b = np.array([b1,b2,b3])
    h = np.array([h1,h2,h3])
    zsi = np.array([zsi1,zsi2,zsi3])
    ysi = np.array([ysi1,ysi2,ysi3])
    
    cs_props = stp.cs(b,h,zsi,ysi)

Stäbe mit linear veränderlicher Höhe
====================================
Für die Berechnung der Querschnittswerte für Stäbe mit linear veränderlicher Höhe ist es nützlich die Vektoren :math:`\vec{z_{si}}` und :math:`\vec{y_{si}}` 
in Abhängigkeit der Vektoren :math:`\vec{h}` bzw. :math:`\vec{b}` anzuschreiben. 

I - Querschnitt
---------------
Für einen I-Querschnitt kann der Vektor :math:`\vec{z_{si}}` über eine Matrix Vektor Multiplikation, 
in Abhängigkeit von :math:`\vec{h}` errechnet werden. 
Für einen Bezugspunkt an der Oberkante oder einen Bezugspunkt im Schwerpunkt des Querschnitts ergeben sich:

.. math::
    :label: eq-i-querschnitt

    \vec{z_{si,OK}}=\left[\begin{array}{ccc}
                            1/2&0&0\\
                            1&1/2&0\\
                            1&1&1/2
                            \end{array}
                            \right]\cdot\vec{h} \quad \text{bzw.} \quad
    \vec{z_{si,SP}}=\left[\begin{array}{ccc}
                        -1/2&-1/2&0\\
                        0&0&0\\
                        0&1/2&1/2
                        \end{array}
                        \right]\cdot\vec{h}

Dabei sind die Ergebnisse, bis auf den Abstand zum Bezugspunkt, ident.

(todo Abbildung)::

    import numpy as np
    import stanpy as stp

    ha, hb, hc = 0.2, 0.3, 0.3, 0.4 # m 
    hx = ha+(hb-ha)/l*x # m 

    b = np.array([b1,b2,b3])
    h = np.array([h1,h2,h3])
    zsi_OK = np.array([1/2,0,0],
                   [1,1/2,0],
                   [1,1,1/2]).dot(h)

    zsi_SP = np.array([-1/2,-1/2,0],
                   [0,0,0],
                   [0,1/2,1/2]).dot(h)   

    cs_props_OK = stp.cs(b,h,zsi_OK) # Iy,zs, Iz, ys, Iyz, Iy_main, Iz_main, A
    cs_props_SP = stp.cs(b,h,zsi_SP) # Iy,zs, Iz, ys, Iyz, Iy_main, Iz_main, A

    (todo: prints)
 
H - Querschnitt
---------------
Analog zu dem I-Querschnitt kann der Vektor :math:`\vec{y_{si}}` über eine Matrix Vektor Multiplikation, 
in Abhängigkeit von :math:`\vec{b}` errechnet werden. 
Für einen Bezugspunkt am Linken Rand des H-Querschnitts ergibt sich:

.. math::
    :label: eq-h-querschnitt

    \vec{y_{si}}=\left[\begin{array}{ccc}
                            1/2&0&0\\
                            1&1/2&0\\
                            1&1&1/2
                            \end{array}
                            \right]\cdot\vec{b}

(todo Skizze)::

    import numpy as np
    import stanpy as stp

    b = np.array([b1,b2,b3])
    h = np.array([h1,h2,h3])
    ysi = np.array([1/2,0,0], 
                   [1,1/2,0],
                   [1,1,1/2])
                   .dot(b)   
    
    results = stp.QS(b=b,h=h,ysi=ysi) # Iy,zs, Iz, ys, Iyz, Iy_main, Iz_main, A

Kasten - Querschnitt
--------------------
Für Kastenquerschnitte ergibt sich die Matrix Vektor Multiplikation analog zu :eq:`eq-i-querschnitt` und :eq:`eq-h-querschnitt`.

.. math::
    :label: eq-kasten-querschnitt

    \vec{z_{si}}=\left[\begin{array}{cccc}
                            1/2&0&0&0\\
                            1&1/2&0&0\\
                            1&0&1/2&0\\
                            1&0&1&1/2
                            \end{array}
                            \right]\cdot\vec{h} \qquad
    \vec{y_{si}}=\left[\begin{array}{cccc}
                            1/2&0&0&0\\
                            0&1/2&0&0\\
                            1&0&-1/2&0\\
                            0&0&0&1/2
                            \end{array}
                            \right]\cdot\vec{b}

(todo Skizze)::

    import numpy as np
    import stanpy as stp

    b = np.array([b1,b2,b3])
    h = np.array([h1,h2,h3])

    zsi = np.array([1/2,0,0,0], # Obergurt
                   [1,1/2,0,0], # Steg links
                   [1,0,1/2,0], # Steg rechts
                   [1,0,1,1/2]) # Untergrut
                   .dot(h)   
    
    ysi = np.array([1/2,0,0,0], # Obergurt
                   [0,1/2,0,0], # Steg links
                   [1,0,-1/2,0], # Steg rechts
                   [0,0,0,1/2]) # Untergrut
                   .dot(b)   

    results = stp.QS(b,h,zsi,ysi) # Iy,zs, Iz, ys, Iyz, Iy_main, Iz_main, A

I - Querschnitt (verstärkt)
---------------------------
(todo Abbildungen)::

    (todo: still a placeholer)
    import numpy as np
    import stanpy as stp

    b = np.array([b1,b2,b3,b4])
    h = np.array([h1,h2,h3,h4])

    zsi = np.array([1/2,0,0,0], # Obergurt
                   [1,1/2,0,0], # Steg links
                   [1,0,1/2,0], # Steg rechts
                   [1,0,1,1/2]) # Untergrut
                   .dot(h)   
    
    ysi = np.array([1/2,0,0,0], # Obergurt
                   [0,1/2,0,0], # Steg links
                   [1,0,-1/2,0], # Steg rechts
                   [0,0,0,1/2]) # Untergrut
                   .dot(b)   

    results = stp.QS(b,h,zsi,ysi) # Iy,zs, Iz, ys, Iyz, Iy_main, Iz_main, A

.. meta::
    :description lang=de:
        Examples of document structure features in pydata-sphinx-theme.