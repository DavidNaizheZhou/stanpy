
***************************************************
Querschnittswerte für Stäbe mit veränderlicher Höhe
***************************************************
.. Note:: 
    todos: Abbildungen, Definition der Vektoren h und b, Struktur

In diesem Abschnitt werden die Querschnittswerte für verschieden zusammengesetzte Rechtecksquerschnitte mit veränderlicher Höhe berechnet.

Rechtecksquerschnitte mit veränderlicher Höhe
=============================================
::

    import numpy as np
    import stanpy as stp

    b, ha, hb, hc = 0.2, 0.3, 0.3, 0.4 # m 
    hx = ha+(hb-ha)/l*x # m 
    
    cs_pros = stp.cs(b, hx)
    

Zusammengesetzte Rechtecksquerschnitte mit veränderlicher Höhe
==============================================================
Für zusammengesetzte Rechtecksquerschnitte müssen die jeweiligen Breiten, Höhen 
sowie Schwerpunktsabstände in y- und z-Richtung in Listen oder Arrays zusammengefasst werden. 
::

    import numpy as np
    import stanpy as stp

    ha, hb, hc = 0.2, 0.3, 0.3, 0.4 # m 
    hx = ha+(hb-ha)/l*x # m 

    b = np.array([b1,b2,b3])
    h = np.array([h1,hx,h3])
    zsi = np.array([zsi1,zsi2,zsi3])
    ysi = np.array([ysi1,ysi2,ysi3])
    
    cs_props = stp.cs(b,h,zsi,ysi)


Matrix Vektor Notation 
======================
In manchen Fällen kann es nützlich sein die Vektoren :math:`\vec{z_{si}}` und :math:`\vec{y_{si}}` 
in Abhängigkeit der Vektoren :math:`\vec{h}` und :math:`\vec{b}` anzuschreiben. 
Beispielsweise muss für einen Stab mit linear veränderlicher Höhe lediglich ein, von x Abhängiger, Eintrag
im Vektor :math:`\vec{h}` eingetragen werden. Analog gilt das auch für einen Stab mit linear veränderlicher Breite.

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

Verstärkter - I Querschnitt
---------------------------
(todo Skizze)::

    (todo: still a placeholer)
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

.. meta::
    :description lang=de:
        Examples of document structure features in pydata-sphinx-theme.