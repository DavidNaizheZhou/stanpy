
*****************
Querschnittswerte
*****************

Doppelsymmetrischer Querschnitt
===============================
In diesem Beispiel werden die Querschnittswerte für einen doppelsymmetrischen Querschnitt mittels den stanpy Funktionen berechnet.
Dafür werden als Inputgrößen Vektoren vom Datentyp np.array übergeben.
Dabei beschreiben die Komponenten von :math:`\vec{b}` die Breiten und die Komponenten von :math:`\vec{h}` die Höhen der einzelnen Querschnitte. 
Konkret können die Querschnittswerte für einen doppelsymmetrischen I-Querschnitt wie folgt berechnet werden (todo Skizze)::

    import numpy as np
    import stanpy as stp

    b = np.array([b1,b2,b3])
    h = np.array([h1,h2,h3])
    
    results = stp.QS(b,h)

Zusammengesetzter Querschnitt
=============================

In diesem Beispiel werden die Querschnittswerte für einen zusammengesetzen Querschnitt mittels den stanpy Funktionen berechnet.
Dafür werden als Inputgrößen Vektoren vom Datentyp np.array übergeben.
Die Komponenten von :math:`\vec{b}` die Breiten und die Komponenten von :math:`\vec{h}` die Höhen der einzelnen Querschnitte. 
Die Vektoren :math:`\vec{z_{si}}` und :math:`\vec{y_{si}}` beschreiben die Abstände der Schwerpunkte zu einem beliebig wählbaren  
Konkret können die Querschnittswerte für einen zusammengestzen I-Querschnitt wie folgt berechnet werden (todo Skizze)::

    import numpy as np
    import stanpy as stp

    b = np.array([b1,b2,b3])
    h = np.array([h1,h2,h3])
    zsi = np.array([zsi1,zsi2,zsi3])
    ysi = np.array([ysi1,ysi2,ysi3])
    
    results = stp.QS(b,h,zsi,ysi)

.. meta::
    :description lang=de:
        Examples of document structure features in pydata-sphinx-theme.