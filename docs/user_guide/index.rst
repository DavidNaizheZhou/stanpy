==========
User Guide
==========

Verwende stanpy in einem Projekt mit::

    import stanpy as stp

Was ist stanpy?
---------------
Stanpy ist ein fundamentales Package der Baustatik.
Es soll Ingenieure unterstützen baustatische Problemstellungen zu untersuchen.
Ziel ist es ein möglichst intuitives und fehlerresistentes bearbeiten der Problemstellungen nach traditionellen 
Lösunsansätzen zu ermöglichen. Das Package soll modular aufgebaut sein, sodass es dem Detailierungsgrad einer
Handstatik entsprechen kann (stanpy ist keine Blackbox).

In einer ersten Phase werden folgende Features dokumentiert, getestet und implementiert:

* Berechnung von Querschnittswerten
* Übertragungsbeziehungen nach Theorie I Ordnung
* Übertragungsbeziehungen mit veränderlicher Querschnittshöhe nach Theorie I Ordnung
* Übertragungsbeziehungen nach Theorie II Ordnung
* Drehwinkelverfahren nach Theorie I Ordnung
* Drehwinkelverfahren mit veränderlicher Querschnittshöhe  nach Theorie I Ordnung
* Drehwinkelverfahren nach Theorie II Ordnung

Todo's:

* Darstellung von Schnittgrößen
* Darstellung von Verformungen
* Kinematischer Verschiebungsplan/Geklappter Verschiebungsplan
* Kraftgrößenverfahren
* Dreimomentengleichung (mit Erweiterungen – Stabsehendrehwinkel, Auflagerverschiebungen)
* Momentenfortleitung
* Prinzip der virtuellen Verschiebungen
* Allgemeines Verschiebungsgrößenverfahren! 
* Fließgelenktheorie I. Ordnung
* Traglastermittlung mit Hilfe der Traglastsätze
* …



Inhaltsverzeichnis
------------------

.. toctree::
    :maxdepth: 2
    :caption: User Guide

    cross_section
    cross_section_height


.. meta::
    :description lang=en:
        A collection of examples for demonstrating the features of the stanpy.