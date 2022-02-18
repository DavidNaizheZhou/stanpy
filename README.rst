.. image:: https://img.shields.io/pypi/v/stanpy.svg
        :target: https://pypi.python.org/pypi/stanpy

.. image:: https://readthedocs.org/projects/stanpy/badge/?version=latest
        :target: https://stanpy.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

========
Overview
========

Structural analysis libary in python

* Free software: MIT license

Installation
============

::

    pip install stanpy

You can also install the in-development version with::

    pip install git+ssh://git@https://github.com/davidnaizhezhou/stanpy@main

Documentation
=============


https://python-stanpy.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
