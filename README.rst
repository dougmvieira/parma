========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/parma/badge/?style=flat
    :target: https://readthedocs.org/projects/parma
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/dougmvieira/parma.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/dougmvieira/parma

.. |codecov| image:: https://codecov.io/github/dougmvieira/parma/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/dougmvieira/parma

.. |version| image:: https://img.shields.io/pypi/v/parma.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/parma

.. |commits-since| image:: https://img.shields.io/github/commits-since/dougmvieira/parma/v0.1.1.svg
    :alt: Commits since latest release
    :target: https://github.com/dougmvieira/parma/compare/v0.1.1...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/parma.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/parma

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/parma.svg
    :alt: Supported versions
    :target: https://pypi.org/project/parma

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/parma.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/parma


.. end-badges

Multivariate polyharmonic spline interpolation

* Free software: MIT license

Installation
============

::

    pip install parma

Documentation
=============


https://parma.readthedocs.io/


Development
===========

To run the all tests run::

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
