Installation
============

This package works with Python 3.6 through 3.8 in Linux, Mac and Windows.

You can install the latest stable version of this library directly from `PyPI <https://pypi.org/project/d3m-interface/>`__
using PIP:

::

    $ pip install d3m-interface


AutoML systems via containers
------------------------------

For this option, you need to have `Docker <https://docs.docker.com/get-docker/>`__ or `Singularity <https://sylabs.io/guides/3.5/user-guide/introduction.html>`__
installed on your operating system.

Everything you need to deploy, `d3m-interface`
will simply run on the Docker/Singularity engine as a container. You can see
`here <https://gitlab.com/ViDA-NYU/d3m/d3m_interface/-/blob/master/d3m_interface/automl_interface.py#L61>`__ how
this container is set up to deploy different AutoML systems.

Note that `d3m-interface` uses the pre-built Docker images of the D3M AutoML systems. For AlphaD3M, you
can see `here <https://gitlab.com/ViDA-NYU/d3m/alphad3m/-/blob/master/Dockerfile>`__ how the Docker image is built.
For the other D3M AutoML systems, you can find more information :doc:`here <automls_supported>`.


Once the installation is completed, you need to pull manually the Docker image of the D3M AutoML system.

For AlphaD3M run:

::

   $ # for docker
   $ docker pull registry.gitlab.com/vida-nyu/d3m/alphad3m:latest

or

::

   $ # for singularity
   $ singularity pull docker://registry.gitlab.com/vida-nyu/d3m/alphad3m:latest


AutoML systems via PyPI
------------------------
For this option, you don't need to install neither Docker nor Singularity.  Once the installation of `d3m-interface`
is completed, you need to install the PyPI version of the AutoML system and primitives.

For AlphaD3M run:

::

     $ pip install alphad3m


To install the primitives available on PyPI, run this command:

::

     $ pip install d3m-common-primitives d3m-sklearn-wrap dsbox-corex dsbox-primitives sri-d3m distil-primitives d3m-esrnn d3m-nbeats --no-binary pmdarima


Currently, this version has support for classification, regression and forecasting tasks (using a limited set of primitives).
It supports tabular, text and image data types. This package works with Python 3.8 in Linux and Mac.

On non-Linux platforms, you will need `swig` to compile pyrfr. You can obtain swig from `homebrew <https://formulae.brew.sh/formula/swig@3>`__, `anaconda <https://anaconda.org/anaconda/swig>`__, or `the swig website <http://www.swig.org/download.html>`__.
