Installation
============

Linux, Mac and Windows
----------------------

This package works with Python 3.6 through 3.8. You need to have `Docker <https://docs.docker.com/get-docker/>`__
installed on your operating system.

You can install the latest stable version of this library directly from `PyPI <https://pypi.org/project/d3m-interface/>`__
using PIP (only for Linux and Mac):

::

    $ pip install d3m-interface

To install the latest development version (for Linux, Mac and Windows):

::

    $ pip install git+https://gitlab.com/ViDA-NYU/d3m/d3m_interface.git@devel#egg=d3m_interface


After the installation on Windows, you need to download manually the Docker image of the D3M AutoML system. You can
download it for AlphaD3M using:

::

    $ docker pull registry.gitlab.com/vida-nyu/d3m/alphad3m:master

How Docker is Used in D3M Interface
-----------------------------------

Docker creates containers instead of full-blown virtual machines. So, everything you need to deploy, `d3m-interface`
will simply run on the Docker engine as a container. You can see
`here <https://gitlab.com/ViDA-NYU/d3m/d3m_interface/-/blob/master/d3m_interface/automl_interface.py#L561>`__ how
this container is set up to deploy different AutoML systems.

Note that `d3m-interface` uses the pre-built Docker images (latest version) of the D3M AutoML systems. For AlphaD3M, you
can see `here <https://gitlab.com/ViDA-NYU/d3m/alphad3m/-/blob/master/Dockerfile>`__ how the Docker image is built.
For the other D3M AutoML systems, you can find more information :doc:`here <automls_supported>`.
