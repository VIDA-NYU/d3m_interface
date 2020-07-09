D3M Interface Library
=====================
Library to use D3M AutoML Systems. This repository contains an implementation to integrate 
 D3M TA2s with Jupyter Notebooks using the TA3-TA2 API. 

[Documentation is available here](https://d3m-interface.readthedocs.io/en/latest/)

## Installation
This package works with Python 3.6. You need to have [Docker](https://docs.docker.com/get-docker/) installed on your operating system.
Since it uses the D3M core package, you also need to install libcurl4-openssl-dev (for Debian/Ubuntu).

You can install the latest stable version of this library from [PyPI](https://pypi.org/project/d3m-interface/):

```
$ pip3 install d3m-interface
```

To install the latest development version:

```
$ pip3 install git+https://gitlab.com/ViDA-NYU/d3m/d3m_interface.git
```


Getting `ImportError: pycurl`? 
See this [page](https://gitlab.com/ViDA-NYU/d3m/d3m_interface/-/wikis/Pycurl-problem).