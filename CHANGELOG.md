Changelog
=========
0.9.0 (2022-05-20)
-------------------
* Add support for raw datasets of text, image, audio, and video.
* Add support for raw time-series classification datasets.


0.8.0 (2022-05-05)
-------------------
* Add support for raw time-series forecasting datasets.
* Add support for raw clustering datasets.
* Allow loading primitive types for PipelineProfiler.
* Allow using relative paths.
* Use an example with csv datasets in 'Getting started' section.


0.7.1 (2022-03-29)
-------------------
* Unlock d3m and d3m-automl-rpc dependencies.


0.7.0 (2022-03-22)
-------------------
* Allow users to specify the directory for static files.


0.6.0 (2022-03-21)
--------------------
* Read version from \_\_init\_\_.
* Remove "Indices and tables" section from documentation.
* Verify if image is downloaded before running the search.
* Fix multiple issues about documentation.


0.5.0 (2022-02-18)
-------------------
* Added verbose parameter to show/hide logs from AutoMLs.
* Added CHANGELOG file.
* Added instructions for releases.
* Add destructor to runtime objects.
* Add support to load static_files folder
* Add new examples for text, image and time-series tasks.
* Avoid errors when imported from non-jupyter environments.


0.4.1 (2022-02-11)
------------------
* Run automl engines without executing a container.
* Pick port at random with Docker.
* Use unique names for Docker containers.


0.4.0 (2022-01-04)
------------------
* Added LocalRuntime to run in the same container.
* Test multiple Python versions on CI.
* Added a CI.
* Added support to to blacklist and whitelist primitives
* Change devel version to a PEP-440.


0.3 (2021-06-03)
----------------
* Added support for Singularity.
* Implemented classes for different runtime modes.
* Fixed try-except eating KeyboardInterrupt.


0.2 (2021-01-29)
----------------
* Added a method to plot leaderboard.
* Added support for SaveFittedSolution method.
* Calculate confidence in pipelines.
* Reorganized installation section.


0.1.0 (2020-06-02)
------------------
* First release on PyPI.