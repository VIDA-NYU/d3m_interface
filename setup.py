import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="d3m_interface",
    version="0.1.16",
    author="Roque Lopez",
    author_email="rlopez@nyu.edu",
    description="Library to use D3M AutoML Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/ViDA-NYU/d3m/d3m_interface",
    packages=setuptools.find_packages(),
    license='Apache-2.0',
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: Apache Software License',
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'ta3ta2-api @ git+https://gitlab.com/datadrivendiscovery/ta3ta2-api.git@1c06dc505ad517becde0866e1aa9925513ad599c',
        'd3m @ git+https://gitlab.com/datadrivendiscovery/d3m.git@devel#egg=d3m',
        'pandas==1.0.3',
        'SQLAlchemy==1.2.16',
        'pipelineprofiler==0.1.15',
        'data-profile-viewer==0.2.0',
        'datamart_profiler==0.6.2'
    ],
    python_requires='>=3.6',
)
