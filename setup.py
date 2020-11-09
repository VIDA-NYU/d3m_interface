import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="d3m_interface",
    version="0.1.18",
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
        'ta3ta2-api==2020.6.2',
        'd3m==2020.5.18',
        'pandas==1.0.3',
        'pipelineprofiler==0.1.15',
        'data-profile-viewer==0.2.0',
        'visual-text-explorer==0.1.2',
        'datamart_profiler==0.6.2'
    ],
    python_requires='>=3.6',
)
