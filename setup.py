import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="d3m_interface",
    version="0.1.3",
    author="Roque Lopez",
    author_email="rlopez@nyu.edu",
    description="Library to use D3M AutoML Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/ViDA-NYU/d3m/d3m_interface",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'gitdb2==2.0.6',
        'grpcio',
        'grpcio-tools',
        'ta3ta2-api==2020.6.2',
        'd3m==2020.5.18'
    ],
    python_requires='>=3.6',
)