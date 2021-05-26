import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='d3m_interface',
    version='0.2.6',
    author='Roque Lopez',
    author_email='rlopez@nyu.edu',
    description='Library to use D3M AutoML Systems',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://gitlab.com/ViDA-NYU/d3m/d3m_interface',
    packages=setuptools.find_packages(),
    license='Apache-2.0',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'd3m-automl-rpc @ git+https://gitlab.com/datadrivendiscovery/automl-rpc.git@dev-dist-python',
        'd3m @ git+https://gitlab.com/datadrivendiscovery/d3m.git@b72cdf5ed55de775b2553a86179fc1db6ba09c09',
        'pandas==1.1.0',
        'numpy==1.18.2',
        'scikit-learn==0.22.2',
        'lime==0.2.0.1',
        'pipelineprofiler==0.1.16',
        'datamart_profiler==0.8.1',
        'data-profile-viewer==0.2.5',
        'visual-text-explorer==0.1.9',
        'dataclasses==0.7',
    ],
    python_requires='>=3.6',
    include_package_data=True,
    package_data={'d3m_interface': ['resource/*.json']}
)
