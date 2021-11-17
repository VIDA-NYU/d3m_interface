#!/usr/bin/env python3
import os

import nbconvert.preprocessors
import nbformat
import sys


if __name__ == '__main__':
    notebook_file, = sys.argv[1:]

    # Read notebook
    with open(notebook_file) as fp:
        nb = nbformat.read(fp, as_version=4)

    # Insert a cell that sets up logging (will only work on POSIX)
    nb['cells'].insert(0, nbformat.NotebookNode({
        'cell_type': 'code',
        'execution_count': 1,
        'metadata': {},
        'source': (
            'import logging\n'
            + 'logging.basicConfig(level=logging.INFO, stream=open("/dev/stderr", "w"))'
        ),
    }))

    # Execute the notebook
    ep = nbconvert.preprocessors.ExecutePreprocessor(timeout=3600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': os.getcwd()}})
