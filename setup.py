#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#from __future__ import absolute_import, print_function

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    return io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ).read()


setup(
    name='jupyter_al_annotator',
    version='0.1.0',
    description='Widget for active learning annotation in Jupyter IDE.',
    author='ISA RAS',
    author_email='',
    license='MIT',
    python_requires='>=3.5',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    keywords='development active learning machine learning annotation corpus',
    zip_safe=False,
    install_requires=[
        'sklearn', 'numpy', 'pandas', 
        'pillow', 'scipy'
    ],
    dependency_links=['git+https://github.com/windj007/libact']
)
