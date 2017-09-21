#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#from __future__ import absolute_import, print_function

import io
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

# print(find_packages())
# 1 / 0

setup(
    name='actleto',
    version='0.1.0',
    description='Toolbox for rapid dataset creation and classifier training with active machine learning',
    author='ISA RAS',
    author_email='',
    license='MIT',
    python_requires='>=3.5',
    packages=find_packages(),
#     package_dir={'': 'actleto'},
    include_package_data=True,
    keywords='development active machine learning annotation corpus',
    zip_safe=False,
    package_dir={'examples': 'examples'},
    package_data={'examples': ['*.ipynb']},
    install_requires=['cython', 
                      'numpy>=1.12.1',
                      'pandas>=0.20.1',
                      'scikit-learn>=0.18',
                      'scipy>=0.19.0',
                      'Pillow>=4.2.1',
                      'ipywidgets>=4',
                      'annoy'],
    dependency_links=['git+https://github.com/windj007/libact/#egg=libact']
)
