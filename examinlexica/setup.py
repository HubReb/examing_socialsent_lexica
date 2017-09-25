#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Run this setup to use examinlexica. 

If setup fails, check if
    - setup_requires package are installed
    - setuptools is installed on your system

If setup still fails, install packages (install_requires) yourself and either
set PYTHONPATH variable or run setup script again.


If you change any python code in this package, make sure to run this setup script
afterward!
'''

try:
    from setuptools import setup
except:
    from distutils.core import setup

setup(name='examinlexica',
      version='1.0',
      url='https://gitlab.cl.uni-heidelberg.de/hubert/examining_socialsent_lexica',
      author='Rebekka Hubert',
      author_email='hubert@cl.uni-heidelberg.de',
      packages=[
          'examinlexica',
          'examinlexica.clusteredData',
          'examinlexica.evaluate',
          'examinlexica.original'
      ],
      package_data={'examinlexica' : ['historical/*.tsv', 'subreddits/*.tsv']},
      setup_requires=['numpy', 'scipy'],
      install_requires=['hdbscan', 'matplotlib', 'pandas', 'scikit-learn', 'cython'],
     )
